"""The Colab bootstrap orchestrates download -> ts2-sft-data -> ts2-sft --resume
as one idempotent command. These tests drive that orchestration against fixture
artifacts with an injected/monkeypatched download (no network), so they verify
the wiring and the skip-on-warm-VM behavior the real run depends on.
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import huggingface_hub
import pytest
import requests
from huggingface_hub.utils import RepositoryNotFoundError

from tinystories_v2.data import run as data_run
from tinystories_v2.tokenizer import run as tokenizer_run

# scripts/ is not an importable package; load the bootstrap module by path.
_spec = importlib.util.spec_from_file_location(
    "sft_colab", Path(__file__).parent.parent / "scripts" / "sft_colab.py")
boot = importlib.util.module_from_spec(_spec)
sys.modules["sft_colab"] = boot
_spec.loader.exec_module(boot)


@pytest.fixture
def hub_and_configs(tmp_path, fixture_path):
    """Build a real tokenizer + data-prep split (the 'Hub' source the bootstrap
    downloads from) and write sft_data + sft configs pointing at local artifact
    paths that do not exist yet, so prepare() must download then build."""
    hub = tmp_path / "hub"
    tokenizer_run({"out_dir": str(hub / "tokenizer"), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    data_run({
        "out_dir": str(hub / "data"), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.4, "sft": 0.4,
                   "pref": 0.1, "eval": 0.1},
    })
    art = tmp_path / "artifacts"
    tokenizer_dst = art / "tokenizer_full" / "tokenizer.json"
    split_dst = art / "data_prep_full" / "splits" / "sft.jsonl"
    examples_dst = art / "sft_data_full" / "examples.jsonl"

    sft_data_cfg = tmp_path / "sft_data.toml"
    sft_data_cfg.write_text(
        f'out_dir = "{art / "sft_data_full"}"\n'
        f'tokenizer = "{tokenizer_dst}"\n'
        f'sft_split = "{split_dst}"\n'
        f"max_examples = 0\n", encoding="utf-8")
    sft_cfg = tmp_path / "sft.toml"
    sft_cfg.write_text(
        f'out_dir = "{art / "sft_full"}"\n\n'
        f'[data]\nexamples_path = "{examples_dst}"\n'
        f'tokenizer_path = "{tokenizer_dst}"\n', encoding="utf-8")

    sources = {
        (boot.TOKENIZER_REPO, "tokenizer.json"): hub / "tokenizer" / "tokenizer.json",
        (boot.DATA_REPO, "splits/sft.jsonl"): hub / "data" / "splits" / "sft.jsonl",
    }
    return {"sft_data_cfg": sft_data_cfg, "sft_cfg": sft_cfg,
            "tokenizer_dst": tokenizer_dst, "split_dst": split_dst,
            "examples_dst": examples_dst, "sources": sources}


def _fake_download(sources, calls=None):
    def download(repo_id, filename, local_dir):
        if calls is not None:
            calls.append((repo_id, filename))
        dst = Path(local_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[(repo_id, filename)], dst)
        return dst
    return download


def test_prepare_downloads_then_builds_examples(hub_and_configs):
    calls = []
    examples = boot.prepare(
        hub_and_configs["sft_data_cfg"], hub_and_configs["sft_cfg"],
        download=_fake_download(hub_and_configs["sources"], calls))
    assert {filename for _, filename in calls} == {"tokenizer.json", "splits/sft.jsonl"}
    assert examples == hub_and_configs["examples_dst"]
    records = [json.loads(line) for line in
               examples.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records
    assert set(records[0]) == {"prompt_hash", "input_ids", "loss_mask", "n_prompt_tokens"}


def test_prepare_is_idempotent_on_a_warm_vm(hub_and_configs):
    boot.prepare(hub_and_configs["sft_data_cfg"], hub_and_configs["sft_cfg"],
                 download=_fake_download(hub_and_configs["sources"]))

    def boom(*args, **kwargs):
        raise AssertionError("download must not run when artifacts already exist")

    before = hub_and_configs["examples_dst"].stat().st_mtime_ns
    boot.prepare(hub_and_configs["sft_data_cfg"], hub_and_configs["sft_cfg"], download=boom)
    assert hub_and_configs["examples_dst"].stat().st_mtime_ns == before  # not rebuilt


def test_main_skip_train_prepares_without_training(hub_and_configs, monkeypatch):
    monkeypatch.setattr(boot, "download_file", _fake_download(hub_and_configs["sources"]))
    trained = []
    monkeypatch.setattr(boot.sft, "run", lambda *a, **k: trained.append(True))
    boot.main(["--sft-data-config", str(hub_and_configs["sft_data_cfg"]),
               "--sft-config", str(hub_and_configs["sft_cfg"]), "--skip-train"])
    assert hub_and_configs["examples_dst"].exists()
    assert trained == []


def test_main_trains_with_resume_after_prepare(hub_and_configs, monkeypatch):
    monkeypatch.setattr(boot, "download_file", _fake_download(hub_and_configs["sources"]))
    resume_flags = []

    def fake_run(config, resume=False):
        resume_flags.append(resume)
        return {"step": 1, "loss": 0.5}

    monkeypatch.setattr(boot.sft, "run", fake_run)
    boot.main(["--sft-data-config", str(hub_and_configs["sft_data_cfg"]),
               "--sft-config", str(hub_and_configs["sft_cfg"])])
    assert hub_and_configs["examples_dst"].exists()
    assert resume_flags == [True]  # ts2-sft invoked with resume=True


def test_download_file_falls_back_from_model_to_dataset_repo(tmp_path, monkeypatch):
    src = tmp_path / "src.txt"
    src.write_text("payload", encoding="utf-8")
    seen = []

    not_found = RepositoryNotFoundError("no such model repo", response=requests.Response())

    def fake_hub_download(*, repo_id, filename, repo_type, local_dir):
        seen.append(repo_type)
        if repo_type == "model":
            raise not_found
        dst = Path(local_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return str(dst)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hub_download)
    out = boot.download_file("owner/repo", "f.txt", tmp_path / "dst")
    assert out.read_text(encoding="utf-8") == "payload"
    assert seen == ["model", "dataset"]  # tried model first, then dataset
