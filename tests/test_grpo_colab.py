"""The GRPO Colab bootstrap orchestrates download -> ts2-grpo --resume as one
idempotent command. These tests drive that orchestration against fixture
artifacts with an injected/monkeypatched download (no network), verifying the
wiring and the skip-on-warm-VM behavior the real run depends on.
"""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

from tinystories_v2.data import run as data_run
from tinystories_v2.tokenizer import run as tokenizer_run

# scripts/ is not an importable package; load the bootstrap module by path.
_spec = importlib.util.spec_from_file_location(
    "grpo_colab", Path(__file__).parent.parent / "scripts" / "grpo_colab.py")
boot = importlib.util.module_from_spec(_spec)
sys.modules["grpo_colab"] = boot
_spec.loader.exec_module(boot)


@pytest.fixture
def hub_and_config(tmp_path, fixture_path):
    hub = tmp_path / "hub"
    tokenizer_run({"out_dir": str(hub / "tokenizer"), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    data_run({
        "out_dir": str(hub / "data"), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.1, "sft": 0.6,
                   "pref": 0.2, "eval": 0.1},
    })

    art = tmp_path / "artifacts"
    tokenizer_dst = art / "tokenizer_full" / "tokenizer.json"
    pref_dst = art / "data_prep_full" / "splits" / "pref.jsonl"
    grpo_cfg = tmp_path / "grpo.toml"
    grpo_cfg.write_text(
        f'out_dir = "{art / "grpo_full"}"\n\n'
        f'[data]\npref_split = "{pref_dst}"\n'
        f'tokenizer_path = "{tokenizer_dst}"\n', encoding="utf-8")

    sources = {
        (boot.TOKENIZER_REPO, "tokenizer.json"): hub / "tokenizer" / "tokenizer.json",
        (boot.DATA_REPO, boot.PREF_SPLIT_FILENAME): hub / "data" / "splits" / "pref.jsonl",
    }
    return {"grpo_cfg": grpo_cfg, "tokenizer_dst": tokenizer_dst,
            "pref_dst": pref_dst, "sources": sources}


def _fake_download(sources, calls=None):
    def download(repo_id, filename, local_dir):
        if calls is not None:
            calls.append((repo_id, filename))
        dst = Path(local_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[(repo_id, filename)], dst)
        return dst
    return download


def test_prepare_downloads_tokenizer_and_pref_split(hub_and_config):
    calls = []
    pref = boot.prepare(hub_and_config["grpo_cfg"],
                        download=_fake_download(hub_and_config["sources"], calls))
    assert {filename for _, filename in calls} == {"tokenizer.json", boot.PREF_SPLIT_FILENAME}
    assert pref == hub_and_config["pref_dst"]
    assert hub_and_config["tokenizer_dst"].exists() and pref.exists()


def test_prepare_is_idempotent_on_a_warm_vm(hub_and_config):
    boot.prepare(hub_and_config["grpo_cfg"],
                 download=_fake_download(hub_and_config["sources"]))

    def boom(*args, **kwargs):
        raise AssertionError("download must not run when artifacts already exist")

    boot.prepare(hub_and_config["grpo_cfg"], download=boom)


def test_main_skip_train_prepares_without_training(hub_and_config, monkeypatch):
    monkeypatch.setattr(boot, "download_file", _fake_download(hub_and_config["sources"]))
    trained = []
    monkeypatch.setattr(boot.grpo, "run", lambda *a, **k: trained.append(True))
    boot.main(["--grpo-config", str(hub_and_config["grpo_cfg"]), "--skip-train"])
    assert hub_and_config["pref_dst"].exists()
    assert trained == []


def test_main_trains_with_resume_after_prepare(hub_and_config, monkeypatch):
    monkeypatch.setattr(boot, "download_file", _fake_download(hub_and_config["sources"]))
    resume_flags = []

    def fake_run(config, resume=False):
        resume_flags.append(resume)
        return {"step": 1, "loss": 0.5, "reward_mean": 0.2, "kl": 0.01}

    monkeypatch.setattr(boot.grpo, "run", fake_run)
    boot.main(["--grpo-config", str(hub_and_config["grpo_cfg"])])
    assert hub_and_config["pref_dst"].exists()
    assert resume_flags == [True]  # ts2-grpo invoked with resume=True
