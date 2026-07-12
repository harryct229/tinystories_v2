"""The eval Colab bootstrap orchestrates download -> ts2-eval as one idempotent
command. Driven against fixture artifacts with an injected/monkeypatched
download (no network)."""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

from tinystories_v2.data import run as data_run
from tinystories_v2.tokenizer import run as tokenizer_run

# scripts/ is not an importable package; load the bootstrap module by path.
_spec = importlib.util.spec_from_file_location(
    "eval_colab", Path(__file__).parent.parent / "scripts" / "eval_colab.py")
boot = importlib.util.module_from_spec(_spec)
sys.modules["eval_colab"] = boot
_spec.loader.exec_module(boot)


@pytest.fixture
def hub_and_config(tmp_path, fixture_path):
    """Build a 'Hub' with a tokenizer + a data repo (eval split), and an eval
    config pointing at local artifact paths that do not exist yet."""
    hub = tmp_path / "hub"
    tokenizer_run({"out_dir": str(hub / "tokenizer"), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    data_run({
        "out_dir": str(hub / "data"), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.1, "sft": 0.6,
                   "pref": 0.1, "eval": 0.2},
    })

    art = tmp_path / "artifacts"
    tokenizer_dst = art / "tokenizer_full" / "tokenizer.json"
    eval_dst = art / "data_prep_full" / "splits" / "eval.jsonl"
    eval_cfg = tmp_path / "eval.toml"
    eval_cfg.write_text(
        f'out_dir = "{art / "eval_full"}"\n\n'
        f'[data]\neval_split = "{eval_dst}"\n'
        f'tokenizer = "{tokenizer_dst}"\n', encoding="utf-8")

    sources = {
        (boot.TOKENIZER_REPO, "tokenizer.json"): hub / "tokenizer" / "tokenizer.json",
        (boot.DATA_REPO, boot.EVAL_FILENAME): hub / "data" / "splits" / "eval.jsonl",
    }
    return {"eval_cfg": eval_cfg, "tokenizer_dst": tokenizer_dst,
            "eval_dst": eval_dst, "sources": sources}


def _fake_download(sources, calls=None):
    def download(repo_id, filename, local_dir):
        if calls is not None:
            calls.append((repo_id, filename))
        dst = Path(local_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[(repo_id, filename)], dst)
        return dst
    return download


def test_prepare_downloads_tokenizer_and_eval_split(hub_and_config):
    calls = []
    tokenizer_path, eval_path = boot.prepare(
        hub_and_config["eval_cfg"],
        download=_fake_download(hub_and_config["sources"], calls))
    assert {filename for _, filename in calls} == {"tokenizer.json", boot.EVAL_FILENAME}
    assert tokenizer_path == hub_and_config["tokenizer_dst"]
    assert eval_path == hub_and_config["eval_dst"]
    assert tokenizer_path.exists() and eval_path.exists()


def test_prepare_is_idempotent_on_a_warm_vm(hub_and_config):
    boot.prepare(hub_and_config["eval_cfg"],
                 download=_fake_download(hub_and_config["sources"]))

    def boom(*args, **kwargs):
        raise AssertionError("download must not run when artifacts already exist")

    boot.prepare(hub_and_config["eval_cfg"], download=boom)


def test_main_skip_eval_prepares_without_running(hub_and_config, monkeypatch):
    monkeypatch.setattr(boot, "download_file", _fake_download(hub_and_config["sources"]))
    ran = []
    monkeypatch.setattr(boot.eval, "run", lambda *a, **k: ran.append(True))
    boot.main(["--eval-config", str(hub_and_config["eval_cfg"]), "--skip-eval"])
    assert hub_and_config["eval_dst"].exists()
    assert ran == []


def test_main_runs_eval_after_prepare(hub_and_config, monkeypatch):
    monkeypatch.setattr(boot, "download_file", _fake_download(hub_and_config["sources"]))
    ran = []
    monkeypatch.setattr(boot.eval, "run", lambda *a, **k: ran.append(True) or {})
    boot.main(["--eval-config", str(hub_and_config["eval_cfg"])])
    assert hub_and_config["eval_dst"].exists()
    assert ran == [True]
