"""The reward Colab bootstrap orchestrates download -> ts2-reward --resume as one
idempotent command. These tests drive that orchestration against fixture
artifacts with an injected/monkeypatched download (no network), verifying the
wiring and the skip-on-warm-VM behavior the real run depends on.
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from tinystories_v2.judge import SlotCoverageFakeJudge, judge_with_order_swap
from tinystories_v2.slot_prompt import SLOT_FIELDS
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run
from tinystories_v2.data import run as data_run

# scripts/ is not an importable package; load the bootstrap module by path.
_spec = importlib.util.spec_from_file_location(
    "reward_colab", Path(__file__).parent.parent / "scripts" / "reward_colab.py")
boot = importlib.util.module_from_spec(_spec)
sys.modules["reward_colab"] = boot
_spec.loader.exec_module(boot)

_BLAND = ["A plain note with nothing much to say.",
          "Some words that go nowhere in particular."]


@pytest.fixture
def hub_and_config(tmp_path, fixture_path):
    """Build a real tokenizer + a separable fake-Judge pairs.jsonl (the 'Hub'
    source the bootstrap downloads from) and a reward config pointing at local
    artifact paths that do not exist yet, so prepare() must download."""
    hub = tmp_path / "hub"
    tokenizer_run({"out_dir": str(hub / "tokenizer"), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    data_run({
        "out_dir": str(hub / "data"), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.1, "sft": 0.7,
                   "pref": 0.1, "eval": 0.1},
    })
    with open(hub / "data" / "splits" / "sft.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    judge = SlotCoverageFakeJudge()
    pairs_src = hub / "pairs.jsonl"
    with pairs_src.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            scaffold = Scaffold(**{fld: row[fld] for fld in SLOT_FIELDS})
            chosen = (f"{scaffold.character}, a {scaffold.trait} one in "
                      f"{scaffold.setting}, met {scaffold.conflict}. "
                      f"{scaffold.resolution}. The moral: {scaffold.moral}.")
            pair = judge_with_order_swap(judge, scaffold, chosen, _BLAND[i % len(_BLAND)])
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

    art = tmp_path / "artifacts"
    tokenizer_dst = art / "tokenizer_full" / "tokenizer.json"
    pairs_dst = art / "pref_full" / "pairs.jsonl"
    reward_cfg = tmp_path / "reward.toml"
    reward_cfg.write_text(
        f'out_dir = "{art / "reward_full"}"\n\n'
        f'[data]\npairs_path = "{pairs_dst}"\n'
        f'tokenizer_path = "{tokenizer_dst}"\n', encoding="utf-8")

    sources = {
        (boot.TOKENIZER_REPO, "tokenizer.json"): hub / "tokenizer" / "tokenizer.json",
        (boot.PAIRS_REPO, boot.PAIRS_FILENAME): pairs_src,
    }
    return {"reward_cfg": reward_cfg, "tokenizer_dst": tokenizer_dst,
            "pairs_dst": pairs_dst, "sources": sources}


def _fake_download(sources, calls=None):
    def download(repo_id, filename, local_dir):
        if calls is not None:
            calls.append((repo_id, filename))
        dst = Path(local_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[(repo_id, filename)], dst)
        return dst
    return download


def test_prepare_downloads_tokenizer_and_pairs(hub_and_config):
    calls = []
    pairs = boot.prepare(hub_and_config["reward_cfg"],
                         download=_fake_download(hub_and_config["sources"], calls))
    assert {filename for _, filename in calls} == {"tokenizer.json", boot.PAIRS_FILENAME}
    assert pairs == hub_and_config["pairs_dst"]
    assert hub_and_config["tokenizer_dst"].exists() and pairs.exists()


def test_prepare_is_idempotent_on_a_warm_vm(hub_and_config):
    boot.prepare(hub_and_config["reward_cfg"],
                 download=_fake_download(hub_and_config["sources"]))

    def boom(*args, **kwargs):
        raise AssertionError("download must not run when artifacts already exist")

    boot.prepare(hub_and_config["reward_cfg"], download=boom)  # no download call


def test_main_skip_train_prepares_without_training(hub_and_config, monkeypatch):
    monkeypatch.setattr(boot, "download_file", _fake_download(hub_and_config["sources"]))
    trained = []
    monkeypatch.setattr(boot.reward, "run", lambda *a, **k: trained.append(True))
    boot.main(["--reward-config", str(hub_and_config["reward_cfg"]), "--skip-train"])
    assert hub_and_config["pairs_dst"].exists()
    assert trained == []


def test_main_trains_with_resume_after_prepare(hub_and_config, monkeypatch):
    monkeypatch.setattr(boot, "download_file", _fake_download(hub_and_config["sources"]))
    resume_flags = []

    def fake_run(config, resume=False):
        resume_flags.append(resume)
        return {"step": 1, "loss": 0.5, "heldout_accuracy": 0.9}

    monkeypatch.setattr(boot.reward, "run", fake_run)
    boot.main(["--reward-config", str(hub_and_config["reward_cfg"])])
    assert hub_and_config["pairs_dst"].exists()
    assert resume_flags == [True]  # ts2-reward invoked with resume=True
