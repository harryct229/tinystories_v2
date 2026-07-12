import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from tinystories_v2.data import run as data_run
from tinystories_v2.dpo import run as dpo_run
from tinystories_v2.eval import load_stage_model
from tinystories_v2.judge import SlotCoverageFakeJudge, judge_with_order_swap
from tinystories_v2.model import FableLM
from tinystories_v2.pretrain import run as pretrain_run
from tinystories_v2.slot_prompt import SLOT_FIELDS
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run

TOY_MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 128, "ffn_hidden": 192}

_BLAND = ["A plain note with nothing much to say.",
          "Some words that go nowhere in particular."]


def _separable_pairs(rows):
    """Order-swap-consistent fake-Judge pairs where chosen mentions every slot
    value and rejected is bland — a learnable, separable preference signal."""
    judge = SlotCoverageFakeJudge()
    pairs = []
    for i, row in enumerate(rows):
        scaffold = Scaffold(**{f: row[f] for f in SLOT_FIELDS})
        chosen = (f"{scaffold.character}, a {scaffold.trait} one in "
                  f"{scaffold.setting}, met {scaffold.conflict}. "
                  f"{scaffold.resolution}. The moral: {scaffold.moral}.")
        pair = judge_with_order_swap(judge, scaffold, chosen, _BLAND[i % len(_BLAND)])
        assert pair is not None and pair.chosen == chosen
        pairs.append(pair)
    return pairs


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, fixture_path):
    base = tmp_path_factory.mktemp("dpo_stage_inputs")
    data_dir, tok_dir = base / "data", base / "tok"
    data_run({
        "out_dir": str(data_dir), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.1, "sft": 0.7,
                   "pref": 0.1, "eval": 0.1},
    })
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    with open(data_dir / "splits" / "sft.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    pairs = _separable_pairs(rows)
    pairs_path = base / "pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
    return {"pairs_path": str(pairs_path),
            "tokenizer": str(tok_dir / "tokenizer.json"), "n_pairs": len(pairs)}


def dpo_toy_config(out_dir, prepared, init_dir, model=None, **train_overrides) -> dict:
    train = {
        "steps": 60, "micro_batch_size": 8, "grad_accum": 1,
        "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
        "weight_decay": 0.0, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
        "precision": "fp32", "seed": 1337,
        "checkpoint_every": 20, "log_every": 1, "keep_last": 0,
    }
    train.update(train_overrides)
    return {
        "out_dir": str(out_dir),
        "model": dict(model or TOY_MODEL),
        "data": {"pairs_path": prepared["pairs_path"],
                 "tokenizer_path": prepared["tokenizer"]},
        "init": {"local_dir": str(init_dir)},
        "split": {"holdout_frac": 0.25, "seed": 20260712},
        "dpo": {"beta": 0.1},
        "train": train,
        "wandb": {"enabled": False},
    }


def read_metrics(out_dir) -> list[dict]:
    lines = (Path(out_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_toy_dpo_shifts_policy_toward_chosen(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = dpo_toy_config(tmp_path / "out", prepared, init)
    summary = dpo_run(config)
    assert summary["heldout_margin"] > 0.0        # policy prefers chosen over the reference
    metrics = read_metrics(config["out_dir"])
    assert len(metrics) == 60
    assert {"step", "loss", "lr", "margin", "pairs_seen"} <= metrics[0].keys()
    assert metrics[-1]["loss"] < metrics[0]["loss"]   # DPO loss fell


def test_manifest_records_beta_margin_and_split(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = dpo_toy_config(tmp_path / "out", prepared, init, steps=6, checkpoint_every=3)
    dpo_run(config)
    manifest = json.loads(
        (Path(config["out_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "dpo"
    assert manifest["beta"] == 0.1
    assert isinstance(manifest["heldout_margin"], float)
    split = manifest["pair_split"]
    assert split["seed"] == 20260712 and split["holdout_frac"] == 0.25
    assert split["n_train"] + split["n_holdout"] == split["n_pairs"] == prepared["n_pairs"]
    assert manifest["pairs_path"] == prepared["pairs_path"]


def test_output_checkpoint_is_eval_drop_in(tmp_path, prepared, make_init_checkpoint):
    # Criterion 5: the DPO checkpoint loads through eval.load_stage_model exactly
    # like base/SFT — a plain FableLM, no DPO-specific eval code.
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = dpo_toy_config(tmp_path / "out", prepared, init, steps=6, checkpoint_every=6)
    dpo_run(config)
    model = load_stage_model({"name": "dpo", "local_dir": config["out_dir"]}, "cpu")
    assert isinstance(model, FableLM)


def test_init_from_a_real_checkpoint(tmp_path, prepared, fixture_path):
    model64 = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
               "n_heads": 2, "context": 64, "ffn_hidden": 192}
    pre_dir = tmp_path / "pretrain"
    pretrain_run({
        "out_dir": str(pre_dir), "model": dict(model64),
        "data": {"split_path": str(fixture_path),
                 "tokenizer_path": prepared["tokenizer"],
                 "packed_path": str(tmp_path / "packed" / "pretrain.bin")},
        "train": {"steps": 2, "micro_batch_size": 4, "grad_accum": 1,
                  "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                  "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
                  "grad_clip": 1.0, "precision": "fp32", "seed": 1,
                  "checkpoint_every": 2, "log_every": 1, "keep_last": 0},
        "wandb": {"enabled": False},
    })
    config = dpo_toy_config(tmp_path / "dpo_out", prepared, pre_dir,
                            model=model64, steps=3, checkpoint_every=3)
    summary = dpo_run(config)
    assert summary["step"] == 3 and math.isfinite(summary["loss"])


def to_toml(config: dict) -> str:
    """Serialize the nested DPO config as TOML (stdlib has no writer)."""
    lines = [f'out_dir = "{config["out_dir"]}"']
    for section in ("model", "data", "init", "split", "dpo", "train", "wandb", "hub"):
        if section not in config:
            continue
        lines.append(f"[{section}]")
        for key, value in config[section].items():
            if isinstance(value, bool):
                lines.append(f"{key} = {str(value).lower()}")
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"


def test_cli_entrypoint_runs_standalone(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = dpo_toy_config(tmp_path / "out", prepared, init, steps=2, checkpoint_every=2)
    config_file = tmp_path / "cfg.toml"
    config_file.write_text(to_toml(config), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.dpo", "--config", str(config_file)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (Path(config["out_dir"]) / "checkpoints" / "step_000002.pt").exists()
