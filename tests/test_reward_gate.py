import json
from pathlib import Path

import pytest

from tinystories_v2.gate import (
    DEFAULT_ACCURACY_GATE, RewardGateError, check_reward_gate, load_reward_manifest,
)


def _write_manifest(dir_path: Path, **fields) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    manifest = {"stage": "reward_model", "heldout_accuracy": 0.75, **fields}
    (dir_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dir_path


def test_default_gate_matches_design():
    assert DEFAULT_ACCURACY_GATE == 0.68


def test_above_gate_returns_accuracy(tmp_path):
    rm = _write_manifest(tmp_path / "rm", heldout_accuracy=0.72)
    assert check_reward_gate(rm) == 0.72


def test_below_gate_raises_with_clear_message(tmp_path):
    rm = _write_manifest(tmp_path / "rm", heldout_accuracy=0.60)
    with pytest.raises(RewardGateError, match="below the gate"):
        check_reward_gate(rm)


def test_custom_gate_threshold(tmp_path):
    rm = _write_manifest(tmp_path / "rm", heldout_accuracy=0.72)
    assert check_reward_gate(rm, gate=0.70) == 0.72
    with pytest.raises(RewardGateError, match="below the gate"):
        check_reward_gate(rm, gate=0.80)


def test_nan_accuracy_raises(tmp_path):
    rm = _write_manifest(tmp_path / "rm", heldout_accuracy=float("nan"))
    with pytest.raises(RewardGateError, match="undefined"):
        check_reward_gate(rm)


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(RewardGateError, match="no Reward Model manifest"):
        check_reward_gate(tmp_path / "nope")


def test_non_reward_manifest_raises(tmp_path):
    rm = tmp_path / "rm"
    rm.mkdir()
    (rm / "manifest.json").write_text(json.dumps({"stage": "sft"}), encoding="utf-8")
    with pytest.raises(RewardGateError, match="not a Reward Model artifact"):
        load_reward_manifest(rm)


def test_gate_reads_a_real_trained_artifact(tmp_path, fixture_path, make_init_checkpoint):
    # End to end: train a toy Reward Model, then gate its real artifact. A
    # permissive gate passes; an impossible one refuses.
    import json as _json

    from tinystories_v2.data import run as data_run
    from tinystories_v2.judge import SlotCoverageFakeJudge, judge_with_order_swap
    from tinystories_v2.reward import run as reward_run
    from tinystories_v2.slot_prompt import SLOT_FIELDS
    from tinystories_v2.slots import Scaffold
    from tinystories_v2.tokenizer import run as tokenizer_run

    data_dir, tok_dir = tmp_path / "data", tmp_path / "tok"
    data_run({"out_dir": str(data_dir), "max_extraction_failures": 0,
              "source": {"kind": "jsonl", "path": str(fixture_path)},
              "splits": {"seed": "fixture-v1", "pretrain": 0.1, "sft": 0.7,
                         "pref": 0.1, "eval": 0.1}})
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer_path = str(tok_dir / "tokenizer.json")
    with open(data_dir / "splits" / "sft.jsonl", encoding="utf-8") as f:
        rows = [_json.loads(line) for line in f if line.strip()]
    judge = SlotCoverageFakeJudge()
    pairs_path = tmp_path / "pairs.jsonl"
    bland = ["A plain note with nothing much to say.",
             "Some words that go nowhere in particular."]
    with pairs_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            scaffold = Scaffold(**{fld: row[fld] for fld in SLOT_FIELDS})
            chosen = (f"{scaffold.character}, a {scaffold.trait} one in "
                      f"{scaffold.setting}, met {scaffold.conflict}. "
                      f"{scaffold.resolution}. The moral: {scaffold.moral}.")
            pair = judge_with_order_swap(judge, scaffold, chosen, bland[i % len(bland)])
            f.write(_json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

    model = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 128, "ffn_hidden": 192}
    init = make_init_checkpoint(tmp_path / "init", model, tokenizer_path)
    out_dir = tmp_path / "reward_out"
    reward_run({
        "out_dir": str(out_dir), "model": dict(model),
        "data": {"pairs_path": str(pairs_path), "tokenizer_path": tokenizer_path},
        "init": {"local_dir": str(init)},
        "split": {"holdout_frac": 0.25, "seed": 20260712},
        "train": {"steps": 60, "micro_batch_size": 8, "grad_accum": 1,
                  "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                  "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
                  "grad_clip": 1.0, "precision": "fp32", "seed": 1337,
                  "checkpoint_every": 60, "log_every": 20, "keep_last": 0},
        "wandb": {"enabled": False},
    })
    assert check_reward_gate(out_dir, gate=0.5) > 0.5     # separable pairs clear a low gate
    with pytest.raises(RewardGateError, match="below the gate"):
        check_reward_gate(out_dir, gate=1.01)             # nothing clears an impossible gate
