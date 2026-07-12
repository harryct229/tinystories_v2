import subprocess
import sys
from pathlib import Path

from tinystories_v2.data import run as data_run
from tinystories_v2.pretrain import run as pretrain_run
from tinystories_v2.tokenizer import run as tokenizer_run

MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
         "n_heads": 2, "context": 128, "ffn_hidden": 192}

SLOTS = ["--character", "fox", "--trait", "sly", "--setting", "a green wood",
         "--conflict", "a locked gate", "--resolution", "the fox waited",
         "--moral", "patience wins"]


def _toy_checkpoint(tmp_path, fixture_path):
    """Any checkpoint works for the demo; a 2-step toy pretrain is cheapest.
    Its stored config carries data.tokenizer_path so the demo finds the tokenizer."""
    tok_dir = tmp_path / "tok"
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    out = tmp_path / "pretrain"
    pretrain_run({
        "out_dir": str(out), "model": dict(MODEL),
        "data": {"split_path": str(fixture_path),
                 "tokenizer_path": str(tok_dir / "tokenizer.json"),
                 "packed_path": str(tmp_path / "packed" / "pretrain.bin")},
        "train": {"steps": 2, "micro_batch_size": 4, "grad_accum": 1,
                  "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                  "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
                  "grad_clip": 1.0, "precision": "fp32", "seed": 1,
                  "checkpoint_every": 2, "log_every": 1, "keep_last": 0},
        "wandb": {"enabled": False},
    })
    return out / "checkpoints"


def test_demo_generates_from_six_slot_values_on_cpu(tmp_path, fixture_path):
    ckpts = _toy_checkpoint(tmp_path, fixture_path)
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.demo",
         "--checkpoint", str(ckpts), *SLOTS,
         "--max-new-tokens", "16", "--seed", "3"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "fox" in result.stdout          # Scaffold header echoes the slots
    assert "Fable" in result.stdout


def test_demo_requires_all_six_slots_or_sample_eval(tmp_path, fixture_path):
    ckpts = _toy_checkpoint(tmp_path, fixture_path)
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.demo",
         "--checkpoint", str(ckpts), "--character", "fox"],  # only one slot
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "six slots" in (result.stderr + result.stdout)


def test_demo_samples_scaffold_from_eval_split(tmp_path, fixture_path):
    ckpts = _toy_checkpoint(tmp_path, fixture_path)
    data_dir = tmp_path / "data"
    data_run({
        "out_dir": str(data_dir), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.4, "sft": 0.3,
                   "pref": 0.1, "eval": 0.2},
    })
    eval_split = data_dir / "splits" / "eval.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.demo",
         "--checkpoint", str(ckpts), "--sample-eval", str(eval_split),
         "--max-new-tokens", "16", "--seed", "0"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Fable" in result.stdout
