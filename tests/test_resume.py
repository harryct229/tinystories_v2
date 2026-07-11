"""Kill-and-resume: the checkpoint-resume contract, end to end.

Sized so the run is slow enough to SIGKILL mid-flight: d_model 128 / 4 layers
/ ctx 128 / micro-batch 8 is ~0.9M params and ~50-150 ms per CPU step, so 50
steps gives a multi-second window. checkpoint_every=5 guarantees several
checkpoints exist before the kill.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from tinystories_v2.checkpoint import latest_checkpoint, load_checkpoint
from tinystories_v2.pretrain import run
from tinystories_v2.tokenizer import run as run_tokenizer

STEPS = 50
CHECKPOINT_EVERY = 5
KILL_AFTER_STEP = 10  # SIGKILL once this checkpoint appears


def resume_config(base: Path, fixture_path: Path, tokenizer_path: Path,
                  out_name: str) -> dict:
    return {
        "out_dir": str(base / out_name),
        "model": {"vocab_size": 512, "d_model": 128, "n_layers": 4,
                  "n_heads": 4, "context": 128, "ffn_hidden": 384},
        "data": {"split_path": str(fixture_path),
                 "tokenizer_path": str(tokenizer_path),
                 "packed_path": str(base / "packed" / "pretrain.bin")},
        "train": {"steps": STEPS, "micro_batch_size": 8, "grad_accum": 1,
                  "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                  "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
                  "grad_clip": 1.0, "precision": "fp32", "seed": 1337,
                  "checkpoint_every": CHECKPOINT_EVERY, "log_every": 1,
                  "keep_last": 0},
        "wandb": {"enabled": False},
    }


def to_toml(config: dict) -> str:
    lines = [f'out_dir = "{config["out_dir"]}"']
    for section in ("model", "data", "train", "wandb"):
        lines.append(f"[{section}]")
        for key, value in config[section].items():
            if isinstance(value, bool):
                lines.append(f"{key} = {str(value).lower()}")
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"


def read_metrics(out_dir: str) -> dict[int, float]:
    lines = (Path(out_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return {row["step"]: row["loss"] for row in map(json.loads, lines)}


def test_killed_run_resumes_to_identical_final_state(tmp_path, fixture_path):
    tok_dir = tmp_path / "tok"
    run_tokenizer({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer_path = tok_dir / "tokenizer.json"

    # Reference: the same config, never interrupted (shares the packed binary).
    reference = resume_config(tmp_path, fixture_path, tokenizer_path, "reference")
    run(reference)

    # Interrupted: identical config except out_dir, run as a subprocess.
    interrupted = resume_config(tmp_path, fixture_path, tokenizer_path, "interrupted")
    config_file = tmp_path / "interrupted.toml"
    config_file.write_text(to_toml(interrupted), encoding="utf-8")
    ckpt_dir = Path(interrupted["out_dir"]) / "checkpoints"
    kill_marker = ckpt_dir / f"step_{KILL_AFTER_STEP:06d}.pt"

    proc = subprocess.Popen(
        [sys.executable, "-m", "tinystories_v2.pretrain", "--config", str(config_file)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 120
        while not kill_marker.exists():
            if proc.poll() is not None:
                pytest.fail(
                    f"stage finished (rc={proc.returncode}) before the kill window; "
                    f"enlarge the toy model or lower KILL_AFTER_STEP"
                )
            if time.monotonic() > deadline:
                pytest.fail("timed out waiting for the kill-marker checkpoint")
            time.sleep(0.01)
        os.kill(proc.pid, signal.SIGKILL)
    finally:
        proc.kill()  # idempotent if already SIGKILLed; guards the timeout path
        proc.wait(timeout=30)

    killed_at = load_checkpoint(latest_checkpoint(ckpt_dir))["step"]
    assert KILL_AFTER_STEP <= killed_at < STEPS  # it really died mid-run

    # Resume with the one flag; must continue from the recorded step to the end.
    run(interrupted, resume=True)

    # Training state matches the uninterrupted run bitwise.
    final_ref = load_checkpoint(
        latest_checkpoint(Path(reference["out_dir"]) / "checkpoints"))
    final_res = load_checkpoint(latest_checkpoint(ckpt_dir))
    assert final_res["step"] == final_ref["step"] == STEPS
    assert final_res["tokens_seen"] == final_ref["tokens_seen"]
    for key, tensor in final_ref["model"].items():
        assert torch.equal(final_res["model"][key], tensor), key

    # Post-resume losses replay the reference exactly, from the resumed step on.
    ref_losses, res_losses = read_metrics(reference["out_dir"]), read_metrics(
        interrupted["out_dir"])
    for step in range(killed_at + 1, STEPS + 1):
        assert res_losses[step] == ref_losses[step], step
