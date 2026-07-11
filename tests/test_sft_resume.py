"""Kill-and-resume: the SFT checkpoint-resume contract, end to end.

Sized so the run is slow enough to SIGKILL mid-flight: d_model 128 / 4 layers /
ctx 128, 50 steps, checkpoint_every 5 gives several checkpoints before the kill.
Both runs share one init checkpoint and one examples.jsonl, so batches (a pure
function of seed/step/micro_step) and starting weights are identical.
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
from tinystories_v2.data import run as data_run
from tinystories_v2.sft import run as sft_run
from tinystories_v2.sft_data import run as sft_data_run
from tinystories_v2.tokenizer import run as tokenizer_run

STEPS = 50
CHECKPOINT_EVERY = 5
KILL_AFTER_STEP = 10

MODEL = {"vocab_size": 512, "d_model": 128, "n_layers": 4,
         "n_heads": 4, "context": 128, "ffn_hidden": 384}


def sft_config(out_dir, examples_path, tokenizer_path, init_dir) -> dict:
    return {
        "out_dir": str(out_dir),
        "model": dict(MODEL),
        "data": {"examples_path": str(examples_path),
                 "tokenizer_path": str(tokenizer_path)},
        "init": {"local_dir": str(init_dir)},
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
    for section in ("model", "data", "init", "train", "wandb"):
        lines.append(f"[{section}]")
        for key, value in config[section].items():
            if isinstance(value, bool):
                lines.append(f"{key} = {str(value).lower()}")
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"


def read_metrics(out_dir) -> dict[int, float]:
    lines = (Path(out_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return {row["step"]: row["loss"] for row in map(json.loads, lines)}


def test_killed_sft_resumes_to_identical_final_state(
        tmp_path, fixture_path, make_init_checkpoint):
    # Build the shared inputs once: a real examples.jsonl and one init checkpoint.
    data_dir, tok_dir, sd_dir = tmp_path / "data", tmp_path / "tok", tmp_path / "sd"
    data_run({
        "out_dir": str(data_dir), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.4, "sft": 0.4,
                   "pref": 0.1, "eval": 0.1},
    })
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer_path = tok_dir / "tokenizer.json"
    sft_data_run({"out_dir": str(sd_dir), "tokenizer": str(tokenizer_path),
                  "sft_split": str(data_dir / "splits" / "sft.jsonl"),
                  "max_examples": 0})
    examples_path = sd_dir / "examples.jsonl"
    init_dir = make_init_checkpoint(tmp_path / "init", MODEL, tokenizer_path)

    # Reference: identical config, never interrupted.
    reference = sft_config(tmp_path / "reference", examples_path, tokenizer_path, init_dir)
    sft_run(reference)

    # Interrupted: run as a subprocess and SIGKILL once the kill-marker appears.
    interrupted = sft_config(tmp_path / "interrupted", examples_path,
                             tokenizer_path, init_dir)
    config_file = tmp_path / "interrupted.toml"
    config_file.write_text(to_toml(interrupted), encoding="utf-8")
    ckpt_dir = Path(interrupted["out_dir"]) / "checkpoints"
    kill_marker = ckpt_dir / f"step_{KILL_AFTER_STEP:06d}.pt"

    proc = subprocess.Popen(
        [sys.executable, "-m", "tinystories_v2.sft", "--config", str(config_file)],
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
        proc.kill()
        proc.wait(timeout=30)

    killed_at = load_checkpoint(latest_checkpoint(ckpt_dir))["step"]
    assert KILL_AFTER_STEP <= killed_at < STEPS

    sft_run(interrupted, resume=True)

    final_ref = load_checkpoint(
        latest_checkpoint(Path(reference["out_dir"]) / "checkpoints"))
    final_res = load_checkpoint(latest_checkpoint(ckpt_dir))
    assert final_res["step"] == final_ref["step"] == STEPS
    assert final_res["tokens_seen"] == final_ref["tokens_seen"]
    for key, tensor in final_ref["model"].items():
        assert torch.equal(final_res["model"][key], tensor), key

    ref_losses = read_metrics(reference["out_dir"])
    res_losses = read_metrics(interrupted["out_dir"])
    for step in range(killed_at + 1, STEPS + 1):
        assert res_losses[step] == ref_losses[step], step
