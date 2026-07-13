"""Kill-and-resume: the GRPO checkpoint-resume contract, end to end.

Sized so the run is slow enough to SIGKILL mid-flight: d_model 128 / 4 layers /
ctx 128, prompts_per_step 4 x group_size 6 rollouts per step (rollout sampling
has no KV cache and dominates), 30 steps, checkpoint_every 3. Both runs share
one SFT init, one frozen Reward Model, and one pref split, so the Scaffold batch
(a pure function of seed/step), the per-rollout sampling (a pure function of
seed/step/prompt_index), the frozen reference (a pure function of the fixed SFT
checkpoint), and the frozen Reward Model (a pure function of the reward artifact)
are identical across the two runs.
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
from tinystories_v2.grpo import run as grpo_run
from tinystories_v2.judge import SlotCoverageFakeJudge, judge_with_order_swap
from tinystories_v2.reward import run as reward_run
from tinystories_v2.slot_prompt import SLOT_FIELDS
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run

STEPS = 30
CHECKPOINT_EVERY = 3
KILL_AFTER_STEP = 6

MODEL = {"vocab_size": 512, "d_model": 128, "n_layers": 4,
         "n_heads": 4, "context": 128, "ffn_hidden": 384}

_BLAND = ["A plain note with nothing much to say.",
          "Some words that go nowhere in particular."]


def _write_pairs(path, rows):
    judge = SlotCoverageFakeJudge()
    with path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            scaffold = Scaffold(**{fld: row[fld] for fld in SLOT_FIELDS})
            chosen = (f"{scaffold.character}, a {scaffold.trait} one in "
                      f"{scaffold.setting}, met {scaffold.conflict}. "
                      f"{scaffold.resolution}. The moral: {scaffold.moral}.")
            pair = judge_with_order_swap(judge, scaffold, chosen, _BLAND[i % len(_BLAND)])
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")


def grpo_config(out_dir, pref_split, tokenizer_path, init_dir, reward_dir) -> dict:
    return {
        "out_dir": str(out_dir), "model": dict(MODEL),
        "data": {"pref_split": str(pref_split), "tokenizer_path": str(tokenizer_path)},
        "init": {"local_dir": str(init_dir)},
        "reward": {"local_dir": str(reward_dir), "gate": 0.5},
        "grpo": {"group_size": 6, "clip_eps": 0.2, "kl_beta": 0.03,
                 "ppo_epochs": 2, "adv_eps": 1e-6},
        "sampling": {"max_new_tokens": 24, "temperature": 1.0, "top_p": 1.0},
        "train": {"steps": STEPS, "prompts_per_step": 4, "peak_lr": 1e-3,
                  "warmup_frac": 0.1, "min_lr_frac": 0.1, "weight_decay": 0.0,
                  "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0, "precision": "fp32",
                  "seed": 1337, "checkpoint_every": CHECKPOINT_EVERY, "log_every": 1,
                  "keep_last": 0},
        "wandb": {"enabled": False},
    }


def to_toml(config: dict) -> str:
    lines = [f'out_dir = "{config["out_dir"]}"']
    for section in ("model", "data", "init", "reward", "grpo", "sampling",
                    "train", "wandb"):
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


def _train_toy_rm(reward_dir, init_dir, pairs_path, tokenizer_path):
    reward_run({
        "out_dir": str(reward_dir), "model": dict(MODEL),
        "data": {"pairs_path": str(pairs_path), "tokenizer_path": str(tokenizer_path)},
        "init": {"local_dir": str(init_dir)},
        "split": {"holdout_frac": 0.25, "seed": 20260712},
        "train": {"steps": 40, "micro_batch_size": 8, "grad_accum": 1,
                  "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                  "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
                  "precision": "fp32", "seed": 1337, "checkpoint_every": 40,
                  "log_every": 20, "keep_last": 0},
        "wandb": {"enabled": False},
    })


def test_killed_grpo_run_resumes_to_identical_final_state(
        tmp_path, fixture_path, make_init_checkpoint):
    data_dir, tok_dir = tmp_path / "data", tmp_path / "tok"
    data_run({
        "out_dir": str(data_dir), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.1, "sft": 0.6,
                   "pref": 0.2, "eval": 0.1},
    })
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer_path = tok_dir / "tokenizer.json"
    pref_split = data_dir / "splits" / "pref.jsonl"
    with open(data_dir / "splits" / "sft.jsonl", encoding="utf-8") as f:
        sft_rows = [json.loads(line) for line in f if line.strip()]
    init_dir = make_init_checkpoint(tmp_path / "init", MODEL, tokenizer_path)
    pairs_path = tmp_path / "pairs.jsonl"
    _write_pairs(pairs_path, sft_rows)
    reward_dir = tmp_path / "rm"
    _train_toy_rm(reward_dir, init_dir, pairs_path, tokenizer_path)

    # Reference: identical config, never interrupted.
    reference = grpo_config(tmp_path / "reference", pref_split, tokenizer_path,
                            init_dir, reward_dir)
    grpo_run(reference)

    # Interrupted: run as a subprocess and SIGKILL once the kill-marker appears.
    interrupted = grpo_config(tmp_path / "interrupted", pref_split, tokenizer_path,
                              init_dir, reward_dir)
    config_file = tmp_path / "interrupted.toml"
    config_file.write_text(to_toml(interrupted), encoding="utf-8")
    ckpt_dir = Path(interrupted["out_dir"]) / "checkpoints"
    kill_marker = ckpt_dir / f"step_{KILL_AFTER_STEP:06d}.pt"

    proc = subprocess.Popen(
        [sys.executable, "-m", "tinystories_v2.grpo", "--config", str(config_file)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 180
        while not kill_marker.exists():
            if proc.poll() is not None:
                pytest.fail(
                    f"stage finished (rc={proc.returncode}) before the kill window; "
                    f"enlarge the toy model or lower KILL_AFTER_STEP")
            if time.monotonic() > deadline:
                pytest.fail("timed out waiting for the kill-marker checkpoint")
            time.sleep(0.01)
        os.kill(proc.pid, signal.SIGKILL)
    finally:
        proc.kill()
        proc.wait(timeout=30)

    killed_at = load_checkpoint(latest_checkpoint(ckpt_dir))["step"]
    assert KILL_AFTER_STEP <= killed_at < STEPS

    # Snapshot the pre-kill checkpoints so we can prove resume continues from
    # them rather than silently recomputing the prefix from scratch.
    pre_kill_stats = {
        p.name: (p.stat().st_mtime_ns, p.stat().st_size)
        for p in ckpt_dir.glob("step_*.pt")
    }

    grpo_run(interrupted, resume=True)

    for name, (mtime_ns, size) in pre_kill_stats.items():
        p = ckpt_dir / name
        assert p.exists(), f"pre-kill checkpoint {name} disappeared after resume"
        stat = p.stat()
        assert (stat.st_mtime_ns, stat.st_size) == (mtime_ns, size), (
            f"pre-kill checkpoint {name} was rewritten — resume recomputed "
            f"the prefix instead of continuing from it")

    final_ref = load_checkpoint(
        latest_checkpoint(Path(reference["out_dir"]) / "checkpoints"))
    final_res = load_checkpoint(latest_checkpoint(ckpt_dir))
    assert final_res["step"] == final_ref["step"] == STEPS
    assert final_res["rollouts_seen"] == final_ref["rollouts_seen"]
    for key, tensor in final_ref["model"].items():
        assert torch.equal(final_res["model"][key], tensor), key

    ref_losses = read_metrics(reference["out_dir"])
    res_losses = read_metrics(interrupted["out_dir"])
    for step in range(killed_at + 1, STEPS + 1):
        assert res_losses[step] == ref_losses[step], step
