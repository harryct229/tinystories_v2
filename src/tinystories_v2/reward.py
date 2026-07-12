"""Reward Model stage: distill Judge preferences into a scalar reward with
hand-written Bradley-Terry loss (issue 05).

Invoke standalone:
    ts2-reward --config configs/reward_fixture.toml [--resume]
    (or: python -m tinystories_v2.reward --config ...)

Initializes the backbone from an SFT checkpoint ([init] section), attaches a
fresh scalar head, and trains on order-swap-consistent preference pairs (issue
10 schema). Reuses issue 02's checkpoint-resume contract, optimizer conventions
(build_optimizer), LR schedule (lr_at), precision knob, W&B logging, and Hub
sync verbatim; only the data source (preference pairs) and the loss (Bradley-
Terry over chosen/rejected scores) differ.

Holds out a deterministic slice of pairs and records held-out pair accuracy and
the split recipe in the manifest (schema: docs/schemas/reward-model-artifact-v1.md).
The accuracy gate that protects RLAIF lives in tinystories_v2.gate and reads
that manifest.

Artifacts in <out_dir>:
    checkpoints/step_XXXXXX.pt   full training state (atomic; resume contract)
    metrics.jsonl                one line per log_every steps: loss, lr, accuracy, pairs_seen
    manifest.json                stage, version, final step/loss, heldout_accuracy,
                                 pair_split recipe, pairs_path, config

Determinism contract: backbone init is loaded from a fixed checkpoint, the
held-out split is a pure function of (n_pairs, holdout_frac, split_seed),
batches are a pure function of (seed, step, micro_step), and optimizer state
round-trips, so an interrupted-and-resumed run reproduces the uninterrupted run
exactly (fp32 CPU; asserted by tests/test_reward_resume.py).
"""

import argparse
import json
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinystories_v2 import __version__
from tinystories_v2.checkpoint import (
    latest_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint,
)
from tinystories_v2.config import load_config, load_env
from tinystories_v2.hub import fetch_from, try_sync_to
from tinystories_v2.model import ModelConfig
from tinystories_v2.preferences import PreferencePair, validate_preference_pair
from tinystories_v2.pretrain import build_optimizer, lr_at
from tinystories_v2.reward_model import (
    RewardModel, bradley_terry_loss, pad_sequences, pair_accuracy, score_sequences,
)
from tinystories_v2.slot_prompt import encode_example
from tinystories_v2.tracking import MetricsLogger


def load_pairs(path: str | Path) -> list[PreferencePair]:
    """Read a preference-pair jsonl, validating each line against schema v1."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(validate_preference_pair(json.loads(line)))
    return pairs


def encode_pairs(tokenizer: Tokenizer, pairs: list[PreferencePair]) -> list[dict]:
    """Precompute (chosen_ids, rejected_ids) per pair via the Slot Prompt encoder
    (each is the full <|character|>…<|fable|>{body}<|end|> sequence)."""
    encoded = []
    for pair in pairs:
        chosen = encode_example(tokenizer, pair.scaffold, pair.chosen).input_ids
        rejected = encode_example(tokenizer, pair.scaffold, pair.rejected).input_ids
        encoded.append({"chosen_ids": chosen, "rejected_ids": rejected})
    return encoded


def split_pairs(encoded: list[dict], holdout_frac: float,
                seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic train/holdout split: a seeded permutation, the last
    round(n*holdout_frac) held out. A pure function of (n, holdout_frac, seed) so
    a resumed run reproduces the same split (and thus the same held-out accuracy)."""
    n = len(encoded)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    n_holdout = round(n * holdout_frac)
    train = [encoded[i] for i in perm[:n - n_holdout]]
    holdout = [encoded[i] for i in perm[n - n_holdout:]] if n_holdout else []
    return train, holdout


def get_pair_batch(train: list[dict], micro_batch_size: int, context: int, *,
                   seed: int, step: int, micro_step: int,
                   device: str = "cpu") -> tuple[torch.Tensor, ...]:
    """A (chosen_idx, chosen_len, rejected_idx, rejected_len) micro-batch sampled
    with replacement; a pure function of (seed, step, micro_step) for resume."""
    generator = torch.Generator()
    generator.manual_seed(((seed * 1_000_003 + step) * 1_009 + micro_step) % 2**63)
    idx = torch.randint(0, len(train), (micro_batch_size,), generator=generator)
    picks = idx.tolist()
    chosen_idx, chosen_len = pad_sequences(
        [train[i]["chosen_ids"] for i in picks], context, device)
    rejected_idx, rejected_len = pad_sequences(
        [train[i]["rejected_ids"] for i in picks], context, device)
    return chosen_idx, chosen_len, rejected_idx, rejected_len


@torch.no_grad()
def evaluate_accuracy(model: RewardModel, holdout: list[dict], *,
                      device: str = "cpu") -> float:
    """Held-out pair accuracy: fraction of holdout pairs the model scores
    chosen > rejected. Returns NaN for an empty holdout."""
    if not holdout:
        return float("nan")
    chosen = score_sequences(model, [p["chosen_ids"] for p in holdout], device=device)
    rejected = score_sequences(model, [p["rejected_ids"] for p in holdout], device=device)
    return pair_accuracy(chosen, rejected).item()
