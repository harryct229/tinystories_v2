"""Evaluation stage: cross-family win-rates, reference-free metrics, and a
qualitative sample sheet over stage checkpoints (issue 07).

Invoke standalone:
    ts2-eval --config configs/eval_fixture.toml
    (or: python -m tinystories_v2.eval --config ...)

For each Scaffold in the held-out eval split, generate one seeded completion
per configured stage checkpoint (base / SFT / optional RLAIF) using identical
Scaffolds and sampling settings, then:
  - win-rates: score every stage pair with the config-selected cross-family
    eval Judge (issue 10) under order-swapped double judging;
  - reference-free metrics: issue 11's Self-BLEU, Distinct-n, Flesch Reading
    Ease per stage, plus held-out perplexity of each checkpoint on the eval
    fables;
  - sample sheet: the first sample_sheet_k eval Scaffolds rendered by every
    stage side by side.

Artifacts in <out_dir>:
    results.json   eval_judge_id, sampling, per-pair win-rate tables (counts),
                   per-stage metric tables, and the config (schema:
                   docs/schemas/eval-results-v1.md)
    report.md      report-pastable win-rate tables, metric tables, and the
                   embedded sample sheet
"""

import argparse
import itertools
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinystories_v2 import __version__
from tinystories_v2.checkpoint import latest_checkpoint, load_checkpoint
from tinystories_v2.config import load_config, load_env
from tinystories_v2.generate import sample
from tinystories_v2.hub import fetch_file_from, fetch_from, try_sync_to
from tinystories_v2.judge import Verdict, build_judge, normalize_text
from tinystories_v2.metrics import (
    distinct_n, mean_distinct_n, mean_flesch_reading_ease, self_bleu,
    tokenize_words,
)
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.perplexity import perplexity
from tinystories_v2.pref_data import scaffold_seed
from tinystories_v2.slot_prompt import END_TOKEN, SLOT_FIELDS, render_prompt
from tinystories_v2.slots import Scaffold


def _degenerate(fable_a: str, fable_b: str) -> bool:
    """True when the Judge could not accept this pair: empty or effectively
    identical candidates (the Judge seam's own candidate normalization)."""
    a, b = normalize_text(fable_a), normalize_text(fable_b)
    return not a or not b or a == b


def stage_win(judge, scaffold: Scaffold, fable_a: str, fable_b: str) -> str:
    """Order-swapped double judging of two stages' fables for one Scaffold.

    Returns "a"/"b" only when the same candidate is preferred under both
    presentation orders (position bias cancels); otherwise "tie". Assumes
    non-degenerate candidates — callers skip degenerate pairs first."""
    first = judge.compare(scaffold, fable_a, fable_b)
    swapped = judge.compare(scaffold, fable_b, fable_a)
    if first is Verdict.A and swapped is Verdict.B:
        return "a"
    if first is Verdict.B and swapped is Verdict.A:
        return "b"
    return "tie"


def win_rate_table(judge, scaffolds: list[Scaffold], stage_a: str,
                   fables_a: list[str], stage_b: str,
                   fables_b: list[str]) -> dict:
    """Tally wins/ties/skips of stage_a vs stage_b over aligned per-Scaffold
    completions. Degenerate pairs (empty or identical) the Judge cannot compare
    are skipped, not counted as ties."""
    if not (len(scaffolds) == len(fables_a) == len(fables_b)):
        raise ValueError("scaffolds and both fable lists must align")
    wins_a = wins_b = ties = skipped = 0
    for scaffold, fa, fb in zip(scaffolds, fables_a, fables_b):
        if _degenerate(fa, fb):
            skipped += 1
            continue
        outcome = stage_win(judge, scaffold, fa, fb)
        if outcome == "a":
            wins_a += 1
        elif outcome == "b":
            wins_b += 1
        else:
            ties += 1
    return {"stage_a": stage_a, "stage_b": stage_b, "wins_a": wins_a,
            "wins_b": wins_b, "ties": ties, "skipped": skipped,
            "n": len(scaffolds)}


def all_pairwise_win_rates(judge, scaffolds: list[Scaffold],
                           stage_fables: dict[str, list[str]]) -> list[dict]:
    """A win_rate_table for every unordered stage pair, in stage_fables order."""
    names = list(stage_fables)
    return [win_rate_table(judge, scaffolds, a, stage_fables[a],
                           b, stage_fables[b])
            for a, b in itertools.combinations(names, 2)]
