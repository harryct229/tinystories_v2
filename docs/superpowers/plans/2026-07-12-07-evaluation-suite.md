# Evaluation Suite Implementation Plan (issue 07)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `ts2-eval` stage that produces the report's evidence — cross-family win-rates between stage checkpoints, the paper's reference-free metrics per stage, and a qualitative sample sheet — from one TOML config.

**Architecture:** A new `src/tinystories_v2/eval.py` stage mirrors the established stage convention (`pref_data.py`, `reward.py`): read one TOML config, load N stage checkpoints (base / SFT / optional RLAIF), generate one seeded completion per held-out eval Scaffold from each checkpoint using **identical Scaffolds and sampling settings**, score every stage pair with a **cross-family eval Judge** (Llama-3.1-8B-Instruct) through issue 10's Judge interface under order-swapped double judging, compute issue 11's reference-free metrics + held-out perplexity per stage, and write a machine-readable `results.json` plus a report-pastable `report.md` (win-rate tables, metric tables, and an embedded sample sheet). Pure helpers (win-rate tally, metric aggregation, report/sample-sheet rendering) are TDD'd in isolation before the `run` entrypoint wires them together. A thin Colab bootstrap + notebook drives the real run.

**Tech Stack:** Python ≥3.11, PyTorch (generation + perplexity), `tokenizers`, `huggingface_hub` (artifact sync), `transformers` (real eval Judge only, via the `[judge]` extra — never imported on the CPU path). No new third-party dependencies.

## Global Constraints

- **Eval Judge is cross-family**: Llama-3.1-8B-Instruct (fp16 on L4, 4-bit on T4), selected via issue 10's Judge interface (`build_judge`, `kind = "transformers"`). It is **never** the Qwen Judge that produced the reward signal (self-preference bias, per the dataset paper).
- **Order-swapped double judging** applies to every win-rate comparison: a stage wins a Scaffold only when it is preferred under both presentation orders; otherwise the comparison is a tie.
- **Identical Scaffolds and sampling settings** are used across all stage checkpoints; this is asserted by a test.
- **Reference-free metrics** come from issue 11's `tinystories_v2.metrics` and `tinystories_v2.perplexity` — no reimplementation — so the numbers stay directly comparable to the dataset paper's tables.
- **Works with base+SFT alone**; the RLAIF column appears only when a third stage checkpoint is configured.
- **The whole stage runs on CPU** with the fake Judge and toy checkpoints (no GPU, network, or model download).
- **The eval-Judge identity is recorded** in the results artifact (`results["eval_judge_id"]`), so "who judged" is never ambiguous in the report.
- **Stage convention** (PRD): one TOML config → artifacts under its `out_dir`; stages share nothing in memory.
- **Colab notebook stays thin**: setup + one-command bootstrap invocation only — no `def`/`class`/`import torch`/`for`/`while` in cell source, no committed outputs, no secrets (`hf_` substring forbidden). Enforced by `tests/test_notebook.py`.
- Package version string comes from `tinystories_v2.__version__`; artifact `stage` field is `"eval"`.

## File Structure

- **Create `src/tinystories_v2/eval.py`** — the entire eval stage: win-rate primitives, metric aggregation, report/sample-sheet rendering, stage-model loading, per-stage generation, the `run` entrypoint, and `main`. One file, one responsibility (the evaluation stage), matching how `pref_data.py` and `reward.py` each hold a whole stage.
- **Create `configs/eval_fixture.toml`** — toy CPU config (fake Judge, fixture artifacts) for local smoke runs and docs.
- **Create `configs/eval_full.toml`** — real Colab run config (Llama eval Judge, base+SFT stages, Hub sources).
- **Create `scripts/eval_colab.py`** — one-command Colab bootstrap (download tokenizer + eval split, then run the stage), mirroring `scripts/reward_colab.py`.
- **Create `notebooks/eval_colab.ipynb`** — thin wrapper that invokes `scripts/eval_colab.py`.
- **Create `docs/schemas/eval-results-v1.md`** — the `results.json` contract.
- **Create tests** `tests/test_eval_winrate.py`, `tests/test_eval_metrics.py`, `tests/test_eval_report.py`, `tests/test_eval_generate.py`, `tests/test_eval_stage.py`, `tests/test_eval_colab.py`.
- **Modify `pyproject.toml:40`** — add the `ts2-eval` entry point.
- **Modify `tests/test_notebook.py`** (append) — thinness + no-secrets tests for the eval notebook.
- **Modify `PROGRESS.md`** — flip issue 07 to code-complete and add a Log entry.

Interfaces reused verbatim (do not modify): `tinystories_v2.judge.build_judge` / `Verdict` / `normalize_text`; `tinystories_v2.generate.sample`; `tinystories_v2.checkpoint.{latest_checkpoint,load_checkpoint}`; `tinystories_v2.model.{FableLM,ModelConfig}`; `tinystories_v2.slot_prompt.{render_prompt,SLOT_FIELDS,END_TOKEN}`; `tinystories_v2.slots.Scaffold`; `tinystories_v2.metrics.{mean_distinct_n,distinct_n,self_bleu,mean_flesch_reading_ease,tokenize_words}`; `tinystories_v2.perplexity.perplexity`; `tinystories_v2.pref_data.scaffold_seed`; `tinystories_v2.hub.{fetch_from,fetch_file_from,try_sync_to}`; `tinystories_v2.config.{load_config,load_env}`; `tinystories_v2.hub_download.download_file`.

---

### Task 1: Win-rate primitives

Pure comparison logic: given two stages' aligned fable lists over the same Scaffolds, tally wins/ties/skips under order-swapped double judging. No models, no torch — testable with a fake Judge.

**Files:**
- Create: `src/tinystories_v2/eval.py`
- Test: `tests/test_eval_winrate.py`

**Interfaces:**
- Consumes: `tinystories_v2.judge.Verdict`, `tinystories_v2.judge.normalize_text`, a `Judge` (anything with `.compare(scaffold, a, b) -> Verdict` and `.judge_id`); `tinystories_v2.slots.Scaffold`.
- Produces:
  - `stage_win(judge, scaffold: Scaffold, fable_a: str, fable_b: str) -> str` — returns `"a"`, `"b"`, or `"tie"`. Assumes non-degenerate candidates (caller pre-checks).
  - `win_rate_table(judge, scaffolds: list[Scaffold], stage_a: str, fables_a: list[str], stage_b: str, fables_b: list[str]) -> dict` — keys `{"stage_a","stage_b","wins_a","wins_b","ties","skipped","n"}`.
  - `all_pairwise_win_rates(judge, scaffolds: list[Scaffold], stage_fables: dict[str, list[str]]) -> list[dict]` — one `win_rate_table` per unordered stage pair, in `stage_fables` insertion order.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_winrate.py`:

```python
"""Order-swapped win-rate tallies over aligned per-stage fable lists (issue 07)."""

from tinystories_v2.eval import (
    all_pairwise_win_rates,
    stage_win,
    win_rate_table,
)
from tinystories_v2.judge import PositionBiasedFakeJudge, SlotCoverageFakeJudge
from tinystories_v2.slots import Scaffold

SCAFFOLD = Scaffold("fox", "sly", "a wood", "a locked gate",
                    "the fox shared", "sharing brings friends")
# SlotCoverageFakeJudge prefers whichever candidate mentions more slot values.
RICH = ("The sly fox in a wood met a locked gate; the fox shared, and sharing "
        "brings friends.")
BLAND = "A plain note with nothing much to say."


def test_stage_win_picks_the_slot_rich_side_consistently():
    assert stage_win(SlotCoverageFakeJudge(), SCAFFOLD, RICH, BLAND) == "a"
    assert stage_win(SlotCoverageFakeJudge(), SCAFFOLD, BLAND, RICH) == "b"


def test_stage_win_is_a_tie_when_order_swap_is_inconsistent():
    # PositionBiasedFakeJudge always answers "A": it prefers position, not
    # content, so the two presentation orders disagree -> a tie.
    assert stage_win(PositionBiasedFakeJudge(), SCAFFOLD, RICH, BLAND) == "tie"


def test_win_rate_table_counts_wins_and_skips_degenerate_pairs():
    scaffolds = [SCAFFOLD, SCAFFOLD, SCAFFOLD]
    fables_a = [RICH, RICH, "identical"]
    fables_b = [BLAND, BLAND, "IDENTICAL"]  # third pair is degenerate (casefold-equal)
    table = win_rate_table(SlotCoverageFakeJudge(), scaffolds,
                           "sft", fables_a, "base", fables_b)
    assert table == {"stage_a": "sft", "stage_b": "base", "wins_a": 2,
                     "wins_b": 0, "ties": 0, "skipped": 1, "n": 3}


def test_win_rate_table_rejects_misaligned_lists():
    import pytest
    with pytest.raises(ValueError, match="align"):
        win_rate_table(SlotCoverageFakeJudge(), [SCAFFOLD], "a", [RICH],
                       "b", [BLAND, RICH])


def test_all_pairwise_win_rates_covers_each_unordered_stage_pair():
    scaffolds = [SCAFFOLD]
    stage_fables = {"base": [BLAND], "sft": [RICH], "rlaif": [RICH]}
    tables = all_pairwise_win_rates(SlotCoverageFakeJudge(), scaffolds, stage_fables)
    pairs = {(t["stage_a"], t["stage_b"]) for t in tables}
    assert pairs == {("base", "sft"), ("base", "rlaif"), ("sft", "rlaif")}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_winrate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tinystories_v2.eval'` (or ImportError on the three names).

- [ ] **Step 3: Write minimal implementation**

Create `src/tinystories_v2/eval.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_winrate.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/eval.py tests/test_eval_winrate.py
git commit -m "feat: order-swapped win-rate primitives for the eval stage"
```

---

### Task 2: Reference-free metric aggregation

Aggregate issue 11's metrics over one stage's fables into a single dict, robust to wordless generations (a toy or early checkpoint can emit an empty body).

**Files:**
- Modify: `src/tinystories_v2/eval.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: `tinystories_v2.metrics.{mean_distinct_n, distinct_n, self_bleu, mean_flesch_reading_ease, tokenize_words}`.
- Produces:
  - `reference_free_metrics(fables: list[str], *, self_bleu_sample_size: int | None = None, self_bleu_seed: int = 0) -> dict` — keys `{"mean_distinct_1","distinct_2","self_bleu","mean_flesch_reading_ease","n_usable"}`. Fables with no word tokens are dropped before aggregation; a metric that is undefined for the usable set is `None` (`distinct_2` when the pooled set has no bigram; `self_bleu` when fewer than two usable fables). `perplexity` is **not** computed here — the `run` entrypoint adds it per stage.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_metrics.py`:

```python
"""Per-stage reference-free metric aggregation for the eval stage (issue 07)."""

from tinystories_v2.eval import reference_free_metrics


def test_reference_free_metrics_reports_the_paper_family():
    fables = [
        "The fox shared the grapes and learned that sharing brings friends.",
        "A wise owl watched the moon and taught the mouse to be patient.",
        "The greedy dog dropped his bone chasing a shadow in the pond.",
    ]
    m = reference_free_metrics(fables)
    assert set(m) == {"mean_distinct_1", "distinct_2", "self_bleu",
                      "mean_flesch_reading_ease", "n_usable"}
    assert m["n_usable"] == 3
    assert 0.0 < m["mean_distinct_1"] <= 1.0
    assert isinstance(m["distinct_2"], float)
    assert isinstance(m["self_bleu"], float)
    assert isinstance(m["mean_flesch_reading_ease"], float)


def test_reference_free_metrics_drops_wordless_fables():
    # An empty / whitespace body contributes no words and is dropped; with only
    # one usable fable, Self-BLEU is undefined (None) but Distinct-1 is defined.
    m = reference_free_metrics(["Hello there, small friend.", "", "   "])
    assert m["n_usable"] == 1
    assert m["self_bleu"] is None
    assert m["distinct_2"] is None  # single 4-word fable, but guarded either way
    assert isinstance(m["mean_distinct_1"], float)


def test_reference_free_metrics_all_none_when_no_usable_fables():
    m = reference_free_metrics(["", "   ", "\n"])
    assert m["n_usable"] == 0
    assert m["mean_distinct_1"] is None
    assert m["self_bleu"] is None
    assert m["mean_flesch_reading_ease"] is None


def test_reference_free_metrics_forwards_self_bleu_subsampling():
    fables = [f"story number {i} about a small brave animal today" for i in range(10)]
    full = reference_free_metrics(fables)
    subsampled = reference_free_metrics(fables, self_bleu_sample_size=4, self_bleu_seed=7)
    assert isinstance(full["self_bleu"], float)
    assert isinstance(subsampled["self_bleu"], float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_metrics.py -q`
Expected: FAIL — `ImportError: cannot import name 'reference_free_metrics'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/tinystories_v2/eval.py` (after `all_pairwise_win_rates`):

```python
def reference_free_metrics(fables: list[str], *,
                           self_bleu_sample_size: int | None = None,
                           self_bleu_seed: int = 0) -> dict:
    """Aggregate issue 11's reference-free metrics over one stage's fables.

    Wordless generations (an empty body from an early/toy checkpoint) carry no
    lexical signal and are dropped first. A metric undefined for the usable set
    is None: distinct_2 when no fable has a bigram, self_bleu with fewer than
    two usable fables, and every metric when nothing is usable. Distinct-1 is
    the paper's per-Fable mean (mean_distinct_n); distinct_2 is pooled
    (distinct_n) so short fables don't make it undefined."""
    usable = [f for f in fables if tokenize_words(f)]
    metrics = {
        "n_usable": len(usable),
        "mean_distinct_1": None,
        "distinct_2": None,
        "self_bleu": None,
        "mean_flesch_reading_ease": None,
    }
    if not usable:
        return metrics
    metrics["mean_distinct_1"] = mean_distinct_n(usable, 1)
    metrics["mean_flesch_reading_ease"] = mean_flesch_reading_ease(usable)
    try:
        metrics["distinct_2"] = distinct_n(usable, 2)
    except ValueError:
        metrics["distinct_2"] = None  # no fable has two tokens
    if len(usable) >= 2:
        metrics["self_bleu"] = self_bleu(
            usable, sample_size=self_bleu_sample_size, seed=self_bleu_seed)
    return metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_metrics.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/eval.py tests/test_eval_metrics.py
git commit -m "feat: robust per-stage reference-free metric aggregation"
```

---

### Task 3: Report and sample-sheet rendering

Turn the computed win-rate + metric dicts and the per-stage fables into report-pastable Markdown, with a standalone qualitative sample sheet embedded.

**Files:**
- Modify: `src/tinystories_v2/eval.py`
- Test: `tests/test_eval_report.py`

**Interfaces:**
- Consumes: `tinystories_v2.slot_prompt.SLOT_FIELDS`, `tinystories_v2.slots.Scaffold`; the `results` dict shape produced later by `run` (keys `eval_judge_id`, `n_scaffolds`, `win_rates`, `metrics`).
- Produces:
  - `sample_sheet_md(scaffolds: list[Scaffold], stage_fables: dict[str, list[str]], k: int) -> str`.
  - `render_report(results: dict, sample_sheet: str) -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_report.py`:

```python
"""Markdown rendering of the eval report and sample sheet (issue 07)."""

from tinystories_v2.eval import render_report, sample_sheet_md
from tinystories_v2.slots import Scaffold

SCAFFOLD = Scaffold("fox", "sly", "a wood", "a locked gate",
                    "the fox shared", "sharing brings friends")


def test_sample_sheet_shows_each_stage_side_by_side():
    scaffolds = [SCAFFOLD, SCAFFOLD]
    stage_fables = {"base": ["base body one", "base body two"],
                    "sft": ["sft body one", "sft body two"]}
    sheet = sample_sheet_md(scaffolds, stage_fables, k=1)
    assert "Scaffold 1" in sheet
    assert "Scaffold 2" not in sheet          # k=1 truncates
    assert "sly" in sheet and "the fox shared" in sheet  # slot values rendered
    assert "### base" in sheet and "### sft" in sheet
    assert "base body one" in sheet and "sft body one" in sheet


def test_sample_sheet_marks_empty_bodies():
    sheet = sample_sheet_md([SCAFFOLD], {"base": [""]}, k=1)
    assert "_(empty)_" in sheet


def test_render_report_has_winrate_and_metric_tables_with_counts():
    results = {
        "eval_judge_id": "fake:slot-coverage-v1",
        "n_scaffolds": 3,
        "win_rates": [{"stage_a": "base", "stage_b": "sft", "wins_a": 1,
                       "wins_b": 2, "ties": 0, "skipped": 0, "n": 3}],
        "metrics": {
            "base": {"mean_distinct_1": 0.5, "distinct_2": 0.9,
                     "self_bleu": 0.1, "mean_flesch_reading_ease": 80.0,
                     "perplexity": 42.0, "n_usable": 3},
            "sft": {"mean_distinct_1": 0.6, "distinct_2": None,
                    "self_bleu": None, "mean_flesch_reading_ease": 78.9,
                    "perplexity": 30.0, "n_usable": 1},
        },
    }
    report = render_report(results, "SAMPLE-SHEET-BODY")
    assert "fake:slot-coverage-v1" in report          # eval judge identity
    assert "| base | sft | 1 | 2 | 0 | 0 | 3 |" in report   # win-rate counts
    assert "| base |" in report and "42.000" in report      # metric row
    assert "n/a" in report                            # None -> n/a
    assert report.rstrip().endswith("SAMPLE-SHEET-BODY")     # sheet embedded last
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_report.py -q`
Expected: FAIL — `ImportError: cannot import name 'render_report'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/tinystories_v2/eval.py` (after `reference_free_metrics`):

```python
def sample_sheet_md(scaffolds: list[Scaffold],
                    stage_fables: dict[str, list[str]], k: int) -> str:
    """The first k eval Scaffolds rendered by every stage side by side."""
    names = list(stage_fables)
    lines = ["# Qualitative sample sheet", ""]
    for i, scaffold in enumerate(scaffolds[:k]):
        lines.append(f"## Scaffold {i + 1}")
        for field in SLOT_FIELDS:
            lines.append(f"- **{field}**: {getattr(scaffold, field)}")
        lines.append("")
        for name in names:
            lines.append(f"### {name}")
            lines.append(stage_fables[name][i].strip() or "_(empty)_")
            lines.append("")
    return "\n".join(lines)


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_report(results: dict, sample_sheet: str) -> str:
    """Report-pastable Markdown: win-rate tables, metric tables, sample sheet."""
    lines = [
        "# Evaluation report",
        "",
        f"Eval Judge: `{results['eval_judge_id']}`",
        f"Held-out Scaffolds: {results['n_scaffolds']}",
        "",
        "## Win-rates (order-swapped double judging)",
        "",
        "| A | B | A wins | B wins | ties | skipped | n |",
        "| - | - | ------ | ------ | ---- | ------- | - |",
    ]
    for w in results["win_rates"]:
        lines.append(
            f"| {w['stage_a']} | {w['stage_b']} | {w['wins_a']} | "
            f"{w['wins_b']} | {w['ties']} | {w['skipped']} | {w['n']} |")
    lines += [
        "",
        "## Reference-free metrics",
        "",
        "| stage | Distinct-1 | Distinct-2 | Self-BLEU | Flesch | Perplexity |",
        "| ----- | ---------- | ---------- | --------- | ------ | ---------- |",
    ]
    for name, m in results["metrics"].items():
        lines.append(
            f"| {name} | {_fmt(m['mean_distinct_1'])} | {_fmt(m['distinct_2'])} "
            f"| {_fmt(m['self_bleu'])} | {_fmt(m['mean_flesch_reading_ease'])} "
            f"| {_fmt(m['perplexity'])} |")
    return "\n".join(lines) + "\n\n" + sample_sheet
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_report.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/eval.py tests/test_eval_report.py
git commit -m "feat: eval report and sample-sheet Markdown rendering"
```

---

### Task 4: Stage-model loading and identical-across-stages generation

Load each stage checkpoint (fetching from the Hub on a fresh VM) and generate exactly one seeded completion per Scaffold per stage — driven so that **every stage sees the identical Scaffolds, per-Scaffold seeds, and sampling settings**. This is the criterion that comparisons are apples-to-apples.

**Files:**
- Modify: `src/tinystories_v2/eval.py`
- Test: `tests/test_eval_generate.py`

**Interfaces:**
- Consumes: `tinystories_v2.checkpoint.{latest_checkpoint,load_checkpoint}`, `tinystories_v2.model.{FableLM,ModelConfig}`, `tinystories_v2.hub.fetch_from`, `tinystories_v2.generate.sample`, `tinystories_v2.slot_prompt.{render_prompt,END_TOKEN}`.
- Produces:
  - `load_stage_model(stage_cfg: dict, device: str) -> FableLM` — `stage_cfg` has `name`, `local_dir`, optional `hub_source`.
  - `generate_stage_fables(model, tokenizer, scaffolds: list[Scaffold], seeds: list[int], sampling: dict, *, device: str = "cpu") -> list[str]` — one decoded fable body per Scaffold (prompt prefix and `<|end|>` excluded); a Scaffold whose Slot Prompt exceeds `model.config.context` yields `""`.
  - `generate_all_stages(stage_models: dict[str, FableLM], tokenizer, scaffolds: list[Scaffold], seeds: list[int], sampling: dict, *, device: str = "cpu", generate_fn=None) -> dict[str, list[str]]` — calls `generate_fn(model, tokenizer, scaffolds, seeds, sampling, device=device)` once per stage with the **same** `scaffolds`, `seeds`, and `sampling`; `generate_fn` defaults to `generate_stage_fables` and is injectable for tests.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_generate.py`:

```python
"""Generation is identical across stages (same Scaffolds/seeds/sampling) — the
apples-to-apples criterion — and stage-model loading errors are explicit."""

import pytest

from tinystories_v2.eval import generate_all_stages, load_stage_model
from tinystories_v2.slots import Scaffold

SCAFFOLDS = [Scaffold("fox", "sly", "a wood", "a gate", "it shared", "share"),
             Scaffold("owl", "wise", "a barn", "a storm", "it waited", "wait")]
SEEDS = [11, 22]
SAMPLING = {"max_new_tokens": 8, "temperature": 0.8, "top_p": 0.95, "seed": 1337}


def test_generate_all_stages_feeds_identical_inputs_to_every_stage():
    calls = []

    def spy(model, tokenizer, scaffolds, seeds, sampling, *, device="cpu"):
        calls.append({"scaffolds": scaffolds, "seeds": seeds, "sampling": sampling})
        # A stage-distinct but Scaffold-aligned canned completion.
        return [f"{model}:{s.character}" for s in scaffolds]

    stage_models = {"base": "M_BASE", "sft": "M_SFT"}
    out = generate_all_stages(stage_models, tokenizer=None, scaffolds=SCAFFOLDS,
                              seeds=SEEDS, sampling=SAMPLING, generate_fn=spy)

    assert list(out) == ["base", "sft"]
    assert out["base"] == ["M_BASE:fox", "M_BASE:owl"]
    assert out["sft"] == ["M_SFT:fox", "M_SFT:owl"]
    # Both stages saw the SAME Scaffolds, seeds, and sampling object.
    assert calls[0]["scaffolds"] is calls[1]["scaffolds"] is SCAFFOLDS
    assert calls[0]["seeds"] is calls[1]["seeds"] is SEEDS
    assert calls[0]["sampling"] is calls[1]["sampling"] is SAMPLING


def test_load_stage_model_raises_without_a_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="no checkpoint for stage 'base'"):
        load_stage_model({"name": "base", "local_dir": str(tmp_path / "missing")},
                         device="cpu")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_generate.py -q`
Expected: FAIL — `ImportError: cannot import name 'generate_all_stages'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/tinystories_v2/eval.py` (after `render_report`):

```python
def load_stage_model(stage_cfg: dict, device: str) -> FableLM:
    """Load one stage's FableLM checkpoint, fetching the artifact from the Hub
    first if the local checkpoint is absent (fresh VM). Every stage checkpoint
    is a plain FableLM (base/SFT/RLAIF share the architecture)."""
    local_dir = Path(stage_cfg["local_dir"])
    ckpt_dir = local_dir / "checkpoints"
    if latest_checkpoint(ckpt_dir) is None and stage_cfg.get("hub_source"):
        fetch_from(stage_cfg["hub_source"], local_dir)
    ckpt = latest_checkpoint(ckpt_dir)
    if ckpt is None:
        raise ValueError(
            f"no checkpoint for stage {stage_cfg['name']!r} under {ckpt_dir}; "
            f"point [[stages]].local_dir (and optionally hub_source) at the "
            f"stage artifact")
    state = load_checkpoint(ckpt)
    model = FableLM(ModelConfig(**state["config"]["model"]))
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def generate_stage_fables(model, tokenizer, scaffolds: list[Scaffold],
                          seeds: list[int], sampling: dict, *,
                          device: str = "cpu") -> list[str]:
    """One seeded completion per Scaffold, decoded to a fable body (prompt
    prefix and <|end|> excluded). A Slot Prompt longer than the model context
    yields "" so the caller can skip it rather than crash the whole eval."""
    end_id = tokenizer.token_to_id(END_TOKEN)
    fables = []
    for scaffold, seed in zip(scaffolds, seeds):
        prompt_ids = tokenizer.encode(render_prompt(scaffold)).ids
        if len(prompt_ids) > model.config.context:
            fables.append("")
            continue
        seq = sample(
            model, prompt_ids, num_samples=1,
            max_new_tokens=sampling["max_new_tokens"],
            temperature=sampling["temperature"], top_p=sampling["top_p"],
            seed=seed, end_id=end_id, device=device)[0]
        fables.append(tokenizer.decode(seq[len(prompt_ids):]).strip())
    return fables


def generate_all_stages(stage_models: dict[str, FableLM], tokenizer,
                        scaffolds: list[Scaffold], seeds: list[int],
                        sampling: dict, *, device: str = "cpu",
                        generate_fn=None) -> dict[str, list[str]]:
    """Generate per-stage completions with identical Scaffolds, seeds, and
    sampling across every checkpoint (the apples-to-apples eval contract).
    generate_fn is injectable for tests; it defaults to generate_stage_fables."""
    gen = generate_fn or generate_stage_fables
    return {name: gen(model, tokenizer, scaffolds, seeds, sampling, device=device)
            for name, model in stage_models.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_generate.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/eval.py tests/test_eval_generate.py
git commit -m "feat: stage-model loading and identical-across-stages generation"
```

---

### Task 5: The `run` entrypoint, configs, entry point, and schema

Wire the primitives into the full stage: load stages + eval split + tokenizer + Judge, generate, compute win-rates + metrics + perplexity, and write `results.json` + `report.md`. This task delivers the whole CPU acceptance path.

**Files:**
- Modify: `src/tinystories_v2/eval.py` (add `_read_split`, `_encode_eval_tokens`, `run`, `main`)
- Modify: `pyproject.toml:40` (add `ts2-eval` under `[project.scripts]`)
- Create: `configs/eval_fixture.toml`, `configs/eval_full.toml`
- Create: `docs/schemas/eval-results-v1.md`
- Test: `tests/test_eval_stage.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4, plus `tinystories_v2.judge.build_judge`, `tinystories_v2.perplexity.perplexity`, `tinystories_v2.pref_data.scaffold_seed`, `tinystories_v2.hub.{fetch_file_from,try_sync_to}`, `tinystories_v2.config.{load_config,load_env}`.
- Produces:
  - `run(config: dict, *, generate_fn=None) -> dict` — returns the `results` dict and writes `results.json` + `report.md` under `config["out_dir"]`; syncs to `config["hub"]["target"]` when set.
  - `main(argv: list[str] | None = None) -> None` — `--config` only (single-pass; no `--resume`).
  - Results dict shape (also `results.json`): `{"stage":"eval","package_version",str,"eval_judge_id":str,"sampling":{max_new_tokens,temperature,top_p,seed},"eval_scaffolds":[prompt_hash,...],"n_scaffolds":int,"stages":[name,...],"win_rates":[table,...],"metrics":{name:{mean_distinct_1,distinct_2,self_bleu,mean_flesch_reading_ease,n_usable,perplexity}},"config":dict}`.

- [ ] **Step 1: Add the entry point and both configs**

Edit `pyproject.toml` — after the `ts2-pref-data` line (`pyproject.toml:40`), add:

```toml
ts2-eval = "tinystories_v2.eval:main"
```

Create `configs/eval_fixture.toml`:

```toml
# Toy CPU evaluation smoke against fixture artifacts — local sanity runs / docs.
# Assumes upstream fixture stages have produced the eval split, a tokenizer, and
# at least one checkpoint. Uses the deterministic fake Judge (no GPU/network).
out_dir = "artifacts/eval_fixture"
max_eval_scaffolds = 16
sample_sheet_k = 4

[data]
eval_split = "artifacts/data_prep_fixture/splits/eval.jsonl"
tokenizer = "artifacts/tokenizer_fixture/tokenizer.json"

[[stages]]
name = "base"
local_dir = "artifacts/pretrain_fixture"

[[stages]]
name = "sft"
local_dir = "artifacts/sft_fixture"

[sampling]
max_new_tokens = 32
temperature = 0.8
top_p = 0.95
seed = 1337

[metrics]
self_bleu_sample_size = 0
self_bleu_seed = 0

[judge]
kind = "fake_slot_coverage"
```

Create `configs/eval_full.toml`:

```toml
# Real evaluation run on Colab Pro. Cross-family eval Judge:
# Llama-3.1-8B-Instruct fp16 on the L4 (on a T4, use 4-bit — set
# model_id/precision accordingly). Never the Qwen Judge that produced the
# reward signal (self-preference bias). Order-swapped double judging is applied
# inside the stage. Stage checkpoints are pulled from the Hub on a fresh VM via
# each [[stages]].hub_source. Add the RLAIF stage block once issue 06 lands.
out_dir = "artifacts/eval_full"
max_eval_scaffolds = 200       # cap so a run finishes within an L4 session
sample_sheet_k = 8

[data]
eval_split = "artifacts/data_prep_full/splits/eval.jsonl"
hub_source = "hf://congthanh991/tinystories-v2-data"
tokenizer = "artifacts/tokenizer_full/tokenizer.json"
tokenizer_hub_source = "hf://congthanh991/tinystories-v2-tokenizer"

[[stages]]
name = "base"
local_dir = "artifacts/pretrain_full"
hub_source = "hf://congthanh991/tinystories-v2-pretrain"

[[stages]]
name = "sft"
local_dir = "artifacts/sft_full"
hub_source = "hf://congthanh991/tinystories-v2-sft"

# [[stages]]
# name = "rlaif"
# local_dir = "artifacts/grpo_full"
# hub_source = "hf://congthanh991/tinystories-v2-grpo"

[sampling]
max_new_tokens = 400
temperature = 0.8
top_p = 0.95
seed = 1337

[metrics]
self_bleu_sample_size = 200
self_bleu_seed = 0

[judge]
kind = "transformers"
model_id = "meta-llama/Llama-3.1-8B-Instruct"
precision = "fp16"
device = "cuda"
max_new_tokens = 4

[hub]
target = "hf://congthanh991/tinystories-v2-eval"
```

- [ ] **Step 2: Write the failing stage test**

Create `tests/test_eval_stage.py`:

```python
"""End-to-end eval stage on CPU with the fake Judge and toy checkpoints:
produces the results artifact (win-rate tables with counts, metric tables,
sample sheet), records the eval-Judge identity, and adds the RLAIF column only
when a third stage is configured."""

import json
import subprocess
import sys
from pathlib import Path

from tinystories_v2 import eval as eval_stage
from tinystories_v2.data import run as data_run
from tinystories_v2.tokenizer import run as tokenizer_run

TOY_MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 128, "ffn_hidden": 192}


def _prepare_inputs(base, fixture_path, make_init_checkpoint, stage_names):
    """A tokenizer, an eval split, and one toy checkpoint per stage name."""
    data_dir, tok_dir = base / "data", base / "tok"
    data_run({
        "out_dir": str(data_dir), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.1, "sft": 0.6,
                   "pref": 0.1, "eval": 0.2},
    })
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer_path = tok_dir / "tokenizer.json"
    stages = []
    for name in stage_names:
        stage_dir = base / name
        make_init_checkpoint(stage_dir, TOY_MODEL, tokenizer_path)
        stages.append({"name": name, "local_dir": str(stage_dir)})
    return {
        "out_dir": str(base / "out"),
        "max_eval_scaffolds": 16,
        "sample_sheet_k": 3,
        "data": {"eval_split": str(data_dir / "splits" / "eval.jsonl"),
                 "tokenizer": str(tokenizer_path)},
        "stages": stages,
        "sampling": {"max_new_tokens": 24, "temperature": 0.8, "top_p": 0.95,
                     "seed": 1337},
        "metrics": {"self_bleu_sample_size": 0, "self_bleu_seed": 0},
        "judge": {"kind": "fake_slot_coverage"},
    }


def test_eval_stage_produces_the_results_artifact(tmp_path, fixture_path,
                                                  make_init_checkpoint):
    config = _prepare_inputs(tmp_path, fixture_path, make_init_checkpoint,
                             ["base", "sft"])
    results = eval_stage.run(config)

    out = Path(config["out_dir"])
    assert (out / "results.json").exists() and (out / "report.md").exists()
    saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert saved["stage"] == "eval"
    # Criterion: the eval-Judge identity is recorded in the artifact.
    assert saved["eval_judge_id"] == "fake:slot-coverage-v1"
    assert results["eval_judge_id"] == saved["eval_judge_id"]
    # Win-rate table for the single base-vs-sft pair, with counts summing to n.
    assert saved["stages"] == ["base", "sft"]
    assert len(saved["win_rates"]) == 1
    w = saved["win_rates"][0]
    assert (w["stage_a"], w["stage_b"]) == ("base", "sft")
    assert w["wins_a"] + w["wins_b"] + w["ties"] + w["skipped"] == w["n"]
    assert w["n"] == saved["n_scaffolds"]
    # Metric table per stage, with perplexity attached.
    assert set(saved["metrics"]) == {"base", "sft"}
    for m in saved["metrics"].values():
        assert set(m) >= {"mean_distinct_1", "distinct_2", "self_bleu",
                          "mean_flesch_reading_ease", "n_usable", "perplexity"}
        assert isinstance(m["perplexity"], float)
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## Win-rates" in report and "Qualitative sample sheet" in report


def test_rlaif_column_appears_only_with_a_third_stage(tmp_path, fixture_path,
                                                      make_init_checkpoint):
    two = _prepare_inputs(tmp_path / "two", fixture_path, make_init_checkpoint,
                          ["base", "sft"])
    assert set(eval_stage.run(two)["metrics"]) == {"base", "sft"}

    three = _prepare_inputs(tmp_path / "three", fixture_path, make_init_checkpoint,
                            ["base", "sft", "rlaif"])
    r = eval_stage.run(three)
    assert set(r["metrics"]) == {"base", "sft", "rlaif"}
    # C(3,2) = 3 pairwise comparisons.
    assert len(r["win_rates"]) == 3


def test_identical_scaffolds_and_sampling_across_stages(tmp_path, fixture_path,
                                                        make_init_checkpoint):
    # Criterion asserted here at the stage level: every stage's generate_fn call
    # receives the same Scaffolds and sampling settings.
    config = _prepare_inputs(tmp_path, fixture_path, make_init_checkpoint,
                             ["base", "sft"])
    seen = []

    def spy(model, tokenizer, scaffolds, seeds, sampling, *, device="cpu"):
        seen.append((tuple(scaffolds), tuple(seeds), tuple(sorted(sampling.items()))))
        return ["a small fable body about a brave animal" for _ in scaffolds]

    eval_stage.run(config, generate_fn=spy)
    assert len(seen) == 2 and seen[0] == seen[1]  # base and sft got identical inputs


def test_cli_entrypoint_runs_standalone(tmp_path, fixture_path, make_init_checkpoint):
    config = _prepare_inputs(tmp_path, fixture_path, make_init_checkpoint,
                             ["base", "sft"])
    config_file = tmp_path / "eval.toml"
    config_file.write_text(eval_stage._to_toml(config), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.eval", "--config", str(config_file)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (Path(config["out_dir"]) / "results.json").exists()
```

Note: the CLI test uses a small `_to_toml` helper the stage exposes for tests (arrays-of-tables serialization; stdlib has no TOML writer). It is added in Step 3.

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_stage.py -q`
Expected: FAIL — `AttributeError: module 'tinystories_v2.eval' has no attribute 'run'`.

- [ ] **Step 4: Implement `run`, `main`, and the test helper**

Add to `src/tinystories_v2/eval.py` (after `generate_all_stages`):

```python
def _read_split(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _encode_eval_tokens(tokenizer, rows: list[dict]) -> list[int]:
    """Flatten the eval fables into one held-out token stream for perplexity."""
    ids: list[int] = []
    for row in rows:
        ids.extend(tokenizer.encode(row["fable"]).ids)
    return ids


def run(config: dict, *, generate_fn=None) -> dict:
    load_env()  # HF token for hub fetch/sync — never printed
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = config["data"]
    split_path = Path(data["eval_split"])
    if not split_path.exists() and data.get("hub_source"):
        fetch_file_from(data["hub_source"], "splits/eval.jsonl", split_path)
    rows = _read_split(split_path)
    max_scaffolds = config.get("max_eval_scaffolds", 0)
    if max_scaffolds:
        rows = rows[:max_scaffolds]
    if not rows:
        raise ValueError(f"no eval Scaffolds in {split_path}")

    tokenizer_path = Path(data["tokenizer"])
    if not tokenizer_path.exists() and data.get("tokenizer_hub_source"):
        fetch_file_from(data["tokenizer_hub_source"], "tokenizer.json", tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    scaffolds = [Scaffold(**{f: row[f] for f in SLOT_FIELDS}) for row in rows]
    sampling = config["sampling"]
    seeds = [scaffold_seed(sampling["seed"], row["prompt_hash"]) for row in rows]

    stage_models = {s["name"]: load_stage_model(s, device) for s in config["stages"]}
    stage_fables = generate_all_stages(
        stage_models, tokenizer, scaffolds, seeds, sampling,
        device=device, generate_fn=generate_fn)

    judge = build_judge(config["judge"])
    win_rates = all_pairwise_win_rates(judge, scaffolds, stage_fables)

    metrics_cfg = config.get("metrics", {})
    sample_size = metrics_cfg.get("self_bleu_sample_size") or None
    eval_ids = _encode_eval_tokens(tokenizer, rows)
    metrics = {}
    for name, model in stage_models.items():
        m = reference_free_metrics(
            stage_fables[name], self_bleu_sample_size=sample_size,
            self_bleu_seed=metrics_cfg.get("self_bleu_seed", 0))
        m["perplexity"] = perplexity(
            model, eval_ids, block_size=model.config.context, device=device)
        metrics[name] = m

    sheet = sample_sheet_md(scaffolds, stage_fables, config.get("sample_sheet_k", 8))
    results = {
        "stage": "eval",
        "package_version": __version__,
        "eval_judge_id": judge.judge_id,
        "sampling": {key: sampling[key]
                     for key in ("max_new_tokens", "temperature", "top_p", "seed")},
        "eval_scaffolds": [row["prompt_hash"] for row in rows],
        "n_scaffolds": len(rows),
        "stages": list(stage_models),
        "win_rates": win_rates,
        "metrics": metrics,
        "config": config,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2),
                                          encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(results, sheet),
                                       encoding="utf-8")
    hub_target = config.get("hub", {}).get("target")
    if hub_target:
        try_sync_to(hub_target, out_dir)
    print(f"eval done: {len(rows)} Scaffolds, {len(stage_models)} stages, "
          f"judge {judge.judge_id}")
    return results


def _to_toml(config: dict) -> str:
    """Serialize an eval config to TOML for tests (stdlib has no writer).
    Handles the top-level scalars, [[stages]] array-of-tables, and the nested
    [data]/[sampling]/[metrics]/[judge]/[hub] sections used by this stage."""
    def scalar(value) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, str):
            return f'"{value}"'
        return str(value)

    lines = []
    for key, value in config.items():
        if not isinstance(value, (dict, list)):
            lines.append(f"{key} = {scalar(value)}")
    for stage in config["stages"]:
        lines.append("[[stages]]")
        for key, value in stage.items():
            lines.append(f"{key} = {scalar(value)}")
    for section in ("data", "sampling", "metrics", "judge", "hub"):
        if section not in config:
            continue
        lines.append(f"[{section}]")
        for key, value in config[section].items():
            lines.append(f"{key} = {scalar(value)}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create the results schema doc**

Create `docs/schemas/eval-results-v1.md`:

```markdown
# eval-results-v1

The evaluation stage (`ts2-eval`, issue 07) writes two artifacts to its
`out_dir`: `results.json` (machine-readable, this schema) and `report.md`
(report-pastable win-rate tables, metric tables, and the qualitative sample
sheet). Stages share nothing in memory; downstream readers consume `results.json`.

## `results.json`

| Field | Type | Meaning |
|-------|------|---------|
| `stage` | string | Always `"eval"`. |
| `package_version` | string | `tinystories_v2.__version__` at run time. |
| `eval_judge_id` | string | Identity of the cross-family eval Judge (`judge.judge_id`). Records "who judged" so the report is never ambiguous. Never a Qwen Judge id. |
| `sampling` | object | Shared decoding settings applied to **every** stage: `max_new_tokens`, `temperature`, `top_p`, `seed`. |
| `eval_scaffolds` | string[] | The held-out eval Scaffold `prompt_hash`es scored, in order — the same set for every stage. |
| `n_scaffolds` | int | `len(eval_scaffolds)`. |
| `stages` | string[] | Stage names in config order (e.g. `["base","sft"]`, `["base","sft","rlaif"]`). |
| `win_rates` | object[] | One entry per unordered stage pair. Each: `stage_a`, `stage_b`, `wins_a`, `wins_b`, `ties`, `skipped`, `n`. A win requires consistency under order-swapped double judging; `ties` are inconsistent verdicts; `skipped` are degenerate (empty/identical) comparisons; `wins_a + wins_b + ties + skipped == n`. |
| `metrics` | object | Keyed by stage name. Each value: `mean_distinct_1`, `distinct_2`, `self_bleu`, `mean_flesch_reading_ease` (issue 11's reference-free metrics; `null` when undefined for the usable set), `n_usable` (fables with word tokens), and `perplexity` (held-out perplexity of that checkpoint on the eval fables). |
| `config` | object | The exact TOML config, echoed for provenance. |

## Guarantees

- The eval Judge is selected via issue 10's Judge interface (`kind` in `[judge]`)
  and is cross-family (Llama-3.1-8B-Instruct for real runs), never the Qwen
  Judge that produced the reward signal.
- All stages are scored on identical Scaffolds and identical sampling settings.
- The RLAIF column is present only when a third `[[stages]]` block is configured.
```

- [ ] **Step 6: Run the stage test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_stage.py -q`
Expected: PASS (4 passed).

- [ ] **Step 7: Reinstall the package so the new entry point resolves, then commit**

Run: `.venv/bin/pip install -e . -q && .venv/bin/ts2-eval --help`
Expected: the argparse usage line for `ts2-eval` prints (exit 0).

```bash
git add src/tinystories_v2/eval.py pyproject.toml configs/eval_fixture.toml \
        configs/eval_full.toml docs/schemas/eval-results-v1.md tests/test_eval_stage.py
git commit -m "feat: ts2-eval stage entrypoint, configs, and results schema"
```

---

### Task 6: Colab bootstrap and thin notebook

One-command real-run bootstrap (`scripts/eval_colab.py`) plus a thin notebook, mirroring the reward/SFT Colab pattern.

**Files:**
- Create: `scripts/eval_colab.py`
- Create: `notebooks/eval_colab.ipynb`
- Test: `tests/test_eval_colab.py`
- Modify: `tests/test_notebook.py` (append eval-notebook thinness + no-secrets tests)

**Interfaces:**
- Consumes: `tinystories_v2.eval` (as `eval` module), `tinystories_v2.config.{load_config,load_env}`, `tinystories_v2.hub_download.download_file` (monkeypatched in tests).
- Produces:
  - `prepare(eval_config, *, download=None) -> tuple[Path, Path]` — ensures the tokenizer and eval split are present locally (download if missing); returns `(tokenizer_path, eval_split_path)`. Idempotent (existence-guarded).
  - `main(argv: list[str] | None = None) -> None` — `--eval-config` (default `configs/eval_full.toml`), `--skip-eval` (download only).

- [ ] **Step 1: Write the failing bootstrap test**

Create `tests/test_eval_colab.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_colab.py -q`
Expected: FAIL — `FileNotFoundError`/`spec_from_file_location` fails because `scripts/eval_colab.py` does not exist.

- [ ] **Step 3: Write the bootstrap**

Create `scripts/eval_colab.py`:

```python
"""One-command real evaluation run for Colab (issue 07).

Turns a fresh L4 VM into a running eval job with a single command. Idempotent:
re-running after an L4 preemption skips already-present downloads and re-runs
the (single-pass) eval stage. The stage pulls each stage checkpoint from its
[[stages]].hub_source on a fresh VM.

Steps:
  1. load .env secrets (HF_TOKEN) so Hub download/sync work
  2. download tokenizer.json + the held-out eval split (issue 01's data repo)
     from the Hub (retry-wrapped) if the local copies are absent
  3. run the eval stage (ts2-eval): generate per-stage completions, score
     cross-family win-rates, compute reference-free metrics + perplexity, and
     write results.json + report.md (synced to [hub].target when configured)

Run on the VM (in-kernel, survives disconnects — never nohup-detach):
    python scripts/eval_colab.py            # download + eval
    python scripts/eval_colab.py --skip-eval    # download only
    colab exec -f scripts/eval_colab.py

See docs/colab-notes.md for the CLI gotchas. The eval split lives in issue 01's
data repo (`DATA_REPO`); edit the constants below if it moves.
"""

import argparse
from pathlib import Path

from tinystories_v2 import eval
from tinystories_v2.config import load_config, load_env
from tinystories_v2.hub_download import download_file  # noqa: F401 — monkeypatched in tests

TOKENIZER_REPO = "congthanh991/tinystories-v2-tokenizer"
DATA_REPO = "congthanh991/tinystories-v2-data"
EVAL_FILENAME = "splits/eval.jsonl"
DEFAULT_EVAL_CONFIG = "configs/eval_full.toml"


def prepare(eval_config, *, download=None) -> tuple[Path, Path]:
    """Ensure the tokenizer + eval split are present (download if missing).
    Returns (tokenizer_path, eval_split_path). `download` is injectable for
    tests; it defaults to download_file. Idempotent: each step is guarded by an
    existence check, so re-running on a warm VM is a no-op up to the eval run."""
    if download is None:
        download = download_file
    cfg = load_config(eval_config)
    tokenizer_path = Path(cfg["data"]["tokenizer"])
    eval_path = Path(cfg["data"]["eval_split"])   # .../<data_dir>/splits/eval.jsonl

    if not tokenizer_path.exists():
        print(f"[eval_colab] downloading tokenizer -> {tokenizer_path}")
        download(TOKENIZER_REPO, "tokenizer.json", tokenizer_path.parent)
    if not eval_path.exists():
        print(f"[eval_colab] downloading eval split -> {eval_path}")
        # download_file writes to local_dir/filename, so the data dir (parent of
        # splits/) is the local_dir and the filename keeps its "splits/" prefix.
        download(DATA_REPO, EVAL_FILENAME, eval_path.parent.parent)
    return tokenizer_path, eval_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG, type=Path)
    parser.add_argument("--skip-eval", action="store_true",
                        help="download the tokenizer + eval split only, then stop")
    args = parser.parse_args(argv)

    load_env()  # HF_TOKEN reaches Hub download/sync; never printed
    prepare(args.eval_config)
    if args.skip_eval:
        print("[eval_colab] --skip-eval: inputs ready; skipping eval")
        return
    print("[eval_colab] starting evaluation (ts2-eval)")
    eval.run(load_config(args.eval_config))
    print("[eval_colab] done: results.json + report.md written")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the bootstrap test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_colab.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Create the thin notebook**

Create `notebooks/eval_colab.ipynb` with this exact JSON (thin wrapper; no outputs, no secrets, invokes the bootstrap):

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# TinyStories v2 — Evaluation (issue 07)\n",
    "\n",
    "Cross-family win-rates, reference-free metrics, and the qualitative sample\n",
    "sheet. One-command bootstrap: downloads the tokenizer + held-out eval split,\n",
    "pulls each stage checkpoint from the Hub, and writes `results.json` +\n",
    "`report.md`. Real runs need the Judge extra (Llama-3.1-8B-Instruct)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!git clone https://github.com/congthanh991/tinystories_v2.git\n",
    "%cd tinystories_v2\n",
    "!pip install -q -e '.[judge]'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Upload `.env` (HF_TOKEN) before running the next cell — see\n",
    "`docs/colab-notes.md`. The eval Judge is Llama-3.1-8B-Instruct; edit\n",
    "`configs/eval_full.toml` for a T4 (4-bit) or to add the RLAIF stage."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!python scripts/eval_colab.py --eval-config configs/eval_full.toml"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {"gpuType": "L4", "provenance": []},
  "kernelspec": {"display_name": "Python 3", "name": "python3"},
  "language_info": {"name": "python"}
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
```

- [ ] **Step 6: Append the notebook thinness tests**

Add to the end of `tests/test_notebook.py`:

```python
EVAL_NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "eval_colab.ipynb"


def test_eval_notebook_is_thin():
    cells = json.loads(EVAL_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert 1 <= len(code_cells) <= 4
    source = "\n".join("".join(c["source"]) for c in code_cells)
    for forbidden in ("def ", "class ", "import torch", "for ", "while "):
        assert forbidden not in source, forbidden
    # Turnkey: the notebook invokes the one-command bootstrap, not the stage.
    assert "scripts/eval_colab.py" in source


def test_eval_notebook_has_no_secrets_or_outputs():
    text = EVAL_NOTEBOOK.read_text(encoding="utf-8")
    assert "hf_" not in text
    cells = json.loads(text)["cells"]
    assert all(not c.get("outputs") for c in cells)
```

- [ ] **Step 7: Run the notebook tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_notebook.py -q`
Expected: PASS (all notebook tests, including the two new eval ones).

- [ ] **Step 8: Commit**

```bash
git add scripts/eval_colab.py notebooks/eval_colab.ipynb tests/test_eval_colab.py \
        tests/test_notebook.py
git commit -m "feat: one-command eval Colab bootstrap and thin notebook"
```

---

### Task 7: Full-suite verification and progress update

Run the whole test suite to confirm nothing regressed, then record issue 07 as code-complete.

**Files:**
- Modify: `PROGRESS.md`

**Interfaces:** none (documentation + verification only).

- [ ] **Step 1: Run the entire test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — all pre-existing tests plus the new `test_eval_*` files (win-rate, metrics, report, generate, stage, colab) and the two eval notebook tests. Zero failures.

- [ ] **Step 2: Update the issue board and Now section in `PROGRESS.md`**

In the `## Issue board` table, change the issue 07 row from:

```markdown
| 07 | Evaluation suite | 03 ✅, 10 ✅, 11 ✅ | 🟢 ready (code work) |
```

to:

```markdown
| 07 | Evaluation suite | 03 ✅, 10 ✅, 11 ✅ | ✅ code complete (real run needs stage checkpoints on Hub) |
```

In the `## Now` list, update the highest-leverage line so 07 is no longer listed as ready code-work (it moves to done); leave 06, 08, 09 as the remaining code-work grabs.

- [ ] **Step 3: Add a Log entry at the top of the `## Log` list in `PROGRESS.md`**

```markdown
- **2026-07-12** — Issue 07 (evaluation suite) code complete: `eval.py` stage
  (`ts2-eval`) — order-swapped cross-family win-rates over stage checkpoints,
  issue 11's reference-free metrics + held-out perplexity per stage, and a
  report-pastable sample sheet, writing `results.json` (schema:
  `docs/schemas/eval-results-v1.md`) + `report.md`. Generation feeds every
  stage identical Scaffolds/seeds/sampling (asserted); the eval Judge is
  config-selected via the issue 10 interface (Llama-3.1-8B-Instruct for real
  runs, never the Qwen reward Judge) and its identity is recorded in the
  artifact. Works with base+SFT alone; the RLAIF column appears when a third
  `[[stages]]` block is configured. `configs/eval_{fixture,full}.toml`, the
  one-command `scripts/eval_colab.py` bootstrap, and a thin `eval_colab.ipynb`
  landed with tests green (win-rate/metric/report/generate units, a CPU stage
  test on toy checkpoints with the fake Judge, bootstrap orchestration, and
  notebook thinness).
```

- [ ] **Step 4: Commit**

```bash
git add PROGRESS.md
git commit -m "docs: mark issue 07 (evaluation suite) code complete"
```

---

## Self-Review

**1. Spec coverage** (issue 07 acceptance criteria):
- "Eval stage runs on CPU with the fake Judge and toy checkpoints, producing a results artifact with win-rate tables (with counts), metric tables, and the sample sheet" → Task 5 `test_eval_stage_produces_the_results_artifact` (fake Judge, toy checkpoints, asserts `results.json`/`report.md`, win-rate counts summing to `n`, per-stage metric tables, sample-sheet section).
- "Eval judge is config-selected via the Judge interface; a test guards that the eval-judge identity is recorded in the results artifact" → `build_judge(config["judge"])` in `run`; `results["eval_judge_id"]`; asserted in `test_eval_stage_produces_the_results_artifact`.
- "Comparisons use identical Scaffolds and sampling settings across checkpoints, asserted by a test" → Task 4 `test_generate_all_stages_feeds_identical_inputs_to_every_stage` (unit) + Task 5 `test_identical_scaffolds_and_sampling_across_stages` (stage-level spy).
- "Works with only base+SFT present; RLAIF column appears when a third checkpoint is configured" → Task 5 `test_rlaif_column_appears_only_with_a_third_stage`.
- "Thin Colab notebook exists for the real eval run" → Task 6 notebook + `test_eval_notebook_is_thin`.
- PRD extras: cross-family Judge = Llama-3.1-8B-Instruct (`configs/eval_full.toml`, never Qwen — schema guarantee); order-swapped double judging (`stage_win`); Self-BLEU / Distinct-n / Flesch + held-out perplexity via issue 11 (`reference_free_metrics` + `perplexity`); qualitative sample sheet (`sample_sheet_md`).

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N" — every code and test step contains complete, runnable content. Config files, notebook JSON, and schema doc are given in full.

**3. Type consistency:** Names are stable across tasks — `stage_win`/`win_rate_table`/`all_pairwise_win_rates` (Task 1) → consumed by `run` (Task 5); `reference_free_metrics` keys `mean_distinct_1`/`distinct_2`/`self_bleu`/`mean_flesch_reading_ease`/`n_usable` (Task 2) → rendered by `render_report` and extended with `perplexity` in `run` (Tasks 3, 5); `sample_sheet_md`/`render_report` (Task 3) → called by `run` (Task 5); `load_stage_model`/`generate_stage_fables`/`generate_all_stages` with the `generate_fn(model, tokenizer, scaffolds, seeds, sampling, device=...)` signature (Task 4) → used identically by `run` and the stage spy (Tasks 4, 5); `prepare` returning `(tokenizer_path, eval_split_path)` and constants `TOKENIZER_REPO`/`DATA_REPO`/`EVAL_FILENAME` (Task 6) → referenced by `tests/test_eval_colab.py`. `run(config, *, generate_fn=None)` matches every call site. The `_to_toml` helper used by the CLI test is defined in the same module (Task 5).

**Note for the executor:** commands assume the repo's virtualenv at `.venv/` (adjust the `python`/`pytest` prefix if your shell already has it activated). Task 5 Step 7's `pip install -e .` is required before `ts2-eval` resolves on PATH; the module-form CLI test (`python -m tinystories_v2.eval`) works without it.
