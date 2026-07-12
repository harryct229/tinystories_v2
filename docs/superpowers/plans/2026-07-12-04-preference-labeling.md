# Preference Labeling Stage (Issue 04) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ts2-pref-data`, the resumable offline stage that samples N completions per pref-split Scaffold from an SFT checkpoint, labels 2–3 pairs per Scaffold with the config-selected Judge through order-swap consistency filtering, and accumulates one growing, schema-valid preference-pair artifact across Colab sessions synced to the Hub.

**Scope note:** Code work only. The real labeling run consumes issue 03's real SFT checkpoint on `hf://congthanh991/tinystories-v2-sft` — which now exists (issue 03's run completed 2026-07-12: `checkpoints/step_000800.pt`, final masked loss 1.083). This plan delivers the stage, tests (CPU, fake Judge, fixture data), configs, and the thin Colab notebook; the real labeling run is a one-cell job as soon as this branch lands.

**Architecture:** One new stage module `src/tinystories_v2/pref_data.py` following the repo's stage convention (one TOML config → artifacts in `out_dir`). It composes existing seams: `generate.sample` (issue 02) for completions, `judge.build_judge` + `judge_with_order_swap` (issue 10) for labeling, `preferences.validate_preference_pair` for the v1 record, `hub.sync_to/fetch_from` for Hub persistence. Kill-safety comes from a per-Scaffold commit protocol: append pairs to `pairs.jsonl` + fsync, then atomically replace `progress.json`; resume truncates any uncommitted trailing lines. Per-Scaffold sampling seeds derive from `(seed, prompt_hash)` so a resumed run is byte-identical to an uninterrupted one.

**Tech Stack:** Python 3.11+ (repo venv is 3.14), PyTorch (CPU in tests), `tokenizers`, `huggingface_hub`, pytest. No new dependencies.

## Global Constraints

- No new dependencies; imports limited to what `pyproject.toml` already declares (`torch`, `tokenizers`, `huggingface_hub`, stdlib).
- All tests run on CPU with **no GPU, no network, no model download** (fake Judges only; sampling model is a tiny random-weight `FableLM`).
- Design-doc sampling defaults (docs/DESIGN.md "Preference data"), pinned verbatim in `configs/pref_data_full.toml`: `num_completions = 4`, `temperature = 1.0`, `top_p = 0.95`, `pairs_per_scaffold = 3` (design says 2–3 pairs; default to 3).
- Real Judge default (docs/DESIGN.md): `Qwen/Qwen3-8B`, `fp16`, `device = "cuda"` on L4; T4 fallback is `Qwen/Qwen3-4B-Instruct-2507` (config edit, documented in the notebook).
- Preference-pair schema v1 is strict: exactly five top-level fields, extra fields rejected (`src/tinystories_v2/preferences.py`). Therefore resume bookkeeping (prompt hashes, counters) lives in `progress.json`, **never** inside pair records.
- Stage convention: entrypoint reads one TOML via `load_config`, writes artifacts under `out_dir`; stages couple only through artifacts on disk.
- Colab notebooks stay thin (enforced by tests/test_notebook.py conventions): 1–4 code cells; code cells must not contain `def `, `class `, `import torch`, `for `, or `while `; no cell outputs committed; the literal lowercase string `hf_` must not appear anywhere in the notebook file.
- Hub repos (private, under `congthanh991`): data splits `tinystories-v2-data`, tokenizer `tinystories-v2-tokenizer`, SFT checkpoint `tinystories-v2-sft`, and the new pairs artifact `tinystories-v2-pref-pairs`.
- Style: match the repo — module docstring explains invocation + artifacts + contracts; ~88-col lines; double quotes; comments only for non-obvious constraints.
- Branch: `issue-04-preference-labeling` off `main`.

## File Structure

- **Create** `src/tinystories_v2/pref_data.py` — the entire stage: pair scheduling, per-Scaffold seeding, completion sampling, pair labeling, kill-safe progress store, `run()`/`main()`. One file per the repo's one-module-per-stage pattern.
- **Modify** `src/tinystories_v2/hub.py` — add `fetch_file_from` (fetch a single file from a sync target; needed because `fetch_from` snapshots a whole repo and the data repo contains the ~1 GB pretrain split).
- **Modify** `pyproject.toml` — register `ts2-pref-data` console script.
- **Create** `configs/pref_data_fixture.toml` — CPU smoke config (fake Judge, fixture artifacts).
- **Create** `configs/pref_data_full.toml` — real L4 run config (Qwen3-8B Judge, Hub sources/target, design defaults).
- **Create** `notebooks/pref_data_colab.ipynb` — thin Colab wrapper.
- **Create** `tests/test_pref_data.py` — unit tests: pair schedule, seeds, sampling, labeling, progress store.
- **Create** `tests/test_pref_data_stage.py` — stage tests: artifact contract, determinism, config-selected Judge, resume, Hub wiring (local-path targets), config pinning.
- **Create** `tests/test_pref_data_resume.py` — subprocess SIGKILL kill-and-resume test.
- **Modify** `tests/test_hub_sync.py` — `fetch_file_from` tests.
- **Modify** `tests/test_notebook.py` — thinness/secrets tests for the new notebook.
- **Modify** `PROGRESS.md` and `.scratch/tinystories-v2-pipeline/issues/04-judge-seam-preference-labeling.md` — status updates.

Final import block of `src/tinystories_v2/pref_data.py` once all tasks land (tasks add imports incrementally; this is the end state for reference):

```python
import argparse
import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinystories_v2 import __version__
from tinystories_v2.checkpoint import latest_checkpoint, load_checkpoint
from tinystories_v2.config import load_config, load_env
from tinystories_v2.generate import sample
from tinystories_v2.hub import fetch_file_from, fetch_from, try_sync_to
from tinystories_v2.judge import (
    Judge, build_judge, judge_with_order_swap, normalize_text,
)
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.preferences import SCHEMA_VERSION, PreferencePair
from tinystories_v2.slot_prompt import END_TOKEN, SLOT_FIELDS, render_prompt
from tinystories_v2.slots import Scaffold
```

---

### Task 1: Pair schedule and per-Scaffold seed helpers

**Files:**
- Create: `src/tinystories_v2/pref_data.py`
- Test: `tests/test_pref_data.py`

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces:
  - `pair_indices(n_completions: int, n_pairs: int) -> list[tuple[int, int]]` — deterministic round-robin schedule of completion-index pairs, each tuple `(i, j)` with `i < j`; raises `ValueError` if `n_completions < 2` or `n_pairs` not in `[1, C(n,2)]`.
  - `scaffold_seed(seed: int, prompt_hash: str) -> int` — deterministic per-Scaffold sampling seed in `[0, 2**63)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pref_data.py`:

```python
"""Unit seams of the preference-labeling stage: pair scheduling, per-Scaffold
seeding, completion sampling, order-swap pair labeling, and the kill-safe
progress store."""

import pytest

from tinystories_v2.pref_data import pair_indices, scaffold_seed


def test_pair_indices_covers_all_completions_before_repeating():
    # Round-robin: the first two pairs are disjoint, so N=4 completions all
    # appear before any completion is reused.
    assert pair_indices(4, 3) == [(0, 3), (1, 2), (0, 2)]


def test_pair_indices_full_schedule_is_every_unique_pair():
    schedule = pair_indices(4, 6)
    assert len(set(schedule)) == 6
    assert all(a < b for a, b in schedule)


def test_pair_indices_handles_odd_completion_counts():
    assert pair_indices(3, 3) == [(1, 2), (0, 2), (0, 1)]


def test_pair_indices_rejects_bad_counts():
    with pytest.raises(ValueError):
        pair_indices(4, 7)   # only C(4,2) = 6 pairs exist
    with pytest.raises(ValueError):
        pair_indices(4, 0)
    with pytest.raises(ValueError):
        pair_indices(1, 1)


def test_scaffold_seed_is_deterministic_and_input_sensitive():
    assert scaffold_seed(1337, "abc") == scaffold_seed(1337, "abc")
    assert scaffold_seed(1337, "abc") != scaffold_seed(1337, "abd")
    assert scaffold_seed(1, "abc") != scaffold_seed(2, "abc")
    assert 0 <= scaffold_seed(1337, "abc") < 2**63
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pref_data.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'tinystories_v2.pref_data'`

- [ ] **Step 3: Create the module with the two helpers**

Create `src/tinystories_v2/pref_data.py`:

```python
"""Preference-labeling stage: pref-split Scaffolds -> Judge-labeled preference
pairs, the Reward Model training data (issue 04).

Invoke standalone:
    ts2-pref-data --config configs/pref_data_fixture.toml [--resume]
    (or: python -m tinystories_v2.pref_data --config ...)

For each Scaffold in the pref split: sample [sampling].num_completions from
the SFT checkpoint (issue 02's generation utility), form pairs_per_scaffold
pairs on a round-robin schedule, and label each pair through issue 10's
order-swap consistency filter with the config-selected Judge. Only consistent
verdicts become records (docs/schemas/preference-pair-v1.md).

Artifacts in <out_dir>:
    pairs.jsonl      one schema-valid preference-pair v1 record per line
    progress.json    the commit point: committed line count, done prompt
                     hashes, counters
    manifest.json    stage, version, counters, discard_rate, judge_id, config

Commit protocol (kill-safe, one Scaffold at a time): append the Scaffold's
retained pairs to pairs.jsonl and fsync, then atomically replace
progress.json. A crash between the two leaves uncommitted trailing lines that
the next resume truncates before continuing — never a duplicate, never a lost
pair. Per-Scaffold sampling seeds derive from (seed, prompt_hash), so a
resumed run reproduces exactly what the uninterrupted run would have written,
and labeling accumulates across Colab sessions into one growing artifact
synced to [hub].target.
"""

import hashlib


def pair_indices(n_completions: int, n_pairs: int) -> list[tuple[int, int]]:
    """The first n_pairs of a round-robin (circle method) schedule over all
    C(n,2) index pairs. Scheduling rounds use every completion at most once,
    so small n_pairs spreads comparisons across completions instead of
    reusing completion 0 (what plain itertools.combinations order would do)."""
    if n_completions < 2:
        raise ValueError("need at least two completions to form pairs")
    max_pairs = n_completions * (n_completions - 1) // 2
    if not 1 <= n_pairs <= max_pairs:
        raise ValueError(
            f"pairs_per_scaffold must be in [1, {max_pairs}] for "
            f"{n_completions} completions, got {n_pairs}"
        )
    ids = list(range(n_completions))
    if len(ids) % 2:
        ids.append(-1)  # bye slot for odd counts
    half = len(ids) // 2
    schedule = []
    for _ in range(len(ids) - 1):
        for a, b in zip(ids[:half], reversed(ids[half:])):
            if -1 not in (a, b):
                schedule.append((min(a, b), max(a, b)))
        ids = [ids[0]] + [ids[-1]] + ids[1:-1]
    return schedule[:n_pairs]


def scaffold_seed(seed: int, prompt_hash: str) -> int:
    """Per-Scaffold sampling seed: a pure function of (seed, prompt_hash), so
    completions are independent of processing order and a resumed run
    regenerates exactly what the uninterrupted run would have (mirrors
    data.assign_split's hashing)."""
    digest = hashlib.sha256(f"{seed}:{prompt_hash}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2**63
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_pref_data.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/pref_data.py tests/test_pref_data.py
git commit -m "feat: pair schedule and per-scaffold seed for preference labeling"
```

---

### Task 2: Completion sampling and order-swap pair labeling

**Files:**
- Modify: `src/tinystories_v2/pref_data.py`
- Modify: `src/tinystories_v2/judge.py` (promote `_normalize` to public `normalize_text`)
- Test: `tests/test_pref_data.py`

**Interfaces:**
- Consumes: `pair_indices` (Task 1); `generate.sample`, `slot_prompt.render_prompt`/`END_TOKEN`, `judge.judge_with_order_swap`, `judge.Judge`, `preferences.PreferencePair` (existing).
- Produces:
  - `sample_completions(model: FableLM, tokenizer: Tokenizer, scaffold: Scaffold, *, num_completions: int, max_new_tokens: int, temperature: float, top_p: float, seed: int, device: str = "cpu") -> list[str]` — decoded, stripped fable bodies (prompt and specials excluded).
  - `label_scaffold(judge: Judge, scaffold: Scaffold, completions: list[str], n_pairs: int) -> tuple[list[PreferencePair], dict[str, int]]` — retained pairs plus counters dict with exactly the keys `kept`, `discarded_inconsistent`, `skipped_degenerate`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pref_data.py` (extend the imports at the top of the file as shown):

```python
import torch
from tokenizers import Tokenizer

from tinystories_v2.judge import PositionBiasedFakeJudge, SlotCoverageFakeJudge
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.pref_data import label_scaffold, sample_completions
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run

TOY_MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 256, "ffn_hidden": 192}


@pytest.fixture(scope="module")
def toy_tokenizer(tmp_path_factory, fixture_path) -> Tokenizer:
    out = tmp_path_factory.mktemp("tok")
    tokenizer_run({"out_dir": str(out), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    return Tokenizer.from_file(str(out / "tokenizer.json"))


@pytest.fixture(scope="module")
def toy_model() -> FableLM:
    torch.manual_seed(0)
    return FableLM(ModelConfig(**TOY_MODEL))


@pytest.fixture(scope="module")
def toy_scaffold() -> Scaffold:
    return Scaffold(character="fox", trait="greedy", setting="a dense forest",
                    conflict="loses food to a trick",
                    resolution="the trickster is exposed",
                    moral="honesty is the best policy")


def test_sample_completions_is_seed_deterministic(toy_model, toy_tokenizer,
                                                  toy_scaffold):
    kwargs = dict(num_completions=4, max_new_tokens=16, temperature=1.0,
                  top_p=0.95, device="cpu")
    first = sample_completions(toy_model, toy_tokenizer, toy_scaffold,
                               seed=7, **kwargs)
    second = sample_completions(toy_model, toy_tokenizer, toy_scaffold,
                                seed=7, **kwargs)
    assert first == second
    assert len(first) == 4
    assert all(isinstance(text, str) for text in first)


def test_label_scaffold_keeps_consistent_pairs_and_counts_degenerate(
        toy_scaffold):
    completions = [
        "The greedy fox in a dense forest learned honesty is the best policy.",
        "A bird flew.",
        "A bird flew.",   # duplicate of index 1 -> the (1, 2) pair is degenerate
        "Fish swam in a dense forest.",
    ]
    pairs, counters = label_scaffold(
        SlotCoverageFakeJudge(), toy_scaffold, completions, 3)
    # pair_indices(4, 3) == [(0, 3), (1, 2), (0, 2)]; completion 0 has the
    # highest slot coverage, so it wins both non-degenerate pairs.
    assert counters == {"kept": 2, "discarded_inconsistent": 0,
                        "skipped_degenerate": 1}
    assert [pair.chosen for pair in pairs] == [completions[0], completions[0]]
    assert all(pair.verdict.consistent for pair in pairs)


def test_label_scaffold_discards_all_position_biased_verdicts(toy_scaffold):
    pairs, counters = label_scaffold(
        PositionBiasedFakeJudge(), toy_scaffold,
        ["Alpha text.", "Beta text.", "Gamma text.", "Delta text."], 3)
    assert pairs == []
    assert counters == {"kept": 0, "discarded_inconsistent": 3,
                        "skipped_degenerate": 0}


def test_label_scaffold_skips_empty_completions(toy_scaffold):
    pairs, counters = label_scaffold(
        SlotCoverageFakeJudge(), toy_scaffold,
        ["", "Beta text.", "  ", "Delta text."], 3)
    # Schedule (0,3), (1,2), (0,2): every pair touches an empty completion.
    assert pairs == []
    assert counters == {"kept": 0, "discarded_inconsistent": 0,
                        "skipped_degenerate": 3}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pref_data.py -v`
Expected: FAIL — `ImportError: cannot import name 'label_scaffold' from 'tinystories_v2.pref_data'`

- [ ] **Step 3: Implement `sample_completions` and `label_scaffold`**

First, in `src/tinystories_v2/judge.py`, promote the private normalization
helper so the labeling stage shares it instead of duplicating it (reviewer
finding, human-approved): replace

```python
def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())
```

with

```python
def normalize_text(text: str) -> str:
    """Whitespace-collapsed casefold: the seam's notion of candidate equality,
    shared with the labeling stage's degeneracy check."""
    return " ".join(text.casefold().split())
```

and update its call sites in `_validate_candidates` and `_coverage_score`
(`_normalize(` → `normalize_text(`).

Then, in `src/tinystories_v2/pref_data.py`, replace the import block

```python
import hashlib
```

with

```python
import hashlib

from tokenizers import Tokenizer

from tinystories_v2.generate import sample
from tinystories_v2.judge import Judge, judge_with_order_swap, normalize_text
from tinystories_v2.model import FableLM
from tinystories_v2.preferences import PreferencePair
from tinystories_v2.slot_prompt import END_TOKEN, render_prompt
from tinystories_v2.slots import Scaffold
```

and append after `scaffold_seed`:

```python
def sample_completions(model: FableLM, tokenizer: Tokenizer,
                       scaffold: Scaffold, *, num_completions: int,
                       max_new_tokens: int, temperature: float, top_p: float,
                       seed: int, device: str = "cpu") -> list[str]:
    """N seeded completions of a Scaffold's Slot Prompt: decoded fable bodies
    (the prompt prefix and special tokens are excluded; decode drops <|end|>)."""
    prompt_ids = tokenizer.encode(render_prompt(scaffold)).ids
    sequences = sample(
        model, prompt_ids, num_samples=num_completions,
        max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
        seed=seed, end_id=tokenizer.token_to_id(END_TOKEN), device=device,
    )
    return [tokenizer.decode(seq[len(prompt_ids):]).strip()
            for seq in sequences]


def _degenerate(fable_a: str, fable_b: str) -> bool:
    """True when the Judge could not accept this pair: empty or effectively
    identical candidates (the Judge seam's own candidate normalization)."""
    a, b = normalize_text(fable_a), normalize_text(fable_b)
    return not a or not b or a == b


def label_scaffold(judge: Judge, scaffold: Scaffold, completions: list[str],
                   n_pairs: int) -> tuple[list[PreferencePair], dict[str, int]]:
    """Form the round-robin pairs and label each through order-swap
    consistency filtering. Degenerate pairs are skipped before the Judge sees
    them; inconsistent verdicts are discarded (position-bias filter)."""
    pairs: list[PreferencePair] = []
    counters = {"kept": 0, "discarded_inconsistent": 0, "skipped_degenerate": 0}
    for i, j in pair_indices(len(completions), n_pairs):
        fable_a, fable_b = completions[i], completions[j]
        if _degenerate(fable_a, fable_b):
            counters["skipped_degenerate"] += 1
            continue
        pair = judge_with_order_swap(judge, scaffold, fable_a, fable_b)
        if pair is None:
            counters["discarded_inconsistent"] += 1
        else:
            counters["kept"] += 1
            pairs.append(pair)
    return pairs, counters
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_pref_data.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/pref_data.py tests/test_pref_data.py
git commit -m "feat: completion sampling and order-swap pair labeling helpers"
```

---

### Task 3: Kill-safe progress store and single-file Hub fetch

**Files:**
- Modify: `src/tinystories_v2/pref_data.py`
- Modify: `src/tinystories_v2/hub.py`
- Test: `tests/test_pref_data.py`, `tests/test_hub_sync.py`

**Interfaces:**
- Consumes: `PreferencePair.to_dict()` (existing).
- Produces:
  - `@dataclass Progress` with fields `pairs_written: int = 0`, `done: list[str]` (default empty), `counters: dict[str, int]` (default empty).
  - `load_progress(out_dir: Path) -> Progress` — reads `progress.json` if present, then truncates `pairs.jsonl` to the committed line count (dropping uncommitted trailing lines); raises `ValueError` (message contains "corrupt") when committed pairs are missing.
  - `commit_scaffold(out_dir: Path, progress: Progress, prompt_hash: str, pairs: list[PreferencePair], counters: dict[str, int]) -> None` — appends pair lines (fsync), mutates `progress` in place, atomically replaces `progress.json`.
  - `hub.fetch_file_from(target: str, relative_path: str, dest: Path) -> None` — copies one file out of an `hf://` repo or local-directory target.

- [ ] **Step 1: Write the failing progress-store tests**

Append to `tests/test_pref_data.py` (add these imports at the top: `import json`, and `from tinystories_v2.pref_data import Progress, commit_scaffold, load_progress`, `from tinystories_v2.preferences import VerdictMetadata, validate_preference_pair`, `from tinystories_v2.preferences import PreferencePair`):

```python
def make_pair(tag: str) -> PreferencePair:
    return PreferencePair(
        scaffold=Scaffold(character="fox", trait="greedy",
                          setting="a dense forest",
                          conflict="loses food to a trick",
                          resolution="the trickster is exposed",
                          moral="honesty is the best policy"),
        chosen=f"The fox learned honesty ({tag}).",
        rejected=f"A fox went home ({tag}).",
        verdict=VerdictMetadata(judge_id="fake:slot-coverage-v1",
                                first_pass="A", swapped_pass="B",
                                consistent=True),
    )


def test_commit_and_reload_round_trip(tmp_path):
    progress = load_progress(tmp_path)
    assert progress == Progress()
    commit_scaffold(tmp_path, progress, "hash-1", [make_pair("one")],
                    {"kept": 1})
    commit_scaffold(tmp_path, progress, "hash-2", [],
                    {"skipped_degenerate": 3})
    reloaded = load_progress(tmp_path)
    assert reloaded == Progress(pairs_written=1, done=["hash-1", "hash-2"],
                                counters={"kept": 1, "skipped_degenerate": 3})
    lines = (tmp_path / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    validate_preference_pair(json.loads(lines[0]))


def test_load_truncates_uncommitted_trailing_lines(tmp_path):
    progress = load_progress(tmp_path)
    commit_scaffold(tmp_path, progress, "hash-1", [make_pair("one")],
                    {"kept": 1})
    with (tmp_path / "pairs.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"uncommitted": ')   # crash mid-append, before the commit
    reloaded = load_progress(tmp_path)
    assert reloaded.pairs_written == 1
    lines = (tmp_path / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    validate_preference_pair(json.loads(lines[0]))


def test_load_rejects_missing_committed_pairs(tmp_path):
    progress = load_progress(tmp_path)
    commit_scaffold(tmp_path, progress, "hash-1", [make_pair("one")],
                    {"kept": 1})
    (tmp_path / "pairs.jsonl").unlink()
    with pytest.raises(ValueError, match="corrupt"):
        load_progress(tmp_path)
```

- [ ] **Step 2: Write the failing `fetch_file_from` test**

Append to `tests/test_hub_sync.py` (extend its `from tinystories_v2.hub import ...` line with `fetch_file_from`):

```python
def test_fetch_file_from_local_target(tmp_path):
    src = tmp_path / "artifact"
    (src / "splits").mkdir(parents=True)
    (src / "splits" / "pref.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    dest = tmp_path / "elsewhere" / "pref.jsonl"
    fetch_file_from(str(src), "splits/pref.jsonl", dest)
    assert dest.read_text(encoding="utf-8") == '{"x": 1}\n'


def test_fetch_file_from_hf_dispatches_to_hf_hub_download(tmp_path, monkeypatch):
    calls = {}

    def fake_download(*, repo_id, filename, repo_type):
        calls.update(repo_id=repo_id, filename=filename, repo_type=repo_type)
        src = tmp_path / "downloaded.jsonl"
        src.write_text('{"y": 2}\n', encoding="utf-8")
        return str(src)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    dest = tmp_path / "dest" / "pref.jsonl"
    fetch_file_from("hf://someone/some-repo", "splits/pref.jsonl", dest)
    assert calls == {"repo_id": "someone/some-repo",
                     "filename": "splits/pref.jsonl", "repo_type": "model"}
    assert dest.read_text(encoding="utf-8") == '{"y": 2}\n'
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_pref_data.py tests/test_hub_sync.py -v`
Expected: FAIL — `ImportError: cannot import name 'Progress'` and `ImportError: cannot import name 'fetch_file_from'`

- [ ] **Step 4: Implement the progress store**

In `src/tinystories_v2/pref_data.py`, replace

```python
import hashlib

from tokenizers import Tokenizer
```

with

```python
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from tokenizers import Tokenizer
```

and append after `label_scaffold`:

```python
@dataclass
class Progress:
    """The stage's resume state. pairs_written is the commit point: the number
    of pairs.jsonl lines that are durably part of the artifact."""

    pairs_written: int = 0
    done: list[str] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)


def _truncate_pairs(pairs_path: Path, committed_lines: int) -> None:
    """Drop uncommitted trailing lines (a crash between append and commit)."""
    if not pairs_path.exists():
        if committed_lines:
            raise ValueError(
                f"{pairs_path} is missing but progress.json committed "
                f"{committed_lines} pairs; the artifact is corrupt"
            )
        return
    lines = pairs_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) < committed_lines:
        raise ValueError(
            f"{pairs_path} has {len(lines)} lines but progress.json committed "
            f"{committed_lines}; the artifact is corrupt"
        )
    if len(lines) > committed_lines:
        tmp = pairs_path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(lines[:committed_lines]), encoding="utf-8")
        tmp.replace(pairs_path)


def load_progress(out_dir: Path) -> Progress:
    out_dir = Path(out_dir)
    progress_path = out_dir / "progress.json"
    if progress_path.exists():
        raw = json.loads(progress_path.read_text(encoding="utf-8"))
        progress = Progress(pairs_written=raw["pairs_written"],
                            done=list(raw["done"]),
                            counters=dict(raw["counters"]))
    else:
        progress = Progress()
    _truncate_pairs(out_dir / "pairs.jsonl", progress.pairs_written)
    return progress


def commit_scaffold(out_dir: Path, progress: Progress, prompt_hash: str,
                    pairs: list[PreferencePair],
                    counters: dict[str, int]) -> None:
    """Append this Scaffold's retained pairs, then advance the commit point.

    Order matters for kill-safety: pair lines are appended and fsynced first;
    the atomic progress.json replace is the commit. A crash in between leaves
    trailing lines that the next load_progress truncates away."""
    out_dir = Path(out_dir)
    with (out_dir / "pairs.jsonl").open("a", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    progress.pairs_written += len(pairs)
    progress.done.append(prompt_hash)
    for key, value in counters.items():
        progress.counters[key] = progress.counters.get(key, 0) + value
    payload = json.dumps({"pairs_written": progress.pairs_written,
                          "done": progress.done,
                          "counters": progress.counters}, indent=2)
    tmp = out_dir / "progress.json.tmp"
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(out_dir / "progress.json")
```

- [ ] **Step 5: Implement `fetch_file_from` in hub.py**

Append to `src/tinystories_v2/hub.py` (after `fetch_from`, before `try_sync_to`):

```python
def fetch_file_from(target: str, relative_path: str, dest: Path) -> None:
    """Fetch one file from a sync target (additive, like fetch_from) — for
    repos where a full snapshot would pull far more than needed (e.g. the
    data repo's ~1 GB pretrain split when only splits/pref.jsonl is wanted)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if target.startswith(_HF_PREFIX):
        load_env()
        repo_id = target[len(_HF_PREFIX):]
        downloaded = huggingface_hub.hf_hub_download(
            repo_id=repo_id, filename=relative_path, repo_type="model"
        )
        shutil.copy2(downloaded, dest)
    else:
        shutil.copy2(Path(target) / relative_path, dest)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_pref_data.py tests/test_hub_sync.py -v`
Expected: all pass (12 in test_pref_data.py, prior hub tests + 2 new)

- [ ] **Step 7: Commit**

```bash
git add src/tinystories_v2/pref_data.py src/tinystories_v2/hub.py tests/test_pref_data.py tests/test_hub_sync.py
git commit -m "feat: kill-safe labeling progress store and single-file hub fetch"
```

---

### Task 4: The `ts2-pref-data` stage — run(), CLI, fixture config

**Files:**
- Modify: `src/tinystories_v2/pref_data.py`
- Modify: `pyproject.toml:38` (scripts table)
- Create: `configs/pref_data_fixture.toml`
- Test: `tests/test_pref_data_stage.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3; `checkpoint.latest_checkpoint`/`load_checkpoint`, `judge.build_judge`, `hub.fetch_from`/`fetch_file_from`/`try_sync_to`, `config.load_config`/`load_env` (existing).
- Produces:
  - `run(config: dict, resume: bool = False) -> dict` — executes the stage; returns `{"scaffolds": <len(progress.done)>, "pairs": <progress.pairs_written>}`. Raises `ValueError` (message contains "resume") when `progress.json` exists and `resume` is False.
  - `main(argv: list[str] | None = None) -> None` — argparse CLI with `--config` (required) and `--resume` flag.
  - Config schema (TOML): top-level `out_dir`, optional `max_scaffolds` (0 = all), optional `sync_every` (default 25); `[data]` `pref_split` + optional `hub_source`, `tokenizer`, `tokenizer_hub_source`; `[checkpoint]` `local_dir` + optional `hub_source`; `[sampling]` `num_completions`, `pairs_per_scaffold`, `temperature`, `top_p`, `max_new_tokens`, `seed` (all required); `[judge]` passed verbatim to `build_judge`; optional `[hub]` `target`.
  - `manifest.json` keys: `stage` ("pref_data"), `package_version`, `schema_version` (pair schema, 1), `scaffolds_total`, `scaffolds_done`, `counters` (kept / discarded_inconsistent / skipped_degenerate / skipped_long_prompt), `discard_rate`, `judge_id`, `config`.

- [ ] **Step 1: Write the failing stage tests**

Create `tests/test_pref_data_stage.py`:

```python
"""Preference-labeling stage behavior on CPU fixture artifacts with fake
Judges: a schema-valid growing artifact, config-selected Judge, determinism,
and resume semantics."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tinystories_v2.data import run as data_run
from tinystories_v2.pref_data import run as pref_run
from tinystories_v2.preferences import validate_preference_pair
from tinystories_v2.tokenizer import run as tokenizer_run

MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
         "n_heads": 2, "context": 256, "ffn_hidden": 192}


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, fixture_path):
    """Run the data-prep and tokenizer stages once so labeling has real
    upstream artifacts (stages couple only through artifacts)."""
    base = tmp_path_factory.mktemp("pref_inputs")
    data_dir, tok_dir = base / "data", base / "tok"
    data_run({
        "out_dir": str(data_dir), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        # pref weighted high so the split has plenty of Scaffolds from ~120 Fables.
        "splits": {"seed": "fixture-v1", "pretrain": 0.3, "sft": 0.2,
                   "pref": 0.4, "eval": 0.1},
    })
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    return {"pref_split": str(data_dir / "splits" / "pref.jsonl"),
            "tokenizer": str(tok_dir / "tokenizer.json"),
            "data_root": str(data_dir), "tok_root": str(tok_dir)}


@pytest.fixture
def init_dir(tmp_path, prepared, make_init_checkpoint):
    return make_init_checkpoint(tmp_path / "init", MODEL,
                                prepared["tokenizer"])


def make_config(out_dir, prepared, init_dir, *,
                judge_kind="fake_slot_coverage", max_scaffolds=6,
                temperature=1.0) -> dict:
    return {
        "out_dir": str(out_dir),
        "max_scaffolds": max_scaffolds,
        "data": {"pref_split": prepared["pref_split"]},
        "checkpoint": {"local_dir": str(init_dir)},
        "sampling": {"num_completions": 4, "pairs_per_scaffold": 3,
                     "temperature": temperature, "top_p": 0.95,
                     "max_new_tokens": 32, "seed": 1337},
        "judge": {"kind": judge_kind},
    }


def to_toml(config: dict) -> str:
    lines = [f'out_dir = "{config["out_dir"]}"',
             f"max_scaffolds = {config['max_scaffolds']}"]
    for section in ("data", "checkpoint", "sampling", "judge"):
        lines.append(f"[{section}]")
        for key, value in config[section].items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"


def read_pairs(out_dir) -> list[dict]:
    text = (Path(out_dir) / "pairs.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_manifest(out_dir) -> dict:
    return json.loads(
        (Path(out_dir) / "manifest.json").read_text(encoding="utf-8"))


def test_artifact_is_schema_valid_with_rates_in_manifest(tmp_path, prepared,
                                                         init_dir):
    out = tmp_path / "out"
    result = pref_run(make_config(out, prepared, init_dir))
    records = read_pairs(out)
    assert records, "the consistent fake Judge must retain pairs"
    for record in records:
        validate_preference_pair(record)   # raises on any schema violation
    manifest = read_manifest(out)
    assert manifest["stage"] == "pref_data"
    assert manifest["schema_version"] == 1
    assert manifest["judge_id"] == "fake:slot-coverage-v1"
    assert manifest["scaffolds_done"] == result["scaffolds"] == 6
    counters = manifest["counters"]
    assert counters["kept"] == len(records) == result["pairs"]
    kept = counters.get("kept", 0)
    discarded = counters.get("discarded_inconsistent", 0)
    assert manifest["discard_rate"] == pytest.approx(
        discarded / max(kept + discarded, 1))


def test_two_fresh_runs_are_byte_identical(tmp_path, prepared, init_dir):
    for name in ("run1", "run2"):
        pref_run(make_config(tmp_path / name, prepared, init_dir))
    assert (tmp_path / "run1" / "pairs.jsonl").read_bytes() == \
        (tmp_path / "run2" / "pairs.jsonl").read_bytes()


def test_judge_is_config_selected_and_biased_judge_discards_all(
        tmp_path, prepared, init_dir):
    out = tmp_path / "out"
    pref_run(make_config(out, prepared, init_dir,
                         judge_kind="fake_position_biased"))
    assert read_pairs(out) == []
    manifest = read_manifest(out)
    assert manifest["judge_id"] == "fake:position-a-v1"
    assert manifest["counters"].get("kept", 0) == 0
    assert manifest["counters"]["discarded_inconsistent"] > 0
    assert manifest["discard_rate"] == 1.0


def test_greedy_sampling_yields_only_degenerate_pairs(tmp_path, prepared,
                                                      init_dir):
    # temperature 0.0 makes all N completions identical (argmax), so every
    # pair is degenerate: nothing reaches the Judge, nothing is kept.
    out = tmp_path / "out"
    pref_run(make_config(out, prepared, init_dir, temperature=0.0))
    manifest = read_manifest(out)
    assert manifest["counters"].get("kept", 0) == 0
    assert manifest["counters"].get("discarded_inconsistent", 0) == 0
    assert manifest["counters"]["skipped_degenerate"] > 0


def test_max_scaffolds_caps_progress(tmp_path, prepared, init_dir):
    out = tmp_path / "out"
    result = pref_run(make_config(out, prepared, init_dir, max_scaffolds=2))
    assert result["scaffolds"] == 2
    assert read_manifest(out)["scaffolds_done"] == 2


def test_resume_after_cap_matches_uninterrupted_run(tmp_path, prepared,
                                                    init_dir):
    reference = make_config(tmp_path / "ref", prepared, init_dir)
    pref_run(reference)

    partial = make_config(tmp_path / "partial", prepared, init_dir,
                          max_scaffolds=2)
    pref_run(partial)
    partial["max_scaffolds"] = 6
    pref_run(partial, resume=True)

    assert (tmp_path / "partial" / "pairs.jsonl").read_bytes() == \
        (tmp_path / "ref" / "pairs.jsonl").read_bytes()
    assert read_manifest(tmp_path / "partial")["counters"] == \
        read_manifest(tmp_path / "ref")["counters"]


def test_resume_discards_uncommitted_trailing_garbage(tmp_path, prepared,
                                                      init_dir):
    reference = make_config(tmp_path / "ref", prepared, init_dir)
    pref_run(reference)

    partial = make_config(tmp_path / "partial", prepared, init_dir,
                          max_scaffolds=2)
    pref_run(partial)
    with (tmp_path / "partial" / "pairs.jsonl").open(
            "a", encoding="utf-8") as f:
        f.write('{"crashed mid-')
    partial["max_scaffolds"] = 6
    pref_run(partial, resume=True)

    assert (tmp_path / "partial" / "pairs.jsonl").read_bytes() == \
        (tmp_path / "ref" / "pairs.jsonl").read_bytes()


def test_fresh_run_refuses_an_existing_labeling_dir(tmp_path, prepared,
                                                    init_dir):
    config = make_config(tmp_path / "out", prepared, init_dir,
                         max_scaffolds=2)
    pref_run(config)
    with pytest.raises(ValueError, match="resume"):
        pref_run(config)


def test_empty_pref_split_is_an_error(tmp_path, prepared, init_dir):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    config = make_config(tmp_path / "out", prepared, init_dir)
    config["data"] = {"pref_split": str(empty)}
    with pytest.raises(ValueError, match="no Scaffolds"):
        pref_run(config)


def test_cli_entrypoint_runs_standalone(tmp_path, prepared, init_dir):
    out = tmp_path / "cli_out"
    config_file = tmp_path / "cfg.toml"
    config_file.write_text(
        to_toml(make_config(out, prepared, init_dir, max_scaffolds=2)),
        encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.pref_data",
         "--config", str(config_file)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (out / "pairs.jsonl").exists()
    assert (out / "manifest.json").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pref_data_stage.py -v`
Expected: FAIL — `ImportError: cannot import name 'run' from 'tinystories_v2.pref_data'`

- [ ] **Step 3: Implement `run()` and `main()`**

In `src/tinystories_v2/pref_data.py`, replace the import block

```python
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from tokenizers import Tokenizer

from tinystories_v2.generate import sample
from tinystories_v2.judge import Judge, judge_with_order_swap, normalize_text
from tinystories_v2.model import FableLM
from tinystories_v2.preferences import PreferencePair
from tinystories_v2.slot_prompt import END_TOKEN, render_prompt
from tinystories_v2.slots import Scaffold
```

with

```python
import argparse
import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinystories_v2 import __version__
from tinystories_v2.checkpoint import latest_checkpoint, load_checkpoint
from tinystories_v2.config import load_config, load_env
from tinystories_v2.generate import sample
from tinystories_v2.hub import fetch_file_from, fetch_from, try_sync_to
from tinystories_v2.judge import (
    Judge, build_judge, judge_with_order_swap, normalize_text,
)
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.preferences import SCHEMA_VERSION, PreferencePair
from tinystories_v2.slot_prompt import END_TOKEN, SLOT_FIELDS, render_prompt
from tinystories_v2.slots import Scaffold
```

and append at the end of the module:

```python
def _read_split(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_sampler(config: dict, device: str) -> tuple[FableLM, Tokenizer]:
    """Load the SFT (or any FableLM) checkpoint to sample completions from,
    plus its tokenizer. Fetches missing pieces from the Hub (fresh VM):
    the checkpoint artifact via [checkpoint].hub_source, the tokenizer via
    [data].tokenizer_hub_source (the checkpoint's recorded tokenizer_path is
    a local path that does not exist on a fresh VM)."""
    ckpt_cfg = config["checkpoint"]
    local_dir = Path(ckpt_cfg["local_dir"])
    if (latest_checkpoint(local_dir / "checkpoints") is None
            and ckpt_cfg.get("hub_source")):
        fetch_from(ckpt_cfg["hub_source"], local_dir)
    ckpt_path = latest_checkpoint(local_dir / "checkpoints")
    if ckpt_path is None:
        raise ValueError(
            f"no step_*.pt checkpoint under {local_dir / 'checkpoints'}; "
            f"point [checkpoint].local_dir (and optionally hub_source) at "
            f"the SFT artifact"
        )
    state = load_checkpoint(ckpt_path)
    model = FableLM(ModelConfig(**state["config"]["model"]))
    model.load_state_dict(state["model"])

    data = config["data"]
    tokenizer_path = Path(
        data.get("tokenizer") or state["config"]["data"]["tokenizer_path"])
    if not tokenizer_path.exists() and data.get("tokenizer_hub_source"):
        fetch_file_from(data["tokenizer_hub_source"], "tokenizer.json",
                        tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"sampling from checkpoint {ckpt_path}")
    return model.to(device).eval(), tokenizer


def run(config: dict, resume: bool = False) -> dict:
    load_env()  # HF token for hub sync/fetch — never printed
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    hub_target = config.get("hub", {}).get("target")

    # A fresh run over an existing labeling dir would append duplicates into
    # pairs.jsonl — hours of Judge time silently corrupted. Refuse instead.
    if not resume and progress_path.exists():
        raise ValueError(
            f"{progress_path} already exists; pass --resume to continue that "
            f"labeling run, or remove {out_dir} to start over"
        )
    if resume and not progress_path.exists() and hub_target:
        try:
            fetch_from(hub_target, out_dir)  # fresh VM: pull prior sessions
        except Exception as err:  # noqa: BLE001 — first run: the repo may not exist yet
            warnings.warn(
                f"could not fetch a prior labeling run from {hub_target!r}; "
                f"starting fresh: {err}", stacklevel=2)
    progress = load_progress(out_dir)

    data = config["data"]
    split_path = Path(data["pref_split"])
    if not split_path.exists() and data.get("hub_source"):
        fetch_file_from(data["hub_source"], "splits/pref.jsonl", split_path)
    rows = _read_split(split_path)
    if not rows:
        raise ValueError(f"no Scaffolds in {split_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = _load_sampler(config, device)
    judge = build_judge(config["judge"])

    sampling = config["sampling"]
    max_scaffolds = config.get("max_scaffolds", 0)
    sync_every = config.get("sync_every", 25)

    def write_manifest() -> None:
        kept = progress.counters.get("kept", 0)
        discarded = progress.counters.get("discarded_inconsistent", 0)
        manifest = {
            "stage": "pref_data",
            "package_version": __version__,
            "schema_version": SCHEMA_VERSION,  # preference-pair schema
            "scaffolds_total": len(rows),
            "scaffolds_done": len(progress.done),
            "counters": dict(progress.counters),
            "discard_rate": discarded / max(kept + discarded, 1),
            "judge_id": judge.judge_id,
            "config": config,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")

    done = set(progress.done)
    for row in rows:
        if max_scaffolds and len(progress.done) >= max_scaffolds:
            break
        prompt_hash = row["prompt_hash"]
        if prompt_hash in done:
            continue
        scaffold = Scaffold(**{f: row[f] for f in SLOT_FIELDS})
        prompt_ids = tokenizer.encode(render_prompt(scaffold)).ids
        if len(prompt_ids) > model.config.context:
            pairs, counters = [], {"skipped_long_prompt": 1}
        else:
            completions = sample_completions(
                model, tokenizer, scaffold,
                num_completions=sampling["num_completions"],
                max_new_tokens=sampling["max_new_tokens"],
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                seed=scaffold_seed(sampling["seed"], prompt_hash),
                device=device,
            )
            pairs, counters = label_scaffold(
                judge, scaffold, completions, sampling["pairs_per_scaffold"])
        commit_scaffold(out_dir, progress, prompt_hash, pairs, counters)
        done.add(prompt_hash)
        write_manifest()
        print(f"[{len(progress.done)}/{len(rows)}] "
              f"kept={progress.counters.get('kept', 0)} "
              f"discarded={progress.counters.get('discarded_inconsistent', 0)}")
        if hub_target and len(progress.done) % sync_every == 0:
            try_sync_to(hub_target, out_dir)

    write_manifest()
    if hub_target:
        try_sync_to(hub_target, out_dir)
    return {"scaffolds": len(progress.done), "pairs": progress.pairs_written}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted labeling run in out_dir")
    args = parser.parse_args(argv)
    run(load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Register the console script**

In `pyproject.toml`, replace

```toml
ts2-demo = "tinystories_v2.demo:main"
```

with

```toml
ts2-demo = "tinystories_v2.demo:main"
ts2-pref-data = "tinystories_v2.pref_data:main"
```

- [ ] **Step 5: Create the fixture config**

Create `configs/pref_data_fixture.toml`:

```toml
# Toy CPU labeling smoke against fixture artifacts — local sanity runs and docs.
# Assumes these upstream stages have run:
#   ts2-data-prep --config configs/data_prep_fixture.toml
#   ts2-tokenizer --config configs/tokenizer_fixture.toml
#   ts2-pretrain  --config configs/pretrain_fixture.toml   (the sampling checkpoint)
# Uses the deterministic fake Judge: no GPU, network, or model download.
# Caveat: the fixture pipeline's context is 64, so Scaffold prompts that exceed
# it are counted under skipped_long_prompt instead of labeled — this is a
# wiring smoke, not a quality run. Stage behavior is guarded by
# tests/test_pref_data*.py, which use a context-256 checkpoint.
out_dir = "artifacts/pref_data_fixture"
max_scaffolds = 0

[data]
pref_split = "artifacts/data_prep_fixture/splits/pref.jsonl"

[checkpoint]
local_dir = "artifacts/pretrain_fixture"

[sampling]
num_completions = 4
pairs_per_scaffold = 3
temperature = 1.0
top_p = 0.95
max_new_tokens = 32
seed = 1337

[judge]
kind = "fake_slot_coverage"
```

- [ ] **Step 6: Run the stage tests to verify they pass**

Run: `pytest tests/test_pref_data_stage.py -v`
Expected: 10 passed

- [ ] **Step 7: Reinstall so the console script registers, and smoke it**

Run: `uv pip install -e . && ts2-pref-data --help`
Expected: usage text showing `--config` and `--resume`

- [ ] **Step 8: Run the whole suite**

Run: `pytest -q`
Expected: all tests pass (was 145 before this issue; now more, 0 failures)

- [ ] **Step 9: Commit**

```bash
git add src/tinystories_v2/pref_data.py pyproject.toml configs/pref_data_fixture.toml tests/test_pref_data_stage.py
git commit -m "feat: ts2-pref-data preference-labeling stage (CPU fixture path)"
```

---

### Task 5: Hub wiring tests and the real-run config

**Files:**
- Create: `configs/pref_data_full.toml`
- Test: `tests/test_pref_data_stage.py` (append)

**Interfaces:**
- Consumes: `run()` from Task 4 (its Hub hooks are already implemented; this task proves them with local-path targets — `hub.sync_to`/`fetch_from` treat non-`hf://` targets as local dirs, so no network is touched).
- Produces: `configs/pref_data_full.toml` pinning the design-doc defaults; no new code.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pref_data_stage.py` (add `import shutil` and `from tinystories_v2.config import load_config` to its imports):

```python
CONFIG_DIR = Path(__file__).parents[1] / "configs"


def test_hub_target_mirrors_artifact_and_fresh_vm_resume_completes(
        tmp_path, prepared, init_dir):
    reference = make_config(tmp_path / "ref", prepared, init_dir)
    pref_run(reference)

    mirror = tmp_path / "mirror"
    out = tmp_path / "out"
    config = make_config(out, prepared, init_dir, max_scaffolds=2)
    config["hub"] = {"target": str(mirror)}
    pref_run(config)
    for name in ("pairs.jsonl", "progress.json", "manifest.json"):
        assert (mirror / name).exists(), name

    shutil.rmtree(out)  # a fresh Colab VM has no local artifact
    config["max_scaffolds"] = 6
    pref_run(config, resume=True)  # pulls prior session from the mirror
    assert (out / "pairs.jsonl").read_bytes() == \
        (tmp_path / "ref" / "pairs.jsonl").read_bytes()
    assert (mirror / "pairs.jsonl").read_bytes() == \
        (tmp_path / "ref" / "pairs.jsonl").read_bytes()


def test_missing_split_and_tokenizer_fetched_from_hub_sources(
        tmp_path, prepared, init_dir):
    reference = make_config(tmp_path / "ref", prepared, init_dir)
    pref_run(reference)

    config = make_config(tmp_path / "out", prepared, init_dir)
    config["data"] = {
        "pref_split": str(tmp_path / "fetched" / "splits" / "pref.jsonl"),
        "hub_source": prepared["data_root"],
        "tokenizer": str(tmp_path / "fetched" / "tokenizer.json"),
        "tokenizer_hub_source": prepared["tok_root"],
    }
    pref_run(config)
    assert (tmp_path / "out" / "pairs.jsonl").read_bytes() == \
        (tmp_path / "ref" / "pairs.jsonl").read_bytes()


def test_missing_checkpoint_fetched_from_hub_source(tmp_path, prepared,
                                                    init_dir):
    config = make_config(tmp_path / "out", prepared, init_dir,
                         max_scaffolds=2)
    config["checkpoint"] = {"local_dir": str(tmp_path / "ckpt_local"),
                            "hub_source": str(init_dir)}
    result = pref_run(config)
    assert result["scaffolds"] == 2


def test_full_config_pins_design_doc_defaults():
    config = load_config(CONFIG_DIR / "pref_data_full.toml")
    assert config["sampling"]["num_completions"] == 4
    assert config["sampling"]["temperature"] == 1.0
    assert config["sampling"]["top_p"] == 0.95
    assert config["sampling"]["pairs_per_scaffold"] == 3
    assert config["judge"]["kind"] == "transformers"
    assert config["judge"]["model_id"] == "Qwen/Qwen3-8B"
    assert config["judge"]["precision"] == "fp16"
    assert config["hub"]["target"].startswith("hf://")
    assert config["checkpoint"]["hub_source"].startswith("hf://")
    assert config["data"]["hub_source"].startswith("hf://")
    assert config["data"]["tokenizer_hub_source"].startswith("hf://")


def test_fixture_config_selects_the_fake_judge():
    config = load_config(CONFIG_DIR / "pref_data_fixture.toml")
    assert config["judge"]["kind"] == "fake_slot_coverage"
    assert config["sampling"]["num_completions"] == 4
```

- [ ] **Step 2: Run the tests to verify the config test fails**

Run: `pytest tests/test_pref_data_stage.py -v`
Expected: `test_full_config_pins_design_doc_defaults` FAILS with `FileNotFoundError` for `configs/pref_data_full.toml`; the three Hub-wiring tests PASS (the hooks landed in Task 4 — these tests lock the behavior in).

- [ ] **Step 3: Create the full config**

Create `configs/pref_data_full.toml`:

```toml
# Real preference-labeling run on Colab Pro (design-doc defaults: N=4
# completions per Scaffold at temperature 1.0 / top-p 0.95, 3 pairs per
# Scaffold, Qwen3-8B fp16 Judge on the L4 — on a T4 set
# model_id = "Qwen/Qwen3-4B-Instruct-2507").
# Prerequisites on the Hub, all fetched automatically on a fresh VM: the
# issue 03 SFT checkpoint, the pref split, and the tokenizer.
# Resumable offline batch: rerun with --resume after a preemption and the run
# continues from progress.json (pulled back from [hub].target); pairs
# accumulate across sessions into one growing artifact.
out_dir = "artifacts/pref_data_full"
max_scaffolds = 0
sync_every = 25

[data]
pref_split = "artifacts/data_prep_full/splits/pref.jsonl"
hub_source = "hf://congthanh991/tinystories-v2-data"
tokenizer = "artifacts/tokenizer_full/tokenizer.json"
tokenizer_hub_source = "hf://congthanh991/tinystories-v2-tokenizer"

[checkpoint]
local_dir = "artifacts/sft_full"
hub_source = "hf://congthanh991/tinystories-v2-sft"

[sampling]
num_completions = 4
pairs_per_scaffold = 3
temperature = 1.0
top_p = 0.95
max_new_tokens = 400
seed = 1337

[judge]
kind = "transformers"
model_id = "Qwen/Qwen3-8B"
precision = "fp16"
device = "cuda"
enable_thinking = false
max_new_tokens = 4

[hub]
target = "hf://congthanh991/tinystories-v2-pref-pairs"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_pref_data_stage.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add configs/pref_data_full.toml tests/test_pref_data_stage.py
git commit -m "feat: hub wiring tests and full L4 labeling config"
```

---

### Task 6: Kill-and-resume integrity test

**Files:**
- Test: `tests/test_pref_data_resume.py`

**Interfaces:**
- Consumes: `run()` and the CLI module path `tinystories_v2.pref_data` (Task 4); `make_init_checkpoint` conftest fixture.
- Produces: the acceptance-criterion proof — after SIGKILL and resume, the artifact is byte-identical to an uninterrupted run (no duplicate, no lost pairs).

- [ ] **Step 1: Write the test**

Create `tests/test_pref_data_resume.py`:

```python
"""Kill-and-resume: the preference-labeling commit protocol, end to end.

Sized so the run is slow enough to SIGKILL mid-flight: a d_model-128 sampler
generating 128 new tokens x 4 completions takes a noticeable fraction of a
second per Scaffold, giving several commits inside the kill window.
Per-Scaffold sampling seeds make each Scaffold's work independent of history,
so the resumed artifact must be byte-identical to an uninterrupted reference
run — a duplicated pair or a lost pair would break equality.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tinystories_v2.data import run as data_run
from tinystories_v2.pref_data import run as pref_run
from tinystories_v2.tokenizer import run as tokenizer_run

MODEL = {"vocab_size": 512, "d_model": 128, "n_layers": 4,
         "n_heads": 4, "context": 256, "ffn_hidden": 384}
MAX_SCAFFOLDS = 12
KILL_AFTER_DONE = 3


def pref_config(out_dir, pref_split, init_dir) -> dict:
    return {
        "out_dir": str(out_dir),
        "max_scaffolds": MAX_SCAFFOLDS,
        "data": {"pref_split": str(pref_split)},
        "checkpoint": {"local_dir": str(init_dir)},
        "sampling": {"num_completions": 4, "pairs_per_scaffold": 3,
                     "temperature": 1.0, "top_p": 0.95,
                     "max_new_tokens": 128, "seed": 1337},
        "judge": {"kind": "fake_slot_coverage"},
    }


def to_toml(config: dict) -> str:
    lines = [f'out_dir = "{config["out_dir"]}"',
             f"max_scaffolds = {config['max_scaffolds']}"]
    for section in ("data", "checkpoint", "sampling", "judge"):
        lines.append(f"[{section}]")
        for key, value in config[section].items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"


def done_count(progress_path: Path) -> int:
    try:
        return len(
            json.loads(progress_path.read_text(encoding="utf-8"))["done"])
    except (FileNotFoundError, json.JSONDecodeError):
        return 0   # not yet written


def test_killed_labeling_resumes_to_identical_artifact(
        tmp_path, fixture_path, make_init_checkpoint):
    # Shared inputs: one pref split, one tokenizer, one sampling checkpoint.
    data_dir, tok_dir = tmp_path / "data", tmp_path / "tok"
    data_run({
        "out_dir": str(data_dir), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.3, "sft": 0.2,
                   "pref": 0.4, "eval": 0.1},
    })
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    pref_split = data_dir / "splits" / "pref.jsonl"
    init_dir = make_init_checkpoint(tmp_path / "init", MODEL,
                                    tok_dir / "tokenizer.json")

    # Reference: identical config, never interrupted.
    reference = pref_config(tmp_path / "reference", pref_split, init_dir)
    pref_run(reference)

    # Interrupted: run as a subprocess, SIGKILL after a few commits.
    interrupted = pref_config(tmp_path / "interrupted", pref_split, init_dir)
    config_file = tmp_path / "interrupted.toml"
    config_file.write_text(to_toml(interrupted), encoding="utf-8")
    progress_path = Path(interrupted["out_dir"]) / "progress.json"

    proc = subprocess.Popen(
        [sys.executable, "-m", "tinystories_v2.pref_data",
         "--config", str(config_file)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 120
        while done_count(progress_path) < KILL_AFTER_DONE:
            if proc.poll() is not None:
                pytest.fail(
                    f"stage finished (rc={proc.returncode}) before the kill "
                    f"window; enlarge MODEL or lower KILL_AFTER_DONE")
            if time.monotonic() > deadline:
                pytest.fail("timed out waiting for committed Scaffolds")
            time.sleep(0.005)
        os.kill(proc.pid, signal.SIGKILL)
    finally:
        proc.kill()
        proc.wait(timeout=30)

    killed_at = done_count(progress_path)
    assert KILL_AFTER_DONE <= killed_at < MAX_SCAFFOLDS

    pref_run(interrupted, resume=True)

    ref_out = Path(reference["out_dir"])
    res_out = Path(interrupted["out_dir"])
    assert (res_out / "pairs.jsonl").read_bytes() == \
        (ref_out / "pairs.jsonl").read_bytes()
    ref_manifest = json.loads(
        (ref_out / "manifest.json").read_text(encoding="utf-8"))
    res_manifest = json.loads(
        (res_out / "manifest.json").read_text(encoding="utf-8"))
    assert res_manifest["counters"] == ref_manifest["counters"]
    assert res_manifest["scaffolds_done"] == \
        ref_manifest["scaffolds_done"] == MAX_SCAFFOLDS
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_pref_data_resume.py -v`
Expected: 1 passed (in roughly 30–90 s on a laptop CPU). If it fails with "stage finished before the kill window", raise `max_new_tokens` to 256 in `pref_config` and rerun — do not lower `KILL_AFTER_DONE` below 2.

- [ ] **Step 3: Run the whole suite**

Run: `pytest -q`
Expected: all pass, 0 failures

- [ ] **Step 4: Commit**

```bash
git add tests/test_pref_data_resume.py
git commit -m "test: kill-and-resume integrity for preference labeling"
```

---

### Task 7: Colab notebook, notebook tests, progress docs

**Files:**
- Create: `notebooks/pref_data_colab.ipynb`
- Modify: `tests/test_notebook.py` (append)
- Modify: `PROGRESS.md`
- Modify: `.scratch/tinystories-v2-pipeline/issues/04-judge-seam-preference-labeling.md`

**Interfaces:**
- Consumes: `ts2-pref-data` CLI (Task 4), `configs/pref_data_full.toml` (Task 5).
- Produces: the thin real-run notebook (acceptance criterion 6) and updated status docs.

- [ ] **Step 1: Write the failing notebook tests**

Append to `tests/test_notebook.py`:

```python
PREF_NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "pref_data_colab.ipynb"


def test_pref_data_notebook_is_thin():
    cells = json.loads(PREF_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert 1 <= len(code_cells) <= 4
    source = "\n".join("".join(c["source"]) for c in code_cells)
    for forbidden in ("def ", "class ", "import torch", "for ", "while "):
        assert forbidden not in source, forbidden
    assert "ts2-pref-data" in source
    assert "--resume" in source
    assert "[judge]" in source  # real Judge needs the transformers extra


def test_pref_data_notebook_has_no_secrets_or_outputs():
    text = PREF_NOTEBOOK.read_text(encoding="utf-8")
    assert "hf_" not in text  # no literal HF token prefixes
    cells = json.loads(text)["cells"]
    assert all(not c.get("outputs") for c in cells)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_notebook.py -v`
Expected: the two new tests FAIL with `FileNotFoundError` for `pref_data_colab.ipynb`; the four existing tests pass.

- [ ] **Step 3: Create the notebook**

Create `notebooks/pref_data_colab.ipynb` with exactly this JSON:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Preference labeling on Colab Pro (L4 preferred)\n",
    "\n",
    "Thin wrapper per docs/DESIGN.md: clone → install → secrets → run stage.\n",
    "All logic lives in the package; edit `configs/pref_data_full.toml` in the repo, not here.\n",
    "\n",
    "Prerequisites on the Hub, fetched automatically on a fresh VM: the issue 03 SFT\n",
    "checkpoint, the pref split, and the tokenizer. The Judge is Qwen3-8B fp16 (fits the\n",
    "L4's 24 GB); on a T4 set `model_id = \"Qwen/Qwen3-4B-Instruct-2507\"` in the config.\n",
    "\n",
    "The job is a resumable offline batch: pairs accumulate on the Hub across sessions.\n",
    "After a preemption just rerun the last cell — `--resume` continues from progress.json.\n",
    "\n",
    "Before running: set `HF_TOKEN` in Colab **Secrets** (key icon, left sidebar) and set the repo URL below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "REPO_URL = \"https://github.com/harryct229/tinystories_v2.git\"\n",
    "!git clone {REPO_URL}\n",
    "%cd tinystories_v2\n",
    "!pip install -q -e '.[judge]'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from google.colab import userdata\n",
    "os.environ[\"HF_TOKEN\"] = userdata.get(\"HF_TOKEN\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!ts2-pref-data --config configs/pref_data_full.toml --resume"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
```

- [ ] **Step 4: Run the notebook tests to verify they pass**

Run: `pytest tests/test_notebook.py -v`
Expected: 6 passed

- [ ] **Step 5: Update the issue file status**

In `.scratch/tinystories-v2-pipeline/issues/04-judge-seam-preference-labeling.md`, replace

```
Status: ready-for-agent
```

with

```
Status: code-complete — real labeling run ready (issue 03's SFT checkpoint is on the Hub)
```

and tick all six acceptance checkboxes (`- [ ]` → `- [x]`) — each is verified by tests: (1) `test_artifact_is_schema_valid_with_rates_in_manifest`, (2) same test + `test_judge_is_config_selected_and_biased_judge_discards_all`, (3) `test_killed_labeling_resumes_to_identical_artifact`, (4) `test_judge_is_config_selected_and_biased_judge_discards_all`, (5) `test_full_config_pins_design_doc_defaults`, (6) `test_pref_data_notebook_is_thin`.

- [ ] **Step 6: Update PROGRESS.md**

Note: if PROGRESS.md changed since this plan was written, merge these edits with the current state rather than applying blindly. Three edits:

1. In the issue board, replace

```
| 04 | Preference labeling stage | 03 ✅code, 10 ✅ | 🟢 ready (code work) |
```

with

```
| 04 | Preference labeling stage | 03 ✅, 10 ✅ | ✅ code complete (real run ready — 03's SFT checkpoint on Hub) |
```

2. In the "Now" section, replace

```
- 🟢 Highest-leverage grabs now: **issue 04** (unblocked by 03), and the
  ready code-work issues **05, 07, 08, 09**.
```

with

```
- ✅ **Issue 04 (preference labeling) code complete** — `ts2-pref-data`
  samples N completions per pref Scaffold, labels round-robin pairs through
  the order-swap filter, and accumulates a kill-safe, Hub-synced
  `pairs.jsonl`. The real labeling run is ready — issue 03's SFT checkpoint
  is on the Hub.
- 🟢 Highest-leverage grabs now: the ready code-work issues **05, 07, 08, 09**.
```

3. Prepend to the Log section (directly under `## Log`):

```
- **2026-07-12** — Issue 04 (preference labeling) code complete:
  `pref_data.py` stage (`ts2-pref-data`) with per-Scaffold seeded sampling,
  round-robin pairing, order-swap consistency filtering, a per-Scaffold
  append+fsync/atomic-commit resume protocol (SIGKILL test proves byte-identical
  recovery), single-file Hub fetches for the pref split and tokenizer,
  `configs/pref_data_{fixture,full}.toml`, and a thin
  `pref_data_colab.ipynb`. The real run is ready — issue 03's SFT checkpoint
  (step_000800, masked loss 1.083) is on the Hub.
```

Also update the `_Last updated:` line's date if it differs from the execution date.

- [ ] **Step 7: Run the whole suite one last time**

Run: `pytest -q`
Expected: all pass, 0 failures

- [ ] **Step 8: Commit**

```bash
git add notebooks/pref_data_colab.ipynb tests/test_notebook.py PROGRESS.md .scratch/tinystories-v2-pipeline/issues/04-judge-seam-preference-labeling.md
git commit -m "docs: labeling Colab notebook, notebook tests, progress update"
```
