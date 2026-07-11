# Issue 11 — Reference-Free Metrics Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the dataset paper's reference-free metrics (Self-BLEU, Distinct-n, Flesch Reading Ease) plus a held-out perplexity helper as a pure, deterministic library consumed later by the eval suite (issue 07) and GRPO diversity monitoring (issue 06).

**Architecture:** `metrics.py` holds the three text metrics on plain lists of strings using only the standard library — one shared word-tokenization convention, no torch import ever. `perplexity.py` holds the one metric that needs a model; it imports PyTorch lazily inside the function so importing the package never requires torch, and it accepts any module mapping `(batch, seq_len)` int64 ids to `(batch, seq_len, vocab_size)` logits — the contract the hand-written Llama-style model (issue 02) and toy test models both satisfy.

**Tech Stack:** Python ≥3.11 stdlib (`re`, `math`, `random`, `collections`, `itertools`), PyTorch (CPU) as a dev-extra test dependency only, pytest, uv.

## Global Constraints

Copied from `.scratch/tinystories-v2-pipeline/issues/11-reference-free-metrics.md`, the PRD, `CONTEXT.md`, and the dataset paper (`2504.20605v2.md`). Every task's requirements implicitly include these.

- Text metrics accept plain lists (any `Sequence`) of strings — no coupling to stage artifacts, configs, or model classes. The perplexity helper's only model coupling is its `model` argument.
- The library has no model, GPU, or network dependencies: `import tinystories_v2.metrics` and `import tinystories_v2.perplexity` must both succeed and complete without importing torch (guarded by a subprocess test). Torch is imported lazily inside `perplexity()` only.
- All tests run on laptop CPU with no GPU, network, or model download.
- Deterministic given the same inputs: identical calls return identical values. The only sampling (Self-BLEU cost control) is driven by explicit `sample_size` and `seed` parameters using `random.Random(seed)` — consuming stages (issues 06/07) wire these to their configs; the library itself reads no config.
- Paper conventions (`2504.20605v2.md`): Self-BLEU evaluates each fable against all the others as references, lower = more diverse; Distinct-n is the proportion of unique n-grams (paper reports n = 1), higher = richer vocabulary; Flesch Reading Ease is `206.835 − 1.015 × (words/sentences) − 84.6 × (syllables/words)`, higher = easier, paper's fables average 78.9.
- `torch>=2.9` goes into the existing `dev` optional-dependency extra (2.9 is the first release with Python 3.14 wheels; the repo venv is Python 3.14 — verified `uv` resolves torch 2.13.0 for it). `transformers` stays out of `dev`; issue 10's separate `judge` extra remains the home for real-Judge inference dependencies. Torch here is CPU-only for toy-model tests, which does not violate the no-network/no-weights test constraint.
- Use `CONTEXT.md` vocabulary in code and docs: Fable (never "story"/"tale"), Scaffold, Judge, RLAIF.
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`, matching repository history.

## File Structure

```text
pyproject.toml
    # add torch>=2.9 to the [dev] extra for the perplexity tests
src/tinystories_v2/
    metrics.py
        # tokenize_words, distinct_n, self_bleu, flesch_reading_ease,
        # mean_flesch_reading_ease — stdlib only, torch never imported
    perplexity.py
        # perplexity(model, token_ids, ...) with lazy torch import
tests/
    test_metrics.py
        # hand-computed values + edge cases for all three text metrics
    test_perplexity.py
        # toy models, hand-rolled NLL cross-check, lazy-import guard
```

The existing `tests/conftest.py` fixtures (`fixture_records`: 120 real TF1-EN-3M records with `fable` text) provide real Fable text without network access. No existing file changes except `pyproject.toml`.

All hand-computed expected values below were verified against a scratch reference implementation before this plan was written.

---

### Task 1: Word tokenization and Distinct-n

**Files:**
- Create: `src/tinystories_v2/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing from this repo (stdlib only).
- Produces (later tasks and issues 06/07 rely on these exact names):
  - `tokenize_words(text: str) -> list[str]` — casefolded word tokens, the counting convention shared by every text metric.
  - `distinct_n(fables: Sequence[str], n: int = 1) -> float` — unique/total n-grams pooled over the set.
  - `ValueError` on empty input, a fable with no words, `n < 1`, or zero total n-grams.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics.py`:

```python
import pytest

from tinystories_v2.metrics import distinct_n, tokenize_words


def test_tokenize_words_casefolds_and_keeps_apostrophes():
    assert tokenize_words("Don't STOP, little fox!") == [
        "don't",
        "stop",
        "little",
        "fox",
    ]


def test_distinct_1_hand_computed():
    # Pooled unigrams: the, cat, sat, the, dog, sat -> 4 distinct / 6 total.
    assert distinct_n(["The cat sat.", "The dog sat."], n=1) == pytest.approx(
        4 / 6
    )


def test_distinct_2_hand_computed():
    # Bigrams: (the,cat) (cat,sat) (the,dog) (dog,sat) -> 4 distinct / 4.
    assert distinct_n(["The cat sat.", "The dog sat."], n=2) == pytest.approx(
        1.0
    )


def test_distinct_1_identical_fables_halves():
    assert distinct_n(["a b", "a b"], n=1) == pytest.approx(0.5)


def test_distinct_works_on_a_single_fable():
    assert distinct_n(["a b a"], n=1) == pytest.approx(2 / 3)


def test_short_fables_contribute_zero_ngrams():
    # "x" is shorter than n=2, so only "a b c" contributes bigrams.
    assert distinct_n(["a b c", "x"], n=2) == pytest.approx(1.0)


def test_distinct_rejects_empty_set():
    with pytest.raises(ValueError, match="at least one"):
        distinct_n([])


def test_distinct_rejects_fable_without_words():
    with pytest.raises(ValueError, match="no words"):
        distinct_n(["a real fable", "!!!"])


def test_distinct_rejects_all_fables_shorter_than_n():
    with pytest.raises(ValueError, match="no 3-grams"):
        distinct_n(["a b", "c d"], n=3)


def test_distinct_rejects_n_below_one():
    with pytest.raises(ValueError, match="at least 1"):
        distinct_n(["a b"], n=0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `rtk .venv/bin/pytest tests/test_metrics.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'tinystories_v2.metrics'`.

- [ ] **Step 3: Implement tokenization and Distinct-n**

Create `src/tinystories_v2/metrics.py`:

```python
"""Reference-free text metrics over plain lists of Fable strings.

Implements the dataset paper's reference-free table (Self-BLEU, Distinct-n,
Flesch Reading Ease) as a pure standard-library module: no model, GPU, or
network dependencies, and deterministic for identical inputs. Consumed by
the eval suite (issue 07) and GRPO diversity monitoring (issue 06).

Word convention shared by every metric: casefolded runs of letters/digits
with internal apostrophes kept ("Don't" -> "don't").
"""

import re
from collections.abc import Sequence

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def tokenize_words(text: str) -> list[str]:
    """Split text into the casefolded word tokens all metrics count."""
    return _WORD_RE.findall(text.casefold())


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _tokenize_fables(fables: Sequence[str]) -> list[list[str]]:
    tokenized = []
    for index, fable in enumerate(fables):
        tokens = tokenize_words(fable)
        if not tokens:
            raise ValueError(f"fable at index {index} contains no words")
        tokenized.append(tokens)
    return tokenized


def distinct_n(fables: Sequence[str], n: int = 1) -> float:
    """Unique / total n-grams pooled over the set (paper reports n = 1).

    Higher = richer vocabulary. Fables shorter than n tokens contribute
    no n-grams; the set must still yield at least one n-gram overall.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if not fables:
        raise ValueError("distinct_n needs at least one fable")
    pooled: list[tuple[str, ...]] = []
    for tokens in _tokenize_fables(fables):
        pooled.extend(_ngrams(tokens, n))
    if not pooled:
        raise ValueError(f"no {n}-grams: every fable is shorter than n={n}")
    return len(set(pooled)) / len(pooled)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `rtk .venv/bin/pytest tests/test_metrics.py -v`

Expected: `10 passed`.

- [ ] **Step 5: Commit**

```bash
rtk git add src/tinystories_v2/metrics.py tests/test_metrics.py
rtk git commit -m "feat: add word tokenization and Distinct-n metric

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Self-BLEU with seeded sampling

**Files:**
- Modify: `src/tinystories_v2/metrics.py`
- Modify: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `tokenize_words`, `_ngrams`, `_tokenize_fables` from Task 1.
- Produces: `self_bleu(fables: Sequence[str], max_n: int = 4, sample_size: int | None = None, seed: int = 0) -> float` — mean BLEU of each fable against all others; lower = more diverse. `ValueError` on fewer than two fables (after sampling), a fable with no words, `max_n < 1`, or `sample_size < 2`.

BLEU conventions (documented in the docstring so report numbers are reproducible): modified n-gram precision with reference-count clipping; geometric mean over orders `1..min(max_n, len(hypothesis))` so degenerate short fables are scored on the orders they have; any zero precision gives BLEU 0 for that fable (no smoothing); brevity penalty against the reference length closest to the hypothesis length (ties -> shorter).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metrics.py` (and extend the import at the top of the file to `from tinystories_v2.metrics import distinct_n, self_bleu, tokenize_words`, adding `import math` and `import random` above it):

```python
def test_self_bleu_identical_fables_is_maximally_redundant():
    assert self_bleu(
        ["The fox ran home.", "The fox ran home."]
    ) == pytest.approx(1.0)


def test_self_bleu_disjoint_fables_is_zero():
    assert self_bleu(["aa bb cc dd", "ee ff gg hh"]) == 0.0


def test_self_bleu_partial_overlap_hand_computed():
    # Each fable vs the other: p1 = 2/3, p2 = 1/2, equal lengths so no
    # brevity penalty; BLEU-2 = sqrt(2/3 * 1/2) = sqrt(1/3) for both.
    assert self_bleu(
        ["the cat sat", "the cat ran"], max_n=2
    ) == pytest.approx(math.sqrt(1 / 3))


def test_self_bleu_single_token_fables():
    # Degenerate one-word fables score on unigrams only: each "a" finds
    # the other "a" among its references (BLEU 1), "b" finds nothing (0).
    assert self_bleu(["a", "b", "a"]) == pytest.approx(2 / 3)


def test_self_bleu_rejects_fewer_than_two_fables():
    with pytest.raises(ValueError, match="at least two"):
        self_bleu(["only one fable"])


def test_self_bleu_rejects_fable_without_words():
    with pytest.raises(ValueError, match="no words"):
        self_bleu(["a real fable", "!!!"])


def test_self_bleu_sampling_is_seeded_and_deterministic():
    fables = [f"fable number {i} tells of animal {i}" for i in range(6)]
    sampled_directly = random.Random(0).sample(fables, 3)
    result = self_bleu(fables, sample_size=3, seed=0)
    assert result == self_bleu(sampled_directly)
    assert result == self_bleu(fables, sample_size=3, seed=0)


def test_self_bleu_rejects_sample_size_below_two():
    with pytest.raises(ValueError, match="sample_size"):
        self_bleu(["a b", "c d", "e f"], sample_size=1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `rtk .venv/bin/pytest tests/test_metrics.py -v`

Expected: collection fails with `ImportError: cannot import name 'self_bleu'`.

- [ ] **Step 3: Implement Self-BLEU**

In `src/tinystories_v2/metrics.py`, replace the import block:

```python
import re
from collections.abc import Sequence
```

with:

```python
import math
import random
import re
from collections import Counter
from collections.abc import Sequence
```

Then append:

```python
def _modified_precision(
    hypothesis: list[str],
    references: list[list[str]],
    n: int,
) -> tuple[int, int]:
    hyp_counts = Counter(_ngrams(hypothesis, n))
    max_ref_counts: Counter[tuple[str, ...]] = Counter()
    for reference in references:
        for gram, count in Counter(_ngrams(reference, n)).items():
            max_ref_counts[gram] = max(max_ref_counts[gram], count)
    clipped = sum(
        min(count, max_ref_counts[gram])
        for gram, count in hyp_counts.items()
    )
    return clipped, sum(hyp_counts.values())


def _bleu(
    hypothesis: list[str],
    references: list[list[str]],
    max_n: int,
) -> float:
    hyp_len = len(hypothesis)
    log_precisions = []
    for n in range(1, min(max_n, hyp_len) + 1):
        clipped, total = _modified_precision(hypothesis, references, n)
        if clipped == 0:
            return 0.0
        log_precisions.append(math.log(clipped / total))
    ref_len = min(
        (abs(len(reference) - hyp_len), len(reference))
        for reference in references
    )[1]
    brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return brevity * math.exp(sum(log_precisions) / len(log_precisions))


def self_bleu(
    fables: Sequence[str],
    max_n: int = 4,
    sample_size: int | None = None,
    seed: int = 0,
) -> float:
    """Mean BLEU of each fable against all others (lower = more diverse).

    Matches the paper's Self-BLEU usage: every fable is scored as a
    hypothesis with the remaining fables as references, and the scores
    are averaged. BLEU here is the geometric mean of clipped n-gram
    precisions over orders 1..min(max_n, hypothesis length) with a
    brevity penalty against the closest reference length; any zero
    precision scores that fable 0 (no smoothing).

    For cost control on large sets, pass sample_size to score a
    deterministic random.Random(seed) subsample instead of the full set.
    Consuming stages wire sample_size/seed to their configs.
    """
    if max_n < 1:
        raise ValueError("max_n must be at least 1")
    if sample_size is not None and sample_size < 2:
        raise ValueError("sample_size must be at least 2")
    chosen = list(fables)
    if sample_size is not None and sample_size < len(chosen):
        chosen = random.Random(seed).sample(chosen, sample_size)
    if len(chosen) < 2:
        raise ValueError("self_bleu needs at least two fables")
    tokenized = _tokenize_fables(chosen)
    scores = [
        _bleu(hypothesis, tokenized[:i] + tokenized[i + 1 :], max_n)
        for i, hypothesis in enumerate(tokenized)
    ]
    return sum(scores) / len(scores)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `rtk .venv/bin/pytest tests/test_metrics.py -v`

Expected: `18 passed`.

- [ ] **Step 5: Commit**

```bash
rtk git add src/tinystories_v2/metrics.py tests/test_metrics.py
rtk git commit -m "feat: add Self-BLEU diversity metric with seeded sampling

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Flesch Reading Ease

**Files:**
- Modify: `src/tinystories_v2/metrics.py`
- Modify: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `tokenize_words` from Task 1.
- Produces:
  - `flesch_reading_ease(text: str) -> float` — one text; `ValueError` if it has no words.
  - `mean_flesch_reading_ease(fables: Sequence[str]) -> float` — mean per-fable score, matching the paper's per-model reporting; `ValueError` on an empty list.

Counting conventions (documented in docstrings; the hand-computed tests pin them): words are `tokenize_words` tokens; sentences are `[.!?]+`-delimited segments containing at least one word (text with words but no terminal punctuation is one sentence); syllables per word are the number of `[aeiouy]+` vowel runs, minus one when the word ends in a silent `e` (ends with `e` but not `le`, and has more than one run), with a minimum of one.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metrics.py` (extend the metrics import to include `flesch_reading_ease` and `mean_flesch_reading_ease`):

```python
def test_flesch_single_sentence_hand_computed():
    # 6 words, 6 syllables, 1 sentence:
    # 206.835 - 1.015*(6/1) - 84.6*(6/6) = 116.145
    assert flesch_reading_ease("The cat sat on the mat.") == pytest.approx(
        116.145
    )


def test_flesch_multi_sentence_and_syllable_rules_hand_computed():
    # Syllables: the=1 happy=2 fox=1 ran=1 it=1 was=1 little=2 ("le"
    # ending keeps its final vowel run) -> 9 syllables, 7 words,
    # 2 sentences:
    # 206.835 - 1.015*(7/2) - 84.6*(9/7) = 94.51107142857143
    assert flesch_reading_ease(
        "The happy fox ran. It was little."
    ) == pytest.approx(94.51107142857143)


def test_flesch_text_without_terminal_punctuation_is_one_sentence():
    # hello=2 world=1 -> 3 syllables, 2 words, 1 sentence:
    # 206.835 - 1.015*2 - 84.6*1.5 = 77.905
    assert flesch_reading_ease("hello world") == pytest.approx(77.905)


def test_flesch_rejects_text_without_words():
    with pytest.raises(ValueError, match="no words"):
        flesch_reading_ease("?!.")


def test_mean_flesch_is_mean_of_per_fable_scores():
    assert mean_flesch_reading_ease(
        ["The cat sat on the mat.", "hello world"]
    ) == pytest.approx((116.145 + 77.905) / 2)


def test_mean_flesch_rejects_empty_set():
    with pytest.raises(ValueError, match="at least one"):
        mean_flesch_reading_ease([])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `rtk .venv/bin/pytest tests/test_metrics.py -v`

Expected: collection fails with `ImportError: cannot import name 'flesch_reading_ease'`.

- [ ] **Step 3: Implement Flesch Reading Ease**

Append to `src/tinystories_v2/metrics.py`:

```python
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_VOWEL_RUN_RE = re.compile(r"[aeiouy]+")


def _count_syllables(word: str) -> int:
    """Heuristic: vowel runs, dropping a silent final 'e' (but not 'le')."""
    runs = _VOWEL_RUN_RE.findall(word)
    count = len(runs)
    if count > 1 and word.endswith("e") and not word.endswith("le"):
        count -= 1
    return max(count, 1)


def flesch_reading_ease(text: str) -> float:
    """206.835 - 1.015*(words/sentences) - 84.6*(syllables/words).

    Sentences are [.!?]-delimited segments containing at least one word;
    text with words but no terminal punctuation counts as one sentence.
    Higher = easier reading; the paper's Fables average 78.9 (ages 4-7).
    Syllable counts use the documented vowel-run heuristic, so absolute
    values are comparable within this library, not across other tools.
    """
    words = tokenize_words(text)
    if not words:
        raise ValueError("text contains no words")
    sentences = sum(
        1
        for segment in _SENTENCE_SPLIT_RE.split(text)
        if tokenize_words(segment)
    )
    syllables = sum(_count_syllables(word) for word in words)
    return (
        206.835
        - 1.015 * (len(words) / sentences)
        - 84.6 * (syllables / len(words))
    )


def mean_flesch_reading_ease(fables: Sequence[str]) -> float:
    """Mean per-fable Flesch Reading Ease, matching the paper's tables."""
    if not fables:
        raise ValueError("mean_flesch_reading_ease needs at least one fable")
    return sum(flesch_reading_ease(fable) for fable in fables) / len(fables)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `rtk .venv/bin/pytest tests/test_metrics.py -v`

Expected: `24 passed`.

- [ ] **Step 5: Sanity-check all three metrics on real fixture Fables**

Run:

```bash
rtk .venv/bin/python -c "
import json
from pathlib import Path
from tinystories_v2.metrics import distinct_n, mean_flesch_reading_ease, self_bleu

fables = [json.loads(line)['fable'] for line in Path('tests/fixtures/tf1_sample.jsonl').read_text().splitlines() if line.strip()]
print('n fables:', len(fables))
print('self_bleu (20 sampled):', round(self_bleu(fables, sample_size=20, seed=0), 3))
print('distinct_1:', round(distinct_n(fables), 3))
print('mean flesch:', round(mean_flesch_reading_ease(fables), 1))
"
```

Expected: runs in seconds; Self-BLEU in roughly 0.1–0.6, Distinct-1 in roughly 0.05–0.5 (pooled over 120 fables, so lower than the paper's per-fable 0.452), mean Flesch in roughly 60–100. This is a plausibility check against the paper's magnitudes, not an assertion — record the numbers in the commit message body if they look sane; stop and investigate if any metric is at an extreme (0, 1, or negative).

- [ ] **Step 6: Commit**

```bash
rtk git add src/tinystories_v2/metrics.py tests/test_metrics.py
rtk git commit -m "feat: add Flesch Reading Ease metric

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Perplexity helper with lazy torch

**Files:**
- Modify: `pyproject.toml:16-17`
- Create: `src/tinystories_v2/perplexity.py`
- Create: `tests/test_perplexity.py`

**Interfaces:**
- Consumes: `fixture_records` pytest fixture from `tests/conftest.py`; torch from the dev extra.
- Produces: `perplexity(model: Any, token_ids: Any, *, block_size: int, batch_size: int = 8, device: str = "cpu") -> float`, where `token_ids` accepts any `Sequence[int]` or 1-D torch tensor (annotated `Any` because torch types cannot appear at module level in a lazily-importing module).

Contract (issues 02/03/07 rely on this): `model(input_ids)` maps a `(batch, seq_len)` int64 tensor to `(batch, seq_len, vocab_size)` logits — any checkpoint or toy module qualifies. `token_ids` is the flat 1-D tokenized held-out text. It is split into non-overlapping blocks of `block_size`; each block's inputs are `ids[i : i+block_size]` and targets are `ids[i+1 : i+block_size+1]` (inputs truncated to the target length at the tail), so every token after the first is predicted exactly once. Returns `exp(total NLL / total predicted tokens)`. Inputs are moved to `device`; the model must already live there and be in eval mode. `ValueError` on fewer than two tokens, non-1-D input, `block_size < 1`, or `batch_size < 1`; `RuntimeError` with an install hint if torch is missing.

- [ ] **Step 1: Add torch to the dev extra**

In `pyproject.toml`, replace:

```toml
[project.optional-dependencies]
dev = ["pytest>=8"]
```

with:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8",
    # CPU-only, for the perplexity helper's toy-model tests (issue 11).
    # >=2.9 is the first release with Python 3.14 wheels (repo venv is 3.14).
    "torch>=2.9",
]
```

(If a `judge` extra already exists from issue 10, leave it untouched — `transformers` stays out of `dev`.)

Then install from the repo root:

Run: `uv pip install -e '.[dev]'`

Expected: resolves and installs `torch==2.13.x` (plus `sympy`, `networkx`, `setuptools`) into `.venv` with no errors.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_perplexity.py`:

```python
import math
import subprocess
import sys

import pytest
import torch

from tinystories_v2.perplexity import perplexity


class UniformLogitsModel(torch.nn.Module):
    """Zero logits everywhere, so perplexity must equal vocab_size."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        return torch.zeros(batch, seq_len, self.vocab_size)


class ToyByteModel(torch.nn.Module):
    """Tiny seeded byte-vocab model standing in for a real checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(256, 8)
        self.head = torch.nn.Linear(8, 256)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids))


@pytest.fixture()
def fixture_byte_ids(fixture_records) -> list[int]:
    # Real fixture Fable text, tokenized with the trivial byte vocab so
    # the test needs no trained tokenizer artifact.
    return list(fixture_records[0]["fable"].encode("utf-8"))


def test_uniform_model_perplexity_is_vocab_size():
    token_ids = list(range(11)) * 3
    result = perplexity(UniformLogitsModel(11), token_ids, block_size=7)
    assert result == pytest.approx(11.0)


def test_matches_hand_rolled_nll_on_fixture(fixture_byte_ids):
    model = ToyByteModel()
    result = perplexity(model, fixture_byte_ids, block_size=16, batch_size=4)

    ids = torch.tensor(fixture_byte_ids, dtype=torch.long)
    total_nll = 0.0
    total_targets = 0
    with torch.inference_mode():
        for start in range(0, ids.numel() - 1, 16):
            targets = ids[start + 1 : start + 17]
            inputs = ids[start : start + 16][: targets.numel()]
            log_probs = torch.log_softmax(
                model(inputs.unsqueeze(0))[0], dim=-1
            )
            picked = log_probs[torch.arange(targets.numel()), targets]
            total_nll -= float(picked.sum())
            total_targets += targets.numel()

    assert result == pytest.approx(math.exp(total_nll / total_targets))


def test_batch_size_never_changes_the_result(fixture_byte_ids):
    model = ToyByteModel()
    single = perplexity(model, fixture_byte_ids, block_size=16, batch_size=1)
    batched = perplexity(model, fixture_byte_ids, block_size=16, batch_size=5)
    assert batched == pytest.approx(single)


def test_deterministic_across_calls(fixture_byte_ids):
    model = ToyByteModel()
    first = perplexity(model, fixture_byte_ids, block_size=16)
    second = perplexity(model, fixture_byte_ids, block_size=16)
    assert first == second


def test_rejects_fewer_than_two_tokens():
    with pytest.raises(ValueError, match="at least two"):
        perplexity(UniformLogitsModel(11), [5], block_size=4)


@pytest.mark.parametrize(
    "kwargs",
    [{"block_size": 0}, {"block_size": 4, "batch_size": 0}],
)
def test_rejects_non_positive_sizes(kwargs):
    with pytest.raises(ValueError, match="at least 1"):
        perplexity(UniformLogitsModel(11), [1, 2, 3], **kwargs)


def test_import_never_eagerly_pulls_torch():
    code = (
        "import sys\n"
        "import tinystories_v2.metrics\n"
        "import tinystories_v2.perplexity\n"
        "assert 'torch' not in sys.modules, 'torch imported eagerly'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `rtk .venv/bin/pytest tests/test_perplexity.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'tinystories_v2.perplexity'`.

- [ ] **Step 4: Implement the perplexity helper**

Create `src/tinystories_v2/perplexity.py`:

```python
"""Held-out perplexity for any next-token model (toy or real checkpoints).

Contract: model(input_ids) maps a (batch, seq_len) int64 tensor to
(batch, seq_len, vocab_size) logits. The flat token id sequence is split
into non-overlapping blocks of block_size, so every token after the first
is predicted exactly once. PyTorch is imported lazily: importing this
module (like tinystories_v2.metrics) never requires torch.
"""

import itertools
import math
from typing import Any


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "perplexity requires PyTorch; "
            "install with: uv pip install -e '.[dev]'"
        ) from exc
    return torch


def _equal_length_batches(
    blocks: list[tuple[Any, Any]],
    batch_size: int,
):
    """Batch consecutive equal-length blocks (only the last can be short)."""
    for _, group in itertools.groupby(
        blocks, key=lambda pair: pair[0].numel()
    ):
        members = list(group)
        for start in range(0, len(members), batch_size):
            yield members[start : start + batch_size]


def perplexity(
    model: Any,
    token_ids: Any,
    *,
    block_size: int,
    batch_size: int = 8,
    device: str = "cpu",
) -> float:
    """exp(mean next-token NLL) of a flat held-out token id sequence.

    Works with any checkpoint or toy module satisfying the logits
    contract in the module docstring. Inputs are moved to `device`; the
    model must already live there and be in eval mode. Deterministic:
    no sampling, evaluated under torch.inference_mode().
    """
    torch = _require_torch()
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    ids = torch.as_tensor(token_ids, dtype=torch.long)
    if ids.dim() != 1:
        raise ValueError("token_ids must be a flat 1-D sequence")
    if ids.numel() < 2:
        raise ValueError("token_ids must hold at least two tokens")

    blocks = []
    for start in range(0, ids.numel() - 1, block_size):
        targets = ids[start + 1 : start + block_size + 1]
        inputs = ids[start : start + block_size][: targets.numel()]
        blocks.append((inputs, targets))

    total_nll = 0.0
    total_targets = 0
    with torch.inference_mode():
        for batch in _equal_length_batches(blocks, batch_size):
            inputs = torch.stack([pair[0] for pair in batch]).to(device)
            targets = torch.stack([pair[1] for pair in batch]).to(device)
            logits = model(inputs)
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_nll += float(nll)
            total_targets += targets.numel()
    return math.exp(total_nll / total_targets)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `rtk .venv/bin/pytest tests/test_perplexity.py -v`

Expected: `8 passed` (7 test functions, one parametrized twice, plus the subprocess import guard).

- [ ] **Step 6: Commit**

```bash
rtk git add pyproject.toml src/tinystories_v2/perplexity.py tests/test_perplexity.py
rtk git commit -m "feat: add held-out perplexity helper with lazy torch

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Audit acceptance criteria and close the local issue

**Files:**
- Modify: `.scratch/tinystories-v2-pipeline/issues/11-reference-free-metrics.md:3,23-29`

**Interfaces:**
- Consumes: every public interface and test delivered by Tasks 1–4.
- Produces: a checked local issue whose acceptance statements are backed by named tests and a clean full-suite result.

- [ ] **Step 1: Run the issue-specific acceptance suite**

Run: `rtk .venv/bin/pytest tests/test_metrics.py tests/test_perplexity.py -v`

Expected: `32 passed` — 24 metric tests plus 8 perplexity tests.

- [ ] **Step 2: Run repository-wide verification and whitespace checks**

```bash
rtk .venv/bin/pytest -q
rtk git diff --check
```

Expected: all tests pass with zero failures (54 total if no other issue has landed since this plan was written: 22 pre-existing + 32 new; a higher count is fine, failures are not). `git diff --check` prints nothing.

- [ ] **Step 3: Mark the issue complete**

In `.scratch/tinystories-v2-pipeline/issues/11-reference-free-metrics.md`, change:

```markdown
Status: ready-for-agent
```

to:

```markdown
Status: complete
```

Replace the acceptance checklist with:

```markdown
- [x] Each text metric is tested against hand-computed values on tiny inputs (test_distinct_1_hand_computed, test_self_bleu_partial_overlap_hand_computed, test_flesch_single_sentence_hand_computed, and siblings)
- [x] Edge cases covered: empty set, single fable, identical fables (Self-BLEU extreme), degenerate short texts (test_distinct_rejects_empty_set, test_distinct_works_on_a_single_fable, test_self_bleu_rejects_fewer_than_two_fables, test_self_bleu_identical_fables_is_maximally_redundant, test_self_bleu_single_token_fables, test_short_fables_contribute_zero_ngrams)
- [x] Perplexity helper matches a hand-rolled loss computation on the fixture with a toy model, on CPU (test_matches_hand_rolled_nll_on_fixture; test_uniform_model_perplexity_is_vocab_size pins the closed-form case)
- [x] Metrics accept plain lists of strings — no coupling to stage artifacts or model classes (except the perplexity helper's model argument); test_import_never_eagerly_pulls_torch guards that neither module needs torch to import
- [x] Deterministic given the same inputs (test_self_bleu_sampling_is_seeded_and_deterministic, test_deterministic_across_calls); Self-BLEU sampling is seeded via explicit sample_size/seed parameters that consuming stages wire to their configs
```

- [ ] **Step 4: Commit the acceptance audit**

```bash
rtk git add .scratch/tinystories-v2-pipeline/issues/11-reference-free-metrics.md
rtk git commit -m "docs: complete issue 11 reference-free metrics

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
