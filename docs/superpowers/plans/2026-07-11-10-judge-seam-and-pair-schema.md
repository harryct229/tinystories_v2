# Issue 10 — Judge Seam and Preference-Pair Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a model-agnostic Judge seam with offline fakes, reusable order-swap filtering, config-selected Qwen implementations, and a versioned preference-pair record that downstream RLAIF stages can consume before real labels exist.

**Architecture:** `preferences.py` owns the JSON-compatible, versioned preference-pair contract and strict parsing boundary; it has no dependency on model inference. `judge.py` owns the `Judge` protocol, deterministic fakes, pairwise rubric, swap-consistency orchestration, and a lazily loaded Transformers adapter selected by configuration, so importing or constructing any Judge remains CPU-only and offline until real inference is explicitly requested.

**Tech Stack:** Python ≥3.11, stdlib dataclasses/`enum.StrEnum`/`typing.Protocol`/JSON/TOML, PyTorch and Hugging Face Transformers as an optional real-Judge extra, pytest, uv.

## Global Constraints

Copied from `.scratch/tinystories-v2-pipeline/issues/10-judge-seam-and-pair-schema.md`, the PRD, `CONTEXT.md`, `docs/DESIGN.md`, ADR-0005, and the official Qwen model cards. Every task's requirements implicitly include these.

- The public seam is exactly `Judge.compare(scaffold: Scaffold, fable_a: str, fable_b: str) -> Verdict` plus a stable `judge_id`; no model-specific type may leak through it.
- Everything in this issue operates on strings and existing `Scaffold` values. It contains no model-training code and has no dependency on issues 02, 03, or 04.
- The deterministic fake prefers the completion realizing more Scaffold slot values and resolves equal-coverage cases independently of presentation position.
- The intentionally biased fake always prefers the first position, so judging the swapped presentation exposes and discards its result.
- Every retained preference is judged twice: first as `(A, B)`, then as `(B, A)`. It is retained only when the two presentation-relative verdicts are opposite and therefore identify the same underlying completion.
- The rubric contains exactly the paper's four axes: Grammar & Style, Creativity, Moral Clarity, and Prompt Adherence. Prompt Adherence has the highest weight, Moral Clarity is second priority, and age 4–7 suitability is a hard constraint.
- The production L4 model is `Qwen/Qwen3-8B` in fp16; the T4 fallback is `Qwen/Qwen3-4B-Instruct-2507` in fp16. Both are configurations of one `TransformersJudge` code path.
- Use `transformers>=4.51.0`: both official Qwen model cards state that earlier versions fail with `KeyError: 'qwen3'`. References: [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), and [Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating).
- PyTorch and Transformers stay in an optional `judge` dependency extra and are imported lazily. `uv pip install -e '.[dev]'` must not install them or download model weights.
- All automated tests run on laptop CPU with no GPU, network, Hugging Face cache, or model download. Real-GPU inference is a Colab smoke run, not a pytest requirement.
- Preference-pair JSON uses schema version `1` and exact top-level keys: `schema_version`, `scaffold`, `chosen`, `rejected`, and `verdict`.
- Verdict metadata always includes a non-empty Judge identity plus both presentation-relative outcomes and the successful consistency result. Discarded comparisons are returned as `None` and never serialized as training pairs.
- Use `CONTEXT.md` vocabulary in code and docs: Fable, Scaffold, Judge, Reward Model, and RLAIF; never use “critic,” “evaluator,” “scorer,” “story,” or “RLHF” for these concepts.
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`, matching the established repository history.

## File Structure

```text
pyproject.toml
    # add the optional [judge] inference dependencies
configs/
    judge_l4.toml
        # Qwen3-8B fp16 production selection
    judge_t4.toml
        # Qwen3-4B-Instruct-2507 fp16 fallback selection
docs/schemas/
    preference-pair-v1.md
        # normative JSON contract and order-swap semantics
src/tinystories_v2/
    preferences.py
        # versioned pair dataclasses, serialization, strict validation
    judge.py
        # protocol, verdict, fakes, rubric, swap filter, real adapter, factory
tests/
    test_preferences.py
        # schema round-trip and malformed-record rejection
    test_judge.py
        # fixture fakes, swap filtering, rubric rendering, verdict parsing
    test_judge_config.py
        # offline config/factory coverage for both real selections
```

The existing `Scaffold` dataclass in `src/tinystories_v2/slots.py` is reused unchanged. The existing 120-record `tests/fixtures/tf1_sample.jsonl` and `fixture_records` fixture provide real Fables and Scaffolds without network access.

---

### Task 1: Pin and validate preference-pair schema v1

**Files:**
- Create: `src/tinystories_v2/preferences.py`
- Create: `tests/test_preferences.py`
- Create: `docs/schemas/preference-pair-v1.md`

**Interfaces:**
- Consumes: `tinystories_v2.slots.Scaffold(character: str, trait: str, setting: str, conflict: str, resolution: str, moral: str)`.
- Produces:
  - `SCHEMA_VERSION: int = 1`.
  - `VerdictMetadata(judge_id: str, first_pass: str, swapped_pass: str, consistent: bool)`.
  - `PreferencePair(scaffold: Scaffold, chosen: str, rejected: str, verdict: VerdictMetadata, schema_version: int = 1)`.
  - `PreferencePair.to_dict() -> dict[str, Any]`, containing only JSON-compatible values.
  - `validate_preference_pair(record: Mapping[str, object]) -> PreferencePair`.
  - `PreferencePairValidationError(ValueError)` for every schema violation.

- [ ] **Step 1: Write the failing schema tests**

Create `tests/test_preferences.py`:

```python
import copy
import json

import pytest

from tinystories_v2.preferences import (
    PreferencePair,
    PreferencePairValidationError,
    VerdictMetadata,
    validate_preference_pair,
)
from tinystories_v2.slots import Scaffold


def make_pair() -> PreferencePair:
    return PreferencePair(
        scaffold=Scaffold(
            character="fox",
            trait="greedy",
            setting="a dense forest",
            conflict="loses food to a trick",
            resolution="the trickster is exposed",
            moral="honesty is the best policy",
        ),
        chosen=(
            "A greedy fox admitted the trick, returned the food, and learned "
            "that honesty is the best policy."
        ),
        rejected="A fox walked through a forest and went home.",
        verdict=VerdictMetadata(
            judge_id="fake:slot-coverage-v1",
            first_pass="A",
            swapped_pass="B",
            consistent=True,
        ),
    )


def test_schema_round_trips_through_json():
    pair = make_pair()
    payload = json.loads(json.dumps(pair.to_dict()))
    assert validate_preference_pair(payload) == pair
    assert set(payload) == {
        "schema_version",
        "scaffold",
        "chosen",
        "rejected",
        "verdict",
    }


def test_schema_rejects_missing_judge_identity():
    payload = copy.deepcopy(make_pair().to_dict())
    payload["verdict"]["judge_id"] = ""
    with pytest.raises(PreferencePairValidationError, match="judge_id"):
        validate_preference_pair(payload)


def test_schema_rejects_same_winner_in_both_presentations():
    payload = copy.deepcopy(make_pair().to_dict())
    payload["verdict"]["swapped_pass"] = "A"
    with pytest.raises(PreferencePairValidationError, match="opposite"):
        validate_preference_pair(payload)


def test_schema_rejects_identical_completions():
    payload = copy.deepcopy(make_pair().to_dict())
    payload["rejected"] = payload["chosen"]
    with pytest.raises(PreferencePairValidationError, match="must differ"):
        validate_preference_pair(payload)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `rtk .venv/bin/pytest tests/test_preferences.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'tinystories_v2.preferences'`.

- [ ] **Step 3: Implement the versioned dataclasses and strict validator**

Create `src/tinystories_v2/preferences.py`:

```python
"""Versioned preference-pair records shared by RLAIF stages.

The JSON-compatible v1 contract is documented in
docs/schemas/preference-pair-v1.md. Only order-swap-consistent Judge
comparisons become PreferencePair values.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from tinystories_v2.slots import Scaffold

SCHEMA_VERSION = 1
SCAFFOLD_FIELDS = (
    "character",
    "trait",
    "setting",
    "conflict",
    "resolution",
    "moral",
)
TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "scaffold", "chosen", "rejected", "verdict"}
)
VERDICT_FIELDS = frozenset(
    {"judge_id", "first_pass", "swapped_pass", "consistent"}
)
ALLOWED_VERDICTS = frozenset({"A", "B"})


class PreferencePairValidationError(ValueError):
    """Raised when a value does not conform to preference-pair schema v1."""


@dataclass(frozen=True)
class VerdictMetadata:
    judge_id: str
    first_pass: str
    swapped_pass: str
    consistent: bool


@dataclass(frozen=True)
class PreferencePair:
    scaffold: Scaffold
    chosen: str
    rejected: str
    verdict: VerdictMetadata
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scaffold": asdict(self.scaffold),
            "chosen": self.chosen,
            "rejected": self.rejected,
            "verdict": asdict(self.verdict),
        }


def _require_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PreferencePairValidationError(f"{path} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    path: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(str(key) for key in actual - expected)
        raise PreferencePairValidationError(
            f"{path} keys mismatch: missing={missing}, extra={extra}"
        )


def _require_non_empty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreferencePairValidationError(f"{path} must be a non-empty string")
    return value


def validate_preference_pair(record: Mapping[str, object]) -> PreferencePair:
    """Parse and validate one JSON-decoded preference-pair v1 record."""

    top_level = _require_mapping(record, "record")
    _require_exact_keys(top_level, TOP_LEVEL_FIELDS, "record")

    version = top_level["schema_version"]
    if type(version) is not int or version != SCHEMA_VERSION:
        raise PreferencePairValidationError(
            f"schema_version must be exactly {SCHEMA_VERSION}"
        )

    scaffold_record = _require_mapping(top_level["scaffold"], "scaffold")
    _require_exact_keys(
        scaffold_record,
        frozenset(SCAFFOLD_FIELDS),
        "scaffold",
    )
    scaffold_values = {
        field: _require_non_empty_string(
            scaffold_record[field],
            f"scaffold.{field}",
        )
        for field in SCAFFOLD_FIELDS
    }
    scaffold = Scaffold(**scaffold_values)

    chosen = _require_non_empty_string(top_level["chosen"], "chosen")
    rejected = _require_non_empty_string(top_level["rejected"], "rejected")
    if chosen == rejected:
        raise PreferencePairValidationError("chosen and rejected must differ")

    verdict_record = _require_mapping(top_level["verdict"], "verdict")
    _require_exact_keys(verdict_record, VERDICT_FIELDS, "verdict")
    judge_id = _require_non_empty_string(
        verdict_record["judge_id"],
        "verdict.judge_id",
    )
    first_pass = _require_non_empty_string(
        verdict_record["first_pass"],
        "verdict.first_pass",
    )
    swapped_pass = _require_non_empty_string(
        verdict_record["swapped_pass"],
        "verdict.swapped_pass",
    )
    if first_pass not in ALLOWED_VERDICTS:
        raise PreferencePairValidationError(
            "verdict.first_pass must be 'A' or 'B'"
        )
    if swapped_pass not in ALLOWED_VERDICTS:
        raise PreferencePairValidationError(
            "verdict.swapped_pass must be 'A' or 'B'"
        )
    if verdict_record["consistent"] is not True:
        raise PreferencePairValidationError(
            "verdict.consistent must be true for a retained pair"
        )
    if first_pass == swapped_pass:
        raise PreferencePairValidationError(
            "first_pass and swapped_pass must be opposite for a retained pair"
        )

    return PreferencePair(
        scaffold=scaffold,
        chosen=chosen,
        rejected=rejected,
        verdict=VerdictMetadata(
            judge_id=judge_id,
            first_pass=first_pass,
            swapped_pass=swapped_pass,
            consistent=True,
        ),
    )
```

- [ ] **Step 4: Run the schema tests to verify they pass**

Run: `rtk .venv/bin/pytest tests/test_preferences.py -v`

Expected: `4 passed`.

- [ ] **Step 5: Document the normative JSON contract**

Create `docs/schemas/preference-pair-v1.md`:

````markdown
# Preference-Pair Schema v1

Issue 10 pins the record shared by preference labeling, Reward Model
training, DPO, and any fixture-driven RLAIF tests. Each JSONL line is one
retained, order-swap-consistent Judge preference.

## Record

```json
{
  "schema_version": 1,
  "scaffold": {
    "character": "fox",
    "trait": "greedy",
    "setting": "a dense forest",
    "conflict": "loses food to a trick",
    "resolution": "the trickster is exposed",
    "moral": "honesty is the best policy"
  },
  "chosen": "A greedy fox admitted the trick, returned the food, and learned that honesty is the best policy.",
  "rejected": "A fox walked through a forest and went home.",
  "verdict": {
    "judge_id": "fake:slot-coverage-v1",
    "first_pass": "A",
    "swapped_pass": "B",
    "consistent": true
  }
}
```

All five top-level fields are required and additional fields are rejected.
All six Scaffold fields, both completion fields, and `judge_id` are non-empty
strings. `chosen` and `rejected` must differ.

## Order-swap semantics

`first_pass` is relative to the original presentation `(A, B)`.
`swapped_pass` is relative to the second presentation `(B, A)`. Therefore,
opposite labels identify the same underlying Fable:

- `first_pass = "A"` and `swapped_pass = "B"` selects original A.
- `first_pass = "B"` and `swapped_pass = "A"` selects original B.

Equal labels reveal position bias or another inconsistency. Such a comparison
is discarded and is not a preference-pair record. Consequently every stored
v1 record has `consistent = true` and opposite pass labels.

## Consumer contract

Decode each JSONL line and pass it through
`tinystories_v2.preferences.validate_preference_pair` before training. The
helper returns the typed `PreferencePair` or raises
`PreferencePairValidationError`. Consumers must not silently accept another
schema version, infer a missing Judge identity, or reconstruct discarded
pairs.
````

- [ ] **Step 6: Commit the schema boundary**

```bash
rtk git add src/tinystories_v2/preferences.py tests/test_preferences.py docs/schemas/preference-pair-v1.md
rtk git commit -m "feat: pin preference-pair schema v1

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Add the Judge protocol, CPU fakes, and order-swap filter

**Files:**
- Create: `src/tinystories_v2/judge.py`
- Create: `tests/test_judge.py`

**Interfaces:**
- Consumes:
  - `Scaffold` from `tinystories_v2.slots`.
  - `PreferencePair`, `VerdictMetadata`, and `validate_preference_pair` from Task 1.
- Produces:
  - `Verdict.A` / `Verdict.B` string enum values and `Verdict.opposite`.
  - Runtime-checkable `Judge` protocol with `judge_id: str` and `compare(scaffold: Scaffold, fable_a: str, fable_b: str) -> Verdict`.
  - `SlotCoverageFakeJudge(judge_id="fake:slot-coverage-v1")`.
  - `PositionBiasedFakeJudge(judge_id="fake:position-a-v1")`.
  - `judge_with_order_swap(judge: Judge, scaffold: Scaffold, fable_a: str, fable_b: str) -> PreferencePair | None`.

- [ ] **Step 1: Write failing tests through the public Judge seam**

Create `tests/test_judge.py`:

```python
from dataclasses import asdict

from tinystories_v2.judge import (
    Judge,
    PositionBiasedFakeJudge,
    SlotCoverageFakeJudge,
    Verdict,
    judge_with_order_swap,
)
from tinystories_v2.preferences import validate_preference_pair
from tinystories_v2.slots import extract_slots


def fixture_case(fixture_records):
    source = fixture_records[0]
    scaffold = extract_slots(source["prompt"])
    explicit_slots = " ".join(asdict(scaffold).values())
    candidate_a = f"{source['fable']}\n\n{explicit_slots}"
    candidate_b = fixture_records[1]["fable"]
    return scaffold, candidate_a, candidate_b


def test_fakes_implement_judge_interface_on_fixture_fables(fixture_records):
    scaffold, candidate_a, candidate_b = fixture_case(fixture_records)
    consistent = SlotCoverageFakeJudge()
    biased = PositionBiasedFakeJudge()

    assert isinstance(consistent, Judge)
    assert isinstance(biased, Judge)
    assert consistent.compare(scaffold, candidate_a, candidate_b) is Verdict.A
    assert consistent.compare(scaffold, candidate_b, candidate_a) is Verdict.B
    assert biased.compare(scaffold, candidate_a, candidate_b) is Verdict.A
    assert biased.compare(scaffold, candidate_b, candidate_a) is Verdict.A


def test_equal_coverage_tie_break_is_position_independent(fixture_records):
    scaffold = extract_slots(fixture_records[0]["prompt"])
    judge = SlotCoverageFakeJudge()
    first = judge.compare(
        scaffold,
        "An unrelated amber Fable.",
        "An unrelated blue Fable.",
    )
    swapped = judge.compare(
        scaffold,
        "An unrelated blue Fable.",
        "An unrelated amber Fable.",
    )
    assert swapped is first.opposite


def test_consistent_fake_produces_schema_valid_pair(fixture_records):
    scaffold, candidate_a, candidate_b = fixture_case(fixture_records)
    pair = judge_with_order_swap(
        SlotCoverageFakeJudge(),
        scaffold,
        candidate_a,
        candidate_b,
    )

    assert pair is not None
    assert pair.chosen == candidate_a
    assert pair.rejected == candidate_b
    assert pair.verdict.judge_id == "fake:slot-coverage-v1"
    assert pair.verdict.first_pass == "A"
    assert pair.verdict.swapped_pass == "B"
    assert validate_preference_pair(pair.to_dict()) == pair


def test_position_biased_fake_is_discarded(fixture_records):
    scaffold, candidate_a, candidate_b = fixture_case(fixture_records)
    assert (
        judge_with_order_swap(
            PositionBiasedFakeJudge(),
            scaffold,
            candidate_a,
            candidate_b,
        )
        is None
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `rtk .venv/bin/pytest tests/test_judge.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'tinystories_v2.judge'`.

- [ ] **Step 3: Implement the protocol, fakes, and filter**

Create `src/tinystories_v2/judge.py`:

```python
"""Pairwise Judge seam for preference labeling and downstream tests."""

import hashlib
from dataclasses import astuple, dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from tinystories_v2.preferences import (
    PreferencePair,
    VerdictMetadata,
    validate_preference_pair,
)
from tinystories_v2.slots import Scaffold


class Verdict(StrEnum):
    A = "A"
    B = "B"

    @property
    def opposite(self) -> "Verdict":
        return Verdict.B if self is Verdict.A else Verdict.A


@runtime_checkable
class Judge(Protocol):
    @property
    def judge_id(self) -> str:
        raise NotImplementedError

    def compare(
        self,
        scaffold: Scaffold,
        fable_a: str,
        fable_b: str,
    ) -> Verdict:
        raise NotImplementedError


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _validate_candidates(fable_a: str, fable_b: str) -> None:
    if not fable_a.strip() or not fable_b.strip():
        raise ValueError("both candidate Fables must be non-empty")
    if _normalize(fable_a) == _normalize(fable_b):
        raise ValueError("candidate Fables must differ")


def _coverage_score(scaffold: Scaffold, fable: str) -> tuple[int, bytes]:
    normalized_fable = _normalize(fable)
    coverage = sum(
        _normalize(slot_value) in normalized_fable
        for slot_value in astuple(scaffold)
    )
    stable_tie_break = hashlib.sha256(
        normalized_fable.encode("utf-8")
    ).digest()
    return coverage, stable_tie_break


@dataclass(frozen=True)
class SlotCoverageFakeJudge:
    judge_id: str = "fake:slot-coverage-v1"

    def compare(
        self,
        scaffold: Scaffold,
        fable_a: str,
        fable_b: str,
    ) -> Verdict:
        _validate_candidates(fable_a, fable_b)
        score_a = _coverage_score(scaffold, fable_a)
        score_b = _coverage_score(scaffold, fable_b)
        return Verdict.A if score_a > score_b else Verdict.B


@dataclass(frozen=True)
class PositionBiasedFakeJudge:
    judge_id: str = "fake:position-a-v1"

    def compare(
        self,
        scaffold: Scaffold,
        fable_a: str,
        fable_b: str,
    ) -> Verdict:
        return Verdict.A


def judge_with_order_swap(
    judge: Judge,
    scaffold: Scaffold,
    fable_a: str,
    fable_b: str,
) -> PreferencePair | None:
    """Judge both presentations and retain only one consistent preference."""

    _validate_candidates(fable_a, fable_b)
    first_pass = judge.compare(scaffold, fable_a, fable_b)
    swapped_pass = judge.compare(scaffold, fable_b, fable_a)
    if first_pass is swapped_pass:
        return None

    if first_pass is Verdict.A:
        chosen, rejected = fable_a, fable_b
    else:
        chosen, rejected = fable_b, fable_a

    pair = PreferencePair(
        scaffold=scaffold,
        chosen=chosen,
        rejected=rejected,
        verdict=VerdictMetadata(
            judge_id=judge.judge_id,
            first_pass=first_pass.value,
            swapped_pass=swapped_pass.value,
            consistent=True,
        ),
    )
    return validate_preference_pair(pair.to_dict())
```

- [ ] **Step 4: Run the Judge seam tests to verify they pass**

Run: `rtk .venv/bin/pytest tests/test_judge.py -v`

Expected: `4 passed`. No optional Judge dependency, GPU, network, or model cache is used.

- [ ] **Step 5: Commit the test seam**

```bash
rtk git add src/tinystories_v2/judge.py tests/test_judge.py
rtk git commit -m "feat: add Judge protocol fakes and swap filter

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Render the adherence-weighted rubric and parse verdicts

**Files:**
- Modify: `src/tinystories_v2/judge.py`
- Modify: `tests/test_judge.py`

**Interfaces:**
- Consumes: `Scaffold` and `Verdict` from Task 2.
- Produces:
  - `RUBRIC_VERSION: str = "fable-pairwise-v1"`.
  - `render_rubric_prompt(scaffold: Scaffold, fable_a: str, fable_b: str) -> str`.
  - `parse_verdict(raw_output: str) -> Verdict`.
  - `JudgeOutputError(ValueError)` for output other than a single A/B verdict, with an optional `Verdict:` prefix and final period.

- [ ] **Step 1: Add failing rubric and parser tests**

In `tests/test_judge.py`, replace the imports at the top of the file with:

```python
from dataclasses import asdict

import pytest

from tinystories_v2.judge import (
    Judge,
    JudgeOutputError,
    PositionBiasedFakeJudge,
    SlotCoverageFakeJudge,
    Verdict,
    judge_with_order_swap,
    parse_verdict,
    render_rubric_prompt,
)
from tinystories_v2.preferences import validate_preference_pair
from tinystories_v2.slots import extract_slots
```

Then append:

```python
def test_rubric_renders_all_axes_priority_and_age_constraint(fixture_records):
    scaffold = extract_slots(fixture_records[0]["prompt"])
    prompt = render_rubric_prompt(
        scaffold,
        "Candidate A uses simple words and states the moral.",
        "Candidate B has a different ending.",
    )

    for axis in (
        "Grammar & Style",
        "Creativity",
        "Moral Clarity",
        "Prompt Adherence",
    ):
        assert axis in prompt
    assert "Prompt Adherence (HIGHEST WEIGHT)" in prompt
    assert "Moral Clarity (SECOND PRIORITY)" in prompt
    assert "ages 4–7" in prompt
    assert "HARD CONSTRAINT" in prompt
    for slot_value in asdict(scaffold).values():
        assert slot_value in prompt
    assert "Candidate A uses simple words" in prompt
    assert "Candidate B has a different ending" in prompt
    assert "Return exactly one capital letter: A or B." in prompt


@pytest.mark.parametrize(
    ("raw_output", "expected"),
    [
        ("A", Verdict.A),
        ("\nB\n", Verdict.B),
        ("Verdict: A.", Verdict.A),
    ],
)
def test_parse_verdict_accepts_one_unambiguous_label(raw_output, expected):
    assert parse_verdict(raw_output) is expected


@pytest.mark.parametrize("raw_output", ["tie", "A because it follows the moral"])
def test_parse_verdict_rejects_ambiguous_output(raw_output):
    with pytest.raises(JudgeOutputError, match="single A/B verdict"):
        parse_verdict(raw_output)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `rtk .venv/bin/pytest tests/test_judge.py -v`

Expected: collection fails because `JudgeOutputError`, `parse_verdict`, and `render_rubric_prompt` are not yet exported by `tinystories_v2.judge`.

- [ ] **Step 3: Add the rubric renderer and strict output parser**

In `src/tinystories_v2/judge.py`, replace the existing import block:

```python
import hashlib
from dataclasses import astuple, dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable
```

with:

```python
import hashlib
import json
import re
from dataclasses import asdict, astuple, dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable
```

Then append this code immediately after `Verdict`:

```python
RUBRIC_VERSION = "fable-pairwise-v1"


class JudgeOutputError(ValueError):
    """Raised when a real Judge does not return one parseable verdict."""


_VERDICT_RE = re.compile(
    r"\s*(?:verdict\s*:\s*)?([AB])\s*[.]?\s*",
    re.IGNORECASE,
)


def render_rubric_prompt(
    scaffold: Scaffold,
    fable_a: str,
    fable_b: str,
) -> str:
    """Render the pairwise rubric without invoking or importing a model."""

    payload = json.dumps(
        {
            "scaffold": asdict(scaffold),
            "candidate_a": fable_a,
            "candidate_b": fable_b,
        },
        ensure_ascii=False,
        indent=2,
    )
    return (
        "You are the Judge selecting the better moral Fable for children "
        "ages 4–7.\n\n"
        "Compare the candidates on exactly these four axes:\n"
        "1. Prompt Adherence (HIGHEST WEIGHT): faithful realization of all "
        "six Scaffold slots and the requested Fable form.\n"
        "2. Moral Clarity (SECOND PRIORITY): an explicit, relevant ethical "
        "lesson connected to the ending.\n"
        "3. Grammar & Style: correct, fluent, concrete, age-appropriate "
        "language.\n"
        "4. Creativity: an engaging and original narrative realization.\n\n"
        "Age suitability is a HARD CONSTRAINT: reject content whose "
        "vocabulary, syntax, themes, or detail are unsuitable for ages 4–7.\n"
        "Candidate labels are arbitrary. Judge content, never presentation "
        "position. Do not return a tie.\n\n"
        f"INPUT:\n{payload}\n\n"
        "Return exactly one capital letter: A or B."
    )


def parse_verdict(raw_output: str) -> Verdict:
    match = _VERDICT_RE.fullmatch(raw_output)
    if match is None:
        raise JudgeOutputError(
            f"Judge must return a single A/B verdict, got {raw_output[:120]!r}"
        )
    return Verdict(match.group(1).upper())
```

- [ ] **Step 4: Run the complete Judge tests**

Run: `rtk .venv/bin/pytest tests/test_judge.py -v`

Expected: `10 passed`. Rendering and parsing execute without importing Transformers.

- [ ] **Step 5: Commit the rubric**

```bash
rtk git add src/tinystories_v2/judge.py tests/test_judge.py
rtk git commit -m "feat: render pairwise Fable Judge rubric

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Add the lazy Transformers Judge and production selections

**Files:**
- Modify: `pyproject.toml:16-18`
- Modify: `src/tinystories_v2/judge.py`
- Create: `configs/judge_l4.toml`
- Create: `configs/judge_t4.toml`
- Create: `tests/test_judge_config.py`

**Interfaces:**
- Consumes:
  - `Judge`, `Verdict`, `RUBRIC_VERSION`, `render_rubric_prompt`, and `parse_verdict` from Tasks 2–3.
  - Existing `load_config(path) -> dict` from `tinystories_v2.config`.
  - Hugging Face `AutoTokenizer` / `AutoModelForCausalLM` only inside the first real `compare` call.
- Produces:
  - `TransformersJudge(model_id: str, precision: str, device: str, enable_thinking: bool | None = None, max_new_tokens: int = 4)` implementing `Judge`.
  - `build_judge(config: Mapping[str, Any]) -> Judge` supporting `fake_slot_coverage`, `fake_position_biased`, and `transformers`.
  - Stable L4 identity `transformers:Qwen/Qwen3-8B;precision=fp16;thinking=false;rubric=fable-pairwise-v1` and T4 identity `transformers:Qwen/Qwen3-4B-Instruct-2507;precision=fp16;thinking=default;rubric=fable-pairwise-v1`.
  - `configs/judge_l4.toml` selecting `Qwen/Qwen3-8B` fp16 with thinking disabled.
  - `configs/judge_t4.toml` selecting `Qwen/Qwen3-4B-Instruct-2507` fp16 in its native non-thinking mode.

- [ ] **Step 1: Write failing offline configuration tests**

Create `tests/test_judge_config.py`:

```python
from pathlib import Path

import pytest

from tinystories_v2.config import load_config
from tinystories_v2.judge import (
    PositionBiasedFakeJudge,
    SlotCoverageFakeJudge,
    TransformersJudge,
    build_judge,
)

CONFIG_DIR = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize(
    ("filename", "model_id", "enable_thinking"),
    [
        ("judge_l4.toml", "Qwen/Qwen3-8B", False),
        (
            "judge_t4.toml",
            "Qwen/Qwen3-4B-Instruct-2507",
            None,
        ),
    ],
)
def test_real_configs_select_one_lazy_transformers_path(
    filename,
    model_id,
    enable_thinking,
):
    config = load_config(CONFIG_DIR / filename)["judge"]
    judge = build_judge(config)

    assert type(judge) is TransformersJudge
    assert judge.model_id == model_id
    assert judge.precision == "fp16"
    assert judge.device == "cuda"
    assert judge.enable_thinking is enable_thinking
    assert model_id in judge.judge_id
    assert "precision=fp16" in judge.judge_id
    assert "rubric=fable-pairwise-v1" in judge.judge_id


@pytest.mark.parametrize(
    ("kind", "expected_type"),
    [
        ("fake_slot_coverage", SlotCoverageFakeJudge),
        ("fake_position_biased", PositionBiasedFakeJudge),
    ],
)
def test_factory_selects_offline_fakes(kind, expected_type):
    assert type(build_judge({"kind": kind})) is expected_type


def test_factory_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown Judge kind"):
        build_judge({"kind": "remote_api"})


def test_factory_rejects_unknown_precision_without_loading_model():
    with pytest.raises(ValueError, match="precision"):
        build_judge(
            {
                "kind": "transformers",
                "model_id": "Qwen/Qwen3-8B",
                "precision": "int8",
                "device": "cuda",
            }
        )
```

- [ ] **Step 2: Run the configuration tests to verify they fail**

Run: `rtk .venv/bin/pytest tests/test_judge_config.py -v`

Expected: collection fails because `TransformersJudge` and `build_judge` do not exist yet.

- [ ] **Step 3: Add the optional real-Judge dependencies**

In `pyproject.toml`, replace:

```toml
[project.optional-dependencies]
dev = ["pytest>=8"]
```

with:

```toml
[project.optional-dependencies]
dev = ["pytest>=8"]
judge = [
    "torch>=2.3",
    "transformers>=4.51.0",
]
```

Do not install `.[judge]` on the laptop test path. Colab installs it when running a real Judge; the default `.[dev]` environment remains unchanged.

- [ ] **Step 4: Add both production configuration selections**

Create `configs/judge_l4.toml`:

```toml
[judge]
kind = "transformers"
model_id = "Qwen/Qwen3-8B"
precision = "fp16"
device = "cuda"
enable_thinking = false
max_new_tokens = 4
```

Create `configs/judge_t4.toml`:

```toml
[judge]
kind = "transformers"
model_id = "Qwen/Qwen3-4B-Instruct-2507"
precision = "fp16"
device = "cuda"
max_new_tokens = 4
```

- [ ] **Step 5: Implement lazy model loading and the config factory**

In `src/tinystories_v2/judge.py`, replace:

```python
import hashlib
import json
import re
from dataclasses import asdict, astuple, dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable
```

with:

```python
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, astuple, dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable
```

Then append:

```python
class TransformersJudge:
    """Pairwise local-model Judge backed by one lazy Transformers code path."""

    def __init__(
        self,
        model_id: str,
        precision: str,
        device: str,
        enable_thinking: bool | None = None,
        max_new_tokens: int = 4,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if precision not in {"fp16", "bf16"}:
            raise ValueError("precision must be 'fp16' or 'bf16'")
        if not device.strip():
            raise ValueError("device must be a non-empty string")
        if enable_thinking is not None and type(enable_thinking) is not bool:
            raise ValueError("enable_thinking must be bool or omitted")
        if (
            type(max_new_tokens) is not int
            or max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")

        self.model_id = model_id
        self.precision = precision
        self.device = device
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self._backend: tuple[Any, Any, Any] | None = None

    @property
    def judge_id(self) -> str:
        if self.enable_thinking is None:
            thinking_mode = "default"
        else:
            thinking_mode = str(self.enable_thinking).lower()
        return (
            f"transformers:{self.model_id};precision={self.precision};"
            f"thinking={thinking_mode};rubric={RUBRIC_VERSION}"
        )

    def _load_backend(self) -> tuple[Any, Any, Any]:
        if self._backend is not None:
            return self._backend
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "real Judge dependencies are missing; "
                "install with: uv pip install -e '.[judge]'"
            ) from exc

        dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        )
        model = model.to(self.device)
        model.eval()
        self._backend = (torch, tokenizer, model)
        return self._backend

    def compare(
        self,
        scaffold: Scaffold,
        fable_a: str,
        fable_b: str,
    ) -> Verdict:
        _validate_candidates(fable_a, fable_b)
        torch, tokenizer, model = self._load_backend()
        messages = [
            {
                "role": "system",
                "content": (
                    "Follow the pairwise Fable rubric and return only its "
                    "requested verdict label."
                ),
            },
            {
                "role": "user",
                "content": render_rubric_prompt(
                    scaffold,
                    fable_a,
                    fable_b,
                ),
            },
        ]
        template_options: dict[str, Any] = {}
        if self.enable_thinking is not None:
            template_options["enable_thinking"] = self.enable_thinking
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_options,
        ).to(self.device)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        raw_output = tokenizer.decode(
            generated[0][prompt_length:],
            skip_special_tokens=True,
        )
        return parse_verdict(raw_output)


def _required_config_string(
    config: Mapping[str, Any],
    key: str,
) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Judge config {key!r} must be a non-empty string")
    return value.strip()


def build_judge(config: Mapping[str, Any]) -> Judge:
    """Construct a fake or real Judge without loading real model weights."""

    kind = config.get("kind")
    if kind == "fake_slot_coverage":
        return SlotCoverageFakeJudge()
    if kind == "fake_position_biased":
        return PositionBiasedFakeJudge()
    if kind == "transformers":
        device = config.get("device", "cuda")
        if not isinstance(device, str) or not device.strip():
            raise ValueError(
                "Judge config 'device' must be a non-empty string"
            )
        return TransformersJudge(
            model_id=_required_config_string(config, "model_id"),
            precision=_required_config_string(config, "precision"),
            device=device.strip(),
            enable_thinking=config.get("enable_thinking"),
            max_new_tokens=config.get("max_new_tokens", 4),
        )
    raise ValueError(f"unknown Judge kind: {kind!r}")
```

- [ ] **Step 6: Run the factory tests without real-model dependencies**

Run: `rtk .venv/bin/pytest tests/test_judge_config.py -v`

Expected: `6 passed`. The test must finish without importing PyTorch or Transformers and without accessing the network.

- [ ] **Step 7: Run the complete offline suite**

Run: `rtk .venv/bin/pytest -q`

Expected: `42 passed`.

- [ ] **Step 8: Commit the real adapter and configurations**

```bash
rtk git add pyproject.toml src/tinystories_v2/judge.py configs/judge_l4.toml configs/judge_t4.toml tests/test_judge_config.py
rtk git commit -m "feat: add config-selected Qwen Judges

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Audit acceptance criteria and close the local issue

**Files:**
- Modify: `.scratch/tinystories-v2-pipeline/issues/10-judge-seam-and-pair-schema.md:3,41-46`

**Interfaces:**
- Consumes: every public interface and test delivered by Tasks 1–4.
- Produces: a checked local issue whose acceptance statements are backed by named tests and a clean full-suite result.

- [ ] **Step 1: Run the issue-specific acceptance suite**

Run:

```bash
rtk .venv/bin/pytest tests/test_preferences.py tests/test_judge.py tests/test_judge_config.py -v
```

Expected: `20 passed`, including fixture-generated schema-valid pairs, both fake swap outcomes, all rubric assertions, and both real configurations.

- [ ] **Step 2: Run repository-wide verification and whitespace checks**

Run:

```bash
rtk .venv/bin/pytest -q
rtk git diff --check
```

Expected: `42 passed`; `git diff --check` prints nothing.

- [ ] **Step 3: Mark the issue complete**

In `.scratch/tinystories-v2-pipeline/issues/10-judge-seam-and-pair-schema.md`, change:

```markdown
Status: ready-for-agent
```

to:

```markdown
Status: complete
```

Replace the acceptance checklist with:

```markdown
- [x] Judge interface + fakes run on CPU on fixture fables with no GPU, network, or model download
- [x] Order-swap consistency filter is tested through the interface: the position-biased fake yields discarded pairs, the consistent fake yields kept pairs
- [x] Rubric prompt includes the four axes with adherence weighted highest and is covered by a rendering test (no model download in tests)
- [x] Real Judge implementations are config-selected (model id, precision), sharing one code path
- [x] Preference-pair schema is documented, has a validation helper, and a fixture-based test produces schema-valid pairs via the fake Judge
- [x] Verdict metadata records judge identity so downstream artifacts are never ambiguous about who judged
```

- [ ] **Step 4: Commit the acceptance audit**

```bash
rtk git add .scratch/tinystories-v2-pipeline/issues/10-judge-seam-and-pair-schema.md
rtk git commit -m "docs: complete issue 10 Judge seam

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
