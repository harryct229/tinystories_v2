# Slot Prompt Renderer + SFT Dataset Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Slot Prompt format as code — render a Scaffold to the fixed special-token sequence, encode SFT training examples with a masked-loss boundary at `<|fable|>`, parse the inverse, and build the SFT dataset artifact from the data-prep `sft` split (issue 12).

**Architecture:** One pure-format module (`slot_prompt.py`) holds rendering, encoding (with loss mask), and parsing over a `tokenizers.Tokenizer`. One stage module (`sft_data.py`) follows the repo's config→artifacts convention: it reads the tokenizer artifact and the `sft` split, and writes one JSONL example per line plus a `manifest.json`. The token order in `slots.SLOT_SPECIAL_TOKENS` is the format contract issues 03, 04, 06, and 07 depend on — it is never reordered.

**Tech Stack:** Python ≥3.11, `tokenizers` (byte-level BPE, already a core dep), stdlib `tomllib`/`json`/`argparse`, `pytest`. No `torch`/`numpy` in this issue's code or tests — the suite must run on a laptop CPU with no GPU or network (PRD story 25). Run tests with the repo venv: `.venv/bin/pytest ...` (or activate `.venv` and use `pytest`).

## Global Constraints

- **Stage convention:** every stage is an independently invocable entrypoint that reads one TOML config and writes artifacts to `config["out_dir"]`; stages communicate only through artifacts, never in-memory state. Every stage writes a `manifest.json` carrying `stage`, `package_version` (`tinystories_v2.__version__`, currently `"0.1.0"`), and the full `config`.
- **Slot Prompt token order is fixed** — `slots.SLOT_SPECIAL_TOKENS = ("<|character|>", "<|trait|>", "<|setting|>", "<|conflict|>", "<|resolution|>", "<|moral|>", "<|fable|>", "<|end|>")`. Never reorder, rename, or insert in the middle.
- **Loss mask:** masked (`0`) over the conditioning prefix through and including `<|fable|>`; active (`1`) over the fable body and the trailing `<|end|>`.
- **Determinism:** artifacts are byte-identical across two runs on the same inputs — no unseeded randomness, examples emitted in input order, `json.dumps(..., ensure_ascii=False)` for JSONL rows.
- **Vocabulary (CONTEXT.md):** use Fable, Scaffold, Slot Prompt, SFT consistently; never introduce new terms for these.
- **No new heavy dependencies:** stay within the existing core deps (`datasets`, `tokenizers`, `huggingface_hub`); do not import `torch`/`numpy`.
- **TDD, frequent commits, DRY, YAGNI.** Conventional-commit messages (`feat:`, `test:`, `docs:`, `chore:`) matching repo history.

---

## File Structure

- **Create `src/tinystories_v2/slot_prompt.py`** — the format module. Public surface: `SLOT_FIELDS`, `FABLE_TOKEN`, `END_TOKEN`, `SlotPromptError`, `SlotPromptExample`, `ParsedSlotPrompt`, `render_prompt`, `render_example`, `encode_example`, `parse_example`. Depends only on `slots` and a `tokenizers.Tokenizer`.
- **Create `src/tinystories_v2/sft_data.py`** — the SFT dataset builder stage (config→artifact). Public: `SCHEMA_VERSION`, `build_example_record`, `run`, `main`.
- **Create `configs/sft_data_fixture.toml`** — toy config against fixture artifacts, for local sanity runs.
- **Create `configs/sft_data_full.toml`** — real-run config against the full data-prep + tokenizer artifacts.
- **Create `docs/schemas/sft-example-v1.md`** — the artifact record schema (mirrors `docs/schemas/preference-pair-v1.md` style).
- **Create `tests/test_slot_prompt.py`** — renderer, encoder, and parser behavior (uses a fixture-trained tokenizer).
- **Create `tests/test_sft_data_stage.py`** — stage artifact contract, determinism, split-source integrity, CLI.
- **Modify `pyproject.toml`** — add `ts2-sft-data = "tinystories_v2.sft_data:main"` under `[project.scripts]`.

Tasks 1–3 grow `slot_prompt.py` incrementally (render → encode → parse). Task 4 builds the stage on top and ships the docs/configs/entrypoint. Each task ends with an independently testable deliverable and a commit.

---

### Task 1: Slot Prompt renderer (text)

**Files:**
- Create: `src/tinystories_v2/slot_prompt.py`
- Test: `tests/test_slot_prompt.py`

**Interfaces:**
- Consumes: `tinystories_v2.slots.Scaffold` (frozen dataclass with fields `character, trait, setting, conflict, resolution, moral`) and `tinystories_v2.slots.SLOT_SPECIAL_TOKENS`.
- Produces:
  - `SLOT_FIELDS: tuple[str, ...] = ("character", "trait", "setting", "conflict", "resolution", "moral")` — the six conditioning slots in render order.
  - `FABLE_TOKEN = "<|fable|>"`, `END_TOKEN = "<|end|>"`.
  - `class SlotPromptError(ValueError)`.
  - `render_prompt(scaffold: Scaffold) -> str` — the conditioning prefix ending at `<|fable|>` (no body). This is the generation prompt for issue 03's demo.
  - `render_example(scaffold: Scaffold, fable: str) -> str` — full training text: prompt prefix + fable body + `<|end|>`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_slot_prompt.py`:

```python
import pytest

from tinystories_v2.slots import Scaffold
from tinystories_v2.slot_prompt import (
    SlotPromptError,
    render_example,
    render_prompt,
)

SCAFFOLD = Scaffold(
    character="fox",
    trait="greedy",
    setting="a dense forest",
    conflict="loses their food",
    resolution="the trickster is exposed",
    moral="honesty is the best policy",
)


def test_render_prompt_is_exact_slot_sequence_ending_at_fable():
    assert render_prompt(SCAFFOLD) == (
        "<|character|>fox<|trait|>greedy<|setting|>a dense forest"
        "<|conflict|>loses their food<|resolution|>the trickster is exposed"
        "<|moral|>honesty is the best policy<|fable|>"
    )


def test_render_example_appends_fable_body_and_end():
    text = render_example(SCAFFOLD, "One day, a fox schemed.")
    assert text == render_prompt(SCAFFOLD) + "One day, a fox schemed.<|end|>"


def test_render_empty_slot_raises():
    bad = Scaffold(
        character="fox", trait="   ", setting="s",
        conflict="c", resolution="r", moral="m",
    )
    with pytest.raises(SlotPromptError, match="trait"):
        render_prompt(bad)


def test_render_empty_fable_raises():
    with pytest.raises(SlotPromptError, match="fable"):
        render_example(SCAFFOLD, "   ")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_slot_prompt.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tinystories_v2.slot_prompt'`.

- [ ] **Step 3: Write the minimal implementation**

Create `src/tinystories_v2/slot_prompt.py`:

```python
"""Slot Prompt format: render a Scaffold to the reserved special-token
sequence, encode SFT training examples with a masked-loss boundary, and parse
the inverse.

The token order is the format contract every stage from SFT onward relies on
(issues 03, 04, 06, 07). The sequence is exactly:

    <|character|>C<|trait|>T<|setting|>S<|conflict|>Cf<|resolution|>R
    <|moral|>M<|fable|>{fable body}<|end|>

Loss is masked over the conditioning prefix (through and including
`<|fable|>`) and active over the fable body and the trailing `<|end|>`.
"""

from tinystories_v2.slots import Scaffold

# The six conditioning slots in render order — the first six SLOT_SPECIAL_TOKENS
# without their <| |> delimiters. The trailing two specials (<|fable|>, <|end|>)
# frame the fable body, not a slot.
SLOT_FIELDS = ("character", "trait", "setting", "conflict", "resolution", "moral")

FABLE_TOKEN = "<|fable|>"
END_TOKEN = "<|end|>"


class SlotPromptError(ValueError):
    """Raised when a Scaffold cannot be rendered or a token sequence cannot be
    parsed as a Slot Prompt (empty slot, missing marker, wrong order)."""


def _slot_values(scaffold: Scaffold) -> list[str]:
    values = []
    for field in SLOT_FIELDS:
        value = getattr(scaffold, field)
        if not value or not value.strip():
            raise SlotPromptError(f"{field} slot is empty")
        values.append(value)
    return values


def render_prompt(scaffold: Scaffold) -> str:
    """The conditioning prefix ending at <|fable|> (no fable body). Feed this to
    the model and let it complete the fable."""
    values = _slot_values(scaffold)
    parts = [f"<|{field}|>{value}" for field, value in zip(SLOT_FIELDS, values)]
    return "".join(parts) + FABLE_TOKEN


def render_example(scaffold: Scaffold, fable: str) -> str:
    """The full training text: prompt prefix + fable body + <|end|>."""
    if not fable or not fable.strip():
        raise SlotPromptError("fable body is empty")
    return render_prompt(scaffold) + fable + END_TOKEN
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_slot_prompt.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/slot_prompt.py tests/test_slot_prompt.py
git commit -m "feat: add Slot Prompt text renderer"
```

---

### Task 2: Encode SFT example with loss mask

**Files:**
- Modify: `src/tinystories_v2/slot_prompt.py`
- Test: `tests/test_slot_prompt.py`

**Interfaces:**
- Consumes: `render_example` (Task 1), `tinystories_v2.slots.SLOT_SPECIAL_TOKENS`, a trained `tokenizers.Tokenizer` whose reserved specials each encode to a single ID (guaranteed by the issue-01 tokenizer stage).
- Produces:
  - `@dataclass(frozen=True) class SlotPromptExample` with `input_ids: list[int]`, `loss_mask: list[int]`, `n_prompt_tokens: int`, and `to_dict() -> dict`.
  - `encode_example(tokenizer: Tokenizer, scaffold: Scaffold, fable: str) -> SlotPromptExample` — tokenizes the full example; `loss_mask` is `0` through and including `<|fable|>` and `1` after; `n_prompt_tokens` is the count of leading masked tokens.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_slot_prompt.py`. Add these imports at the top (extend the existing `slot_prompt` import):

```python
from tinystories_v2.slot_prompt import (
    FABLE_TOKEN,
    SlotPromptError,
    encode_example,
    render_example,
    render_prompt,
)
from tinystories_v2.slots import SLOT_SPECIAL_TOKENS
from tinystories_v2.tokenizer import iter_corpus, train_tokenizer
```

Then add the tokenizer fixture and the encoder tests (`fixture_path` comes from `tests/conftest.py`):

```python
@pytest.fixture(scope="module")
def tokenizer(fixture_path):
    # Toy vocab; the artifact contract (specials -> single IDs) is identical to
    # the real 8192 tokenizer, and 512 trains in well under a second.
    texts = iter_corpus([str(fixture_path)], "fable")
    return train_tokenizer(texts, vocab_size=512)


def test_special_tokens_encode_to_single_ids_in_order(tokenizer):
    ex = encode_example(tokenizer, SCAFFOLD, "One day, a fox schemed and lost.")
    spec_ids = {t: tokenizer.token_to_id(t) for t in SLOT_SPECIAL_TOKENS}
    seen = [t for i in ex.input_ids for t in SLOT_SPECIAL_TOKENS if spec_ids[t] == i]
    assert seen == list(SLOT_SPECIAL_TOKENS)  # each special once, in order


def test_loss_mask_boundary_is_exactly_at_fable_token(tokenizer):
    ex = encode_example(tokenizer, SCAFFOLD, "One day, a fox schemed and lost.")
    boundary = ex.input_ids.index(tokenizer.token_to_id(FABLE_TOKEN))
    assert ex.n_prompt_tokens == boundary + 1
    assert ex.loss_mask == (
        [0] * (boundary + 1) + [1] * (len(ex.input_ids) - boundary - 1)
    )
    assert ex.loss_mask[boundary] == 0      # <|fable|> itself is masked
    assert ex.loss_mask[boundary + 1] == 1  # first fable-body token is active
    assert ex.loss_mask[-1] == 1            # <|end|> is active


def test_mask_and_ids_are_same_length(tokenizer):
    ex = encode_example(tokenizer, SCAFFOLD, "A short fable body.")
    assert len(ex.loss_mask) == len(ex.input_ids)


def test_example_to_dict_has_schema_fields(tokenizer):
    ex = encode_example(tokenizer, SCAFFOLD, "A short fable body.")
    assert set(ex.to_dict()) == {"input_ids", "loss_mask", "n_prompt_tokens"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_slot_prompt.py -q`
Expected: FAIL — `ImportError: cannot import name 'encode_example'`.

- [ ] **Step 3: Write the minimal implementation**

Add to `src/tinystories_v2/slot_prompt.py`. Update the imports and add the dataclass + function:

```python
from dataclasses import dataclass

from tokenizers import Tokenizer

from tinystories_v2.slots import SLOT_SPECIAL_TOKENS, Scaffold
```

```python
@dataclass(frozen=True)
class SlotPromptExample:
    input_ids: list[int]
    loss_mask: list[int]   # 0 over the prompt prefix, 1 over body + <|end|>
    n_prompt_tokens: int   # leading masked tokens (through <|fable|>)

    def to_dict(self) -> dict:
        return {
            "input_ids": self.input_ids,
            "loss_mask": self.loss_mask,
            "n_prompt_tokens": self.n_prompt_tokens,
        }


def encode_example(
    tokenizer: Tokenizer, scaffold: Scaffold, fable: str
) -> SlotPromptExample:
    """Tokenize a training example and compute its loss mask: masked (0) through
    and including <|fable|>, active (1) over the fable body and <|end|>."""
    text = render_example(scaffold, fable)
    ids = tokenizer.encode(text).ids
    boundary = ids.index(tokenizer.token_to_id(FABLE_TOKEN))  # <|fable|> once
    n_prompt_tokens = boundary + 1
    loss_mask = [0] * n_prompt_tokens + [1] * (len(ids) - n_prompt_tokens)
    return SlotPromptExample(
        input_ids=ids, loss_mask=loss_mask, n_prompt_tokens=n_prompt_tokens
    )
```

Note: keep the existing `from tinystories_v2.slots import Scaffold` line consistent — the file should import `Scaffold`, `SLOT_SPECIAL_TOKENS` from `slots`. Merge the Task 1 import (`from tinystories_v2.slots import Scaffold`) into the single line above.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_slot_prompt.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/slot_prompt.py tests/test_slot_prompt.py
git commit -m "feat: encode SFT examples with masked-loss boundary at <|fable|>"
```

---

### Task 3: Slot Prompt parser (inverse) + malformed handling

**Files:**
- Modify: `src/tinystories_v2/slot_prompt.py`
- Test: `tests/test_slot_prompt.py`

**Interfaces:**
- Consumes: `encode_example` (Task 2), `tinystories_v2.slots.extract_slots` (for fixture round-trip), a trained `tokenizers.Tokenizer`.
- Produces:
  - `@dataclass(frozen=True) class ParsedSlotPrompt` with `scaffold: Scaffold`, `fable: str`.
  - `parse_example(tokenizer: Tokenizer, ids: Sequence[int]) -> ParsedSlotPrompt` — recovers the Scaffold and fable body from a **complete** example sequence. Raises `SlotPromptError` unless all eight specials appear exactly once, in canonical order, with `<|character|>` first and `<|end|>` last. (Callers parsing raw generations should truncate at the first `<|end|>` before calling.)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_slot_prompt.py`. Extend the imports (`parse_example`, `ParsedSlotPrompt` from `slot_prompt`; `extract_slots` from `slots`):

```python
from tinystories_v2.slot_prompt import (
    FABLE_TOKEN,
    ParsedSlotPrompt,
    SlotPromptError,
    encode_example,
    parse_example,
    render_example,
    render_prompt,
)
from tinystories_v2.slots import SLOT_SPECIAL_TOKENS, extract_slots
```

Then add:

```python
def test_render_encode_parse_round_trip(tokenizer):
    fable = "One day, a greedy fox lost his lunch and learned to share."
    ex = encode_example(tokenizer, SCAFFOLD, fable)
    parsed = parse_example(tokenizer, ex.input_ids)
    assert isinstance(parsed, ParsedSlotPrompt)
    assert parsed.scaffold == SCAFFOLD
    assert parsed.fable == fable


def test_round_trip_on_fixture_records(tokenizer, fixture_records):
    for record in fixture_records[:5]:
        scaffold = extract_slots(record["prompt"])
        ex = encode_example(tokenizer, scaffold, record["fable"])
        parsed = parse_example(tokenizer, ex.input_ids)
        assert parsed.scaffold == scaffold
        assert parsed.fable == record["fable"]


def test_parse_missing_slot_raises(tokenizer):
    text = (
        "<|character|>fox<|trait|>greedy<|setting|>a forest"
        "<|conflict|>c<|resolution|>r<|fable|>body<|end|>"  # no <|moral|>
    )
    ids = tokenizer.encode(text).ids
    with pytest.raises(SlotPromptError, match="moral"):
        parse_example(tokenizer, ids)


def test_parse_wrong_order_raises(tokenizer):
    text = (
        "<|trait|>greedy<|character|>fox<|setting|>a forest"  # trait before character
        "<|conflict|>c<|resolution|>r<|moral|>m<|fable|>body<|end|>"
    )
    ids = tokenizer.encode(text).ids
    with pytest.raises(SlotPromptError, match="order"):
        parse_example(tokenizer, ids)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_slot_prompt.py -q`
Expected: FAIL — `ImportError: cannot import name 'parse_example'`.

- [ ] **Step 3: Write the minimal implementation**

Add to `src/tinystories_v2/slot_prompt.py`. Add `from collections.abc import Sequence` to the imports, then:

```python
@dataclass(frozen=True)
class ParsedSlotPrompt:
    scaffold: Scaffold
    fable: str


def parse_example(tokenizer: Tokenizer, ids: Sequence[int]) -> ParsedSlotPrompt:
    """Recover the Scaffold and fable body from a complete example sequence.
    Raises SlotPromptError unless all eight special tokens appear exactly once,
    in canonical order, with <|character|> first and <|end|> last."""
    ids = list(ids)
    special_ids = {tok: tokenizer.token_to_id(tok) for tok in SLOT_SPECIAL_TOKENS}
    for tok, sid in special_ids.items():
        if ids.count(sid) != 1:
            raise SlotPromptError(f"{tok} must appear exactly once")
    positions = [ids.index(special_ids[tok]) for tok in SLOT_SPECIAL_TOKENS]
    if positions != sorted(positions):
        raise SlotPromptError("special tokens are out of order")
    if ids[0] != special_ids["<|character|>"] or ids[-1] != special_ids[END_TOKEN]:
        raise SlotPromptError(
            "sequence must start with <|character|> and end with <|end|>"
        )
    # positions align 1:1 with SLOT_SPECIAL_TOKENS; decode each inter-marker span.
    slot_values = {
        field: tokenizer.decode(ids[start + 1 : end])
        for field, start, end in zip(SLOT_FIELDS, positions[:6], positions[1:7])
    }
    fable = tokenizer.decode(ids[positions[6] + 1 : positions[7]])
    return ParsedSlotPrompt(scaffold=Scaffold(**slot_values), fable=fable)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_slot_prompt.py -q`
Expected: PASS (12 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/slot_prompt.py tests/test_slot_prompt.py
git commit -m "feat: parse Slot Prompt token sequences back to Scaffold and fable"
```

---

### Task 4: SFT dataset builder stage

**Files:**
- Create: `src/tinystories_v2/sft_data.py`
- Create: `docs/schemas/sft-example-v1.md`
- Create: `configs/sft_data_fixture.toml`
- Create: `configs/sft_data_full.toml`
- Modify: `pyproject.toml` (add `[project.scripts]` entry)
- Test: `tests/test_sft_data_stage.py`

**Interfaces:**
- Consumes: `encode_example` and `SLOT_FIELDS` (Tasks 1–2), `parse_example` (Task 3), `tinystories_v2.slots.Scaffold`, `tinystories_v2.config.load_config`, `tinystories_v2.__version__`, `tokenizers.Tokenizer.from_file`, the data-prep `sft` split (`splits/sft.jsonl`, rows keyed `prompt_hash` + six slot fields + `fable`), the tokenizer artifact (`tokenizer.json`).
- Produces:
  - Artifact `examples.jsonl` — one record per line: `{"prompt_hash": str, "input_ids": [int], "loss_mask": [int], "n_prompt_tokens": int}`.
  - Artifact `manifest.json` — `{"stage": "sft_data", "package_version", "schema_version": 1, "count", "config"}`.
  - `SCHEMA_VERSION = 1`, `build_example_record(tokenizer, row) -> dict`, `run(config) -> None`, `main(argv=None) -> None`.
  - Console entrypoint `ts2-sft-data`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sft_data_stage.py`:

```python
import json
import subprocess
import sys

import pytest
from tokenizers import Tokenizer

from tinystories_v2.data import run as data_run
from tinystories_v2.sft_data import run as sft_run
from tinystories_v2.slot_prompt import parse_example
from tinystories_v2.tokenizer import run as tokenizer_run


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, fixture_path):
    """Run the data-prep and tokenizer stages so the SFT builder has real
    upstream artifacts to read (stages couple only through artifacts)."""
    base = tmp_path_factory.mktemp("sft_inputs")
    data_dir = base / "data"
    tok_dir = base / "tok"
    data_run({
        "out_dir": str(data_dir),
        "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        # sft weighted high so the split has plenty of examples from ~120 Fables.
        "splits": {"seed": "fixture-v1", "pretrain": 0.4, "sft": 0.4, "pref": 0.1, "eval": 0.1},
    })
    tokenizer_run({
        "out_dir": str(tok_dir),
        "corpus": [str(fixture_path)],
        "text_field": "fable",
        "vocab_size": 512,
    })
    return {
        "sft_split": str(data_dir / "splits" / "sft.jsonl"),
        "tokenizer": str(tok_dir / "tokenizer.json"),
    }


def make_config(out_dir, prepared) -> dict:
    return {
        "out_dir": str(out_dir),
        "tokenizer": prepared["tokenizer"],
        "sft_split": prepared["sft_split"],
        "max_examples": 0,
    }


def sft_hashes(prepared) -> set:
    with open(prepared["sft_split"], encoding="utf-8") as f:
        return {json.loads(line)["prompt_hash"] for line in f if line.strip()}


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory, prepared):
    out = tmp_path_factory.mktemp("sft_data")
    sft_run(make_config(out, prepared))
    return out


def read_examples(artifact_dir) -> list[dict]:
    lines = (artifact_dir / "examples.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_artifact_contract(artifact_dir, prepared):
    assert (artifact_dir / "examples.jsonl").exists()
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "sft_data"
    assert manifest["schema_version"] == 1
    assert manifest["count"] == len(sft_hashes(prepared))


def test_example_records_have_schema_and_aligned_mask(artifact_dir):
    records = read_examples(artifact_dir)
    assert records
    for rec in records:
        assert set(rec) == {"prompt_hash", "input_ids", "loss_mask", "n_prompt_tokens"}
        assert len(rec["input_ids"]) == len(rec["loss_mask"])
        n = rec["n_prompt_tokens"]
        assert rec["loss_mask"][:n] == [0] * n
        assert set(rec["loss_mask"][n:]) <= {1}
        assert rec["loss_mask"][n:]  # at least one active (fable body) token


def test_examples_parse_back_to_a_scaffold(artifact_dir, prepared):
    tokenizer = Tokenizer.from_file(prepared["tokenizer"])
    parsed = parse_example(tokenizer, read_examples(artifact_dir)[0]["input_ids"])
    assert parsed.fable.strip()


def test_reads_only_the_configured_sft_split(artifact_dir, prepared):
    emitted = {rec["prompt_hash"] for rec in read_examples(artifact_dir)}
    assert emitted == sft_hashes(prepared)


def test_two_runs_are_byte_identical(tmp_path, prepared):
    for name in ("run1", "run2"):
        sft_run(make_config(tmp_path / name, prepared))
    assert (tmp_path / "run1" / "examples.jsonl").read_bytes() == (
        tmp_path / "run2" / "examples.jsonl"
    ).read_bytes()


def test_max_examples_caps_output(tmp_path, prepared):
    out = tmp_path / "capped"
    config = make_config(out, prepared)
    config["max_examples"] = 2
    sft_run(config)
    assert len(read_examples(out)) == 2


def test_cli_entrypoint_runs_standalone(tmp_path, prepared):
    out = tmp_path / "cli_out"
    config_file = tmp_path / "cfg.toml"
    config_file.write_text(
        f'out_dir = "{out}"\ntokenizer = "{prepared["tokenizer"]}"\n'
        f'sft_split = "{prepared["sft_split"]}"\nmax_examples = 0\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.sft_data", "--config", str(config_file)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "examples.jsonl").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_sft_data_stage.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tinystories_v2.sft_data'`.

- [ ] **Step 3: Write the stage implementation**

Create `src/tinystories_v2/sft_data.py`:

```python
"""SFT dataset builder: the data-prep `sft` split -> masked-loss training
examples (issue 12).

Invoke standalone:
    ts2-sft-data --config configs/sft_data_fixture.toml
    (or: python -m tinystories_v2.sft_data --config ...)

Reads the tokenizer artifact and the `sft` split; writes, per the stage
convention:

Artifacts in <out_dir>:
    examples.jsonl   one training example per line: prompt_hash, input_ids,
                     loss_mask, n_prompt_tokens
                     (schema: docs/schemas/sft-example-v1.md)
    manifest.json    stage, version, count, config

Deterministic: examples are emitted in split order with no randomness, so two
runs on the same inputs are byte-identical.
"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from tinystories_v2 import __version__
from tinystories_v2.config import load_config
from tinystories_v2.slot_prompt import SLOT_FIELDS, encode_example
from tinystories_v2.slots import Scaffold

SCHEMA_VERSION = 1


def _read_split(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_example_record(tokenizer: Tokenizer, row: dict) -> dict:
    scaffold = Scaffold(**{field: row[field] for field in SLOT_FIELDS})
    example = encode_example(tokenizer, scaffold, row["fable"])
    return {"prompt_hash": row["prompt_hash"], **example.to_dict()}


def run(config: dict) -> None:
    tokenizer = Tokenizer.from_file(config["tokenizer"])
    split_path = Path(config["sft_split"])
    max_examples = config.get("max_examples", 0)

    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with (out_dir / "examples.jsonl").open("w", encoding="utf-8") as out:
        for row in _read_split(split_path):
            record = build_example_record(tokenizer, row)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if max_examples and count >= max_examples:
                break

    manifest = {
        "stage": "sft_data",
        "package_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "count": count,
        "config": config,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the console entrypoint**

Edit `pyproject.toml` — under `[project.scripts]`, add the third line:

```toml
[project.scripts]
ts2-data-prep = "tinystories_v2.data:main"
ts2-tokenizer = "tinystories_v2.tokenizer:main"
ts2-sft-data = "tinystories_v2.sft_data:main"
```

Then reinstall so the entrypoint and new module resolve for the subprocess test:

Run: `.venv/bin/pip install -e . -q`
Expected: completes with no error.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_sft_data_stage.py -q`
Expected: PASS (7 passed).

- [ ] **Step 6: Write the schema doc**

Create `docs/schemas/sft-example-v1.md`:

```markdown
# SFT-Example Schema v1

Issue 12 pins the record the SFT dataset builder (`tinystories_v2.sft_data`)
writes and the SFT trainer (issue 03) reads. Each JSONL line in
`examples.jsonl` is one masked-loss training example.

## Record

```json
{
  "prompt_hash": "71df0b5f…",
  "input_ids": [4, 812, 7, 233, 5, 91, 6, 44, 2, 118, 3, 77, 8, 501, 320, 9],
  "loss_mask": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
  "n_prompt_tokens": 13
}
```

- `prompt_hash` — the data-prep identity of the source Fable (join key).
- `input_ids` — the tokenized Slot Prompt example, in the fixed order
  `<|character|>…<|moral|><|fable|>{body}<|end|>`. Special tokens each encode
  to a single reserved ID.
- `loss_mask` — same length as `input_ids`; `0` over the conditioning prefix
  (through and including `<|fable|>`), `1` over the fable body and `<|end|>`.
- `n_prompt_tokens` — count of leading masked tokens; equals the index of
  `<|fable|>` plus one, and the number of leading zeros in `loss_mask`.

## Consumer contract

`loss_mask` and `input_ids` are always equal length. Train next-token
prediction only where `loss_mask` is `1` (the model learns to write the Fable,
not to parrot the Slot Prompt). Recover the Scaffold and fable body with
`tinystories_v2.slot_prompt.parse_example(tokenizer, input_ids)`. The builder
is deterministic: two runs over the same `sft_split` and tokenizer produce a
byte-identical `examples.jsonl`.
```

- [ ] **Step 7: Write the config files**

Create `configs/sft_data_fixture.toml`:

```toml
# Toy SFT dataset build against fixture artifacts — local sanity runs and docs.
# Assumes these upstream stages have run:
#   ts2-data-prep --config configs/data_prep_fixture.toml
#   ts2-tokenizer --config configs/tokenizer_fixture.toml
out_dir = "artifacts/sft_data_fixture"
tokenizer = "artifacts/tokenizer_fixture/tokenizer.json"
sft_split = "artifacts/data_prep_fixture/splits/sft.jsonl"
max_examples = 0
```

Create `configs/sft_data_full.toml`:

```toml
# Real SFT dataset build (~50k examples). Assumes these upstream stages have run:
#   ts2-data-prep --config configs/data_prep_full.toml
#   ts2-tokenizer --config configs/tokenizer_full.toml
out_dir = "artifacts/sft_data_full"
tokenizer = "artifacts/tokenizer_full/tokenizer.json"
sft_split = "artifacts/data_prep_full/splits/sft.jsonl"
max_examples = 0
```

- [ ] **Step 8: Run the whole suite to verify nothing regressed**

Run: `.venv/bin/pytest -q`
Expected: PASS — all prior tests plus the 12 `slot_prompt` and 7 `sft_data` tests, green on CPU with no network.

- [ ] **Step 9: Commit**

```bash
git add src/tinystories_v2/sft_data.py tests/test_sft_data_stage.py \
        docs/schemas/sft-example-v1.md configs/sft_data_fixture.toml \
        configs/sft_data_full.toml pyproject.toml
git commit -m "feat: add SFT dataset builder stage and v1 example schema"
```

---

## Acceptance Criteria Coverage

- *Rendered token sequence matches the schema exactly; specials encode as single IDs; order fixed; loss-mask boundary at `<|fable|>`* → Task 2 (`test_special_tokens_encode_to_single_ids_in_order`, `test_loss_mask_boundary_is_exactly_at_fable_token`).
- *Renderer→parser round-trip recovers slots and fable body on fixture data* → Task 3 (`test_render_encode_parse_round_trip`, `test_round_trip_on_fixture_records`).
- *Dataset builder produces a schema-documented artifact from the fixture's sft split via the stage convention, deterministically across two runs* → Task 4 (`docs/schemas/sft-example-v1.md`, `test_artifact_contract`, `test_two_runs_are_byte_identical`, `test_reads_only_the_configured_sft_split`, `test_cli_entrypoint_runs_standalone`).
- *Malformed input handling defined and tested (missing slot, unexpected token order)* → Task 1 (`test_render_empty_slot_raises`, `test_render_empty_fable_raises`) and Task 3 (`test_parse_missing_slot_raises`, `test_parse_wrong_order_raises`).
