# Issue 01 — Walking Skeleton (data-prep + tokenizer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An installable `tinystories_v2` package with the config→artifacts stage convention established and the two cheapest stages — data-prep and tokenizer — working end-to-end against a committed ~120-record real-data fixture.

**Architecture:** Each stage is an independently invocable entrypoint (`ts2-data-prep`, `ts2-tokenizer`, also runnable as `python -m tinystories_v2.<module>`) that reads one declarative TOML config and writes artifacts + a manifest into the config's `out_dir`. Stages communicate only through artifacts. Slot extraction is a pure function in its own module; split membership is a pure hash function of `(seed, prompt_hash)` so splits are deterministic and disjoint by construction.

**Tech Stack:** Python ≥3.11 (stdlib `tomllib` for configs), `tokenizers` (byte-level BPE), `datasets`/`huggingface_hub` (data download only — never used in tests), `pytest`, `uv` for the venv.

## Global Constraints

Copied from the spec (`.scratch/tinystories-v2-pipeline/issues/01-walking-skeleton-data-tokenizer.md`, PRD, `docs/DESIGN.md`, ADR-0003). Every task's requirements implicitly include these.

- Real tokenizer vocab is **8192**; special tokens reserved at creation time, exactly these, in this order: `<|character|> <|trait|> <|setting|> <|conflict|> <|resolution|> <|moral|> <|fable|> <|end|>` (ADR-0003, DESIGN.md).
- Four splits, disjoint by fable, deterministic membership recorded with the artifacts: `pretrain / sft / pref / eval` (DESIGN.md; real-run targets ~1.1M / ~50k / ~4k / ~6k of the 2.8M-row train split).
- Dataset: `klusai/ds-tf1-en-3m` (public, MIT). Relevant columns: `prompt` (verbose template), `prompt_hash`, `fable`. There are **no slot columns** — slots are regex-extracted from `prompt`.
- The real prompt template **differs from the paper**: trait is folded into `Main Character: a <trait> <character>` (no separate Trait line), and the setting reads `a canyon where our story unfolds` (boilerplate suffix must be stripped). Verified against live records on 2026-07-11.
- Test suite must run green on laptop CPU with **no GPU and no network** — all tests read only the committed fixture.
- Secrets (HF token) come from environment/`.env`; `.env` stays gitignored; no secret value may appear in code, config, git history, or test output.
- Stage convention: declarative TOML config in → versioned artifacts in `out_dir` out; no in-memory coupling between stages.
- Use the CONTEXT.md vocabulary in code comments/docstrings: Fable, Scaffold, Slot Prompt, Pretraining, SFT, RLAIF (never "RLHF"), Judge, Reward Model.
- Dependencies for this issue: `datasets`, `tokenizers`, `huggingface_hub`, `pytest` (dev). **No torch yet** — that arrives with issue 02.
- Dev machine: Python 3.14.6 via Homebrew, `uv` available. If any dependency lacks a cp314 wheel, create the venv with `uv venv --python 3.12 .venv` instead and re-run the install — `requires-python` floor stays `>=3.11`.
- Every commit message ends with the trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

## File Structure

```
pyproject.toml                      # package metadata, deps, console scripts
.gitignore                          # extended: venv, artifacts, caches
configs/
    data_prep_fixture.toml          # toy run against the committed fixture
    data_prep_full.toml             # real run (hub source, design-doc fractions)
    tokenizer_fixture.toml          # toy vocab 512 against the fixture
    tokenizer_full.toml             # vocab 8192 against the pretrain split artifact
scripts/
    make_fixture.py                 # one-time, network: builds the committed fixture
src/tinystories_v2/
    __init__.py                     # __version__
    config.py                       # load_config (TOML), load_env (.env secrets)
    slots.py                        # Scaffold dataclass, extract_slots, SLOT_SPECIAL_TOKENS
    data.py                         # data-prep stage: assign_split, run, main
    tokenizer.py                    # tokenizer stage: train_tokenizer, run, main
tests/
    conftest.py                     # fixture_path / fixture_records session fixtures
    fixtures/tf1_sample.jsonl       # committed ~120 real TF1-EN-3M records
    test_config.py
    test_slots.py
    test_data_stage.py
    test_tokenizer_stage.py
```

Repo state at start: one commit (`dffb87d Init`) containing CONTEXT.md, docs/, .scratch/ (PRD + issues), .gitignore. Untracked: `2504.20605v2.md`, `2504.20605v2.pdf` (the dataset paper, 69KB + 403KB).

---

### Task 1: Commit the dataset paper (finish the docs-only initial commit)

The issue requires the repo history to start with the existing docs before any code. The `Init` commit already covers CONTEXT.md, docs/, and the PRD; only the paper files are still untracked.

**Files:**
- Commit (no edits): `2504.20605v2.md`, `2504.20605v2.pdf`

**Interfaces:**
- Consumes: nothing
- Produces: clean `git status` so later `git add` commands can't sweep in strays

- [ ] **Step 1: Verify only the paper files are untracked**

Run: `git -C /Users/thanh/code/tinystories_v2 status --short`
Expected output exactly:
```
?? 2504.20605v2.md
?? 2504.20605v2.pdf
```
If anything else appears, stop and resolve it first — do not blanket-`git add .` later.

- [ ] **Step 2: Commit**

```bash
cd /Users/thanh/code/tinystories_v2
git add 2504.20605v2.md 2504.20605v2.pdf
git commit -m "docs: add TF1-EN-3M dataset paper (arXiv 2504.20605)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Package scaffold + config loader

**Files:**
- Modify: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/tinystories_v2/__init__.py`
- Create: `src/tinystories_v2/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `load_config(path: str | Path) -> dict` — parses a TOML file into a plain dict (later tasks feed this dict to `run()`).
  - `load_env(path: str | Path = ".env") -> None` — loads `KEY=VALUE` lines into `os.environ` without overriding existing values, printing nothing.
  - Console-script declarations `ts2-data-prep` / `ts2-tokenizer` (their target modules land in Tasks 5–6; invoking them before that raises ImportError, which is fine — nothing invokes them yet).
  - `tinystories_v2.__version__ = "0.1.0"`.

- [ ] **Step 1: Extend .gitignore**

Replace the whole file (it currently contains only `.env`) with:

```gitignore
.env
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
artifacts/
dist/
```

`artifacts/` is where stage runs write output by default — never committed.

- [ ] **Step 2: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "tinystories-v2"
version = "0.1.0"
description = "Fable LM trained from scratch with RLAIF (course project; see docs/DESIGN.md)"
requires-python = ">=3.11"
dependencies = [
    "datasets>=2.19",
    "tokenizers>=0.19",
    "huggingface_hub>=0.23",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[project.scripts]
ts2-data-prep = "tinystories_v2.data:main"
ts2-tokenizer = "tinystories_v2.tokenizer:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create the package skeleton**

`src/tinystories_v2/__init__.py`:

```python
"""tinystories_v2: Fable LM training pipeline (see docs/DESIGN.md)."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Create the venv and install editable**

```bash
cd /Users/thanh/code/tinystories_v2
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -c "import tinystories_v2; print(tinystories_v2.__version__)"
```

Expected: install resolves and finishes; final command prints `0.1.0`. (Needs network once for dependency download. If `tokenizers`/`pyarrow` fail to resolve on 3.14, recreate with `uv venv --python 3.12 .venv` and re-run the install.)

- [ ] **Step 5: Write the failing tests**

`tests/test_config.py`:

```python
import os

from tinystories_v2.config import load_config, load_env


def test_load_config_parses_toml(tmp_path):
    cfg_file = tmp_path / "stage.toml"
    cfg_file.write_text('out_dir = "artifacts/x"\n\n[source]\nkind = "jsonl"\n', encoding="utf-8")
    assert load_config(cfg_file) == {"out_dir": "artifacts/x", "source": {"kind": "jsonl"}}


def test_load_env_sets_without_override_or_output(tmp_path, capsys, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("NEW_KEY=fresh\nEXISTING_KEY=changed\n# comment\n", encoding="utf-8")
    monkeypatch.setenv("EXISTING_KEY", "original")
    monkeypatch.delenv("NEW_KEY", raising=False)
    load_env(env_file)
    assert os.environ["NEW_KEY"] == "fresh"
    assert os.environ["EXISTING_KEY"] == "original"
    assert capsys.readouterr() == ("", "")  # secret values must never be printed


def test_load_env_missing_file_is_noop(tmp_path):
    load_env(tmp_path / "absent.env")  # must not raise
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL — collection error `ModuleNotFoundError: No module named 'tinystories_v2.config'` (the module doesn't exist yet).

- [ ] **Step 7: Implement config.py**

`src/tinystories_v2/config.py`:

```python
"""Declarative stage configs (TOML) and .env secret loading.

Every stage entrypoint reads one TOML file and writes artifacts to the
config's out_dir; stages share nothing in memory (PRD stage convention).
"""

import os
import tomllib
from pathlib import Path


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_env(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing values.

    Values are secrets (HF token) — never log or print them.
    """
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 9: Commit**

```bash
cd /Users/thanh/code/tinystories_v2
git add .gitignore pyproject.toml src/tinystories_v2/ tests/test_config.py
git commit -m "feat: package scaffold with TOML config loader and .env handling

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Fixture builder script + committed real-record fixture

One-time network step: download the first 120 real TF1-EN-3M records and commit them. This fixture is both the offline corpus for the whole suite AND the "small committed sample of real records" the acceptance criteria require for slot-extraction validation. 120 records ≈ 300KB — fine to commit (dataset is MIT).

**Files:**
- Create: `scripts/make_fixture.py`
- Create (generated, then committed): `tests/fixtures/tf1_sample.jsonl`

**Interfaces:**
- Consumes: `load_env` from Task 2
- Produces: `tests/fixtures/tf1_sample.jsonl` — one JSON object per line with exactly the keys `prompt_hash`, `prompt`, `fable`. Later tasks locate it via the `fixture_path` pytest fixture (Task 4).

- [ ] **Step 1: Write the script**

`scripts/make_fixture.py`:

```python
"""One-time fixture builder: downloads the first 120 real TF1-EN-3M records.

Writes tests/fixtures/tf1_sample.jsonl with only the fields the pipeline uses
(prompt_hash, prompt, fable). Needs network; the dataset is public so the
HF token in .env is optional but loaded for parity with real stage runs.
The committed output is what the offline test suite runs on.

Run from the repo root:  .venv/bin/python scripts/make_fixture.py
"""

import json
from pathlib import Path

from datasets import load_dataset

from tinystories_v2.config import load_env

N_RECORDS = 120
OUT = Path("tests/fixtures/tf1_sample.jsonl")


def main() -> None:
    load_env()
    rows = load_dataset("klusai/ds-tf1-en-3m", split="train", streaming=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for i, record in enumerate(rows):
            if i >= N_RECORDS:
                break
            row = {k: record[k] for k in ("prompt_hash", "prompt", "fable")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {N_RECORDS} records to {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (network required, one-time)**

Run: `cd /Users/thanh/code/tinystories_v2 && .venv/bin/python scripts/make_fixture.py`
Expected: `wrote 120 records to tests/fixtures/tf1_sample.jsonl` (takes ~10–60s).
If the sandboxed shell blocks network, re-run with sandbox disabled / request approval — this is the one intentional network action in the whole issue.

- [ ] **Step 3: Verify the fixture and eyeball the real template**

```bash
.venv/bin/python - <<'EOF'
import json
lines = open("tests/fixtures/tf1_sample.jsonl", encoding="utf-8").read().splitlines()
assert len(lines) == 120, f"expected 120 lines, got {len(lines)}"
for line in lines:
    rec = json.loads(line)
    assert set(rec) == {"prompt_hash", "prompt", "fable"}, set(rec)
    assert rec["prompt"].strip() and rec["fable"].strip()
print("fixture OK — first record prompt:")
print(repr(json.loads(lines[0])["prompt"][:400]))
EOF
```

Expected: `fixture OK` followed by the raw prompt of record 0. **Read it** — it should contain the labels `Main Character:`, `Setting:`, `Challenge:`, `Outcome:`, `Teaching:` followed by `The fable should`. Note the exact whitespace/dash characters; Task 4's regex tolerates newline-or-space separators and `-`/`–` dashes, but if the raw text uses some other structure entirely, Task 4's regex must be adapted to what you see here.

- [ ] **Step 4: Commit**

```bash
git add scripts/make_fixture.py tests/fixtures/tf1_sample.jsonl
git commit -m "feat: add committed 120-record TF1-EN-3M test fixture and builder script

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Scaffold slot extraction (`slots.py`)

**Files:**
- Create: `src/tinystories_v2/slots.py`
- Create: `tests/conftest.py`
- Test: `tests/test_slots.py`

**Interfaces:**
- Consumes: `tests/fixtures/tf1_sample.jsonl` from Task 3
- Produces:
  - `Scaffold` — frozen dataclass with str fields `character, trait, setting, conflict, resolution, moral` (in that order).
  - `extract_slots(prompt: str) -> Scaffold` — raises `SlotExtractionError(ValueError)` on non-matching prompts.
  - `SLOT_SPECIAL_TOKENS: tuple[str, ...]` — the 8 reserved tokens in design-doc order (Task 6 imports this).
  - pytest session fixtures `fixture_path: Path` and `fixture_records: list[dict]` (Tasks 5–6 tests use both).

- [ ] **Step 1: Write conftest.py**

`tests/conftest.py`:

```python
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "tf1_sample.jsonl"


@pytest.fixture(scope="session")
def fixture_records(fixture_path) -> list[dict]:
    with fixture_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
```

- [ ] **Step 2: Write the failing tests**

`tests/test_slots.py`:

```python
import pytest

from tinystories_v2.slots import Scaffold, SlotExtractionError, extract_slots

# Mirrors the template observed in real TF1-EN-3M records (paper section 3,
# corrected against reality: trait is folded into the Main Character line and
# the setting carries a "where our story unfolds" suffix).
SYNTHETIC_PROMPT = (
    "Create a fable based on the following elements. Weave them naturally into a story: \n"
    "- Main Character: a greedy fox \n"
    "- Setting: a dense forest where our story unfolds \n"
    "- Challenge: loses their food to someone's trick \n"
    "- Outcome: the trickster is exposed \n"
    "- Teaching: honesty is the best policy \n"
    "The fable should: \n"
    "- Be appropriate for age group B (4-7 years)"
)


def test_extract_from_verbose_template():
    assert extract_slots(SYNTHETIC_PROMPT) == Scaffold(
        character="fox",
        trait="greedy",
        setting="a dense forest",
        conflict="loses their food to someone's trick",
        resolution="the trickster is exposed",
        moral="honesty is the best policy",
    )


def test_extract_known_real_record(fixture_records):
    # Expected values read from the live dataset (train row 0, previewed
    # 2026-07-11). If this prompt_hash is missing from the fixture (dataset
    # re-upload), open the fixture's first record, read its prompt, and
    # replace hash + expected values with that record's actual content.
    by_hash = {r["prompt_hash"]: r for r in fixture_records}
    record = by_hash["71df0b5fc187f6e393954bc32cccac0cf9f856e31df8276ea6557c9b1710294e"]
    assert extract_slots(record["prompt"]) == Scaffold(
        character="firefly",
        trait="persuasive",
        setting="a canyon",
        conflict="betrayal by a friend",
        resolution="a lesson is documented for future generations",
        moral="timely help earns lasting loyalty",
    )


def test_extract_every_fixture_record(fixture_records):
    for record in fixture_records:
        scaffold = extract_slots(record["prompt"])
        for field in ("character", "trait", "setting", "conflict", "resolution", "moral"):
            assert getattr(scaffold, field), f"{field} empty for {record['prompt_hash']}"
        assert "where our story unfolds" not in scaffold.setting
        assert "fable should" not in scaffold.moral


def test_non_template_prompt_raises():
    with pytest.raises(SlotExtractionError):
        extract_slots("Write me a story about a dog.")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_slots.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tinystories_v2.slots'`.

- [ ] **Step 4: Implement slots.py**

`src/tinystories_v2/slots.py`:

```python
"""Scaffold slot extraction from TF1-EN-3M's verbose prompt field.

The dataset has no slot columns: the six Scaffold slots are embedded in a
fixed natural-language template. Written against real records, which differ
from the paper: the trait is folded into "Main Character: a <trait>
<character>" and the setting ends with "where our story unfolds" boilerplate.
"""

import re
from dataclasses import dataclass

# Reserved at tokenizer creation (ADR-0003). Order is fixed — downstream
# stages assume it; never reorder or append in the middle.
SLOT_SPECIAL_TOKENS = (
    "<|character|>",
    "<|trait|>",
    "<|setting|>",
    "<|conflict|>",
    "<|resolution|>",
    "<|moral|>",
    "<|fable|>",
    "<|end|>",
)


class SlotExtractionError(ValueError):
    pass


@dataclass(frozen=True)
class Scaffold:
    character: str
    trait: str
    setting: str
    conflict: str
    resolution: str
    moral: str


# Anchored on the template's field labels; tolerant of newline-vs-space
# separators and -/– list dashes. Trait is the first word after the article.
_PROMPT_RE = re.compile(
    r"Main Character:\s*(?:[Aa]n?\s+)?(?P<trait>\S+)\s+(?P<character>.+?)"
    r"\s*[-–]\s*Setting:\s*(?P<setting>.+?)"
    r"\s*[-–]\s*Challenge:\s*(?P<conflict>.+?)"
    r"\s*[-–]\s*Outcome:\s*(?P<resolution>.+?)"
    r"\s*[-–]\s*Teaching:\s*(?P<moral>.+?)"
    r"\s*The fable should",
    re.DOTALL,
)

_SETTING_BOILERPLATE = re.compile(r"\s+where our story unfolds$")


def extract_slots(prompt: str) -> Scaffold:
    match = _PROMPT_RE.search(prompt)
    if match is None:
        raise SlotExtractionError(f"prompt does not match template: {prompt[:120]!r}")
    return Scaffold(
        character=match["character"].strip(),
        trait=match["trait"].strip(),
        setting=_SETTING_BOILERPLATE.sub("", match["setting"].strip()),
        conflict=match["conflict"].strip(),
        resolution=match["resolution"].strip(),
        moral=match["moral"].strip(),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_slots.py -v`
Expected: 4 passed.

If `test_extract_every_fixture_record` fails on some record: print that record's raw prompt, adjust `_PROMPT_RE` to cover the variant (keep it label-anchored), and re-run — that is exactly the "validate against real records" acceptance criterion doing its job. Do not weaken the test.

- [ ] **Step 6: Commit**

```bash
git add src/tinystories_v2/slots.py tests/conftest.py tests/test_slots.py
git commit -m "feat: extract Scaffold slots from verbose TF1-EN-3M prompts

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Data-prep stage (`data.py`)

**Files:**
- Create: `src/tinystories_v2/data.py`
- Create: `configs/data_prep_fixture.toml`
- Create: `configs/data_prep_full.toml`
- Test: `tests/test_data_stage.py`

**Interfaces:**
- Consumes: `extract_slots`, `SlotExtractionError` (Task 4); `load_config`, `load_env` (Task 2); fixture fixtures (Task 4).
- Produces:
  - `SPLIT_NAMES = ("pretrain", "sft", "pref", "eval")`
  - `assign_split(prompt_hash: str, fractions: dict[str, float], seed: str) -> str | None` — pure, order-independent; `None` = record unused.
  - `run(config: dict) -> None` and `main(argv: list[str] | None = None) -> None` (the `ts2-data-prep` console script target).
  - Artifact layout under `out_dir`: `splits/{pretrain,sft,pref,eval}.jsonl` (rows with keys `prompt_hash, character, trait, setting, conflict, resolution, moral, fable` — the verbose prompt is deliberately dropped; SFT renders Slot Prompts from the slots), `membership.json` (`{split: [prompt_hash, ...]}`), `manifest.json` (`{stage, package_version, counts, skipped_extraction_failures, config}`). Issue 02+ consume `splits/*.jsonl`.

- [ ] **Step 1: Write the failing tests**

`tests/test_data_stage.py`:

```python
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import pytest

from tinystories_v2.data import SPLIT_NAMES, run
from tinystories_v2.slots import SlotExtractionError


def make_config(out_dir: Path, source_path: Path) -> dict:
    return {
        "out_dir": str(out_dir),
        "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(source_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.5, "sft": 0.2, "pref": 0.1, "eval": 0.2},
    }


def read_membership(out_dir: Path) -> dict:
    return json.loads((out_dir / "membership.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory, fixture_path):
    out = tmp_path_factory.mktemp("data_prep")
    run(make_config(out, fixture_path))
    return out


def test_produces_all_four_split_artifacts(artifact_dir):
    for name in SPLIT_NAMES:
        assert (artifact_dir / "splits" / f"{name}.jsonl").exists()
    assert (artifact_dir / "membership.json").exists()
    counts = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))["counts"]
    for name in SPLIT_NAMES:
        assert counts[name] > 0, f"{name} split is empty"


def test_splits_disjoint_by_fable(artifact_dir):
    membership = read_membership(artifact_dir)
    for a, b in combinations(SPLIT_NAMES, 2):
        assert not set(membership[a]) & set(membership[b]), f"{a} and {b} overlap"


def test_membership_matches_split_files(artifact_dir):
    membership = read_membership(artifact_dir)
    for name in SPLIT_NAMES:
        lines = (artifact_dir / "splits" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["prompt_hash"] for line in lines] == membership[name]


def test_split_records_carry_scaffold_and_fable(artifact_dir):
    first = (artifact_dir / "splits" / "pretrain.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert set(json.loads(first)) == {
        "prompt_hash", "character", "trait", "setting",
        "conflict", "resolution", "moral", "fable",
    }


def test_two_runs_are_byte_identical(tmp_path, fixture_path):
    for name in ("run1", "run2"):
        run(make_config(tmp_path / name, fixture_path))
    for rel in ["membership.json"] + [f"splits/{n}.jsonl" for n in SPLIT_NAMES]:
        assert (tmp_path / "run1" / rel).read_bytes() == (tmp_path / "run2" / rel).read_bytes(), rel


def test_extraction_failure_budget(tmp_path, fixture_records):
    corrupt = tmp_path / "corrupt.jsonl"
    rows = [
        dict(fixture_records[0]),
        {"prompt_hash": "0" * 64, "prompt": "not the template", "fable": "x"},
    ]
    corrupt.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    all_pretrain = {"seed": "fixture-v1", "pretrain": 1.0, "sft": 0.0, "pref": 0.0, "eval": 0.0}

    strict = make_config(tmp_path / "strict", corrupt)
    strict["splits"] = dict(all_pretrain)
    with pytest.raises(SlotExtractionError):
        run(strict)

    lenient = make_config(tmp_path / "lenient", corrupt)
    lenient["splits"] = dict(all_pretrain)
    lenient["max_extraction_failures"] = 1
    run(lenient)
    manifest = json.loads((tmp_path / "lenient" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["skipped_extraction_failures"] == 1
    assert manifest["counts"]["pretrain"] == 1


def test_cli_entrypoint_runs_standalone(tmp_path, fixture_path):
    out = tmp_path / "cli_out"
    config_file = tmp_path / "cfg.toml"
    config_file.write_text(
        f'out_dir = "{out}"\nmax_extraction_failures = 0\n\n'
        f'[source]\nkind = "jsonl"\npath = "{fixture_path}"\n\n'
        '[splits]\nseed = "fixture-v1"\npretrain = 0.5\nsft = 0.2\npref = 0.1\neval = 0.2\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.data", "--config", str(config_file)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "membership.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_data_stage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tinystories_v2.data'`.

- [ ] **Step 3: Implement data.py**

`src/tinystories_v2/data.py`:

```python
"""Data-prep stage: TF1-EN-3M -> slot-extracted, disjoint-by-fable splits.

Invoke standalone:
    ts2-data-prep --config configs/data_prep_fixture.toml
    (or: python -m tinystories_v2.data --config ...)

Artifacts in <out_dir>:
    splits/{pretrain,sft,pref,eval}.jsonl   one row per Fable:
        prompt_hash + six Scaffold slots + fable text
    membership.json                         {split: [prompt_hash, ...]}
    manifest.json                           stage, version, counts, config

Split membership is a pure function of (seed, prompt_hash), so runs are
deterministic and Fables sharing a Scaffold never straddle splits.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from tinystories_v2 import __version__
from tinystories_v2.config import load_config, load_env
from tinystories_v2.slots import SlotExtractionError, extract_slots

SPLIT_NAMES = ("pretrain", "sft", "pref", "eval")


def assign_split(prompt_hash: str, fractions: dict[str, float], seed: str) -> str | None:
    digest = hashlib.sha256(f"{seed}:{prompt_hash}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    upper = 0.0
    for name in SPLIT_NAMES:
        upper += fractions[name]
        if position < upper:
            return name
    return None  # remainder of the corpus stays unused


def iter_source(source: dict):
    kind = source["kind"]
    if kind == "jsonl":
        with open(source["path"], encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif kind == "hub":
        from datasets import load_dataset  # network path; tests only use jsonl

        load_env()  # HF token, if any, from environment/.env — never printed
        rows = load_dataset(source["dataset"], split=source["split"], streaming=True)
        limit = source.get("limit", 0)
        for i, row in enumerate(rows):
            if limit and i >= limit:
                break
            yield {k: row[k] for k in ("prompt_hash", "prompt", "fable")}
    else:
        raise ValueError(f"unknown source kind: {kind!r}")


def run(config: dict) -> None:
    fractions = {name: float(config["splits"][name]) for name in SPLIT_NAMES}
    if sum(fractions.values()) > 1.0 + 1e-9:
        raise ValueError(f"split fractions sum to more than 1: {fractions}")
    seed = config["splits"]["seed"]
    max_failures = config.get("max_extraction_failures", 0)

    out_dir = Path(config["out_dir"])
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    membership: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    failures = 0
    writers = {
        name: (splits_dir / f"{name}.jsonl").open("w", encoding="utf-8")
        for name in SPLIT_NAMES
    }
    try:
        for record in iter_source(config["source"]):
            split = assign_split(record["prompt_hash"], fractions, seed)
            if split is None:
                continue
            try:
                scaffold = extract_slots(record["prompt"])
            except SlotExtractionError:
                failures += 1
                if failures > max_failures:
                    raise
                continue
            row = {"prompt_hash": record["prompt_hash"], **asdict(scaffold), "fable": record["fable"]}
            writers[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            membership[split].append(record["prompt_hash"])
    finally:
        for writer in writers.values():
            writer.close()

    (out_dir / "membership.json").write_text(json.dumps(membership, indent=2), encoding="utf-8")
    manifest = {
        "stage": "data_prep",
        "package_version": __version__,
        "counts": {name: len(ids) for name, ids in membership.items()},
        "skipped_extraction_failures": failures,
        "config": config,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_data_stage.py -v`
Expected: 7 passed.

- [ ] **Step 5: Add the run configs**

`configs/data_prep_fixture.toml`:

```toml
# Toy run against the committed fixture — local sanity runs and docs.
out_dir = "artifacts/data_prep_fixture"
max_extraction_failures = 0

[source]
kind = "jsonl"
path = "tests/fixtures/tf1_sample.jsonl"

[splits]
seed = "tinystories-v2-splits-v1"
pretrain = 0.5
sft = 0.2
pref = 0.1
eval = 0.2
```

`configs/data_prep_full.toml`:

```toml
# Real run over the 2.8M-row train split -> ~1.1M pretrain / ~50k SFT /
# ~4k pref / ~6k eval Fables (docs/DESIGN.md). Fractions of 2.8M rows;
# the unassigned remainder stays unused. Needs network + HF token in .env.
out_dir = "artifacts/data_prep_full"
max_extraction_failures = 100

[source]
kind = "hub"
dataset = "klusai/ds-tf1-en-3m"
split = "train"

[splits]
seed = "tinystories-v2-splits-v1"
pretrain = 0.393
sft = 0.018
pref = 0.0015
eval = 0.0022
```

- [ ] **Step 6: Smoke-run the stage exactly as a teammate would**

```bash
cd /Users/thanh/code/tinystories_v2
.venv/bin/ts2-data-prep --config configs/data_prep_fixture.toml
cat artifacts/data_prep_fixture/manifest.json
```

Expected: exit 0; manifest shows `"stage": "data_prep"` and four non-zero counts summing to ≤120. (`artifacts/` is gitignored.)

- [ ] **Step 7: Commit**

```bash
git add src/tinystories_v2/data.py tests/test_data_stage.py configs/data_prep_fixture.toml configs/data_prep_full.toml
git commit -m "feat: data-prep stage producing disjoint deterministic Fable splits

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Tokenizer stage (`tokenizer.py`)

**Files:**
- Create: `src/tinystories_v2/tokenizer.py`
- Create: `configs/tokenizer_fixture.toml`
- Create: `configs/tokenizer_full.toml`
- Test: `tests/test_tokenizer_stage.py`

**Interfaces:**
- Consumes: `SLOT_SPECIAL_TOKENS` (Task 4); `load_config` (Task 2); fixture fixtures (Task 4).
- Produces:
  - `train_tokenizer(texts: Iterator[str], vocab_size: int) -> Tokenizer` (from the `tokenizers` library).
  - `run(config: dict) -> None` and `main(argv: list[str] | None = None) -> None` (the `ts2-tokenizer` console script target).
  - Artifact layout under `out_dir`: `tokenizer.json` (load with `tokenizers.Tokenizer.from_file`), `manifest.json` (`{stage, package_version, vocab_size, special_tokens, config}`). Issue 02+ consume `tokenizer.json`.
- Note: the special tokens are **not configurable** — they are baked in from `SLOT_SPECIAL_TOKENS` so the reserved set can never drift from ADR-0003 via a config edit. The test vocab is 512 because 8192 is unreachable from a 120-fable corpus; 8192 lives in the real config and is asserted the same way (manifest/`get_vocab_size()` equals the configured value).

- [ ] **Step 1: Write the failing tests**

`tests/test_tokenizer_stage.py`:

```python
import json
import subprocess
import sys

import pytest
from tokenizers import Tokenizer

from tinystories_v2.slots import SLOT_SPECIAL_TOKENS
from tinystories_v2.tokenizer import run

VOCAB_SIZE = 512  # 8192 needs the real corpus; the artifact contract is identical


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory, fixture_path):
    out = tmp_path_factory.mktemp("tokenizer")
    run({
        "out_dir": str(out),
        "corpus": [str(fixture_path)],
        "text_field": "fable",
        "vocab_size": VOCAB_SIZE,
    })
    return out


@pytest.fixture(scope="module")
def tokenizer(artifact_dir):
    return Tokenizer.from_file(str(artifact_dir / "tokenizer.json"))


def test_artifact_contract(artifact_dir):
    assert (artifact_dir / "tokenizer.json").exists()
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "tokenizer"
    assert manifest["vocab_size"] == VOCAB_SIZE
    assert manifest["special_tokens"] == list(SLOT_SPECIAL_TOKENS)


def test_vocab_size_matches_config(tokenizer):
    assert tokenizer.get_vocab_size() == VOCAB_SIZE


def test_slot_special_tokens_encode_to_single_ids(tokenizer):
    id_lists = [tokenizer.encode(token).ids for token in SLOT_SPECIAL_TOKENS]
    assert all(len(ids) == 1 for ids in id_lists), id_lists
    assert len({ids[0] for ids in id_lists}) == len(SLOT_SPECIAL_TOKENS)


def test_roundtrips_fixture_fables_losslessly(tokenizer, fixture_records):
    for record in fixture_records:
        text = record["fable"]
        assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_roundtrips_slot_prompt_text(tokenizer):
    text = (
        "<|character|>fox<|trait|>greedy<|setting|>a dense forest"
        "<|conflict|>loses their food<|resolution|>the trickster is exposed"
        "<|moral|>honesty is the best policy<|fable|>One day, a fox...<|end|>"
    )
    decoded = tokenizer.decode(tokenizer.encode(text).ids, skip_special_tokens=False)
    assert decoded == text


def test_cli_entrypoint_runs_standalone(tmp_path, fixture_path):
    out = tmp_path / "cli_out"
    config_file = tmp_path / "cfg.toml"
    config_file.write_text(
        f'out_dir = "{out}"\ncorpus = ["{fixture_path}"]\n'
        f'text_field = "fable"\nvocab_size = {VOCAB_SIZE}\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.tokenizer", "--config", str(config_file)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "tokenizer.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_tokenizer_stage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tinystories_v2.tokenizer'`.

- [ ] **Step 3: Implement tokenizer.py**

`src/tinystories_v2/tokenizer.py`:

```python
"""Tokenizer stage: byte-level BPE with Slot Prompt tokens reserved (ADR-0003).

Invoke standalone:
    ts2-tokenizer --config configs/tokenizer_fixture.toml
    (or: python -m tinystories_v2.tokenizer --config ...)

Artifacts in <out_dir>:
    tokenizer.json    load with tokenizers.Tokenizer.from_file
    manifest.json     stage, version, vocab size, special tokens, config

The special tokens come from slots.SLOT_SPECIAL_TOKENS, not from config:
the reserved set is an ADR-0003 invariant, not a tunable.
"""

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from tinystories_v2 import __version__
from tinystories_v2.config import load_config
from tinystories_v2.slots import SLOT_SPECIAL_TOKENS


def iter_corpus(paths: Iterable[str], text_field: str, max_docs: int = 0) -> Iterator[str]:
    seen = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)[text_field]
                seen += 1
                if max_docs and seen >= max_docs:
                    return


def train_tokenizer(texts: Iterator[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SLOT_SPECIAL_TOKENS),
        # Full byte alphabet up front -> any text round-trips losslessly,
        # even bytes absent from the training sample.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def run(config: dict) -> None:
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = iter_corpus(
        config["corpus"], config.get("text_field", "fable"), config.get("max_docs", 0)
    )
    tokenizer = train_tokenizer(texts, config["vocab_size"])
    tokenizer.save(str(out_dir / "tokenizer.json"))
    manifest = {
        "stage": "tokenizer",
        "package_version": __version__,
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": list(SLOT_SPECIAL_TOKENS),
        "config": config,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tokenizer_stage.py -v`
Expected: 6 passed (module-scoped training takes a few seconds).

If `test_roundtrips_slot_prompt_text` fails with stray spaces around special tokens, that is a decoder wiring problem (check `decoders.ByteLevel` is set and `tokenizers>=0.19`) — fix the implementation, do not weaken the test: SFT depends on exact Slot Prompt reconstruction.

- [ ] **Step 5: Add the run configs**

`configs/tokenizer_fixture.toml`:

```toml
# Toy vocab against the committed fixture (8192 is unreachable from 120 Fables).
out_dir = "artifacts/tokenizer_fixture"
corpus = ["tests/fixtures/tf1_sample.jsonl"]
text_field = "fable"
vocab_size = 512
```

`configs/tokenizer_full.toml`:

```toml
# Real tokenizer (ADR-0003): vocab 8192, trained on a 200k-Fable sample of the
# pretrain split produced by: ts2-data-prep --config configs/data_prep_full.toml
out_dir = "artifacts/tokenizer_full"
corpus = ["artifacts/data_prep_full/splits/pretrain.jsonl"]
text_field = "fable"
vocab_size = 8192
max_docs = 200000
```

- [ ] **Step 6: Smoke-run the stage exactly as a teammate would**

```bash
cd /Users/thanh/code/tinystories_v2
.venv/bin/ts2-tokenizer --config configs/tokenizer_fixture.toml
cat artifacts/tokenizer_fixture/manifest.json
```

Expected: exit 0; manifest shows `"stage": "tokenizer"`, `"vocab_size": 512`, and the 8 special tokens.

- [ ] **Step 7: Commit**

```bash
git add src/tinystories_v2/tokenizer.py tests/test_tokenizer_stage.py configs/tokenizer_fixture.toml configs/tokenizer_full.toml
git commit -m "feat: tokenizer stage training byte-level BPE with reserved Slot Prompt tokens

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Acceptance verification (no new code)

Walk the issue's acceptance criteria end-to-end. Fix anything that fails (with a `fix:` commit); otherwise this task produces no commit.

**Files:**
- None (verification only)

**Interfaces:**
- Consumes: everything above
- Produces: confirmation the issue is done

- [ ] **Step 1: Full suite in the dev venv**

Run: `cd /Users/thanh/code/tinystories_v2 && .venv/bin/pytest -v`
Expected: 20 passed (3 config + 4 slots + 7 data + 6 tokenizer), well under a minute.

- [ ] **Step 2: Prove `pip install -e .` works in a fresh venv (the literal acceptance criterion)**

```bash
python3 -m venv "$TMPDIR/ts2-accept"
"$TMPDIR/ts2-accept/bin/pip" install --quiet -e '/Users/thanh/code/tinystories_v2[dev]'
cd /Users/thanh/code/tinystories_v2 && "$TMPDIR/ts2-accept/bin/pytest" -q
rm -rf "$TMPDIR/ts2-accept"
```

Expected: install succeeds, `20 passed`. (Uses stock `pip` deliberately, not uv. Needs network for the install itself; the tests do not.)

- [ ] **Step 3: Secret hygiene checks**

```bash
cd /Users/thanh/code/tinystories_v2
git check-ignore .env                      # expected output: .env
git ls-files .env                          # expected output: (empty)
git grep -I -iE 'hf_[a-z0-9]{30,}' -- ':!*.pdf' && echo "TOKEN LEAKED" || echo "clean"
grep -rn "load_dataset" tests/ || echo "tests never touch the network"
```

Expected: `.env`, empty, `clean`, `tests never touch the network`.

- [ ] **Step 4: Tick off the issue's acceptance criteria**

Confirm each against evidence, and check the boxes in `.scratch/tinystories-v2-pipeline/issues/01-walking-skeleton-data-tokenizer.md` (commit that edit as `docs: check off issue 01 acceptance criteria`):

1. Initial docs commit exists (`git log --reverse --oneline | head -2` shows Init + paper commit before any code); `.env` gitignored, no secret anywhere — Step 3.
2. `pip install -e .` + green suite, CPU-only, no network — Step 2.
3. Four splits, disjointness + determinism tests — `test_splits_disjoint_by_fable`, `test_two_runs_are_byte_identical`.
4. Slot extraction validated against committed real records — `tests/fixtures/tf1_sample.jsonl` (120 real records) + `test_extract_known_real_record`, `test_extract_every_fixture_record`, `test_extract_from_verbose_template`.
5. Tokenizer round-trip / vocab size / single-ID special tokens — `test_roundtrips_fixture_fables_losslessly`, `test_vocab_size_matches_config`, `test_slot_special_tokens_encode_to_single_ids`.
6. Both stages config→artifacts, standalone invocable — `test_cli_entrypoint_runs_standalone` (both stages) + Task 5/6 smoke runs via the `ts2-*` console scripts.
