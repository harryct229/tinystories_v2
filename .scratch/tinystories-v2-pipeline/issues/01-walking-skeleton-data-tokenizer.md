# 01 — Walking skeleton: package scaffold, test fixture, data-prep + tokenizer stages

Status: complete

## Parent

`.scratch/tinystories-v2-pipeline/PRD.md`

## What to build

The first tracer bullet: an installable Python package with the stage-entrypoint
convention established, plus the two cheapest real stages working end-to-end —
data-prep and tokenizer.

Before any code: make the repo's initial commit from the existing docs
(CONTEXT.md, docs/, PRD, paper). Then scaffold the package with the
config→artifacts stage convention from the PRD: every stage is an
independently invocable entrypoint reading a declarative config and writing
versioned artifacts to a directory; stages communicate only through
artifacts.

**Data-prep stage**: downloads TF1-EN-3M (HF token from environment/.env,
never committed or printed), extracts the six Scaffold slots from the
dataset's verbose prompt field (write the extraction against real records —
the paper's template description may not match exactly), and produces the
four disjoint-by-fable splits from the design doc (pretrain / sft / pref /
eval) with deterministic membership recorded alongside the artifacts. Also
produces the committed ~100-fable test fixture the whole suite runs on.

**Tokenizer stage**: trains the custom byte-level BPE (vocab 8192) on a
corpus sample with all Slot Prompt special tokens reserved at creation time,
per ADR-0003.

## Acceptance criteria

- [x] Repo has an initial commit of existing docs, then commits for the scaffold; `.env` remains gitignored and no secret value appears in code, config, or test output
- [x] Package installs with `pip install -e .` and the test suite runs green on laptop CPU with no GPU or network access
- [x] Data-prep stage run on the fixture produces the four splits; a test asserts split disjointness by fable and deterministic membership across two runs
- [x] Slot extraction is validated against at least a handful of real TF1-EN-3M records (recorded as a small committed sample), and a test covers extraction from the verbose prompt format
- [x] Tokenizer stage produces an artifact that round-trips fixture text losslessly; a test asserts vocab size and that every Slot Prompt special token encodes to a single ID
- [x] Both stages follow the config→artifacts convention (invocable standalone, artifacts in a target directory, no in-memory coupling)

## Blocked by

None — can start immediately.
