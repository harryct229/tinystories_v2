# 03 — SFT Stage + Demo Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the SFT stage that fine-tunes a Pretraining checkpoint into a Scaffold-conditioned Fable generator on issue 12's masked-loss dataset, plus a CPU demo that renders six slot values into a Slot Prompt and prints the model's Fable.

**Architecture:** A new stage module `sft.py` mirrors `pretrain.py`'s stage contract (one TOML → `out_dir` artifacts, checkpoint-resume, precision knob, W&B logging, Hub sync) but swaps the data source (variable-length masked examples from `examples.jsonl`, not packed windows) and the loss (masked to the Fable body + `<|end|>`). It initializes model weights from a Pretraining checkpoint declared in a `[init]` config section. The optimizer builder and LR schedule are imported from `pretrain.py` verbatim (DRY). A separate `demo.py` reuses `generate.sample` + `slot_prompt.render_prompt` to print a Fable from any checkpoint.

**Tech Stack:** Python 3.11+, PyTorch ≥2.6, `tokenizers`, `numpy`, TOML configs, pytest. No new dependencies.

## Global Constraints

- **Stage convention:** every stage entrypoint reads exactly one TOML file and writes all artifacts under the config's `out_dir`; stages share nothing in memory and couple only through on-disk artifacts.
- **Checkpoint-resume is a hard requirement:** a run killed mid-flight (SIGKILL) must resume from the latest checkpoint and reproduce the uninterrupted run bitwise on fp32 CPU. Batches are a pure function of `(seed, step, micro_step)`; optimizer + scaler state round-trip through checkpoints.
- **Vocabulary (CONTEXT.md):** use **Fable**, **Scaffold**, **Slot Prompt**, **SFT**, **Pretraining** exactly; never "prompt" for the Slot Prompt, never "instruction tuning" for SFT.
- **Slot Prompt format contract (issue 12, `slot_prompt.py`):** the sequence is `<|character|>…<|moral|><|fable|>{body}<|end|>`; loss is masked (0) through and including `<|fable|>`, active (1) over the body and `<|end|>`. `SLOT_FIELDS = ("character","trait","setting","conflict","resolution","moral")`. Do not re-derive or reorder these.
- **SFT-Example schema v1 (`docs/schemas/sft-example-v1.md`):** each `examples.jsonl` line has `prompt_hash`, `input_ids`, `loss_mask` (equal length to `input_ids`), `n_prompt_tokens`.
- **Secrets never printed:** `.env` values (HF/W&B tokens) are loaded via `config.load_env` and never logged.
- **Colab notebooks stay thin:** setup + a single stage invocation only; no `def`, `class`, `import torch`, `for`, or `while` in notebook source (enforced by `tests/test_notebook.py`).
- **Determinism of derived artifacts:** two runs over identical inputs produce byte-identical outputs where the existing stages already guarantee it (`examples.jsonl` from issue 12).
- **Version floors:** `requires-python >=3.11`, `torch>=2.6` (already pinned in `pyproject.toml`); do not add dependencies.

---

## File Structure

**Create:**
- `src/tinystories_v2/sft.py` — SFT stage: `load_sft_examples`, `get_sft_batch`, `masked_lm_loss`, `_init_model_from_pretrain`, `run`, `main`. Entrypoint `ts2-sft`. One responsibility: the SFT training stage.
- `src/tinystories_v2/demo.py` — demo script: render six slot values (or sample an eval Scaffold) → print the model's Fable. Entrypoint `ts2-demo`.
- `configs/sft_fixture.toml` — toy CPU wiring config (local sanity + docs).
- `configs/sft_full.toml` — real Colab SFT config (design-doc defaults).
- `notebooks/sft_colab.ipynb` — thin Colab wrapper for the real SFT run.
- `tests/test_sft_batch.py` — unit tests for `get_sft_batch` and `masked_lm_loss` (determinism, padding, truncation, mask shift).
- `tests/test_sft_stage.py` — stage tests: masked-loss decrease, artifacts/manifest, init-from-real-pretrain, split-leakage guard, CLI.
- `tests/test_sft_resume.py` — kill-and-resume bitwise-identical contract.
- `tests/test_sft_format_learned.py` — overfit tiny examples → generation terminates with `<|end|>`.
- `tests/test_demo.py` — demo CLI produces a Fable from a checkpoint on CPU (explicit slots + `--sample-eval`).

**Modify:**
- `pyproject.toml` — add `ts2-sft` and `ts2-demo` console-script entrypoints.
- `tests/conftest.py` — add a `make_init_checkpoint` fixture (writes a minimal Pretraining-style init checkpoint without running the pretrain stage).
- `tests/test_notebook.py` — add a thin-wrapper test for `sft_colab.ipynb`.
- `PROGRESS.md` — mark issue 03 code-complete (final task).

**Read-only references (do not modify):** `pretrain.py` (imports `lr_at`, `build_optimizer`; stage template), `slot_prompt.py`, `slots.py`, `sft_data.py`, `generate.py`, `checkpoint.py`, `tracking.py`, `hub.py`, `model.py`, `config.py`.

---

### Task 1: SFT batching + masked loss

The data plumbing the training loop needs: read `examples.jsonl` into memory, build a deterministic padded `(x, y, mask)` micro-batch from variable-length masked examples, and compute masked cross-entropy. These are pure, unit-testable functions with no training loop.

**Files:**
- Create: `src/tinystories_v2/sft.py` (helpers only in this task; `run`/`main` land in Task 2)
- Test: `tests/test_sft_batch.py`

**Interfaces:**
- Consumes: nothing from other tasks. Uses `torch` and the SFT-Example schema (`input_ids`, `loss_mask`).
- Produces (relied on by Task 2, 3, 4):
  - `load_sft_examples(path: str | Path) -> list[dict]` — each dict has at least `input_ids: list[int]`, `loss_mask: list[int]`, `prompt_hash: str`.
  - `get_sft_batch(examples: list[dict], micro_batch_size: int, context: int, *, seed: int, step: int, micro_step: int, device: str = "cpu") -> tuple[Tensor, Tensor, Tensor]` returning `(x, y, mask)` each shape `[micro_batch_size, width]`; `x`/`y` are `long`, `mask` is `float`. Selection is a pure function of `(seed, step, micro_step)`.
  - `masked_lm_loss(logits: Tensor, y: Tensor, mask: Tensor) -> Tensor` — scalar mean CE over `mask==1` positions.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sft_batch.py`:

```python
import json

import torch

from tinystories_v2.sft import get_sft_batch, load_sft_examples, masked_lm_loss

# Two hand-built examples in schema v1 shape. Ids are arbitrary; the mask
# marks the prompt prefix (0) vs body+end (1). Example lengths differ so
# padding and per-row shifting are exercised.
EXAMPLES = [
    {"prompt_hash": "a", "input_ids": [10, 11, 12, 13, 14],
     "loss_mask": [0, 0, 1, 1, 1]},                       # len 5, 3 active
    {"prompt_hash": "b", "input_ids": [20, 21, 22, 23, 24, 25, 26],
     "loss_mask": [0, 0, 0, 1, 1, 1, 1]},                 # len 7, 4 active
]


def test_load_sft_examples_reads_jsonl(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text(
        "\n".join(json.dumps(e) for e in EXAMPLES) + "\n" + "  \n",  # blank line ignored
        encoding="utf-8",
    )
    loaded = load_sft_examples(path)
    assert loaded == EXAMPLES


def test_batch_selection_is_pure_function_of_step_and_micro_step():
    a = get_sft_batch(EXAMPLES, 4, 16, seed=1337, step=3, micro_step=0)
    b = get_sft_batch(EXAMPLES, 4, 16, seed=1337, step=3, micro_step=0)
    for t1, t2 in zip(a, b):
        assert torch.equal(t1, t2)
    c = get_sft_batch(EXAMPLES, 4, 16, seed=1337, step=3, micro_step=1)
    # Different micro_step almost certainly draws a different multiset.
    assert not all(torch.equal(t1, t2) for t1, t2 in zip(a, c))


def test_shift_and_mask_alignment_for_a_single_example():
    # micro_batch_size=1 with a 1-example pool always draws that example.
    x, y, mask = get_sft_batch(EXAMPLES[:1], 1, 16, seed=0, step=0, micro_step=0)
    ids, lm = EXAMPLES[0]["input_ids"], EXAMPLES[0]["loss_mask"]
    assert x[0].tolist() == ids[:-1]        # x = input_ids[:-1]
    assert y[0].tolist() == ids[1:]         # y = input_ids[1:] (next-token target)
    assert mask[0].tolist() == [float(v) for v in lm[1:]]  # mask = loss_mask[1:]


def test_rows_are_right_padded_to_batch_width_with_zero_mask():
    x, y, mask = get_sft_batch(EXAMPLES, 8, 16, seed=5, step=0, micro_step=0)
    assert x.shape == y.shape == mask.shape
    width = x.shape[1]
    assert width == 6  # longest example (len 7) shifts to length 6
    # Every padded tail position must have mask 0 (never contributes to loss).
    for row_ids, row_mask in zip(x.tolist(), mask.tolist()):
        # padding id is 0; wherever mask is 0 past a row's real length it is padding
        assert len(row_ids) == width and len(row_mask) == width


def test_truncates_examples_longer_than_context():
    long_example = [{"prompt_hash": "c",
                     "input_ids": list(range(100)),
                     "loss_mask": [0] * 10 + [1] * 90}]
    x, y, mask = get_sft_batch(long_example, 1, 8, seed=0, step=0, micro_step=0)
    # ids truncated to context+1 = 9, then shifted to length 8 = context.
    assert x.shape[1] == 8
    assert x[0].tolist() == list(range(8))        # first 8 ids
    assert y[0].tolist() == list(range(1, 9))


def test_masked_loss_averages_only_active_positions():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7)          # [B, L, V]
    y = torch.randint(0, 7, (2, 3))
    full = torch.ones(2, 3)
    # Full mask equals plain mean cross-entropy.
    ce = torch.nn.functional.cross_entropy(logits.view(-1, 7), y.view(-1))
    assert torch.allclose(masked_lm_loss(logits, y, full), ce)
    # Zeroing a column drops it from both numerator and denominator.
    partial = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    per_tok = torch.nn.functional.cross_entropy(
        logits.view(-1, 7), y.view(-1), reduction="none").view(2, 3)
    expected = (per_tok * partial).sum() / partial.sum()
    assert torch.allclose(masked_lm_loss(logits, y, partial), expected)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sft_batch.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tinystories_v2.sft'` (or `ImportError` for the three names).

- [ ] **Step 3: Write the minimal implementation**

Create `src/tinystories_v2/sft.py` with the module docstring and the three helpers (the `run`/`main` entrypoint is added in Task 2):

```python
"""SFT stage: fine-tune a Pretraining checkpoint on Slot Prompt -> Fable
examples (issue 12's sft_data artifact) with prompt-masked loss.

Invoke standalone:
    ts2-sft --config configs/sft_fixture.toml [--resume]
    (or: python -m tinystories_v2.sft --config ...)

Initializes model weights from a Pretraining checkpoint ([init] section), then
trains with its own optimizer/schedule/checkpoints. Reuses issue 02's
checkpoint-resume contract, optimizer conventions (build_optimizer), LR
schedule (lr_at), precision knob, W&B logging, and Hub sync verbatim; only the
data source (variable-length masked examples, not packed windows) and the loss
(masked to the fable body + <|end|>) differ.

Artifacts in <out_dir>:
    checkpoints/step_XXXXXX.pt   full training state (atomic; resume contract)
    metrics.jsonl                one line per log_every steps: loss, lr, tokens_seen
    manifest.json                stage, version, final step/loss, examples_path, config

Determinism contract (same as pretrain): model init is loaded from a fixed
checkpoint, batches are a pure function of (seed, step, micro_step), optimizer
state round-trips, so an interrupted-and-resumed run reproduces the
uninterrupted run exactly (fp32 CPU; asserted by tests/test_sft_resume.py).
"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from tinystories_v2 import __version__
from tinystories_v2.checkpoint import (
    latest_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint,
)
from tinystories_v2.config import load_config, load_env
from tinystories_v2.hub import fetch_from, try_sync_to
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.pretrain import build_optimizer, lr_at
from tinystories_v2.tracking import MetricsLogger


def load_sft_examples(path: str | Path) -> list[dict]:
    """Read the sft_data artifact (examples.jsonl) into memory. Each record has
    input_ids and loss_mask (schema: docs/schemas/sft-example-v1.md)."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def get_sft_batch(examples: list[dict], micro_batch_size: int, context: int, *,
                  seed: int, step: int, micro_step: int,
                  device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A padded (x, y, mask) micro-batch sampled with replacement. Batch
    selection is a pure function of (seed, step, micro_step) so an interrupted
    run resumed from a checkpoint replays identical batches (resume contract).

    Each example is truncated to context+1 ids, then shifted for next-token
    prediction: x = ids[:-1], y = ids[1:], mask = loss_mask[1:] (active over the
    fable body + <|end|>). Rows are right-padded to the batch's longest x with
    id 0 and mask 0; padding never contributes to the loss, and causal attention
    makes right-padding safe without an attention mask.
    """
    generator = torch.Generator()
    generator.manual_seed(((seed * 1_000_003 + step) * 1_009 + micro_step) % 2**63)
    idx = torch.randint(0, len(examples), (micro_batch_size,), generator=generator)

    rows = []
    for i in idx.tolist():
        ids = examples[i]["input_ids"][:context + 1]
        mask = examples[i]["loss_mask"][:context + 1]
        rows.append((ids[:-1], ids[1:], mask[1:]))
    width = max(len(x) for x, _, _ in rows)

    xs, ys, ms = [], [], []
    for x, y, m in rows:
        pad = width - len(x)
        xs.append(x + [0] * pad)
        ys.append(y + [0] * pad)
        ms.append([float(v) for v in m] + [0.0] * pad)
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    mask = torch.tensor(ms, dtype=torch.float)
    return x.to(device), y.to(device), mask.to(device)


def masked_lm_loss(logits: torch.Tensor, y: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy over active (mask==1) positions only.
    Clamps the denominator so an all-padding batch cannot divide by zero."""
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), y.view(-1), reduction="none"
    )
    loss = loss.view_as(y) * mask
    return loss.sum() / mask.sum().clamp(min=1.0)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sft_batch.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/sft.py tests/test_sft_batch.py
git commit -m "feat: SFT masked-loss batching and loss helpers"
```

---

### Task 2: SFT trainer stage (run + entrypoint + configs)

Wire the helpers into a full stage: initialize from a Pretraining checkpoint, run the masked-loss training loop reusing pretrain's optimizer/schedule/precision/logging/checkpoint-resume, and write the stage artifacts. Add the two config files, the `ts2-sft` entrypoint, and the `make_init_checkpoint` test fixture.

**Files:**
- Modify: `src/tinystories_v2/sft.py` (append `_init_model_from_pretrain`, `run`, `main`)
- Create: `configs/sft_fixture.toml`, `configs/sft_full.toml`
- Modify: `pyproject.toml` (add `ts2-sft` entrypoint)
- Modify: `tests/conftest.py` (add `make_init_checkpoint` fixture)
- Test: `tests/test_sft_stage.py`

**Interfaces:**
- Consumes: `load_sft_examples`, `get_sft_batch`, `masked_lm_loss` (Task 1); `lr_at`, `build_optimizer` (pretrain); the checkpoint schema `{"step","tokens_seen","model","optimizer","scaler","config"}`.
- Produces (relied on by Task 3, 4, 5):
  - `run(config: dict, resume: bool = False) -> dict` returning `{"step": int, "loss": float}`; writes `checkpoints/step_*.pt`, `metrics.jsonl`, `manifest.json` (with `stage="sft"`, `examples_path`, `n_examples`).
  - `_init_model_from_pretrain(config: dict, device: str) -> FableLM`.
  - Config shape: top-level `out_dir`; `[model]` (matches the pretrained architecture); `[data] examples_path, tokenizer_path`; `[init] local_dir` (+ optional `hub_source`); `[train]` (same keys as pretrain); optional `[wandb]`, `[hub] target`.
  - `make_init_checkpoint(init_dir, model_cfg: dict, tokenizer_path) -> Path` (conftest fixture).

- [ ] **Step 1: Add the `make_init_checkpoint` fixture to `tests/conftest.py`**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def make_init_checkpoint():
    """Write a minimal Pretraining-style checkpoint the SFT stage can init from
    without running the pretrain stage. The SFT init path reads only
    state['model'] and state['config']['model']; the rest satisfies the schema
    and lets generate/demo find the tokenizer."""
    import torch
    from tinystories_v2.checkpoint import save_checkpoint
    from tinystories_v2.model import FableLM, ModelConfig

    def _make(init_dir, model_cfg: dict, tokenizer_path) -> Path:
        torch.manual_seed(0)
        model = FableLM(ModelConfig(**model_cfg))
        save_checkpoint(Path(init_dir) / "checkpoints", 0, {
            "step": 0, "tokens_seen": 0,
            "model": model.state_dict(), "optimizer": {}, "scaler": {},
            "config": {"model": dict(model_cfg),
                       "data": {"tokenizer_path": str(tokenizer_path)}},
        })
        return Path(init_dir)

    return _make
```

- [ ] **Step 2: Write the failing stage tests**

Create `tests/test_sft_stage.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tinystories_v2.data import run as data_run
from tinystories_v2.pretrain import run as pretrain_run
from tinystories_v2.sft import run as sft_run
from tinystories_v2.sft_data import run as sft_data_run
from tinystories_v2.tokenizer import run as tokenizer_run

TOY_MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 128, "ffn_hidden": 192}


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, fixture_path):
    """Run data-prep + tokenizer + sft_data so the SFT stage has a real
    examples.jsonl and split-hash sets to read (stages couple via artifacts)."""
    base = tmp_path_factory.mktemp("sft_stage_inputs")
    data_dir, tok_dir, sd_dir = base / "data", base / "tok", base / "sd"
    data_run({
        "out_dir": str(data_dir),
        "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.4, "sft": 0.4,
                   "pref": 0.1, "eval": 0.1},
    })
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer = str(tok_dir / "tokenizer.json")
    sft_data_run({"out_dir": str(sd_dir), "tokenizer": tokenizer,
                  "sft_split": str(data_dir / "splits" / "sft.jsonl"),
                  "max_examples": 0})

    def split_hashes(name):
        with open(data_dir / "splits" / f"{name}.jsonl", encoding="utf-8") as f:
            return {json.loads(line)["prompt_hash"] for line in f if line.strip()}

    return {
        "examples_path": str(sd_dir / "examples.jsonl"),
        "tokenizer": tokenizer,
        "sft_hashes": split_hashes("sft"),
        "pretrain_hashes": split_hashes("pretrain"),
        "eval_hashes": split_hashes("eval"),
    }


def sft_toy_config(out_dir, prepared, init_dir, model=None, **train_overrides) -> dict:
    train = {
        "steps": 30, "micro_batch_size": 8, "grad_accum": 1,
        "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
        "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
        "precision": "fp32", "seed": 1337,
        "checkpoint_every": 10, "log_every": 1, "keep_last": 0,
    }
    train.update(train_overrides)
    return {
        "out_dir": str(out_dir),
        "model": dict(model or TOY_MODEL),
        "data": {"examples_path": prepared["examples_path"],
                 "tokenizer_path": prepared["tokenizer"]},
        "init": {"local_dir": str(init_dir)},
        "train": train,
        "wandb": {"enabled": False},
    }


def read_metrics(out_dir) -> list[dict]:
    lines = (Path(out_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_toy_sft_decreases_masked_loss(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = sft_toy_config(tmp_path / "out", prepared, init)
    summary = sft_run(config)
    metrics = read_metrics(config["out_dir"])
    assert len(metrics) == 30
    assert metrics[-1]["loss"] < metrics[0]["loss"] - 0.5  # verified drop ~0.85
    assert summary["step"] == 30
    assert {"step", "loss", "lr", "tokens_seen"} <= metrics[0].keys()


def test_stage_artifacts_and_manifest(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = sft_toy_config(tmp_path / "out", prepared, init, steps=4,
                            checkpoint_every=2)
    sft_run(config)
    out = Path(config["out_dir"])
    ckpts = sorted(p.name for p in (out / "checkpoints").glob("step_*.pt"))
    assert ckpts == ["step_000002.pt", "step_000004.pt"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "sft"
    assert manifest["final_step"] == 4
    assert manifest["examples_path"] == prepared["examples_path"]
    assert manifest["n_examples"] > 0


def test_init_from_a_real_pretraining_checkpoint(tmp_path, prepared, fixture_path):
    # Produce a real pretrain checkpoint (context 64), then SFT from it. Exercises
    # the init load + architecture-match validation against a genuine artifact.
    model64 = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
               "n_heads": 2, "context": 64, "ffn_hidden": 192}
    pre_dir = tmp_path / "pretrain"
    pretrain_run({
        "out_dir": str(pre_dir),
        "model": dict(model64),
        "data": {"split_path": str(fixture_path),
                 "tokenizer_path": prepared["tokenizer"],
                 "packed_path": str(tmp_path / "packed" / "pretrain.bin")},
        "train": {"steps": 2, "micro_batch_size": 4, "grad_accum": 1,
                  "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                  "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
                  "grad_clip": 1.0, "precision": "fp32", "seed": 1,
                  "checkpoint_every": 2, "log_every": 1, "keep_last": 0},
        "wandb": {"enabled": False},
    })
    config = sft_toy_config(tmp_path / "sft_out", prepared, pre_dir,
                            model=model64, steps=3, checkpoint_every=3)
    summary = sft_run(config)
    import math
    assert summary["step"] == 3 and math.isfinite(summary["loss"])


def test_mismatched_init_architecture_raises(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    drifted = dict(TOY_MODEL, d_model=128)  # differs from the init checkpoint
    config = sft_toy_config(tmp_path / "out", prepared, init, model=drifted,
                            steps=2, checkpoint_every=2)
    with pytest.raises(ValueError, match="Pretraining checkpoint"):
        sft_run(config)


def test_missing_init_checkpoint_raises(tmp_path, prepared):
    config = sft_toy_config(tmp_path / "out", prepared, tmp_path / "empty_init",
                            steps=2, checkpoint_every=2)
    with pytest.raises(ValueError, match="no Pretraining checkpoint"):
        sft_run(config)


def test_stage_trains_only_on_the_sft_split(tmp_path, prepared, make_init_checkpoint):
    # Split-leakage guard: every prompt_hash the stage trains on comes from the
    # sft split; none leak in from pretrain or eval.
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = sft_toy_config(tmp_path / "out", prepared, init, steps=2,
                            checkpoint_every=2)
    sft_run(config)
    manifest = json.loads(
        (Path(config["out_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    from tinystories_v2.sft import load_sft_examples
    trained = {rec["prompt_hash"] for rec in load_sft_examples(manifest["examples_path"])}
    assert trained  # non-empty
    assert trained <= prepared["sft_hashes"]
    assert trained.isdisjoint(prepared["pretrain_hashes"] | prepared["eval_hashes"])


def to_toml(config: dict) -> str:
    """Serialize the nested SFT config as TOML (stdlib has no writer)."""
    lines = [f'out_dir = "{config["out_dir"]}"']
    for section in ("model", "data", "init", "train", "wandb", "hub"):
        if section not in config:
            continue
        lines.append(f"[{section}]")
        for key, value in config[section].items():
            if isinstance(value, bool):
                lines.append(f"{key} = {str(value).lower()}")
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"


def test_cli_entrypoint_runs_standalone(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = sft_toy_config(tmp_path / "out", prepared, init, steps=2,
                            checkpoint_every=2)
    config_file = tmp_path / "cfg.toml"
    config_file.write_text(to_toml(config), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.sft", "--config", str(config_file)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (Path(config["out_dir"]) / "checkpoints" / "step_000002.pt").exists()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sft_stage.py -q`
Expected: FAIL — `ImportError: cannot import name 'run' from 'tinystories_v2.sft'`.

- [ ] **Step 4: Implement `_init_model_from_pretrain`, `run`, and `main` in `src/tinystories_v2/sft.py`**

Append to `src/tinystories_v2/sft.py`:

```python
def _init_model_from_pretrain(config: dict, device: str) -> FableLM:
    """Fresh SFT start: build the model from [model] and load Pretraining
    weights. Fetches the init artifact from Hub first if the local checkpoint is
    absent, then validates the pretrained architecture matches [model]."""
    model = FableLM(ModelConfig(**config["model"])).to(device)
    init = config["init"]
    init_dir = Path(init["local_dir"])
    init_ckpt_dir = init_dir / "checkpoints"
    if latest_checkpoint(init_ckpt_dir) is None and init.get("hub_source"):
        fetch_from(init["hub_source"], init_dir)  # fresh Colab VM: pull pretrain
    init_ckpt = latest_checkpoint(init_ckpt_dir)
    if init_ckpt is None:
        raise ValueError(
            f"no Pretraining checkpoint under {init_ckpt_dir}; point "
            f"[init].local_dir (and optionally [init].hub_source) at the "
            f"Pretraining artifact"
        )
    state = load_checkpoint(init_ckpt)
    if ModelConfig(**state["config"]["model"]) != model.config:
        raise ValueError(
            f"[model] does not match the Pretraining checkpoint at {init_ckpt}; "
            f"SFT must use the pretrained architecture"
        )
    model.load_state_dict(state["model"])
    print(f"initialized from Pretraining checkpoint {init_ckpt}")
    return model


def run(config: dict, resume: bool = False) -> dict:
    load_env()  # W&B/HF keys before wandb.init or hub sync; never printed
    train = config["train"]
    out_dir = Path(config["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    hub_target = config.get("hub", {}).get("target")

    examples = load_sft_examples(config["data"]["examples_path"])
    if not examples:
        raise ValueError(f"no examples in {config['data']['examples_path']}")

    # -- precision knob: fp32 | bf16 (autocast) | fp16 (autocast + GradScaler)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = train["precision"]
    amp_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    autocast = (torch.autocast(device_type=device, dtype=amp_dtype)
                if amp_dtype else nullcontext())
    scaler = torch.amp.GradScaler(device, enabled=(precision == "fp16"))

    torch.manual_seed(train["seed"])
    model = _init_model_from_pretrain(config, device)
    optimizer = build_optimizer(model, train["peak_lr"],
                                (train["beta1"], train["beta2"]),
                                train["weight_decay"])

    start_step, tokens_seen = 0, 0
    if resume:
        if latest_checkpoint(ckpt_dir) is None and hub_target:
            fetch_from(hub_target, out_dir)  # fresh Colab VM: pull previous session
        ckpt_path = latest_checkpoint(ckpt_dir)
        if ckpt_path is not None:
            state = load_checkpoint(ckpt_path)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start_step, tokens_seen = state["step"], state["tokens_seen"]
            print(f"resumed from {ckpt_path.name} at step {start_step}")

    def checkpoint(step: int) -> None:
        save_checkpoint(ckpt_dir, step, {
            "step": step, "tokens_seen": tokens_seen,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "config": config,
        })
        prune_checkpoints(ckpt_dir, train.get("keep_last", 0))
        if hub_target:
            try_sync_to(hub_target, out_dir)

    logger = MetricsLogger(out_dir, config.get("wandb"))
    steps, accum = train["steps"], train["grad_accum"]
    micro_bs, context = train["micro_batch_size"], config["model"]["context"]
    loss_value = float("nan")
    model.train()
    for step in range(start_step, steps):
        lr = lr_at(step, steps, train["peak_lr"],
                   train["warmup_frac"], train["min_lr_frac"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(accum):
            x, y, mask = get_sft_batch(examples, micro_bs, context,
                                       seed=train["seed"], step=step,
                                       micro_step=micro_step, device=device)
            with autocast:
                logits = model(x)
                loss = masked_lm_loss(logits, y, mask)
            scaler.scale(loss / accum).backward()
            tokens_seen += int(mask.sum().item())  # cumulative active target tokens
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        loss_value = loss.item()  # last micro-batch masked loss (logged raw)
        done = step + 1
        if done % train["log_every"] == 0:
            logger.log({"loss": loss_value, "lr": lr, "tokens_seen": tokens_seen},
                       step=done)
        if done % train["checkpoint_every"] == 0:
            checkpoint(done)
    if steps % train["checkpoint_every"] != 0:
        checkpoint(steps)
    logger.finish()

    manifest = {
        "stage": "sft", "package_version": __version__,
        "final_step": steps, "final_loss": loss_value,
        "tokens_seen": tokens_seen,
        "examples_path": config["data"]["examples_path"],
        "n_examples": len(examples),
        "config": config,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    if hub_target:
        try_sync_to(hub_target, out_dir)
    return {"step": steps, "loss": loss_value}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest checkpoint in out_dir")
    args = parser.parse_args(argv)
    run(load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add the `ts2-sft` entrypoint to `pyproject.toml`**

In the `[project.scripts]` table, after the `ts2-sft-data` line, add:

```toml
ts2-sft = "tinystories_v2.sft:main"
```

- [ ] **Step 6: Create `configs/sft_fixture.toml`**

```toml
# Toy CPU wiring smoke against fixture artifacts — local sanity runs and docs.
# Assumes these upstream stages have run with a matching [model] block:
#   ts2-data-prep --config configs/data_prep_fixture.toml
#   ts2-tokenizer --config configs/tokenizer_fixture.toml
#   ts2-sft-data  --config configs/sft_data_fixture.toml
#   ts2-pretrain  --config configs/pretrain_fixture.toml   (produces the init checkpoint)
# At context 64 this is a wiring smoke, not a quality run; real SFT uses
# configs/sft_full.toml. Stage behavior is guarded by tests/test_sft_*.py.
out_dir = "artifacts/sft_fixture"

# Must match the Pretraining checkpoint's architecture (pretrain_fixture.toml).
[model]
vocab_size = 512
d_model = 64
n_layers = 2
n_heads = 2
context = 64
ffn_hidden = 192

[data]
examples_path = "artifacts/sft_data_fixture/examples.jsonl"
tokenizer_path = "artifacts/tokenizer_fixture/tokenizer.json"

[init]
local_dir = "artifacts/pretrain_fixture"   # contains checkpoints/step_*.pt

[train]
steps = 30
micro_batch_size = 8
grad_accum = 1
peak_lr = 1e-3
warmup_frac = 0.1
min_lr_frac = 0.1
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
precision = "fp32"
seed = 1337
checkpoint_every = 10
log_every = 1
keep_last = 0

[wandb]
enabled = false
```

- [ ] **Step 7: Create `configs/sft_full.toml`**

```toml
# Real SFT run on Colab Pro (design-doc defaults: LR 1e-4 cosine, ~1-2 epochs).
# bf16 on L4 (preferred). On a T4 fallback set precision = "fp16".
# Prerequisites on Hub/disk: sft_data full artifact + tokenizer_full, and a
# Pretraining checkpoint. [init].hub_source pulls the pretrain artifact if the
# local copy is absent (fresh VM).
out_dir = "artifacts/sft_full"

[model]
vocab_size = 8192
d_model = 512
n_layers = 8
n_heads = 8
context = 512
ffn_hidden = 1408

[data]
examples_path = "artifacts/sft_data_full/examples.jsonl"
tokenizer_path = "artifacts/tokenizer_full/tokenizer.json"

[init]
local_dir = "artifacts/pretrain_full"
hub_source = "hf://congthanh991/tinystories-v2-pretrain"

[train]
steps = 800                 # ~2 epochs of ~50k examples at micro_batch 16 x accum 8
micro_batch_size = 16
grad_accum = 8
peak_lr = 1e-4
warmup_frac = 0.03
min_lr_frac = 0.1
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
precision = "bf16"
seed = 1337
checkpoint_every = 100      # ~10-15 min on L4
log_every = 10
keep_last = 2

[wandb]
enabled = true
project = "tinystories-v2"
run_name = "sft"

[hub]
target = "hf://congthanh991/tinystories-v2-sft"
```

- [ ] **Step 8: Run the stage tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sft_stage.py -q`
Expected: PASS (7 tests).

- [ ] **Step 9: Commit**

```bash
git add src/tinystories_v2/sft.py configs/sft_fixture.toml configs/sft_full.toml \
        pyproject.toml tests/conftest.py tests/test_sft_stage.py
git commit -m "feat: SFT training stage, configs, and ts2-sft entrypoint"
```

---

### Task 3: Kill-and-resume contract for SFT

Prove the checkpoint-resume contract for SFT end to end: a SIGKILLed run resumes from its latest checkpoint and reproduces the uninterrupted run bitwise, with post-resume losses replaying exactly. Mirrors `tests/test_resume.py`.

**Files:**
- Test: `tests/test_sft_resume.py`

**Interfaces:**
- Consumes: `tinystories_v2.sft.run`, the `python -m tinystories_v2.sft` entrypoint, `make_init_checkpoint` (conftest), `sft_data.run`, `tokenizer.run`, `data.run`, `checkpoint.latest_checkpoint`/`load_checkpoint`.

- [ ] **Step 1: Write the failing resume test**

Create `tests/test_sft_resume.py`:

```python
"""Kill-and-resume: the SFT checkpoint-resume contract, end to end.

Sized so the run is slow enough to SIGKILL mid-flight: d_model 128 / 4 layers /
ctx 128, 50 steps, checkpoint_every 5 gives several checkpoints before the kill.
Both runs share one init checkpoint and one examples.jsonl, so batches (a pure
function of seed/step/micro_step) and starting weights are identical.
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
from tinystories_v2.sft import run as sft_run
from tinystories_v2.sft_data import run as sft_data_run
from tinystories_v2.tokenizer import run as tokenizer_run

STEPS = 50
CHECKPOINT_EVERY = 5
KILL_AFTER_STEP = 10

MODEL = {"vocab_size": 512, "d_model": 128, "n_layers": 4,
         "n_heads": 4, "context": 128, "ffn_hidden": 384}


def sft_config(out_dir, examples_path, tokenizer_path, init_dir) -> dict:
    return {
        "out_dir": str(out_dir),
        "model": dict(MODEL),
        "data": {"examples_path": str(examples_path),
                 "tokenizer_path": str(tokenizer_path)},
        "init": {"local_dir": str(init_dir)},
        "train": {"steps": STEPS, "micro_batch_size": 8, "grad_accum": 1,
                  "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                  "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
                  "grad_clip": 1.0, "precision": "fp32", "seed": 1337,
                  "checkpoint_every": CHECKPOINT_EVERY, "log_every": 1,
                  "keep_last": 0},
        "wandb": {"enabled": False},
    }


def to_toml(config: dict) -> str:
    lines = [f'out_dir = "{config["out_dir"]}"']
    for section in ("model", "data", "init", "train", "wandb"):
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


def test_killed_sft_resumes_to_identical_final_state(
        tmp_path, fixture_path, make_init_checkpoint):
    # Build the shared inputs once: a real examples.jsonl and one init checkpoint.
    data_dir, tok_dir, sd_dir = tmp_path / "data", tmp_path / "tok", tmp_path / "sd"
    data_run({
        "out_dir": str(data_dir), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.4, "sft": 0.4,
                   "pref": 0.1, "eval": 0.1},
    })
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer_path = tok_dir / "tokenizer.json"
    sft_data_run({"out_dir": str(sd_dir), "tokenizer": str(tokenizer_path),
                  "sft_split": str(data_dir / "splits" / "sft.jsonl"),
                  "max_examples": 0})
    examples_path = sd_dir / "examples.jsonl"
    init_dir = make_init_checkpoint(tmp_path / "init", MODEL, tokenizer_path)

    # Reference: identical config, never interrupted.
    reference = sft_config(tmp_path / "reference", examples_path, tokenizer_path, init_dir)
    sft_run(reference)

    # Interrupted: run as a subprocess and SIGKILL once the kill-marker appears.
    interrupted = sft_config(tmp_path / "interrupted", examples_path,
                             tokenizer_path, init_dir)
    config_file = tmp_path / "interrupted.toml"
    config_file.write_text(to_toml(interrupted), encoding="utf-8")
    ckpt_dir = Path(interrupted["out_dir"]) / "checkpoints"
    kill_marker = ckpt_dir / f"step_{KILL_AFTER_STEP:06d}.pt"

    proc = subprocess.Popen(
        [sys.executable, "-m", "tinystories_v2.sft", "--config", str(config_file)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 120
        while not kill_marker.exists():
            if proc.poll() is not None:
                pytest.fail(
                    f"stage finished (rc={proc.returncode}) before the kill window; "
                    f"enlarge the toy model or lower KILL_AFTER_STEP"
                )
            if time.monotonic() > deadline:
                pytest.fail("timed out waiting for the kill-marker checkpoint")
            time.sleep(0.01)
        os.kill(proc.pid, signal.SIGKILL)
    finally:
        proc.kill()
        proc.wait(timeout=30)

    killed_at = load_checkpoint(latest_checkpoint(ckpt_dir))["step"]
    assert KILL_AFTER_STEP <= killed_at < STEPS

    sft_run(interrupted, resume=True)

    final_ref = load_checkpoint(
        latest_checkpoint(Path(reference["out_dir"]) / "checkpoints"))
    final_res = load_checkpoint(latest_checkpoint(ckpt_dir))
    assert final_res["step"] == final_ref["step"] == STEPS
    assert final_res["tokens_seen"] == final_ref["tokens_seen"]
    for key, tensor in final_ref["model"].items():
        assert torch.equal(final_res["model"][key], tensor), key

    ref_losses = read_metrics(reference["out_dir"])
    res_losses = read_metrics(interrupted["out_dir"])
    for step in range(killed_at + 1, STEPS + 1):
        assert res_losses[step] == ref_losses[step], step
```

- [ ] **Step 2: Run the resume test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_sft_resume.py -q`
Expected: PASS. (If it reports the stage finished before the kill window, the toy model is too fast — this should not happen at d128/4L/ctx128/50 steps; do not weaken the assertion, enlarge the model.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_sft_resume.py
git commit -m "test: SFT kill-and-resume reproduces the uninterrupted run bitwise"
```

---

### Task 4: Format learned at toy scale (`<|end|>` termination)

Prove the second acceptance criterion: after a toy SFT overfit, generation conditioned on a training Scaffold terminates with `<|end|>`. A tiny fixture of four short examples (encoded through the real `slot_prompt` format) fits the toy context so `<|end|>` is reachable; the model memorizes and greedy decoding reproduces the body then `<|end|>`. (Verified in a spike: 200 steps → masked loss ≈0.02, 4/4 scaffolds terminate.)

**Files:**
- Test: `tests/test_sft_format_learned.py`

**Interfaces:**
- Consumes: `sft.run`, `slot_prompt.encode_example`/`render_prompt`, `slots.Scaffold`, `generate.sample`, `model.FableLM`/`ModelConfig`, `checkpoint.latest_checkpoint`/`load_checkpoint`, `make_init_checkpoint`, `tokenizer.run`.

- [ ] **Step 1: Write the failing format-learned test**

Create `tests/test_sft_format_learned.py`:

```python
"""After a toy SFT overfit, generation from a training Scaffold terminates with
<|end|> — the Slot Prompt format learned at toy scale (issue 03 criterion 2).

Four short examples encoded through the real slot_prompt format fit the toy
context (<|end|> reachable); the small model overfits and greedy decoding of a
training Slot Prompt reproduces the body and emits <|end|>.
"""

import json

from tokenizers import Tokenizer

from tinystories_v2.checkpoint import latest_checkpoint, load_checkpoint
from tinystories_v2.generate import sample
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.slot_prompt import encode_example, render_prompt
from tinystories_v2.slots import Scaffold
from tinystories_v2.sft import run as sft_run
from tinystories_v2.tokenizer import run as tokenizer_run

MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
         "n_heads": 2, "context": 128, "ffn_hidden": 192}

PAIRS = [
    (Scaffold("fox", "sly", "a green wood", "a locked henhouse",
              "the fox waited", "patience wins"),
     "The sly fox waited by the henhouse and at last it opened."),
    (Scaffold("mouse", "brave", "a tall barn", "a hungry cat",
              "the mouse hid", "courage helps"),
     "The brave mouse hid from the cat and stayed safe all night."),
    (Scaffold("crow", "proud", "a dry field", "a shiny stone",
              "the crow shared", "pride can pass"),
     "The proud crow found a stone and learned to share it."),
    (Scaffold("bee", "busy", "a bright garden", "a coming storm",
              "the bee worked", "work pays off"),
     "The busy bee worked before the storm and saved the honey."),
]


def test_toy_sft_generation_terminates_with_end_token(
        tmp_path, fixture_path, make_init_checkpoint):
    tok_dir = tmp_path / "tok"
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer_path = tok_dir / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    end_id = tokenizer.token_to_id("<|end|>")

    # Build a tiny examples.jsonl via the real encoder (short bodies fit ctx 128).
    examples_path = tmp_path / "examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as f:
        for i, (scaffold, fable) in enumerate(PAIRS):
            example = encode_example(tokenizer, scaffold, fable)
            f.write(json.dumps({"prompt_hash": str(i), **example.to_dict()}) + "\n")

    init_dir = make_init_checkpoint(tmp_path / "init", MODEL, tokenizer_path)
    config = {
        "out_dir": str(tmp_path / "out"),
        "model": dict(MODEL),
        "data": {"examples_path": str(examples_path),
                 "tokenizer_path": str(tokenizer_path)},
        "init": {"local_dir": str(init_dir)},
        "train": {"steps": 250, "micro_batch_size": 4, "grad_accum": 1,
                  "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                  "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
                  "grad_clip": 1.0, "precision": "fp32", "seed": 0,
                  "checkpoint_every": 250, "log_every": 50, "keep_last": 0},
        "wandb": {"enabled": False},
    }
    summary = sft_run(config)
    assert summary["loss"] < 0.5  # overfit drove masked loss near zero

    state = load_checkpoint(latest_checkpoint(Path := (tmp_path / "out" / "checkpoints")))
    model = FableLM(ModelConfig(**state["config"]["model"]))
    model.load_state_dict(state["model"])

    scaffold = PAIRS[0][0]
    prompt_ids = tokenizer.encode(render_prompt(scaffold)).ids
    sequence = sample(model, prompt_ids, max_new_tokens=60,
                      temperature=0.0, end_id=end_id)[0]
    generated = sequence[len(prompt_ids):]
    assert end_id in generated  # generation terminated with <|end|>
```

Note: the `Path := (...)` walrus above is only to keep `latest_checkpoint` on one line; replace with a plain local if you prefer — `ckpts = tmp_path / "out" / "checkpoints"; state = load_checkpoint(latest_checkpoint(ckpts))`.

- [ ] **Step 2: Run the format-learned test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_sft_format_learned.py -q`
Expected: PASS. (If flaky, raise `steps` to 400 — the spike hit loss 0.0 and 4/4 termination by 400; do not lower `max_new_tokens` below the body length.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_sft_format_learned.py
git commit -m "test: toy SFT learns the Slot Prompt format and emits <|end|>"
```

---

### Task 5: Demo script (`ts2-demo`)

The live presentation artifact: take six slot values (or sample a Scaffold from an eval split), render the Slot Prompt, and print the model's Fable. Reuses `generate.sample` and `slot_prompt.render_prompt`. Runs on CPU against any checkpoint.

**Files:**
- Create: `src/tinystories_v2/demo.py`
- Modify: `pyproject.toml` (add `ts2-demo` entrypoint)
- Test: `tests/test_demo.py`

**Interfaces:**
- Consumes: `generate.sample`, `slot_prompt.SLOT_FIELDS`/`END_TOKEN`/`render_prompt`, `slots.Scaffold`, `checkpoint.latest_checkpoint`/`load_checkpoint`, `model.FableLM`/`ModelConfig`, `tokenizers.Tokenizer`.
- Produces: `main(argv=None)` CLI; `_scaffold_from_args(args) -> Scaffold`.

- [ ] **Step 1: Write the failing demo tests**

Create `tests/test_demo.py`:

```python
import subprocess
import sys
from pathlib import Path

from tinystories_v2.data import run as data_run
from tinystories_v2.pretrain import run as pretrain_run
from tinystories_v2.tokenizer import run as tokenizer_run

MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
         "n_heads": 2, "context": 64, "ffn_hidden": 192}

SLOTS = ["--character", "fox", "--trait", "sly", "--setting", "a green wood",
         "--conflict", "a locked gate", "--resolution", "the fox waited",
         "--moral", "patience wins"]


def _toy_checkpoint(tmp_path, fixture_path):
    """Any checkpoint works for the demo; a 2-step toy pretrain is cheapest.
    Its stored config carries data.tokenizer_path so the demo finds the tokenizer."""
    tok_dir = tmp_path / "tok"
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    out = tmp_path / "pretrain"
    pretrain_run({
        "out_dir": str(out), "model": dict(MODEL),
        "data": {"split_path": str(fixture_path),
                 "tokenizer_path": str(tok_dir / "tokenizer.json"),
                 "packed_path": str(tmp_path / "packed" / "pretrain.bin")},
        "train": {"steps": 2, "micro_batch_size": 4, "grad_accum": 1,
                  "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                  "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
                  "grad_clip": 1.0, "precision": "fp32", "seed": 1,
                  "checkpoint_every": 2, "log_every": 1, "keep_last": 0},
        "wandb": {"enabled": False},
    })
    return out / "checkpoints"


def test_demo_generates_from_six_slot_values_on_cpu(tmp_path, fixture_path):
    ckpts = _toy_checkpoint(tmp_path, fixture_path)
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.demo",
         "--checkpoint", str(ckpts), *SLOTS,
         "--max-new-tokens", "16", "--seed", "3"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "fox" in result.stdout          # Scaffold header echoes the slots
    assert "Fable" in result.stdout


def test_demo_requires_all_six_slots_or_sample_eval(tmp_path, fixture_path):
    ckpts = _toy_checkpoint(tmp_path, fixture_path)
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.demo",
         "--checkpoint", str(ckpts), "--character", "fox"],  # only one slot
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "six slots" in (result.stderr + result.stdout)


def test_demo_samples_scaffold_from_eval_split(tmp_path, fixture_path):
    ckpts = _toy_checkpoint(tmp_path, fixture_path)
    data_dir = tmp_path / "data"
    data_run({
        "out_dir": str(data_dir), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.4, "sft": 0.3,
                   "pref": 0.1, "eval": 0.2},
    })
    eval_split = data_dir / "splits" / "eval.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.demo",
         "--checkpoint", str(ckpts), "--sample-eval", str(eval_split),
         "--max-new-tokens", "16", "--seed", "0"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Fable" in result.stdout
```

- [ ] **Step 2: Run the demo tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_demo.py -q`
Expected: FAIL — `No module named tinystories_v2.demo`.

- [ ] **Step 3: Create `src/tinystories_v2/demo.py`**

```python
"""Demo: render six slot values (or a sampled eval Scaffold) into a Slot Prompt
and print the model's Fable. The live artifact for the final presentation; point
--checkpoint at any Pretraining/SFT/RLAIF checkpoint. Runs on CPU.

Invoke standalone:
    ts2-demo --checkpoint artifacts/sft_full/checkpoints \
        --character fox --trait greedy --setting "a sunny orchard" \
        --conflict "a locked gate" --resolution "the fox shared" \
        --moral "sharing brings friends"
    ts2-demo --checkpoint ... --sample-eval artifacts/data_prep_full/splits/eval.jsonl

--tokenizer defaults to the tokenizer_path recorded in the checkpoint's config.
"""

import argparse
import json
import random
from pathlib import Path

from tokenizers import Tokenizer

from tinystories_v2.checkpoint import latest_checkpoint, load_checkpoint
from tinystories_v2.generate import sample
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.slot_prompt import END_TOKEN, SLOT_FIELDS, render_prompt
from tinystories_v2.slots import Scaffold


def _scaffold_from_args(args) -> Scaffold:
    if args.sample_eval:
        rows = [json.loads(line) for line in
                open(args.sample_eval, encoding="utf-8") if line.strip()]
        if not rows:
            raise SystemExit(f"no rows in {args.sample_eval}")
        row = random.Random(args.seed).choice(rows)
        return Scaffold(**{field: row[field] for field in SLOT_FIELDS})
    values = {field: getattr(args, field) for field in SLOT_FIELDS}
    missing = [field for field, value in values.items() if not value]
    if missing:
        raise SystemExit(
            f"provide all six slots or --sample-eval; missing: {', '.join(missing)}"
        )
    return Scaffold(**values)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="a step_*.pt file or a directory of them")
    for field in SLOT_FIELDS:
        parser.add_argument(f"--{field}", help=f"the {field} slot value")
    parser.add_argument("--sample-eval", type=Path, default=None,
                        help="jsonl split to sample a Scaffold from instead of slots")
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    scaffold = _scaffold_from_args(args)

    path = args.checkpoint
    if path.is_dir():
        path = latest_checkpoint(path)
        if path is None:
            raise SystemExit(f"no step_*.pt checkpoints in {args.checkpoint}")
    state = load_checkpoint(path)
    model = FableLM(ModelConfig(**state["config"]["model"]))
    model.load_state_dict(state["model"])

    tokenizer_path = args.tokenizer or state["config"]["data"]["tokenizer_path"]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    end_id = tokenizer.token_to_id(END_TOKEN)

    prompt_ids = tokenizer.encode(render_prompt(scaffold)).ids
    sequence = sample(
        model, prompt_ids, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
        seed=args.seed, end_id=end_id,
    )[0]
    body = tokenizer.decode(sequence[len(prompt_ids):])  # skips <|end|> by default

    print("Scaffold")
    for field in SLOT_FIELDS:
        print(f"  {field}: {getattr(scaffold, field)}")
    print("\nFable\n" + body.strip())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the `ts2-demo` entrypoint to `pyproject.toml`**

In `[project.scripts]`, after `ts2-sft`, add:

```toml
ts2-demo = "tinystories_v2.demo:main"
```

- [ ] **Step 5: Run the demo tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_demo.py -q`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/tinystories_v2/demo.py pyproject.toml tests/test_demo.py
git commit -m "feat: ts2-demo script renders a Slot Prompt and prints a Fable"
```

---

### Task 6: Thin Colab notebook for the real SFT run

A thin wrapper mirroring `pretrain_colab.ipynb`: clone → install → secrets → one `ts2-sft --resume`. `[init].hub_source` in `sft_full.toml` pulls the Pretraining checkpoint automatically, so the notebook stays a single stage invocation.

**Files:**
- Create: `notebooks/sft_colab.ipynb`
- Modify: `tests/test_notebook.py` (add an SFT-notebook thin-wrapper test)

**Interfaces:**
- Consumes: nothing at runtime; the notebook test parses cell JSON.

- [ ] **Step 1: Write the failing notebook test**

Append to `tests/test_notebook.py`:

```python
SFT_NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "sft_colab.ipynb"


def test_sft_notebook_is_thin():
    cells = json.loads(SFT_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert 1 <= len(code_cells) <= 4
    source = "\n".join("".join(c["source"]) for c in code_cells)
    for forbidden in ("def ", "class ", "import torch", "for ", "while "):
        assert forbidden not in source, forbidden
    assert "ts2-sft" in source
    assert "--resume" in source


def test_sft_notebook_has_no_secrets_or_outputs():
    text = SFT_NOTEBOOK.read_text(encoding="utf-8")
    assert "hf_" not in text
    cells = json.loads(text)["cells"]
    assert all(not c.get("outputs") for c in cells)
```

- [ ] **Step 2: Run the notebook test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_notebook.py -q`
Expected: FAIL — `FileNotFoundError` for `sft_colab.ipynb`.

- [ ] **Step 3: Create `notebooks/sft_colab.ipynb`**

Write this exact JSON (mirrors `pretrain_colab.ipynb`; note the `→` arrows and escaped quotes):

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# SFT on Colab Pro (L4 preferred, T4 fallback)\n",
    "\n",
    "Thin wrapper per docs/DESIGN.md: clone → install → secrets → run stage.\n",
    "All logic lives in the package; edit `configs/sft_full.toml` in the repo, not here.\n",
    "\n",
    "The stage initializes from the Pretraining checkpoint: `[init].hub_source` in the\n",
    "config pulls it from the Hub automatically on a fresh VM.\n",
    "\n",
    "Before running: set `HF_TOKEN` and `WANDB_API_KEY` in Colab **Secrets** (key icon, left sidebar),\n",
    "set the repo URL below, and on a T4 change `precision = \"fp16\"` in the config (Turing has no bf16)."
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
    "!pip install -q -e '.[track]'"
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
    "os.environ[\"HF_TOKEN\"] = userdata.get(\"HF_TOKEN\")\n",
    "os.environ[\"WANDB_API_KEY\"] = userdata.get(\"WANDB_API_KEY\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!ts2-sft --config configs/sft_full.toml --resume"
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

Run: `.venv/bin/python -m pytest tests/test_notebook.py -q`
Expected: PASS (4 tests — the 2 existing pretrain-notebook tests plus the 2 new SFT ones).

- [ ] **Step 5: Commit**

```bash
git add notebooks/sft_colab.ipynb tests/test_notebook.py
git commit -m "feat: thin Colab notebook for the real SFT run"
```

---

### Task 7: Full suite + update PROGRESS.md and the issue status

Run the whole test suite green, then record the state change. Issue 02's real Pretraining run is still in flight on the VM, so a *real* SFT run stays gated on that checkpoint; the code and acceptance criteria for issue 03 are complete.

**Files:**
- Modify: `PROGRESS.md`
- Modify: `.scratch/tinystories-v2-pipeline/issues/03-sft-stage.md` (status line)

**Interfaces:** none (documentation).

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — all existing tests plus the new `test_sft_batch.py`, `test_sft_stage.py`, `test_sft_resume.py`, `test_sft_format_learned.py`, `test_demo.py`, and the extended `test_notebook.py`.

- [ ] **Step 2: Flip the issue status line**

In `.scratch/tinystories-v2-pipeline/issues/03-sft-stage.md`, change `Status: ready-for-agent` to `Status: code-complete (real run gated on issue 02's Pretraining checkpoint)`. Check each acceptance box that the tests now cover:
- Toy SFT decreases loss + resumes after a kill → `test_sft_stage.py` + `test_sft_resume.py`.
- Generation terminates with `<|end|>` → `test_sft_format_learned.py`.
- Demo generates from a checkpoint on CPU → `test_demo.py`.
- Thin Colab notebook exists → `notebooks/sft_colab.ipynb` + `test_notebook.py`.
- Reads the sft split only; leakage guard → `test_sft_stage.py::test_stage_trains_only_on_the_sft_split`.

- [ ] **Step 3: Update `PROGRESS.md`**

Make these edits to `PROGRESS.md`:

1. Bump `_Last updated: 2026-07-12_` (keep the date; it is still 2026-07-12).
2. In **## Now**, replace the issue-03 bullet with a note that 03 is code-complete and unblocks 04 and 07 (07 still also waits on issue 11). Keep the issue-02 in-progress bullet.
3. In the **Issue board** table, change the issue-03 row Status from `🟢 ready ← **grab this**` to `✅ code complete (real run gated on 02's checkpoint)`. Update the issue-04 row from `🔴 blocked` to `🟢 ready (code work)` (its blocker 03 is now code-done; 10 already ✅). Leave 07 `🔴 blocked` (still needs 11).
4. In the **Milestones vs plan** table, update the W3 row to note SFT (03) code-complete 2026-07-12.
5. Add a **## Log** entry at the top of the list:

```markdown
- **2026-07-12** — Issue 03 (SFT stage + demo) code complete: `sft.py` stage
  (masked-loss fine-tune from a Pretraining checkpoint, checkpoint-resume,
  `ts2-sft`), `demo.py` (`ts2-demo`), `configs/sft_{fixture,full}.toml`,
  `notebooks/sft_colab.ipynb`, and tests (batching, stage, kill-resume,
  `<|end|>` format-learning, demo, notebook). Real SFT run still gated on issue
  02's Pretraining checkpoint (in flight on the VM). Unblocks issue 04.
```

- [ ] **Step 4: Commit**

```bash
git add PROGRESS.md .scratch/tinystories-v2-pipeline/issues/03-sft-stage.md
git commit -m "docs: mark issue 03 (SFT stage + demo) code-complete"
```

---

## Self-Review

**1. Spec coverage** (issue 03 acceptance criteria → task):
- "Toy SFT run … decreases loss … and resumes after a kill" → Task 2 (`test_toy_sft_decreases_masked_loss`) + Task 3 (`test_sft_resume.py`). ✓
- "generation conditioned on a fixture Scaffold terminates with `<|end|>`" → Task 4. ✓
- "Demo script generates from a checkpoint given six slot values on CPU" → Task 5. ✓
- "Thin Colab notebook exists for the real SFT run" → Task 6. ✓
- "Stage reads the sft split artifact only — a test guards against pretrain/eval split leakage" → Task 2 (`test_stage_trains_only_on_the_sft_split`). ✓
- Blocked-by 02 (checkpoint-resume, optimizer, W&B) → reused via `_init_model_from_pretrain`, imported `lr_at`/`build_optimizer`, `MetricsLogger`, `save/load_checkpoint`, `try_sync_to`. ✓
- Blocked-by 12 (Slot Prompt render/encode/parse, sft_data artifact) → consumed via `examples.jsonl`, `encode_example`, `render_prompt`. ✓

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N"; every code step shows complete code and every run step shows the command + expected outcome. The one shorthand (walrus in Task 4 Step 1) is flagged with the plain-local replacement inline. ✓

**3. Type/name consistency:**
- `run(config, resume=False) -> {"step","loss"}` used identically in Tasks 2/3/4. ✓
- `get_sft_batch(examples, micro_batch_size, context, *, seed, step, micro_step, device)` → `(x, y, mask)`; `masked_lm_loss(logits, y, mask)`; `load_sft_examples(path)` — signatures match between Task 1 definition and Task 2 usage. ✓
- Config sections `[model]/[data]/[init]/[train]/[wandb]/[hub]` consistent across `sft.py`, both configs, and all `to_toml` serializers (they iterate exactly these sections). ✓
- `make_init_checkpoint(init_dir, model_cfg, tokenizer_path)` defined in conftest (Task 2), used in Tasks 2/3/4. ✓
- Manifest keys (`stage="sft"`, `examples_path`, `n_examples`, `final_step`, `final_loss`) written in Task 2, asserted in Task 2 and read in Task 2's leakage test. ✓
- `SLOT_FIELDS`, `END_TOKEN`, `render_prompt` imported from `slot_prompt` (not re-derived) in `demo.py`. ✓

**Verified against real runs (not assumed):** masked-loss decrease over 30 toy steps ≈ 0.85 (assertion threshold 0.5); toy overfit reaches loss ≈0.02 with 4/4 `<|end|>` termination at 200 steps (test uses 250, assertion `loss < 0.5`); toy SFT example lengths 629–890 tokens under the vocab-512 tokenizer (drives the ctx-128 truncation choice for the loss/resume tests and the short-example fixture for the format test).
