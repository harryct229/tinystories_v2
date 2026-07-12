# 05 — Reward Model Stage + Accuracy Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Reward Model stage — the SFT backbone with its LM head replaced by a scalar head, trained with hand-written Bradley-Terry loss on issue 10's preference pairs — that records held-out pair accuracy in its artifact, plus a shared accuracy gate that later RLAIF (issue 06 GRPO) calls to refuse a below-gate Reward Model.

**Architecture:** A new model module `reward_model.py` wraps `FableLM` as a backbone and attaches a linear scalar head; a sequence's reward is the scalar read from the hidden state at its last real token (right-padding + causal attention make that position padding-invariant). A new stage module `reward.py` mirrors `sft.py`'s stage contract (one TOML → `out_dir` artifacts, checkpoint-resume, precision knob, W&B logging, Hub sync) but swaps the data source (preference pairs, not masked examples) and the loss (Bradley-Terry over chosen/rejected scores). It initializes the backbone from an SFT checkpoint declared in `[init]`, holds out a deterministic slice of pairs, and records held-out accuracy + the split recipe in `manifest.json`. A tiny `gate.py` reads that manifest and raises `RewardGateError` when accuracy is below the gate — the reusable seam issue 06 enforces. The optimizer builder and LR schedule are imported from `pretrain.py` verbatim (DRY).

**Tech Stack:** Python 3.11+, PyTorch ≥2.6, `tokenizers`, `numpy`, TOML configs, pytest. No new dependencies.

## Global Constraints

- **Stage convention:** every stage entrypoint reads exactly one TOML file and writes all artifacts under the config's `out_dir`; stages share nothing in memory and couple only through on-disk artifacts.
- **Checkpoint-resume is a hard requirement:** a run killed mid-flight (SIGKILL) must resume from the latest checkpoint and reproduce the uninterrupted run bitwise on fp32 CPU. Batches are a pure function of `(seed, step, micro_step)`; the held-out split is a pure function of `(n_pairs, holdout_frac, split_seed)`; optimizer + scaler state round-trip through checkpoints.
- **Vocabulary (CONTEXT.md):** use **Fable**, **Scaffold**, **Slot Prompt**, **SFT**, **Reward Model**, **Judge**, **RLAIF** (never "RLHF") exactly. Introduce "Reward Model" before abbreviating; never write bare "RM" or "scorer" in prose.
- **Preference-Pair schema v1 (`docs/schemas/preference-pair-v1.md`, `preferences.py`):** each pairs `.jsonl` line has `schema_version`, `scaffold` (six slots), `chosen`, `rejected`, `verdict`. Decode every line through `tinystories_v2.preferences.validate_preference_pair` before training — never accept another schema version or reconstruct discarded pairs.
- **Slot Prompt format contract (issue 12, `slot_prompt.py`):** score a Fable on its full sequence `<|character|>…<|moral|><|fable|>{body}<|end|>` built by `render_example`. `SLOT_FIELDS = ("character","trait","setting","conflict","resolution","moral")`. Do not re-derive or reorder these.
- **Bradley-Terry loss (ADR-0005, hand-written):** `-log σ(r_chosen − r_rejected)` averaged over the batch; no library RM trainer.
- **Accuracy gate default ~0.68** (design): a Reward Model below the gate means the fix is better Judge labels, not RL. The stage *records* accuracy; downstream RL *enforces* the gate via the shared `gate.check_reward_gate`.
- **Secrets never printed:** `.env` values (HF/W&B tokens) are loaded via `config.load_env` and never logged.
- **Colab notebooks stay thin:** setup + a single stage invocation only; no `def`, `class`, `import torch`, `for`, or `while` in notebook source (enforced by `tests/test_notebook.py`).
- **Version floors:** `requires-python >=3.11`, `torch>=2.6` (already pinned in `pyproject.toml`); do not add dependencies.
- **Tests are CPU-only, seconds each, no network or GPU.** Real code paths at toy scale; no mocking of our own code.

---

## File Structure

**Create:**
- `src/tinystories_v2/reward_model.py` — the `RewardModel` nn.Module (FableLM backbone + scalar head), `pad_sequences`, `score_sequences`, `score_fables`, `bradley_terry_loss`, `pair_accuracy`. The reusable model/scoring library, consumed by the reward stage *and* issue 06's GRPO.
- `src/tinystories_v2/reward.py` — the Reward Model training stage: `load_pairs`, `encode_pairs`, `split_pairs`, `get_pair_batch`, `evaluate_accuracy`, `_init_backbone_from_sft`, `run`, `main`. Entrypoint `ts2-reward`.
- `src/tinystories_v2/gate.py` — the shared accuracy gate: `DEFAULT_ACCURACY_GATE`, `RewardGateError`, `load_reward_manifest`, `check_reward_gate`.
- `configs/reward_fixture.toml` — toy CPU wiring config.
- `configs/reward_full.toml` — real Colab Reward Model config (design-doc defaults).
- `src/tinystories_v2/hub_download.py` — shared retry-wrapped single-file Hub download (`retry`, `download_file`) used by the Colab bootstrap scripts; extracted from `sft_colab.py` so the reward (and later GRPO) bootstrap does not copy it a third time.
- `scripts/reward_colab.py` — one-command Colab bootstrap for the real Reward Model run (download tokenizer + preference pairs, then `ts2-reward --resume`), adapting `scripts/sft_colab.py` per `docs/colab-notes.md`.
- `notebooks/reward_colab.ipynb` — thin Colab wrapper (documented parallel path; the real run uses `scripts/reward_colab.py`).
- `docs/schemas/reward-model-artifact-v1.md` — the manifest metadata + gate contract.
- `tests/test_model_hidden_states.py` — the `FableLM.hidden_states` seam (shape + forward-unchanged).
- `tests/test_reward_model.py` — `RewardModel` scoring: shape `[B]`, batched, padding-invariance, backbone loads SFT weights strictly, Bradley-Terry + accuracy primitives.
- `tests/test_reward_batch.py` — `load_pairs`, `encode_pairs`, `split_pairs` determinism, `get_pair_batch` purity/padding.
- `tests/test_reward_stage.py` — stage: held-out accuracy above chance on synthetically separable fake-Judge pairs; artifacts/manifest record accuracy + split recipe; scoring usable downstream; init-from-real-checkpoint; arch-mismatch raises; CLI.
- `tests/test_reward_resume.py` — kill-and-resume bitwise-identical contract.
- `tests/test_reward_gate.py` — gate: below-gate raises with a clear message; above-gate returns accuracy; non-Reward-Model / missing-accuracy manifests raise.
- `tests/test_reward_colab.py` — the bootstrap orchestration (download → `ts2-reward --resume`) with an injected download; wiring + idempotence + `resume=True`.

**Modify:**
- `src/tinystories_v2/model.py` — extract `FableLM.hidden_states`; `forward` calls it (behavior identical).
- `pyproject.toml` — add the `ts2-reward` console-script entrypoint.
- `scripts/sft_colab.py` — replace its local `retry` + `download_file` with an import from the new `hub_download` module (behavior identical; `tests/test_sft_colab.py` stays green).
- `tests/test_notebook.py` — add thin-wrapper + no-secrets tests for `reward_colab.ipynb`.
- `PROGRESS.md` — mark issue 05 code-complete and unblock issue 06 (final task).

**Read-only references (do not modify):** `sft.py` (stage template), `pretrain.py` (imports `lr_at`, `build_optimizer`), `preferences.py`, `judge.py`, `slot_prompt.py`, `slots.py`, `checkpoint.py`, `tracking.py`, `hub.py`, `config.py`, `generate.py`, `tests/conftest.py` (reuses `make_init_checkpoint`), `scripts/sft_colab.py` + `tests/test_sft_colab.py` (bootstrap template), `docs/colab-notes.md` (real-run procedure).

## Colab Run Procedure (from `docs/colab-notes.md`)

The real Reward Model run is **not** driven from the notebook — it uses the one-command bootstrap over the `colab` CLI, exactly like the issue 03 SFT run. Before running: `git push origin main` (the VM clones from GitHub — local-only commits are missing), then `colab upload .env /content/tinystories_v2/.env` (never pass tokens in an `exec`). Run in-kernel via `colab exec -f scripts/reward_colab.py` (never nohup-detach — idle VMs get reaped); background long commands with a log + `EXIT_` marker and poll in <20 s exec calls; wrap CLI calls in a 3–6 try retry loop. `--resume` is idempotent: after an L4 preemption, re-running the bootstrap pulls the last Hub checkpoint and continues. The Hub is the source of truth (list `checkpoints/step_*.pt` + read `manifest.json`), not the VM. Always `colab stop -s <name>` when done. This procedure is documentation for the executor — it is not exercised by the test suite (which is CPU-only, no network).

---

### Task 1: `FableLM.hidden_states` seam

The Reward Model needs the backbone's `[B, T, d_model]` hidden states (pre-LM-head) so it can attach a scalar head to the same transformer (ADR-0005). Extract those states into a method and have `forward` call it — a pure, behavior-preserving refactor.

**Files:**
- Modify: `src/tinystories_v2/model.py:134-142` (the `forward` method)
- Test: `tests/test_model_hidden_states.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (relied on by Task 2): `FableLM.hidden_states(idx: Tensor) -> Tensor` of shape `[B, T, d_model]` — token embeddings through the final RMSNorm. `FableLM.forward(idx)` is unchanged for all inputs.

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_hidden_states.py`:

```python
import torch

from tinystories_v2.model import FableLM, ModelConfig

CONFIG = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2,
                     context=16, ffn_hidden=64)


def _model():
    torch.manual_seed(0)
    return FableLM(CONFIG).eval()


def test_hidden_states_shape():
    model = _model()
    idx = torch.randint(0, CONFIG.vocab_size, (3, 10))
    hidden = model.hidden_states(idx)
    assert hidden.shape == (3, 10, CONFIG.d_model)


def test_forward_equals_lm_head_of_hidden_states():
    model = _model()
    idx = torch.randint(0, CONFIG.vocab_size, (2, 8))
    with torch.no_grad():
        expected = model.lm_head(model.hidden_states(idx))
        assert torch.equal(model(idx), expected)


def test_hidden_states_respects_context_limit():
    model = _model()
    idx = torch.randint(0, CONFIG.vocab_size, (1, CONFIG.context + 1))
    import pytest
    with pytest.raises(ValueError, match="exceeds context"):
        model.hidden_states(idx)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_model_hidden_states.py -q`
Expected: FAIL — `AttributeError: 'FableLM' object has no attribute 'hidden_states'`.

- [ ] **Step 3: Refactor `forward` in `src/tinystories_v2/model.py`**

Replace the current `forward` method (lines 134-142):

```python
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.size(1) > self.config.context:
            raise ValueError(
                f"sequence length {idx.size(1)} exceeds context {self.config.context}"
            )
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        return self.lm_head(self.final_norm(x))
```

with:

```python
    def hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """Token embeddings through the final RMSNorm: the [B, T, d_model] states
        the LM head reads. Exposed as a seam so the Reward Model (issue 05) can
        attach a scalar head to the same backbone (ADR-0005)."""
        if idx.size(1) > self.config.context:
            raise ValueError(
                f"sequence length {idx.size(1)} exceeds context {self.config.context}"
            )
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        return self.final_norm(x)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.hidden_states(idx))
```

- [ ] **Step 4: Run the new test and the existing model tests to verify no regression**

Run: `.venv/bin/python -m pytest tests/test_model_hidden_states.py tests/test_model.py -q`
Expected: PASS (new tests pass; every existing `test_model.py` test still passes — `forward` is unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/model.py tests/test_model_hidden_states.py
git commit -m "refactor: expose FableLM.hidden_states seam for the Reward Model"
```

---

### Task 2: Reward Model module — scoring + Bradley-Terry primitives

The reusable model library: `RewardModel` (FableLM backbone + scalar head, scoring at the last real token), padded batching, batched scoring, and the hand-written Bradley-Terry loss + accuracy. Pure and unit-testable; no training loop, no I/O.

**Files:**
- Create: `src/tinystories_v2/reward_model.py`
- Test: `tests/test_reward_model.py`

**Interfaces:**
- Consumes: `FableLM.hidden_states` (Task 1); `model.ModelConfig`; `slot_prompt.render_example`; `slots.Scaffold`; `tokenizers.Tokenizer`.
- Produces (relied on by Tasks 3-6):
  - `RewardModel(config: ModelConfig)` — nn.Module with `.config`, `.backbone` (a `FableLM`), `.score_head`.
  - `RewardModel.load_backbone_state_dict(backbone_state: dict) -> None` — strict load of an SFT/Pretraining `state["model"]` into the backbone.
  - `RewardModel.forward(idx: Tensor[B,T], lengths: Tensor[B]) -> Tensor[B]` — scalar reward per sequence.
  - `pad_sequences(sequences: list[list[int]], context: int, device: str) -> tuple[Tensor[B,W], Tensor[B]]` returning `(idx, lengths)` (right-padded with id 0, each truncated to `context`).
  - `score_sequences(model, sequences: list[list[int]], *, device="cpu", batch_size=64) -> Tensor[B]` — no-grad batched scoring; truncates to `model.config.context`.
  - `score_fables(model, tokenizer, items: list[tuple[Scaffold, str]], *, device="cpu") -> list[float]` — render each `(Scaffold, fable)` and score it.
  - `bradley_terry_loss(chosen_scores: Tensor, rejected_scores: Tensor) -> Tensor` — scalar `-log σ(Δ)` mean.
  - `pair_accuracy(chosen_scores: Tensor, rejected_scores: Tensor) -> Tensor` — scalar fraction with `r_chosen > r_rejected`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reward_model.py`:

```python
import torch
from tokenizers import Tokenizer

from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.reward_model import (
    RewardModel, bradley_terry_loss, pad_sequences, pair_accuracy,
    score_fables, score_sequences,
)
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run

CONFIG = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2,
                     context=32, ffn_hidden=64)


def _reward_model():
    torch.manual_seed(0)
    return RewardModel(CONFIG)


def test_forward_returns_one_scalar_per_sequence():
    model = _reward_model()
    idx = torch.randint(0, CONFIG.vocab_size, (5, 12))
    lengths = torch.tensor([12, 10, 8, 6, 4])
    scores = model(idx, lengths)
    assert scores.shape == (5,)
    assert scores.dtype == torch.float32


def test_pad_sequences_shapes_and_lengths():
    idx, lengths = pad_sequences([[1, 2, 3], [4, 5], [6]], context=32, device="cpu")
    assert idx.shape == (3, 3)  # padded to the longest (3)
    assert idx[1].tolist() == [4, 5, 0]  # right-padded with 0
    assert lengths.tolist() == [3, 2, 1]


def test_pad_sequences_truncates_to_context():
    idx, lengths = pad_sequences([list(range(50))], context=8, device="cpu")
    assert idx.shape == (1, 8)
    assert lengths.tolist() == [8]


def test_score_is_padding_invariant():
    # A sequence scores identically alone and inside a longer padded batch:
    # last-real-token pooling + causal attention ignore right-padding.
    model = _reward_model()
    short = [3, 7, 1, 9]
    long = [2, 2, 5, 8, 4, 6, 1]
    alone = score_sequences(model, [short], device="cpu")
    batched = score_sequences(model, [short, long], device="cpu")
    assert torch.allclose(alone[0], batched[0], atol=1e-6)


def test_score_sequences_is_batched():
    model = _reward_model()
    scores = score_sequences(model, [[1, 2], [3, 4, 5], [6]], device="cpu")
    assert scores.shape == (3,)


def test_bradley_terry_loss_and_accuracy():
    chosen = torch.tensor([2.0, 1.0, 0.5])
    rejected = torch.tensor([0.0, 3.0, 0.5])
    # Manual BT: -mean(log σ(chosen - rejected)).
    expected = -torch.nn.functional.logsigmoid(chosen - rejected).mean()
    assert torch.allclose(bradley_terry_loss(chosen, rejected), expected)
    # Accuracy counts strictly-greater: pair 0 wins, pair 1 loses, pair 2 ties (not >).
    assert pair_accuracy(chosen, rejected).item() == 1 / 3


def test_load_backbone_state_dict_strict(tmp_path, fixture_path):
    # A RewardModel loads a FableLM state dict into its backbone strictly; the
    # scalar head keeps its fresh init.
    torch.manual_seed(1)
    sft_like = FableLM(CONFIG)
    model = _reward_model()
    model.load_backbone_state_dict(sft_like.state_dict())
    for key, tensor in sft_like.state_dict().items():
        assert torch.equal(model.backbone.state_dict()[key], tensor), key


def test_score_fables_returns_one_float_each(tmp_path, fixture_path):
    tok_dir = tmp_path / "tok"
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    model = RewardModel(ModelConfig(vocab_size=512, d_model=32, n_layers=2,
                                    n_heads=2, context=128, ffn_hidden=64))
    scaffold = Scaffold("fox", "sly", "a wood", "a gate", "it waited", "patience wins")
    scores = score_fables(model, tokenizer,
                          [(scaffold, "The sly fox waited."),
                           (scaffold, "A different tale entirely.")],
                          device="cpu")
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_reward_model.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tinystories_v2.reward_model'`.

- [ ] **Step 3: Write the implementation**

Create `src/tinystories_v2/reward_model.py`:

```python
"""Reward Model: the SFT backbone with a scalar head, plus the scoring and
Bradley-Terry primitives the reward stage (issue 05) and GRPO (issue 06) share.

The Reward Model reuses FableLM's transformer backbone (ADR-0005) and replaces
the tied LM head with one linear scalar head. A sequence's reward is the scalar
read from the hidden state at its last real (non-pad) token; right-padding plus
causal attention make that position independent of padding, so a Fable scores
identically alone or inside a padded batch.
"""

import torch
from tokenizers import Tokenizer
from torch import nn

from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.slot_prompt import render_example
from tinystories_v2.slots import Scaffold


class RewardModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.backbone = FableLM(config)
        self.score_head = nn.Linear(config.d_model, 1, bias=False)
        nn.init.normal_(self.score_head.weight, mean=0.0, std=0.02)

    def load_backbone_state_dict(self, backbone_state: dict) -> None:
        """Load an SFT/Pretraining state['model'] into the backbone (strict). The
        scalar head keeps its fresh init — it has no pretrained counterpart."""
        self.backbone.load_state_dict(backbone_state)

    def forward(self, idx: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Scalar reward per sequence. idx: [B, T] right-padded token ids;
        lengths: [B] real-token count per row. Returns [B]. Pools the hidden
        state at each row's last real token (lengths-1)."""
        hidden = self.backbone.hidden_states(idx)          # [B, T, d_model]
        last = (lengths - 1).clamp(min=0)
        rows = torch.arange(hidden.size(0), device=hidden.device)
        pooled = hidden[rows, last]                         # [B, d_model]
        return self.score_head(pooled).squeeze(-1)         # [B]


def pad_sequences(sequences: list[list[int]], context: int,
                  device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad token-id lists to a rectangular batch. Each sequence is first
    truncated to `context` (the backbone rejects longer inputs). Returns
    (idx [B, W] long, lengths [B] long)."""
    seqs = [s[:context] for s in sequences]
    lengths = [len(s) for s in seqs]
    width = max(lengths)
    padded = [s + [0] * (width - len(s)) for s in seqs]
    return (torch.tensor(padded, dtype=torch.long, device=device),
            torch.tensor(lengths, dtype=torch.long, device=device))


@torch.no_grad()
def score_sequences(model: RewardModel, sequences: list[list[int]], *,
                    device: str = "cpu", batch_size: int = 64) -> torch.Tensor:
    """Batched no-grad scoring of token-id sequences. Returns [len(sequences)]."""
    model = model.to(device).eval()
    context = model.config.context
    chunks = []
    for start in range(0, len(sequences), batch_size):
        idx, lengths = pad_sequences(sequences[start:start + batch_size],
                                     context, device)
        chunks.append(model(idx, lengths))
    return torch.cat(chunks) if chunks else torch.empty(0, device=device)


def score_fables(model: RewardModel, tokenizer: Tokenizer,
                 items: list[tuple[Scaffold, str]], *,
                 device: str = "cpu") -> list[float]:
    """Score each (Slot Prompt Scaffold, Fable body) on its full rendered
    sequence. The downstream scoring call (GRPO, eval, demos)."""
    sequences = [tokenizer.encode(render_example(scaffold, fable)).ids
                 for scaffold, fable in items]
    return score_sequences(model, sequences, device=device).tolist()


def bradley_terry_loss(chosen_scores: torch.Tensor,
                       rejected_scores: torch.Tensor) -> torch.Tensor:
    """-log σ(r_chosen - r_rejected), averaged. Minimized when the model scores
    every chosen Fable above its rejected partner (ADR-0005, hand-written)."""
    return -torch.nn.functional.logsigmoid(chosen_scores - rejected_scores).mean()


def pair_accuracy(chosen_scores: torch.Tensor,
                  rejected_scores: torch.Tensor) -> torch.Tensor:
    """Fraction of pairs with r_chosen > r_rejected (chance = 0.5)."""
    return (chosen_scores > rejected_scores).float().mean()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_reward_model.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/reward_model.py tests/test_reward_model.py
git commit -m "feat: RewardModel scalar head, scoring, and Bradley-Terry primitives"
```

---

### Task 3: Preference-pair loading, deterministic split, and batching

The stage's data plumbing: read + validate a preference-pair `.jsonl`, precompute chosen/rejected token-id sequences, split deterministically into train/holdout, and build a padded pair micro-batch that is a pure function of `(seed, step, micro_step)`. Pure, unit-testable functions with no training loop.

**Files:**
- Create: `src/tinystories_v2/reward.py` (helpers only in this task; `run`/`main` land in Task 4)
- Test: `tests/test_reward_batch.py`

**Interfaces:**
- Consumes: `preferences.validate_preference_pair`; `slot_prompt.encode_example`; `reward_model.pad_sequences`; the Preference-Pair schema.
- Produces (relied on by Tasks 4, 5):
  - `load_pairs(path: str | Path) -> list[PreferencePair]` — validated pairs.
  - `encode_pairs(tokenizer, pairs) -> list[dict]` — each dict `{"chosen_ids": list[int], "rejected_ids": list[int]}`.
  - `split_pairs(encoded: list, holdout_frac: float, seed: int) -> tuple[list, list]` returning `(train, holdout)`; a pure function of `(len(encoded), holdout_frac, seed)`.
  - `get_pair_batch(train: list, micro_batch_size: int, context: int, *, seed, step, micro_step, device="cpu") -> tuple[Tensor, Tensor, Tensor, Tensor]` returning `(chosen_idx, chosen_len, rejected_idx, rejected_len)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reward_batch.py`:

```python
import json

import torch
from tokenizers import Tokenizer

from tinystories_v2.preferences import PreferencePair, VerdictMetadata
from tinystories_v2.reward import encode_pairs, get_pair_batch, load_pairs, split_pairs
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run


def _pair(chosen: str, rejected: str) -> PreferencePair:
    scaffold = Scaffold("fox", "sly", "a wood", "a gate", "it waited", "patience wins")
    return PreferencePair(
        scaffold=scaffold, chosen=chosen, rejected=rejected,
        verdict=VerdictMetadata(judge_id="fake:slot-coverage-v1",
                                first_pass="A", swapped_pass="B", consistent=True))


def test_load_pairs_validates_schema(tmp_path):
    path = tmp_path / "pairs.jsonl"
    records = [_pair("A good fable.", "bad").to_dict(),
               _pair("Another good one.", "meh").to_dict()]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n  \n",  # blank ignored
                    encoding="utf-8")
    pairs = load_pairs(path)
    assert len(pairs) == 2
    assert pairs[0].chosen == "A good fable."


def test_load_pairs_rejects_bad_schema(tmp_path):
    import pytest

    from tinystories_v2.preferences import PreferencePairValidationError
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps({"schema_version": 1, "bogus": True}) + "\n",
                    encoding="utf-8")
    with pytest.raises(PreferencePairValidationError):
        load_pairs(path)


def _encoded(tmp_path, fixture_path):
    tok_dir = tmp_path / "tok"
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    pairs = [_pair(f"Chosen fable number {i}.", "A plain note.") for i in range(20)]
    return encode_pairs(tokenizer, pairs)


def test_encode_pairs_produces_id_lists(tmp_path, fixture_path):
    encoded = _encoded(tmp_path, fixture_path)
    assert len(encoded) == 20
    assert encoded[0]["chosen_ids"] and encoded[0]["rejected_ids"]
    assert all(isinstance(i, int) for i in encoded[0]["chosen_ids"])


def test_split_is_deterministic_and_disjoint(tmp_path, fixture_path):
    encoded = _encoded(tmp_path, fixture_path)
    train_a, holdout_a = split_pairs(encoded, holdout_frac=0.25, seed=7)
    train_b, holdout_b = split_pairs(encoded, holdout_frac=0.25, seed=7)
    assert len(holdout_a) == 5 and len(train_a) == 15
    assert holdout_a == holdout_b and train_a == train_b       # pure function of seed
    other_seed = split_pairs(encoded, holdout_frac=0.25, seed=99)[1]
    assert other_seed != holdout_a                             # seed actually shuffles
    # Train and holdout are disjoint (compare by chosen_ids identity).
    train_ids = [tuple(p["chosen_ids"]) for p in train_a]
    holdout_ids = [tuple(p["chosen_ids"]) for p in holdout_a]
    assert set(train_ids).isdisjoint(holdout_ids)


def test_get_pair_batch_is_pure_and_padded(tmp_path, fixture_path):
    encoded = _encoded(tmp_path, fixture_path)
    a = get_pair_batch(encoded, 4, 64, seed=1337, step=2, micro_step=0)
    b = get_pair_batch(encoded, 4, 64, seed=1337, step=2, micro_step=0)
    for t1, t2 in zip(a, b):
        assert torch.equal(t1, t2)                             # pure in (seed, step, micro_step)
    c_idx, c_len, r_idx, r_len = a
    assert c_idx.shape[0] == c_len.shape[0] == 4
    assert r_idx.shape[0] == r_len.shape[0] == 4
    assert c_idx.shape[1] == int(c_len.max())                  # padded to longest real length
    different = get_pair_batch(encoded, 4, 64, seed=1337, step=2, micro_step=1)
    assert not all(torch.equal(t1, t2) for t1, t2 in zip(a, different))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_reward_batch.py -q`
Expected: FAIL — `ImportError: cannot import name 'load_pairs' from 'tinystories_v2.reward'` (module does not exist).

- [ ] **Step 3: Write the implementation (helpers only)**

Create `src/tinystories_v2/reward.py` with the module docstring, imports, and the four helpers (the `run`/`main` entrypoint is added in Task 4):

```python
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
from tinystories_v2.gate import DEFAULT_ACCURACY_GATE
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_reward_batch.py -q`
Expected: PASS (5 tests). (`evaluate_accuracy` and the extra imports are exercised by Task 4; they are added now so `reward.py` is complete and imports resolve.)

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/reward.py tests/test_reward_batch.py
git commit -m "feat: reward-stage pair loading, deterministic split, and batching"
```

---

### Task 4: Reward Model trainer stage (run + entrypoint + configs + gate stub + schema)

Wire the helpers into a full stage: init the backbone from an SFT checkpoint, run the Bradley-Terry training loop reusing pretrain's optimizer/schedule/precision/logging/checkpoint-resume, evaluate held-out accuracy, and write the stage artifacts with the accuracy + split recipe recorded. Add both configs, the `ts2-reward` entrypoint, the `gate.py` module (needed by the `run` import and the manifest contract), and the artifact schema doc.

**Files:**
- Modify: `src/tinystories_v2/reward.py` (append `_init_backbone_from_sft`, `run`, `main`)
- Create: `src/tinystories_v2/gate.py`
- Create: `configs/reward_fixture.toml`, `configs/reward_full.toml`
- Create: `docs/schemas/reward-model-artifact-v1.md`
- Modify: `pyproject.toml` (add `ts2-reward` entrypoint)
- Test: `tests/test_reward_stage.py`

**Interfaces:**
- Consumes: `load_pairs`, `encode_pairs`, `split_pairs`, `get_pair_batch`, `evaluate_accuracy` (Task 3); `RewardModel`, `bradley_terry_loss`, `pair_accuracy`, `score_fables` (Task 2); `lr_at`, `build_optimizer` (pretrain); `gate.DEFAULT_ACCURACY_GATE`; `make_init_checkpoint` (conftest — a Pretraining-shaped checkpoint, structurally identical to an SFT checkpoint for backbone init).
- Produces (relied on by Tasks 5, 6):
  - `run(config: dict, resume: bool = False) -> dict` returning `{"step", "loss", "heldout_accuracy"}`; writes `checkpoints/step_*.pt`, `metrics.jsonl`, `manifest.json`.
  - `_init_backbone_from_sft(config: dict, device: str) -> RewardModel`.
  - Config shape: top-level `out_dir`; `[model]` (matches the SFT architecture); `[data] pairs_path, tokenizer_path`; `[init] local_dir` (+ optional `hub_source`); `[split] holdout_frac, seed`; `[train]` (same keys as sft); optional `[wandb]`, `[hub] target`.
  - Manifest: `stage="reward_model"`, `heldout_accuracy: float`, `pair_split: {seed, holdout_frac, n_pairs, n_train, n_holdout}`, `pairs_path`, `n_pairs`, `final_step`, `final_loss`, `config`.
  - Checkpoint state schema: `{"step", "pairs_seen", "model", "optimizer", "scaler", "config"}`.

- [ ] **Step 1: Create `src/tinystories_v2/gate.py`**

(Needed now: `reward.py` imports `DEFAULT_ACCURACY_GATE`, and the stage manifest is the gate's input contract. The gate's own tests come in Task 6.)

```python
"""Shared Reward Model accuracy gate (issue 05).

RLAIF refuses to start against a Reward Model whose held-out pair accuracy is
below the gate: below it, the fix is better Judge labels, not RL — a policy
optimized against a near-chance reward learns noise. GRPO (issue 06) calls
check_reward_gate at startup; the reward stage records the accuracy this reads.
"""

import json
import math
from pathlib import Path

DEFAULT_ACCURACY_GATE = 0.68


class RewardGateError(RuntimeError):
    """Raised when a Reward Model artifact is missing, malformed, or below the
    accuracy gate."""


def load_reward_manifest(reward_dir: str | Path) -> dict:
    """Read and sanity-check a Reward Model artifact's manifest.json."""
    path = Path(reward_dir) / "manifest.json"
    if not path.exists():
        raise RewardGateError(f"no Reward Model manifest at {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "reward_model":
        raise RewardGateError(
            f"{path} is not a Reward Model artifact (stage="
            f"{manifest.get('stage')!r})")
    if "heldout_accuracy" not in manifest:
        raise RewardGateError(
            f"{path} records no heldout_accuracy; re-run the reward stage")
    return manifest


def check_reward_gate(reward_dir: str | Path,
                      gate: float = DEFAULT_ACCURACY_GATE) -> float:
    """Return the Reward Model's held-out accuracy, or raise RewardGateError if it
    is undefined or below `gate`. Downstream RLAIF calls this before training."""
    accuracy = load_reward_manifest(reward_dir)["heldout_accuracy"]
    if accuracy is None or (isinstance(accuracy, float) and math.isnan(accuracy)):
        raise RewardGateError(
            "Reward Model held-out accuracy is undefined (NaN); the holdout "
            "split was empty — lower [split].holdout_frac or add more pairs")
    if accuracy < gate:
        raise RewardGateError(
            f"Reward Model held-out accuracy {accuracy:.3f} is below the gate "
            f"{gate:.2f}: improve Judge labels before RL, do not optimize a "
            f"policy against a near-chance reward")
    return accuracy
```

- [ ] **Step 2: Write the failing stage tests**

Create `tests/test_reward_stage.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from tinystories_v2.checkpoint import latest_checkpoint, load_checkpoint
from tinystories_v2.data import run as data_run
from tinystories_v2.judge import SlotCoverageFakeJudge, judge_with_order_swap
from tinystories_v2.model import ModelConfig
from tinystories_v2.pretrain import run as pretrain_run
from tinystories_v2.reward import run as reward_run
from tinystories_v2.reward_model import RewardModel, score_fables
from tinystories_v2.slot_prompt import SLOT_FIELDS
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run

TOY_MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 128, "ffn_hidden": 192}

# Bland rejected bodies mention no slot value, so SlotCoverageFakeJudge always
# prefers the slot-rich chosen body: the pairs are synthetically separable.
_BLAND = ["A plain note with nothing much to say.",
          "Some words that go nowhere in particular."]


def _separable_pairs(rows):
    """Build order-swap-consistent fake-Judge pairs where chosen mentions every
    slot value and rejected is bland — a learnable, separable signal."""
    judge = SlotCoverageFakeJudge()
    pairs = []
    for i, row in enumerate(rows):
        scaffold = Scaffold(**{f: row[f] for f in SLOT_FIELDS})
        chosen = (f"{scaffold.character}, a {scaffold.trait} one in "
                  f"{scaffold.setting}, met {scaffold.conflict}. "
                  f"{scaffold.resolution}. The moral: {scaffold.moral}.")
        rejected = _BLAND[i % len(_BLAND)]
        pair = judge_with_order_swap(judge, scaffold, chosen, rejected)
        assert pair is not None and pair.chosen == chosen  # judge picked the rich body
        pairs.append(pair)
    return pairs


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, fixture_path):
    """Prepare a tokenizer and a separable fake-Judge pairs.jsonl from the
    fixture's sft split (stages couple via artifacts)."""
    base = tmp_path_factory.mktemp("reward_stage_inputs")
    data_dir, tok_dir = base / "data", base / "tok"
    data_run({
        "out_dir": str(data_dir), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.1, "sft": 0.7,
                   "pref": 0.1, "eval": 0.1},
    })
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer = str(tok_dir / "tokenizer.json")
    with open(data_dir / "splits" / "sft.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    pairs = _separable_pairs(rows)
    pairs_path = base / "pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
    return {"pairs_path": str(pairs_path), "tokenizer": tokenizer, "n_pairs": len(pairs)}


def reward_toy_config(out_dir, prepared, init_dir, model=None, **train_overrides) -> dict:
    train = {
        "steps": 60, "micro_batch_size": 8, "grad_accum": 1,
        "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
        "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
        "precision": "fp32", "seed": 1337,
        "checkpoint_every": 20, "log_every": 1, "keep_last": 0,
    }
    train.update(train_overrides)
    return {
        "out_dir": str(out_dir),
        "model": dict(model or TOY_MODEL),
        "data": {"pairs_path": prepared["pairs_path"],
                 "tokenizer_path": prepared["tokenizer"]},
        "init": {"local_dir": str(init_dir)},
        "split": {"holdout_frac": 0.25, "seed": 20260712},
        "train": train,
        "wandb": {"enabled": False},
    }


def read_metrics(out_dir) -> list[dict]:
    lines = (Path(out_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_toy_reward_model_beats_chance_on_heldout(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = reward_toy_config(tmp_path / "out", prepared, init)
    summary = reward_run(config)
    assert summary["heldout_accuracy"] > 0.8   # well above chance (0.5) on separable pairs
    metrics = read_metrics(config["out_dir"])
    assert len(metrics) == 60
    assert {"step", "loss", "lr", "accuracy", "pairs_seen"} <= metrics[0].keys()
    assert metrics[-1]["loss"] < metrics[0]["loss"]   # Bradley-Terry loss fell


def test_manifest_records_accuracy_and_split_recipe(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = reward_toy_config(tmp_path / "out", prepared, init, steps=6,
                              checkpoint_every=3)
    reward_run(config)
    manifest = json.loads(
        (Path(config["out_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "reward_model"
    assert isinstance(manifest["heldout_accuracy"], float)
    split = manifest["pair_split"]
    assert split["seed"] == 20260712 and split["holdout_frac"] == 0.25
    assert split["n_pairs"] == prepared["n_pairs"]
    assert split["n_train"] + split["n_holdout"] == split["n_pairs"]
    assert split["n_holdout"] == round(prepared["n_pairs"] * 0.25)
    assert manifest["pairs_path"] == prepared["pairs_path"]


def test_scores_are_usable_downstream(tmp_path, prepared, make_init_checkpoint):
    # Criterion 2: a scoring call takes (Slot Prompt Scaffold, Fable) -> scalar,
    # batched, on CPU, from a trained artifact.
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = reward_toy_config(tmp_path / "out", prepared, init, steps=6,
                              checkpoint_every=6)
    reward_run(config)
    state = load_checkpoint(latest_checkpoint(Path(config["out_dir"]) / "checkpoints"))
    model = RewardModel(ModelConfig(**state["config"]["model"]))
    model.load_state_dict(state["model"])
    tokenizer = Tokenizer.from_file(prepared["tokenizer"])
    scaffold = Scaffold("fox", "sly", "a wood", "a gate", "it waited", "patience wins")
    scores = score_fables(model, tokenizer,
                          [(scaffold, "The sly fox waited by the gate."),
                           (scaffold, "A plain note with nothing much to say.")],
                          device="cpu")
    assert len(scores) == 2 and all(isinstance(s, float) for s in scores)


def test_split_recipe_is_reproducible(tmp_path, prepared, make_init_checkpoint):
    # Same split seed -> identical held-out accuracy across two fresh runs.
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    a = reward_run(reward_toy_config(tmp_path / "a", prepared, init, steps=6,
                                     checkpoint_every=6))
    b = reward_run(reward_toy_config(tmp_path / "b", prepared, init, steps=6,
                                     checkpoint_every=6))
    assert a["heldout_accuracy"] == b["heldout_accuracy"]


def test_init_from_a_real_checkpoint(tmp_path, prepared, fixture_path):
    # Init the backbone from a genuine checkpoint (a Pretraining checkpoint is
    # structurally identical to an SFT one for the backbone). Exercises the load
    # + architecture-match validation against a real artifact.
    model64 = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
               "n_heads": 2, "context": 64, "ffn_hidden": 192}
    pre_dir = tmp_path / "pretrain"
    pretrain_run({
        "out_dir": str(pre_dir), "model": dict(model64),
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
    config = reward_toy_config(tmp_path / "reward_out", prepared, pre_dir,
                               model=model64, steps=3, checkpoint_every=3)
    summary = reward_run(config)
    import math
    assert summary["step"] == 3 and math.isfinite(summary["loss"])


def test_mismatched_init_architecture_raises(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    drifted = dict(TOY_MODEL, d_model=128)  # differs from the init checkpoint
    config = reward_toy_config(tmp_path / "out", prepared, init, model=drifted,
                              steps=2, checkpoint_every=2)
    with pytest.raises(ValueError, match="SFT checkpoint"):
        reward_run(config)


def test_missing_init_checkpoint_raises(tmp_path, prepared):
    config = reward_toy_config(tmp_path / "out", prepared, tmp_path / "empty_init",
                              steps=2, checkpoint_every=2)
    with pytest.raises(ValueError, match="no SFT checkpoint"):
        reward_run(config)


def to_toml(config: dict) -> str:
    """Serialize the nested reward config as TOML (stdlib has no writer)."""
    lines = [f'out_dir = "{config["out_dir"]}"']
    for section in ("model", "data", "init", "split", "train", "wandb", "hub"):
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
    config = reward_toy_config(tmp_path / "out", prepared, init, steps=2,
                              checkpoint_every=2)
    config_file = tmp_path / "cfg.toml"
    config_file.write_text(to_toml(config), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.reward", "--config", str(config_file)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (Path(config["out_dir"]) / "checkpoints" / "step_000002.pt").exists()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_reward_stage.py -q`
Expected: FAIL — `ImportError: cannot import name 'run' from 'tinystories_v2.reward'`.

- [ ] **Step 4: Implement `_init_backbone_from_sft`, `run`, and `main` in `src/tinystories_v2/reward.py`**

Append to `src/tinystories_v2/reward.py`:

```python
def _init_backbone_from_sft(config: dict, device: str) -> RewardModel:
    """Fresh Reward Model start: build the model from [model], load SFT backbone
    weights into it, and validate the architecture matches. Fetches the init
    artifact from Hub first if the local checkpoint is absent (fresh VM)."""
    model = RewardModel(ModelConfig(**config["model"])).to(device)
    init = config["init"]
    init_dir = Path(init["local_dir"])
    init_ckpt_dir = init_dir / "checkpoints"
    if latest_checkpoint(init_ckpt_dir) is None and init.get("hub_source"):
        fetch_from(init["hub_source"], init_dir)  # fresh Colab VM: pull SFT
    init_ckpt = latest_checkpoint(init_ckpt_dir)
    if init_ckpt is None:
        raise ValueError(
            f"no SFT checkpoint under {init_ckpt_dir}; point [init].local_dir "
            f"(and optionally [init].hub_source) at the SFT artifact")
    state = load_checkpoint(init_ckpt)
    if ModelConfig(**state["config"]["model"]) != model.config:
        raise ValueError(
            f"[model] does not match the SFT checkpoint at {init_ckpt}; the "
            f"Reward Model must reuse the SFT architecture")
    model.load_backbone_state_dict(state["model"])
    print(f"initialized Reward Model backbone from {init_ckpt}")
    return model


def run(config: dict, resume: bool = False) -> dict:
    load_env()  # W&B/HF keys before wandb.init or hub sync; never printed
    train = config["train"]
    out_dir = Path(config["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    hub_target = config.get("hub", {}).get("target")

    tokenizer = Tokenizer.from_file(config["data"]["tokenizer_path"])
    pairs = load_pairs(config["data"]["pairs_path"])
    if not pairs:
        raise ValueError(f"no preference pairs in {config['data']['pairs_path']}")
    encoded = encode_pairs(tokenizer, pairs)
    split = config["split"]
    train_pairs, holdout_pairs = split_pairs(encoded, split["holdout_frac"], split["seed"])
    if not train_pairs:
        raise ValueError("no training pairs after the holdout split; lower "
                         "[split].holdout_frac or add more pairs")

    # -- precision knob: fp32 | bf16 (autocast) | fp16 (autocast + GradScaler)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = train["precision"]
    amp_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    autocast = (torch.autocast(device_type=device, dtype=amp_dtype)
                if amp_dtype else nullcontext())
    scaler = torch.amp.GradScaler(device, enabled=(precision == "fp16"))

    torch.manual_seed(train["seed"])
    model = _init_backbone_from_sft(config, device)
    optimizer = build_optimizer(model, train["peak_lr"],
                                (train["beta1"], train["beta2"]),
                                train["weight_decay"])

    start_step, pairs_seen = 0, 0
    if resume:
        if latest_checkpoint(ckpt_dir) is None and hub_target:
            try:
                fetch_from(hub_target, out_dir)  # fresh VM: pull previous session
            except Exception as err:  # noqa: BLE001 — first run: repo may not exist yet
                warnings.warn(
                    f"could not fetch a prior Reward Model run from {hub_target!r}; "
                    f"starting fresh: {err}", stacklevel=2)
        ckpt_path = latest_checkpoint(ckpt_dir)
        if ckpt_path is not None:
            state = load_checkpoint(ckpt_path)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start_step, pairs_seen = state["step"], state["pairs_seen"]
            print(f"resumed from {ckpt_path.name} at step {start_step}")

    def checkpoint(step: int) -> None:
        save_checkpoint(ckpt_dir, step, {
            "step": step, "pairs_seen": pairs_seen,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "config": config,
        })
        prune_checkpoints(ckpt_dir, train.get("keep_last", 0))
        if hub_target:
            try_sync_to(hub_target, out_dir)

    logger = MetricsLogger(out_dir, config.get("wandb"))
    steps, accum = train["steps"], train["grad_accum"]
    micro_bs, context = train["micro_batch_size"], config["model"]["context"]
    loss_value, batch_acc = float("nan"), float("nan")
    model.train()
    for step in range(start_step, steps):
        lr = lr_at(step, steps, train["peak_lr"],
                   train["warmup_frac"], train["min_lr_frac"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(accum):
            c_idx, c_len, r_idx, r_len = get_pair_batch(
                train_pairs, micro_bs, context, seed=train["seed"],
                step=step, micro_step=micro_step, device=device)
            with autocast:
                chosen_scores = model(c_idx, c_len)
                rejected_scores = model(r_idx, r_len)
                loss = bradley_terry_loss(chosen_scores, rejected_scores)
            scaler.scale(loss / accum).backward()
            pairs_seen += micro_bs
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        loss_value = loss.item()          # last micro-batch Bradley-Terry loss
        batch_acc = pair_accuracy(chosen_scores.detach(),
                                  rejected_scores.detach()).item()
        done = step + 1
        if done % train["log_every"] == 0:
            logger.log({"loss": loss_value, "lr": lr, "accuracy": batch_acc,
                        "pairs_seen": pairs_seen}, step=done)
        if done % train["checkpoint_every"] == 0:
            checkpoint(done)
    if steps % train["checkpoint_every"] != 0:
        checkpoint(steps)

    heldout_accuracy = evaluate_accuracy(model, holdout_pairs, device=device)
    logger.finish()

    manifest = {
        "stage": "reward_model", "package_version": __version__,
        "final_step": steps, "final_loss": loss_value,
        "heldout_accuracy": heldout_accuracy,
        "pair_split": {"seed": split["seed"], "holdout_frac": split["holdout_frac"],
                       "n_pairs": len(encoded), "n_train": len(train_pairs),
                       "n_holdout": len(holdout_pairs)},
        "pairs_path": config["data"]["pairs_path"], "n_pairs": len(pairs),
        "config": config,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    if hub_target:
        try_sync_to(hub_target, out_dir)
    print(f"held-out pair accuracy: {heldout_accuracy:.3f} "
          f"(gate {DEFAULT_ACCURACY_GATE:.2f})")
    return {"step": steps, "loss": loss_value, "heldout_accuracy": heldout_accuracy}


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

- [ ] **Step 5: Add the `ts2-reward` entrypoint to `pyproject.toml`**

In the `[project.scripts]` table, after the `ts2-sft = ...` line, add:

```toml
ts2-reward = "tinystories_v2.reward:main"
```

- [ ] **Step 6: Create `configs/reward_fixture.toml`**

```toml
# Toy CPU wiring smoke against fixture artifacts — local sanity runs and docs.
# Assumes an SFT-shaped init checkpoint and a preference-pair jsonl exist with a
# matching [model] block. Real runs use configs/reward_full.toml. Stage behavior
# is guarded by tests/test_reward_*.py.
out_dir = "artifacts/reward_fixture"

# Must match the SFT checkpoint's architecture (sft_fixture.toml / pretrain_fixture.toml).
[model]
vocab_size = 512
d_model = 64
n_layers = 2
n_heads = 2
context = 64
ffn_hidden = 192

[data]
pairs_path = "artifacts/pref_fixture/pairs.jsonl"   # issue 04 preference artifact
tokenizer_path = "artifacts/tokenizer_fixture/tokenizer.json"

[init]
local_dir = "artifacts/sft_fixture"   # contains checkpoints/step_*.pt

[split]
holdout_frac = 0.2
seed = 20260712

[train]
steps = 60
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
checkpoint_every = 20
log_every = 1
keep_last = 0

[wandb]
enabled = false
```

- [ ] **Step 7: Create `configs/reward_full.toml`**

```toml
# Real Reward Model run on Colab Pro. bf16 on L4 (preferred); on a T4 fallback
# set precision = "fp16". Reward-Model LR is small (design default 1e-5) — the
# scalar head is tiny and the backbone is already SFT-tuned. Prerequisites on
# Hub/disk: issue 04's preference pairs + tokenizer_full, and the SFT checkpoint.
# [init].hub_source pulls the SFT artifact if the local copy is absent (fresh VM).
out_dir = "artifacts/reward_full"

[model]
vocab_size = 8192
d_model = 512
n_layers = 8
n_heads = 8
context = 512
ffn_hidden = 1408

[data]
pairs_path = "artifacts/pref_full/pairs.jsonl"   # issue 04 preference artifact
tokenizer_path = "artifacts/tokenizer_full/tokenizer.json"

[init]
local_dir = "artifacts/sft_full"
hub_source = "hf://congthanh991/tinystories-v2-sft"

[split]
holdout_frac = 0.1
seed = 20260712

[train]
steps = 400
micro_batch_size = 16
grad_accum = 4
peak_lr = 1e-5
warmup_frac = 0.03
min_lr_frac = 0.1
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
precision = "bf16"
seed = 1337
checkpoint_every = 50      # ~frequent: L4 sessions preempt ~hourly
log_every = 10
keep_last = 2

[wandb]
enabled = true
project = "tinystories-v2"
run_name = "reward"

[hub]
target = "hf://congthanh991/tinystories-v2-reward"
```

- [ ] **Step 8: Create `docs/schemas/reward-model-artifact-v1.md`**

```markdown
# Reward Model Artifact v1

Issue 05 pins the Reward Model stage's `out_dir` artifact: the checkpoints,
metrics, and the `manifest.json` metadata the accuracy gate reads.

## Artifact layout

```
<out_dir>/
  checkpoints/step_XXXXXX.pt   full training state (atomic; resume contract)
  metrics.jsonl                one line per log_every steps
  manifest.json                stage metadata (below)
```

Each `metrics.jsonl` line: `{"step", "loss", "lr", "accuracy", "pairs_seen"}`,
where `loss` is the batch Bradley-Terry loss and `accuracy` is the batch pair
accuracy (chance = 0.5).

## manifest.json

```json
{
  "stage": "reward_model",
  "package_version": "0.1.0",
  "final_step": 400,
  "final_loss": 0.31,
  "heldout_accuracy": 0.74,
  "pair_split": {
    "seed": 20260712,
    "holdout_frac": 0.1,
    "n_pairs": 10000,
    "n_train": 9000,
    "n_holdout": 1000
  },
  "pairs_path": "artifacts/pref_full/pairs.jsonl",
  "n_pairs": 10000,
  "config": { "...": "the full stage config" }
}
```

`heldout_accuracy` is the fraction of held-out pairs the trained Reward Model
scores `chosen > rejected`. The `pair_split` recipe makes the held-out slice
reproducible: the split is a seeded permutation of the encoded pairs with the
last `round(n_pairs * holdout_frac)` held out.

## Gate contract

Downstream RLAIF (issue 06 GRPO) calls
`tinystories_v2.gate.check_reward_gate(reward_dir, gate=0.68)` before training.
It returns `heldout_accuracy` or raises `RewardGateError` when the manifest is
missing/not a Reward Model artifact, `heldout_accuracy` is undefined (NaN, i.e.
an empty holdout), or `heldout_accuracy < gate`. Below the gate the fix is
better Judge labels, not RL.
```

- [ ] **Step 9: Run the stage tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_reward_stage.py -q`
Expected: PASS (9 tests). (If `test_toy_reward_model_beats_chance_on_heldout` is flaky below 0.8, raise `steps` — do not weaken the assertion; the pairs are strongly separable and a d64/2L model overfits them fast.)

- [ ] **Step 10: Commit**

```bash
git add src/tinystories_v2/reward.py src/tinystories_v2/gate.py \
        configs/reward_fixture.toml configs/reward_full.toml \
        docs/schemas/reward-model-artifact-v1.md pyproject.toml \
        tests/test_reward_stage.py
git commit -m "feat: Reward Model training stage, configs, gate module, and ts2-reward"
```

---

### Task 5: Kill-and-resume contract for the Reward Model stage

Prove the checkpoint-resume contract end to end: a SIGKILLed run resumes from its latest checkpoint and reproduces the uninterrupted run bitwise, with post-resume losses replaying exactly. Mirrors `tests/test_sft_resume.py`.

**Files:**
- Test: `tests/test_reward_resume.py`

**Interfaces:**
- Consumes: `reward.run`, the `python -m tinystories_v2.reward` entrypoint, `make_init_checkpoint` (conftest), `data.run`, `tokenizer.run`, the fake Judge, `checkpoint.latest_checkpoint`/`load_checkpoint`.

- [ ] **Step 1: Write the failing resume test**

Create `tests/test_reward_resume.py`:

```python
"""Kill-and-resume: the Reward Model checkpoint-resume contract, end to end.

Sized so the run is slow enough to SIGKILL mid-flight: d_model 128 / 4 layers /
ctx 128, 50 steps (each doing two forward passes — chosen and rejected),
checkpoint_every 5. Both runs share one init checkpoint and one pairs.jsonl, so
batches (a pure function of seed/step/micro_step) and the held-out split (a pure
function of the split seed) are identical.
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
from tinystories_v2.judge import SlotCoverageFakeJudge, judge_with_order_swap
from tinystories_v2.reward import run as reward_run
from tinystories_v2.slot_prompt import SLOT_FIELDS
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run

STEPS = 50
CHECKPOINT_EVERY = 5
KILL_AFTER_STEP = 10

MODEL = {"vocab_size": 512, "d_model": 128, "n_layers": 4,
         "n_heads": 4, "context": 128, "ffn_hidden": 384}

_BLAND = ["A plain note with nothing much to say.",
          "Some words that go nowhere in particular."]


def _write_pairs(path, rows):
    judge = SlotCoverageFakeJudge()
    with path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            scaffold = Scaffold(**{fld: row[fld] for fld in SLOT_FIELDS})
            chosen = (f"{scaffold.character}, a {scaffold.trait} one in "
                      f"{scaffold.setting}, met {scaffold.conflict}. "
                      f"{scaffold.resolution}. The moral: {scaffold.moral}.")
            pair = judge_with_order_swap(judge, scaffold, chosen, _BLAND[i % len(_BLAND)])
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")


def reward_config(out_dir, pairs_path, tokenizer_path, init_dir) -> dict:
    return {
        "out_dir": str(out_dir),
        "model": dict(MODEL),
        "data": {"pairs_path": str(pairs_path), "tokenizer_path": str(tokenizer_path)},
        "init": {"local_dir": str(init_dir)},
        "split": {"holdout_frac": 0.2, "seed": 20260712},
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
    for section in ("model", "data", "init", "split", "train", "wandb"):
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


def test_killed_reward_run_resumes_to_identical_final_state(
        tmp_path, fixture_path, make_init_checkpoint):
    # Build shared inputs once: a real pairs.jsonl and one init checkpoint.
    data_dir, tok_dir = tmp_path / "data", tmp_path / "tok"
    data_run({
        "out_dir": str(data_dir), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.1, "sft": 0.7,
                   "pref": 0.1, "eval": 0.1},
    })
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer_path = tok_dir / "tokenizer.json"
    with open(data_dir / "splits" / "sft.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    pairs_path = tmp_path / "pairs.jsonl"
    _write_pairs(pairs_path, rows)
    init_dir = make_init_checkpoint(tmp_path / "init", MODEL, tokenizer_path)

    # Reference: identical config, never interrupted.
    reference = reward_config(tmp_path / "reference", pairs_path, tokenizer_path, init_dir)
    reward_run(reference)

    # Interrupted: run as a subprocess and SIGKILL once the kill-marker appears.
    interrupted = reward_config(tmp_path / "interrupted", pairs_path,
                                tokenizer_path, init_dir)
    config_file = tmp_path / "interrupted.toml"
    config_file.write_text(to_toml(interrupted), encoding="utf-8")
    ckpt_dir = Path(interrupted["out_dir"]) / "checkpoints"
    kill_marker = ckpt_dir / f"step_{KILL_AFTER_STEP:06d}.pt"

    proc = subprocess.Popen(
        [sys.executable, "-m", "tinystories_v2.reward", "--config", str(config_file)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 120
        while not kill_marker.exists():
            if proc.poll() is not None:
                pytest.fail(
                    f"stage finished (rc={proc.returncode}) before the kill window; "
                    f"enlarge the toy model or lower KILL_AFTER_STEP")
            if time.monotonic() > deadline:
                pytest.fail("timed out waiting for the kill-marker checkpoint")
            time.sleep(0.01)
        os.kill(proc.pid, signal.SIGKILL)
    finally:
        proc.kill()
        proc.wait(timeout=30)

    killed_at = load_checkpoint(latest_checkpoint(ckpt_dir))["step"]
    assert KILL_AFTER_STEP <= killed_at < STEPS

    reward_run(interrupted, resume=True)

    final_ref = load_checkpoint(
        latest_checkpoint(Path(reference["out_dir"]) / "checkpoints"))
    final_res = load_checkpoint(latest_checkpoint(ckpt_dir))
    assert final_res["step"] == final_ref["step"] == STEPS
    assert final_res["pairs_seen"] == final_ref["pairs_seen"]
    for key, tensor in final_ref["model"].items():
        assert torch.equal(final_res["model"][key], tensor), key

    ref_losses = read_metrics(reference["out_dir"])
    res_losses = read_metrics(interrupted["out_dir"])
    for step in range(killed_at + 1, STEPS + 1):
        assert res_losses[step] == ref_losses[step], step
```

- [ ] **Step 2: Run the resume test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_reward_resume.py -q`
Expected: PASS. (If it reports the stage finished before the kill window, the toy model is too fast — enlarge the model or lower `KILL_AFTER_STEP`; do not weaken the bitwise assertion.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_reward_resume.py
git commit -m "test: Reward Model kill-and-resume reproduces the uninterrupted run bitwise"
```

---

### Task 6: Accuracy gate tests

Prove the gate seam issue 06 will enforce: a below-gate Reward Model artifact makes `check_reward_gate` refuse with a clear message; an above-gate artifact passes; malformed/missing manifests refuse. `gate.py` itself was created in Task 4 (the stage imports its default); this task adds its tests, including one against a genuine trained artifact.

**Files:**
- Test: `tests/test_reward_gate.py`

**Interfaces:**
- Consumes: `gate.check_reward_gate`, `gate.load_reward_manifest`, `gate.RewardGateError`, `gate.DEFAULT_ACCURACY_GATE`; `reward.run` (for the real-artifact integration case); `make_init_checkpoint`; the `prepared` fixture pattern from `test_reward_stage.py` (rebuilt locally here to keep the file self-contained).

- [ ] **Step 1: Write the failing gate tests**

Create `tests/test_reward_gate.py`:

```python
import json
from pathlib import Path

import pytest

from tinystories_v2.gate import (
    DEFAULT_ACCURACY_GATE, RewardGateError, check_reward_gate, load_reward_manifest,
)


def _write_manifest(dir_path: Path, **fields) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    manifest = {"stage": "reward_model", "heldout_accuracy": 0.75, **fields}
    (dir_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dir_path


def test_default_gate_matches_design():
    assert DEFAULT_ACCURACY_GATE == 0.68


def test_above_gate_returns_accuracy(tmp_path):
    rm = _write_manifest(tmp_path / "rm", heldout_accuracy=0.72)
    assert check_reward_gate(rm) == 0.72


def test_below_gate_raises_with_clear_message(tmp_path):
    rm = _write_manifest(tmp_path / "rm", heldout_accuracy=0.60)
    with pytest.raises(RewardGateError, match="below the gate"):
        check_reward_gate(rm)


def test_custom_gate_threshold(tmp_path):
    rm = _write_manifest(tmp_path / "rm", heldout_accuracy=0.72)
    assert check_reward_gate(rm, gate=0.70) == 0.72
    with pytest.raises(RewardGateError, match="below the gate"):
        check_reward_gate(rm, gate=0.80)


def test_nan_accuracy_raises(tmp_path):
    rm = _write_manifest(tmp_path / "rm", heldout_accuracy=float("nan"))
    with pytest.raises(RewardGateError, match="undefined"):
        check_reward_gate(rm)


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(RewardGateError, match="no Reward Model manifest"):
        check_reward_gate(tmp_path / "nope")


def test_non_reward_manifest_raises(tmp_path):
    rm = tmp_path / "rm"
    rm.mkdir()
    (rm / "manifest.json").write_text(json.dumps({"stage": "sft"}), encoding="utf-8")
    with pytest.raises(RewardGateError, match="not a Reward Model artifact"):
        load_reward_manifest(rm)


def test_gate_reads_a_real_trained_artifact(tmp_path, fixture_path, make_init_checkpoint):
    # End to end: train a toy Reward Model, then gate its real artifact. A
    # permissive gate passes; an impossible one refuses.
    import json as _json

    from tinystories_v2.data import run as data_run
    from tinystories_v2.judge import SlotCoverageFakeJudge, judge_with_order_swap
    from tinystories_v2.reward import run as reward_run
    from tinystories_v2.slot_prompt import SLOT_FIELDS
    from tinystories_v2.slots import Scaffold
    from tinystories_v2.tokenizer import run as tokenizer_run

    data_dir, tok_dir = tmp_path / "data", tmp_path / "tok"
    data_run({"out_dir": str(data_dir), "max_extraction_failures": 0,
              "source": {"kind": "jsonl", "path": str(fixture_path)},
              "splits": {"seed": "fixture-v1", "pretrain": 0.1, "sft": 0.7,
                         "pref": 0.1, "eval": 0.1}})
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer_path = str(tok_dir / "tokenizer.json")
    with open(data_dir / "splits" / "sft.jsonl", encoding="utf-8") as f:
        rows = [_json.loads(line) for line in f if line.strip()]
    judge = SlotCoverageFakeJudge()
    pairs_path = tmp_path / "pairs.jsonl"
    bland = ["A plain note with nothing much to say.",
             "Some words that go nowhere in particular."]
    with pairs_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            scaffold = Scaffold(**{fld: row[fld] for fld in SLOT_FIELDS})
            chosen = (f"{scaffold.character}, a {scaffold.trait} one in "
                      f"{scaffold.setting}, met {scaffold.conflict}. "
                      f"{scaffold.resolution}. The moral: {scaffold.moral}.")
            pair = judge_with_order_swap(judge, scaffold, chosen, bland[i % len(bland)])
            f.write(_json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

    model = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 128, "ffn_hidden": 192}
    init = make_init_checkpoint(tmp_path / "init", model, tokenizer_path)
    out_dir = tmp_path / "reward_out"
    reward_run({
        "out_dir": str(out_dir), "model": dict(model),
        "data": {"pairs_path": str(pairs_path), "tokenizer_path": tokenizer_path},
        "init": {"local_dir": str(init)},
        "split": {"holdout_frac": 0.25, "seed": 20260712},
        "train": {"steps": 60, "micro_batch_size": 8, "grad_accum": 1,
                  "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                  "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
                  "grad_clip": 1.0, "precision": "fp32", "seed": 1337,
                  "checkpoint_every": 60, "log_every": 20, "keep_last": 0},
        "wandb": {"enabled": False},
    })
    assert check_reward_gate(out_dir, gate=0.5) > 0.5     # separable pairs clear a low gate
    with pytest.raises(RewardGateError, match="below the gate"):
        check_reward_gate(out_dir, gate=1.01)             # nothing clears an impossible gate
```

- [ ] **Step 2: Run the gate tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_reward_gate.py -q`
Expected: PASS (8 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_reward_gate.py
git commit -m "test: Reward Model accuracy gate refuses below-gate artifacts"
```

---

### Task 7: Thin Colab notebook, notebook tests, and PROGRESS

The real-run wrapper and the docs touch-ups: a thin `reward_colab.ipynb`, its thinness/no-secrets tests, and the PROGRESS update marking issue 05 code-complete and unblocking issue 06.

**Files:**
- Create: `notebooks/reward_colab.ipynb`
- Modify: `tests/test_notebook.py`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: nothing new. `test_notebook.py` reuses its existing thin/no-secrets assertions against the new notebook path.

- [ ] **Step 1: Create `notebooks/reward_colab.ipynb`**

Write this exact JSON (thin wrapper — no `def`, `class`, `import torch`, `for`, or `while`; committed with empty outputs and no literal `hf_` token):

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Reward Model on Colab Pro (L4 preferred, T4 fallback)\n",
    "\n",
    "Thin wrapper per docs/DESIGN.md: clone → install → secrets → run stage.\n",
    "All logic lives in the package; edit `configs/reward_full.toml` in the repo, not here.\n",
    "\n",
    "The stage initializes the backbone from the SFT checkpoint: `[init].hub_source` in the\n",
    "config pulls it from the Hub automatically on a fresh VM. It trains on issue 04's\n",
    "preference pairs and records held-out pair accuracy in the artifact's manifest.json.\n",
    "\n",
    "For a CLI-driven run on a fresh VM, prefer the one-command bootstrap\n",
    "`python scripts/reward_colab.py` (it also downloads the tokenizer + preference pairs,\n",
    "which this notebook cell does not); see `docs/colab-notes.md` for the run procedure.\n",
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
    "!ts2-reward --config configs/reward_full.toml --resume"
   ]
  }
 ],
 "metadata": {"language_info": {"name": "python"}},
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Add the notebook tests to `tests/test_notebook.py`**

Append to `tests/test_notebook.py`:

```python
REWARD_NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "reward_colab.ipynb"


def test_reward_notebook_is_thin():
    cells = json.loads(REWARD_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert 1 <= len(code_cells) <= 4
    source = "\n".join("".join(c["source"]) for c in code_cells)
    for forbidden in ("def ", "class ", "import torch", "for ", "while "):
        assert forbidden not in source, forbidden
    assert "ts2-reward" in source
    assert "--resume" in source


def test_reward_notebook_has_no_secrets_or_outputs():
    text = REWARD_NOTEBOOK.read_text(encoding="utf-8")
    assert "hf_" not in text
    cells = json.loads(text)["cells"]
    assert all(not c.get("outputs") for c in cells)
```

- [ ] **Step 3: Run the notebook tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_notebook.py -q`
Expected: PASS (existing pretrain/sft notebook tests + 2 new reward tests).

- [ ] **Step 4: Commit**

```bash
git add notebooks/reward_colab.ipynb tests/test_notebook.py
git commit -m "docs: thin reward Colab notebook and its thinness tests"
```

---

### Task 8: Colab bootstrap script + shared Hub-download helper + PROGRESS

The one-command real-run mechanism (`docs/colab-notes.md`): extract the generic Hub-download helpers into the package (so the reward — and later GRPO — bootstrap doesn't copy them a third time), refactor `sft_colab.py` to use them, add `scripts/reward_colab.py` (download tokenizer + preference pairs, then `ts2-reward --resume`), test its orchestration with an injected download, run the full suite, and mark issue 05 code-complete.

**Files:**
- Create: `src/tinystories_v2/hub_download.py`, `scripts/reward_colab.py`, `tests/test_reward_colab.py`
- Modify: `scripts/sft_colab.py` (use the shared helper), `PROGRESS.md`

**Interfaces:**
- Consumes: `reward.run` (resume path), `config.load_config`/`load_env`, `hub_download.download_file`/`retry`; the `SlotCoverageFakeJudge` + `judge_with_order_swap` pair-builder (test-side, mirroring Task 4).
- Produces: `hub_download.retry(fn, *, attempts, base_delay, what)`, `hub_download.download_file(repo_id, filename, local_dir) -> Path`; `reward_colab.prepare(reward_config, *, download=None) -> Path`, `reward_colab.main(argv=None)`, and module constants `TOKENIZER_REPO`, `PAIRS_REPO`, `PAIRS_FILENAME`, `DEFAULT_REWARD_CONFIG`.

- [ ] **Step 1: Create the shared `src/tinystories_v2/hub_download.py`**

```python
"""Retry-wrapped single-file Hub downloads for the Colab bootstrap scripts.

Shared by scripts/*_colab.py so each one-command bootstrap survives Colab's
intermittent network faults (ConnectionResetError) without duplicating the
logic. The hf_hub_download import stays local to download_file so tests can
monkeypatch huggingface_hub.hf_hub_download.
"""

import time
from pathlib import Path


def retry(fn, *, attempts: int = 5, base_delay: float = 2.0, what: str = "operation"):
    """Call fn(), retrying on any exception with exponential backoff."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as err:  # noqa: BLE001 — transient network faults are the norm here
            if attempt == attempts - 1:
                raise
            delay = base_delay * 2**attempt
            print(f"[hub_download] {what} failed ({err!r}); "
                  f"retry {attempt + 1}/{attempts - 1} in {delay:.0f}s")
            time.sleep(delay)


def download_file(repo_id: str, filename: str, local_dir) -> Path:
    """Download one file from the Hub into local_dir (so it lands at
    local_dir/filename), retry-wrapped and tolerant of the repo being a model or
    a dataset repo. Returns the local path."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    def _fetch() -> str:
        last: Exception | None = None
        for repo_type in ("model", "dataset"):
            try:
                return hf_hub_download(repo_id=repo_id, filename=filename,
                                       repo_type=repo_type, local_dir=str(local_dir))
            except RepositoryNotFoundError as err:  # try the other repo type
                last = err
        raise last  # both repo types missing — surface the last error

    retry(_fetch, what=f"download {repo_id}/{filename}")
    return local_dir / filename
```

- [ ] **Step 2: Refactor `scripts/sft_colab.py` to import the shared helpers**

Delete the local `retry` and `download_file` function definitions from `scripts/sft_colab.py`, and add this import next to its existing imports (after `from tinystories_v2.config import load_config, load_env`):

```python
from tinystories_v2.hub_download import download_file, retry  # noqa: F401 — re-exported for tests
```

Everything else in `sft_colab.py` is unchanged (`prepare` still references the module-global `download_file`; `main` still calls it). `retry` is re-exported so nothing that imported it breaks.

- [ ] **Step 3: Run the existing SFT bootstrap test to confirm no regression**

Run: `.venv/bin/python -m pytest tests/test_sft_colab.py -q`
Expected: PASS (5 tests) — the monkeypatch of `boot.download_file` and of `huggingface_hub.hf_hub_download` still work because `download_file`'s hf import stays local.

- [ ] **Step 4: Write the failing reward-bootstrap test**

Create `tests/test_reward_colab.py`:

```python
"""The reward Colab bootstrap orchestrates download -> ts2-reward --resume as one
idempotent command. These tests drive that orchestration against fixture
artifacts with an injected/monkeypatched download (no network), verifying the
wiring and the skip-on-warm-VM behavior the real run depends on.
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from tinystories_v2.judge import SlotCoverageFakeJudge, judge_with_order_swap
from tinystories_v2.slot_prompt import SLOT_FIELDS
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run
from tinystories_v2.data import run as data_run

# scripts/ is not an importable package; load the bootstrap module by path.
_spec = importlib.util.spec_from_file_location(
    "reward_colab", Path(__file__).parent.parent / "scripts" / "reward_colab.py")
boot = importlib.util.module_from_spec(_spec)
sys.modules["reward_colab"] = boot
_spec.loader.exec_module(boot)

_BLAND = ["A plain note with nothing much to say.",
          "Some words that go nowhere in particular."]


@pytest.fixture
def hub_and_config(tmp_path, fixture_path):
    """Build a real tokenizer + a separable fake-Judge pairs.jsonl (the 'Hub'
    source the bootstrap downloads from) and a reward config pointing at local
    artifact paths that do not exist yet, so prepare() must download."""
    hub = tmp_path / "hub"
    tokenizer_run({"out_dir": str(hub / "tokenizer"), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    data_run({
        "out_dir": str(hub / "data"), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.1, "sft": 0.7,
                   "pref": 0.1, "eval": 0.1},
    })
    with open(hub / "data" / "splits" / "sft.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    judge = SlotCoverageFakeJudge()
    pairs_src = hub / "pairs.jsonl"
    with pairs_src.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            scaffold = Scaffold(**{fld: row[fld] for fld in SLOT_FIELDS})
            chosen = (f"{scaffold.character}, a {scaffold.trait} one in "
                      f"{scaffold.setting}, met {scaffold.conflict}. "
                      f"{scaffold.resolution}. The moral: {scaffold.moral}.")
            pair = judge_with_order_swap(judge, scaffold, chosen, _BLAND[i % len(_BLAND)])
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

    art = tmp_path / "artifacts"
    tokenizer_dst = art / "tokenizer_full" / "tokenizer.json"
    pairs_dst = art / "pref_full" / "pairs.jsonl"
    reward_cfg = tmp_path / "reward.toml"
    reward_cfg.write_text(
        f'out_dir = "{art / "reward_full"}"\n\n'
        f'[data]\npairs_path = "{pairs_dst}"\n'
        f'tokenizer_path = "{tokenizer_dst}"\n', encoding="utf-8")

    sources = {
        (boot.TOKENIZER_REPO, "tokenizer.json"): hub / "tokenizer" / "tokenizer.json",
        (boot.PAIRS_REPO, boot.PAIRS_FILENAME): pairs_src,
    }
    return {"reward_cfg": reward_cfg, "tokenizer_dst": tokenizer_dst,
            "pairs_dst": pairs_dst, "sources": sources}


def _fake_download(sources, calls=None):
    def download(repo_id, filename, local_dir):
        if calls is not None:
            calls.append((repo_id, filename))
        dst = Path(local_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[(repo_id, filename)], dst)
        return dst
    return download


def test_prepare_downloads_tokenizer_and_pairs(hub_and_config):
    calls = []
    pairs = boot.prepare(hub_and_config["reward_cfg"],
                         download=_fake_download(hub_and_config["sources"], calls))
    assert {filename for _, filename in calls} == {"tokenizer.json", boot.PAIRS_FILENAME}
    assert pairs == hub_and_config["pairs_dst"]
    assert hub_and_config["tokenizer_dst"].exists() and pairs.exists()


def test_prepare_is_idempotent_on_a_warm_vm(hub_and_config):
    boot.prepare(hub_and_config["reward_cfg"],
                 download=_fake_download(hub_and_config["sources"]))

    def boom(*args, **kwargs):
        raise AssertionError("download must not run when artifacts already exist")

    boot.prepare(hub_and_config["reward_cfg"], download=boom)  # no download call


def test_main_skip_train_prepares_without_training(hub_and_config, monkeypatch):
    monkeypatch.setattr(boot, "download_file", _fake_download(hub_and_config["sources"]))
    trained = []
    monkeypatch.setattr(boot.reward, "run", lambda *a, **k: trained.append(True))
    boot.main(["--reward-config", str(hub_and_config["reward_cfg"]), "--skip-train"])
    assert hub_and_config["pairs_dst"].exists()
    assert trained == []


def test_main_trains_with_resume_after_prepare(hub_and_config, monkeypatch):
    monkeypatch.setattr(boot, "download_file", _fake_download(hub_and_config["sources"]))
    resume_flags = []

    def fake_run(config, resume=False):
        resume_flags.append(resume)
        return {"step": 1, "loss": 0.5, "heldout_accuracy": 0.9}

    monkeypatch.setattr(boot.reward, "run", fake_run)
    boot.main(["--reward-config", str(hub_and_config["reward_cfg"])])
    assert hub_and_config["pairs_dst"].exists()
    assert resume_flags == [True]  # ts2-reward invoked with resume=True
```

- [ ] **Step 5: Run the reward-bootstrap test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_reward_colab.py -q`
Expected: FAIL — `FileNotFoundError`/`spec_from_file_location` cannot load `scripts/reward_colab.py` (does not exist yet).

- [ ] **Step 6: Create `scripts/reward_colab.py`**

```python
"""One-command real Reward Model run for Colab (issue 05).

Turns a fresh L4 VM into a running Reward Model job with a single command.
Idempotent: safe to re-run after an L4 preemption — it skips already-present
artifacts and `ts2-reward --resume` continues from the last Hub checkpoint.

Steps:
  1. load .env secrets (HF_TOKEN, WANDB_API_KEY) so Hub download/sync + W&B work
  2. download tokenizer.json + the preference pairs (issue 04's artifact) from
     the Hub (retry-wrapped) if the local copies are absent
  3. run the Reward Model stage (ts2-reward, resume=True): fetches the SFT
     checkpoint via [init].hub_source, resumes any prior Reward Model checkpoint
     from the Hub, trains with Bradley-Terry loss, and checkpoints back to the
     Hub every checkpoint_every steps

Run on the VM (in-kernel, survives disconnects — never nohup-detach):
    python scripts/reward_colab.py            # download + train
    python scripts/reward_colab.py --skip-train   # download only
    colab exec -f scripts/reward_colab.py

See docs/colab-notes.md for the CLI gotchas (push main first, .env via upload,
background + poll long commands, retries, always stop the VM). The preference
pairs live in issue 04's Hub repo; override with --pairs-repo / --pairs-file if
it moves.
"""

import argparse
from pathlib import Path

from tinystories_v2 import reward
from tinystories_v2.config import load_config, load_env
from tinystories_v2.hub_download import download_file  # noqa: F401 — monkeypatched in tests

TOKENIZER_REPO = "congthanh991/tinystories-v2-tokenizer"
PAIRS_REPO = "congthanh991/tinystories-v2-pref"   # issue 04's preference artifact
PAIRS_FILENAME = "pairs.jsonl"
DEFAULT_REWARD_CONFIG = "configs/reward_full.toml"


def prepare(reward_config, *, download=None) -> Path:
    """Ensure the tokenizer + preference pairs are present (download if missing).
    Returns the local pairs path. `download` is injectable for tests; it defaults
    to download_file resolved at call time. Idempotent: each step is guarded by
    an existence check, so re-running on a warm VM is a no-op up to training."""
    if download is None:
        download = download_file
    cfg = load_config(reward_config)
    tokenizer_path = Path(cfg["data"]["tokenizer_path"])   # artifacts/tokenizer_full/tokenizer.json
    pairs_path = Path(cfg["data"]["pairs_path"])           # artifacts/pref_full/pairs.jsonl

    if not tokenizer_path.exists():
        print(f"[reward_colab] downloading tokenizer -> {tokenizer_path}")
        download(TOKENIZER_REPO, "tokenizer.json", tokenizer_path.parent)
    if not pairs_path.exists():
        print(f"[reward_colab] downloading preference pairs -> {pairs_path}")
        download(PAIRS_REPO, PAIRS_FILENAME, pairs_path.parent)
    return pairs_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reward-config", default=DEFAULT_REWARD_CONFIG, type=Path)
    parser.add_argument("--skip-train", action="store_true",
                        help="download the tokenizer + pairs only, then stop")
    args = parser.parse_args(argv)

    load_env()  # HF_TOKEN / WANDB_API_KEY reach Hub sync + wandb; never printed
    prepare(args.reward_config)
    if args.skip_train:
        print("[reward_colab] --skip-train: inputs ready; skipping training")
        return
    print("[reward_colab] starting Reward Model training (ts2-reward --resume)")
    summary = reward.run(load_config(args.reward_config), resume=True)
    print(f"[reward_colab] done: step {summary['step']}, loss {summary['loss']:.4f}, "
          f"held-out accuracy {summary['heldout_accuracy']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the reward-bootstrap test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_reward_colab.py -q`
Expected: PASS (4 tests).

- [ ] **Step 8: Run the full suite to confirm the whole chain is green**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — all prior tests plus the new `test_model_hidden_states`, `test_reward_model`, `test_reward_batch`, `test_reward_stage`, `test_reward_resume`, `test_reward_gate`, `test_reward_colab`, the refactored `test_sft_colab`, and the two notebook tests.

- [ ] **Step 9: Update `PROGRESS.md`**

In the **Now** section, add a bullet (after the issue 03 bullet):

```markdown
- ✅ **Issue 05 (Reward Model stage + accuracy gate) code complete** — `ts2-reward`
  stage (Bradley-Terry loss on the SFT backbone + a scalar head, deterministic
  held-out split, checkpoint-resume), the shared `gate.check_reward_gate`
  (~68% default), `configs/reward_{fixture,full}.toml`, the one-command
  `scripts/reward_colab.py` bootstrap + `reward_colab.ipynb`, and the
  `reward-model-artifact-v1` schema all landed with tests green. The real run
  consumes issue 03's SFT checkpoint and issue 04's labeled pairs. Unblocks
  **issue 06** (GRPO), which enforces the gate at startup.
```

In the **Issue board** table, change the issue 05 and issue 06 rows to:

```markdown
| 05 | Reward Model stage + accuracy gate | 02 ✅, 10 ✅ | ✅ code complete (real run needs 03 ckpt + 04 labels) |
| 06 | GRPO stage | 05 ✅code, 11 ✅ | 🟢 ready (code work) |
```

In the **Log**, add a dated entry at the top:

```markdown
- **2026-07-12** — Issue 05 (Reward Model stage + accuracy gate) code complete:
  `reward_model.py` (FableLM backbone + scalar head, last-real-token scoring,
  hand-written Bradley-Terry loss), `reward.py` stage (`ts2-reward`, deterministic
  train/holdout split, checkpoint-resume, held-out accuracy + split recipe in the
  manifest), `gate.py` (`check_reward_gate`, ~68% default), configs, the
  `scripts/reward_colab.py` bootstrap (shared `hub_download` helper, extracted
  from `sft_colab.py`), thin `reward_colab.ipynb`, and
  `docs/schemas/reward-model-artifact-v1.md`. Added a behavior-preserving
  `FableLM.hidden_states` seam. Tests: scoring/padding-invariance, batching,
  separable fake-Judge pairs beat chance on held-out, kill-and-resume bitwise,
  gate refusal, and bootstrap orchestration. Unblocks issue 06 (GRPO).
```

- [ ] **Step 10: Commit**

```bash
git add src/tinystories_v2/hub_download.py scripts/reward_colab.py scripts/sft_colab.py \
        tests/test_reward_colab.py PROGRESS.md
git commit -m "feat: one-command reward Colab bootstrap and shared Hub-download helper"
```

---

## Self-Review

**1. Spec coverage** (issue 05 acceptance criteria → task):
- Toy Reward Model beats chance on separable fake-Judge pairs via the stage entrypoint, asserted by a test → Task 4 (`test_toy_reward_model_beats_chance_on_heldout`).
- Scores usable downstream: `(Slot Prompt Scaffold, Fable) → scalar`, batched, CPU → Task 2 (`score_fables`, `score_sequences`) + Task 4 (`test_scores_are_usable_downstream`).
- Held-out accuracy and pair-split recipe recorded in artifact metadata → Task 4 (`manifest` + `test_manifest_records_accuracy_and_split_recipe`) + schema doc.
- Gate: below-gate refuses with a clear message via a shared gate check; above-gate passes → Tasks 4 (`gate.py`) + 6 (`test_reward_gate.py`).
- Training resumes after a kill; metrics stream to W&B when enabled → Task 5 (bitwise resume) + `MetricsLogger` (W&B path reused verbatim from sft/pretrain).
- Thin Colab notebook for the real run → Task 7; the real CLI-driven run mechanism (one-command bootstrap per `docs/colab-notes.md`) → Task 8 (`scripts/reward_colab.py`, tested via injected download).
- Blocked-by (02 ✅, 10 ✅): the backbone + checkpoint infra (02) and the preference-pair schema (10) are both present and reused.
- Colab run procedure honored: the bootstrap downloads the tokenizer + preference pairs the stage does not fetch itself (avoiding the fresh-VM failure `ts2-reward --resume` alone would hit), `--resume` is idempotent for preemption recovery, and the run procedure (push main, `.env` upload, background+poll, `colab stop`) is captured in the plan's Colab Run Procedure section for the executor.

**2. Placeholder scan:** every code step contains complete, runnable ASCII code; no "TBD"/"add error handling"/"similar to Task N", and no intentionally-broken tokens. The one repeated construction (separable fake-Judge pairs) is written out fully in each test file that needs it (Tasks 4, 5, 6, 8), since tasks may be read out of order.

**3. Type consistency:** `RewardModel(config)`, `.backbone`, `.score_head`, `.load_backbone_state_dict`, `forward(idx, lengths) -> [B]` are used identically in Tasks 2, 4, 5. `pad_sequences(sequences, context, device) -> (idx, lengths)` is defined in Task 2 and imported in Task 3. `split_pairs(encoded, holdout_frac, seed) -> (train, holdout)`, `get_pair_batch(...) -> (chosen_idx, chosen_len, rejected_idx, rejected_len)`, and `evaluate_accuracy(model, holdout, *, device)` are defined in Task 3 and called with matching signatures in Task 4. The checkpoint state key `pairs_seen` (not SFT's `tokens_seen`) is written in Task 4's `run` and asserted in Task 5's resume test. `check_reward_gate(reward_dir, gate=DEFAULT_ACCURACY_GATE) -> float` and manifest field `heldout_accuracy` / `pair_split` are consistent across `gate.py` (Task 4), the schema doc (Task 4), and the gate tests (Task 6).
