# 08 — DPO Fallback Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the pre-committed stage-3 fallback (ADR-0004, DESIGN.md): a DPO training stage that fine-tunes the SFT policy directly on issue 10's `(chosen, rejected)` preference pairs against a frozen SFT reference, with a hand-written DPO loss (ADR-0005), producing a checkpoint that is a drop-in third model for the eval suite (issue 07).

**Architecture:** One new stage module `dpo.py` that is a sibling of `reward.py` and `sft.py` — it reads one TOML config and writes `out_dir` artifacts, reusing issue 02's checkpoint-resume contract, `build_optimizer`/`lr_at` (imported from `pretrain.py`), the precision knob, `MetricsLogger`, and Hub sync verbatim. DPO differs from the Reward Model in two places only: it fine-tunes a plain `FableLM` **policy** (no scalar head) and it needs a second, **frozen** `FableLM` **reference** (the SFT weights). The loss is `-log σ(β·[(logπ_chosen − logπ_rejected) − (logπ_ref_chosen − logπ_ref_rejected)])` where each `logπ` is the summed per-token log-probability of the completion tokens (the fable body + `<|end|>`, masked exactly as SFT masks — reusing `slot_prompt.encode_example`'s `loss_mask`). Both policy and reference init from the SFT checkpoint declared in `[init]`; the reference is **always** re-derived from that fixed checkpoint (never from a resumed policy), so it is a pure function of `[init]` and resume stays bitwise-identical. The output checkpoint stores the policy's `state_dict()` under `state["model"]` plus `state["config"]["model"]`, which is exactly what `eval.load_stage_model` reads — so a DPO stage plugs into the eval suite with zero eval-code changes. Preference-pair loading (`load_pairs`) and the deterministic train/holdout split (`split_pairs`) are imported from `reward.py` (both are generic, schema-level helpers) to stay DRY.

**Tech Stack:** Python 3.11+, PyTorch ≥2.6, `tokenizers`, TOML configs, pytest. No new dependencies.

## Global Constraints

- **Stage convention:** every stage entrypoint reads exactly one TOML file and writes all artifacts under the config's `out_dir`; stages share nothing in memory and couple only through on-disk artifacts.
- **Checkpoint-resume is a hard requirement:** a run killed mid-flight (SIGKILL) must resume from the latest checkpoint and reproduce the uninterrupted run bitwise on fp32 CPU. Batches are a pure function of `(seed, step, micro_step)`; the held-out split is a pure function of `(n_pairs, holdout_frac, split_seed)`; optimizer + scaler state round-trip through checkpoints. The frozen reference is a pure function of the fixed `[init]` SFT checkpoint (it is **not** stored in the DPO checkpoint), so it is identical on a fresh run and on resume.
- **Vocabulary (CONTEXT.md):** use **Fable**, **Scaffold**, **Slot Prompt**, **SFT**, **Reward Model**, **Judge**, **RLAIF** (never "RLHF") exactly. Introduce "DPO" as the pre-committed fallback; do not write "RLHF".
- **Preference-Pair schema v1 (`docs/schemas/preference-pair-v1.md`, `preferences.py`):** each pairs `.jsonl` line has `schema_version`, `scaffold` (six slots), `chosen`, `rejected`, `verdict`. Decode every line through `tinystories_v2.preferences.validate_preference_pair` before training (this is what the reused `reward.load_pairs` does) — never accept another schema version or reconstruct discarded pairs. DPO consumes the **identical** artifact as the Reward Model (issue 05) — no separate labeling path.
- **Slot Prompt format contract (issue 12, `slot_prompt.py`):** build each completion sequence as `<|character|>…<|moral|><|fable|>{body}<|end|>` via `render_example`, and mask the completion via `encode_example`'s `loss_mask` (0 over the prompt prefix through `<|fable|>`, 1 over the body + `<|end|>`). `SLOT_FIELDS = ("character","trait","setting","conflict","resolution","moral")`. Do not re-derive or reorder these.
- **DPO loss (ADR-0005, hand-written):** `-log σ(β·[(logπ_c − logπ_r) − (logπ_ref_c − logπ_ref_r)])` averaged over the batch; no TRL `DPOTrainer`. Default `β = 0.1` (Rafailov et al. 2023 canonical value; note GRPO's `β≈0.03` in DESIGN.md is a different KL mechanism — do not conflate them).
- **Secrets never printed:** `.env` values (HF/W&B tokens) are loaded via `config.load_env` and never logged.
- **Colab notebooks stay thin:** setup + a single stage invocation only; no `def`, `class`, `import torch`, `for`, or `while` in notebook source (enforced by `tests/test_notebook.py`).
- **Version floors:** `requires-python >=3.11`, `torch>=2.6` (already pinned in `pyproject.toml`); do not add dependencies.
- **Tests are CPU-only, seconds each, no network or GPU.** Real code paths at toy scale; no mocking of our own code.

---

## File Structure

**Create:**
- `src/tinystories_v2/dpo.py` — the DPO stage and its primitives: `sequence_logprobs`, `implicit_reward_margins`, `dpo_loss` (pure loss library); `encode_pairs`, `_pad_shifted`, `get_pair_batch` (data); `_load_sft_state`, `_build_model`, `evaluate_margin` (init + held-out metric); `run`, `main` (stage). Reuses `load_pairs`, `split_pairs` from `reward.py` and `build_optimizer`, `lr_at` from `pretrain.py`. Entrypoint `ts2-dpo`.
- `configs/dpo_fixture.toml` — toy CPU wiring config (matches the SFT/pretrain fixture architecture).
- `configs/dpo_full.toml` — real Colab DPO config (design-doc defaults; `β = 0.1`).
- `scripts/dpo_colab.py` — one-command Colab bootstrap for the real DPO run (download tokenizer + preference pairs, then `ts2-dpo --resume`), adapting `scripts/reward_colab.py`.
- `notebooks/dpo_colab.ipynb` — thin Colab wrapper (documented parallel path; the real run uses `scripts/dpo_colab.py`).
- `docs/schemas/dpo-artifact-v1.md` — the DPO `out_dir` artifact + manifest metadata contract.
- `tests/test_dpo_loss.py` — the direct loss test (criterion 1): hand-computed `dpo_loss` on known log-probs; `sequence_logprobs` over a known logits tensor; zero-margin → `log 2`.
- `tests/test_dpo_batch.py` — `encode_pairs` produces id+mask lists; `split_pairs` (reused) determinism/disjointness on DPO-encoded dicts; `get_pair_batch` purity + padding + mask shapes.
- `tests/test_dpo_init.py` — policy + frozen reference init from an SFT checkpoint: arch-match validation, missing-checkpoint error, reference has no grad and equals the SFT init while the policy is trainable.
- `tests/test_dpo_stage.py` — stage (criteria 2, 3, 5): toy run on separable fake-Judge pairs raises held-out margin above 0 and lowers the loss; manifest records `beta`, `heldout_margin`, split recipe, `pairs_path`; the output checkpoint loads via `eval.load_stage_model`; init from a real checkpoint; CLI runs standalone.
- `tests/test_dpo_resume.py` — kill-and-resume bitwise-identical contract (criterion 4).
- `tests/test_dpo_colab.py` — the bootstrap orchestration (download → `ts2-dpo --resume`) with an injected download; wiring + idempotence + `resume=True`.
- `tests/test_dpo_config.py` — both configs parse and carry the required sections/keys (incl. `[dpo].beta`).

**Modify:**
- `pyproject.toml` — add the `ts2-dpo` console-script entrypoint.
- `configs/eval_full.toml` — add a commented `[[stages]]` `dpo` block so the real eval run can wire the DPO checkpoint as the third model (documentation of criterion 5).
- `tests/test_notebook.py` — add thin-wrapper + no-secrets tests for `dpo_colab.ipynb`.
- `PROGRESS.md` — mark issue 08 code-complete (final task).

**Read-only references (do not modify):** `reward.py` (sibling stage template; source of `load_pairs`, `split_pairs`), `reward_model.py` (loss-primitive style), `sft.py` (masked-loss batching template), `pretrain.py` (imports `lr_at`, `build_optimizer`), `model.py` (`FableLM.forward`/`hidden_states`), `slot_prompt.py` (`encode_example`, `render_example`, `SLOT_FIELDS`), `preferences.py`, `slots.py`, `checkpoint.py`, `tracking.py`, `hub.py` (`fetch_from`, `try_sync_to`), `hub_download.py` (`download_file`), `config.py`, `eval.py` (`load_stage_model` — the drop-in seam), `gate.py` (`DEFAULT_ACCURACY_GATE` — informational print only), `tests/conftest.py` (`make_init_checkpoint`, `fixture_path`), `scripts/reward_colab.py` + `tests/test_reward_colab.py` (bootstrap template), `notebooks/reward_colab.ipynb` (notebook template), `docs/colab-notes.md` (real-run procedure).

## Colab Run Procedure (from `docs/colab-notes.md`)

The real DPO run is **not** driven from the notebook — it uses the one-command bootstrap over the `colab` CLI, exactly like the issue 03 SFT and issue 05 Reward Model runs. Before running: `git push origin main` (the VM clones from GitHub — local-only commits are missing), then `colab upload .env /content/tinystories_v2/.env` (never pass tokens in an `exec`). Run in-kernel via `colab exec -f scripts/dpo_colab.py` (never nohup-detach — idle VMs get reaped); background long commands with a log + `EXIT_` marker and poll in <20 s exec calls; wrap CLI calls in a 3–6 try retry loop. `--resume` is idempotent: after an L4 preemption, re-running the bootstrap pulls the last Hub checkpoint and continues. The Hub is the source of truth (list `checkpoints/step_*.pt` + read `manifest.json`), not the VM. The DPO run additionally needs issue 03's SFT checkpoint (fetched via `[init].hub_source`) and issue 04's labeled pairs (downloaded by the bootstrap). Always `colab stop -s <name>` when done. This procedure is documentation for the executor — it is not exercised by the test suite (which is CPU-only, no network).

---

### Task 1: DPO loss + log-probability primitives

The hand-written loss library (ADR-0005), independently testable pure functions. This is acceptance criterion 1's home.

**Files:**
- Create: `src/tinystories_v2/dpo.py`
- Test: `tests/test_dpo_loss.py`

**Interfaces:**
- Consumes: nothing (pure tensor functions).
- Produces:
  - `sequence_logprobs(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor` — `logits [B,T,V]`, `y [B,T]`, `mask [B,T]` → `[B]` summed target log-probs over active positions.
  - `implicit_reward_margins(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta) -> torch.Tensor` — each arg `[B]`, `beta: float` → `[B]` per-pair margin.
  - `dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta) -> torch.Tensor` — scalar loss.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dpo_loss.py`:

```python
import math

import torch

from tinystories_v2.dpo import dpo_loss, implicit_reward_margins, sequence_logprobs


def test_dpo_loss_matches_hand_computation():
    # policy/reference summed completion log-probs for a 2-pair batch.
    pc = torch.tensor([-2.0, -1.0])   # policy chosen
    pr = torch.tensor([-3.0, -4.0])   # policy rejected
    rc = torch.tensor([-2.5, -2.0])   # reference chosen
    rr = torch.tensor([-2.5, -3.0])   # reference rejected
    beta = 0.1
    # logits = (pc - pr) - (rc - rr) = (1.0, 3.0) - (0.0, 1.0) = (1.0, 2.0)
    logits = torch.tensor([1.0, 2.0])
    expected = (-torch.nn.functional.logsigmoid(beta * logits)).mean().item()
    assert math.isclose(dpo_loss(pc, pr, rc, rr, beta).item(), expected, rel_tol=1e-6)


def test_zero_margin_gives_log2_loss_and_zero_margin():
    # policy == reference -> logits 0 -> loss = -log sigma(0) = log 2, margin 0.
    z = torch.zeros(4)
    assert math.isclose(dpo_loss(z, z, z, z, 0.1).item(), math.log(2), rel_tol=1e-6)
    assert torch.allclose(implicit_reward_margins(z, z, z, z, 0.1), torch.zeros(4))


def test_margin_is_beta_scaled_and_signed():
    pc, pr = torch.tensor([0.0]), torch.tensor([-1.0])   # policy prefers chosen
    rc, rr = torch.tensor([0.0]), torch.tensor([0.0])    # reference indifferent
    # margin = 0.1 * ((0 - 0) - (-1 - 0)) = 0.1 * 1.0 = 0.1
    assert math.isclose(implicit_reward_margins(pc, pr, rc, rr, 0.1)[0].item(), 0.1, rel_tol=1e-6)


def test_sequence_logprobs_sums_active_positions_only():
    # 1 row, T=3, V=2. First position is prompt (masked); last two are completion.
    logits = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]])   # [1, 3, 2]
    y = torch.tensor([[1, 0, 1]])
    mask = torch.tensor([[0.0, 1.0, 1.0]])
    logp = torch.log_softmax(logits, dim=-1)
    expected = (logp[0, 1, 0] + logp[0, 2, 1]).item()
    got = sequence_logprobs(logits, y, mask)
    assert got.shape == (1,)
    assert math.isclose(got[0].item(), expected, rel_tol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dpo_loss.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tinystories_v2.dpo'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/tinystories_v2/dpo.py` with the module docstring and the three primitives:

```python
"""DPO fallback stage: fine-tune the SFT policy directly on Judge preference
pairs against a frozen SFT reference, with a hand-written DPO loss (issue 08).

Invoke standalone:
    ts2-dpo --config configs/dpo_fixture.toml [--resume]
    (or: python -m tinystories_v2.dpo --config ...)

The pre-committed stage-3 fallback (ADR-0004): if GRPO is unstable or the Reward
Model can't clear its gate by the schedule checkpoint, ship DPO as the aligned
model. It consumes the *identical* preference-pair artifact as the Reward Model
(issue 05) and produces a plain FableLM checkpoint that is a drop-in third model
for the eval suite (issue 07). Reuses issue 02's checkpoint-resume contract,
optimizer conventions (build_optimizer), LR schedule (lr_at), precision knob,
W&B logging, and Hub sync verbatim (ADR-0005: libraries only at the edges).

Both the policy and the frozen reference initialize from the SFT checkpoint in
[init]; the reference is always re-derived from that fixed checkpoint (never a
resumed policy), so it is a pure function of [init] and resume stays bitwise.

Artifacts in <out_dir> (schema: docs/schemas/dpo-artifact-v1.md):
    checkpoints/step_XXXXXX.pt   full training state (atomic; resume contract)
    metrics.jsonl                one line per log_every steps: loss, lr, margin, pairs_seen
    manifest.json                stage, version, final step/loss, heldout_margin,
                                 beta, pair_split recipe, pairs_path, config
"""

import argparse
import json
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from tinystories_v2 import __version__
from tinystories_v2.checkpoint import (
    latest_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint,
)
from tinystories_v2.config import load_config, load_env
from tinystories_v2.hub import fetch_from, try_sync_to
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.pretrain import build_optimizer, lr_at
from tinystories_v2.reward import load_pairs, split_pairs
from tinystories_v2.slot_prompt import encode_example
from tinystories_v2.tracking import MetricsLogger


def sequence_logprobs(logits: torch.Tensor, y: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """Sum of per-token target log-probs over active (mask==1) positions.

    logits [B, T, V] are next-token scores for inputs x = ids[:-1]; y [B, T] are
    the shifted targets ids[1:]; mask [B, T] is 1 over the fable body + <|end|>
    and 0 over the prompt prefix and right-padding. Returns [B]: the completion
    log-probability log p(completion | prompt) the model assigns to each row."""
    logp = F.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)   # [B, T]
    return (token_logp * mask).sum(dim=-1)                       # [B]


def implicit_reward_margins(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
                            ref_chosen: torch.Tensor, ref_rejected: torch.Tensor,
                            beta: float) -> torch.Tensor:
    """Per-pair DPO implicit-reward margin (Rafailov et al. 2023):
    beta * [ (logπ_c - logπ_ref_c) - (logπ_r - logπ_ref_r) ]. Positive means the
    policy prefers chosen over rejected more than the frozen reference does. [B]."""
    return beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor,
             beta: float) -> torch.Tensor:
    """-log σ(beta * [(logπ_c - logπ_r) - (logπ_ref_c - logπ_ref_r)]), averaged
    (ADR-0005, hand-written; no TRL DPOTrainer). Minimized when the policy raises
    the chosen-minus-rejected completion log-ratio above the frozen reference's."""
    logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    return -F.logsigmoid(beta * logits).mean()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dpo_loss.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/dpo.py tests/test_dpo_loss.py
git commit -m "feat: hand-written DPO loss and completion log-prob primitives (issue 08)"
```

---

### Task 2: Preference-pair encoding and batching

Turn validated preference pairs into padded next-token `(x, y, mask)` micro-batches for chosen and rejected completions, deterministically. Reuses `load_pairs` / `split_pairs` from `reward.py`.

**Files:**
- Modify: `src/tinystories_v2/dpo.py`
- Test: `tests/test_dpo_batch.py`

**Interfaces:**
- Consumes: `reward.load_pairs`, `reward.split_pairs`, `slot_prompt.encode_example`, `preferences.PreferencePair`.
- Produces:
  - `encode_pairs(tokenizer, pairs: list[PreferencePair]) -> list[dict]` — each dict has `chosen_ids`, `chosen_mask`, `rejected_ids`, `rejected_mask` (all `list[int]`; masks are 0/1).
  - `_pad_shifted(ids_list: list[list[int]], mask_list: list[list[int]], context: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]` — `(x, y, mask)` right-padded to the batch's longest `x`.
  - `get_pair_batch(train: list[dict], micro_batch_size: int, context: int, *, seed: int, step: int, micro_step: int, device: str = "cpu") -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]` — `(chosen_xyz, rejected_xyz)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dpo_batch.py`:

```python
import torch
from tokenizers import Tokenizer

from tinystories_v2.dpo import encode_pairs, get_pair_batch
from tinystories_v2.preferences import PreferencePair, VerdictMetadata
from tinystories_v2.reward import split_pairs
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run


def _pair(chosen: str, rejected: str) -> PreferencePair:
    scaffold = Scaffold("fox", "sly", "a wood", "a gate", "it waited", "patience wins")
    return PreferencePair(
        scaffold=scaffold, chosen=chosen, rejected=rejected,
        verdict=VerdictMetadata(judge_id="fake:slot-coverage-v1",
                                first_pass="A", swapped_pass="B", consistent=True))


def _encoded(tmp_path, fixture_path):
    tok_dir = tmp_path / "tok"
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    pairs = [_pair(f"Chosen fable number {i}.", "A plain note.") for i in range(20)]
    return encode_pairs(tokenizer, pairs)


def test_encode_pairs_produces_ids_and_masks(tmp_path, fixture_path):
    encoded = _encoded(tmp_path, fixture_path)
    assert len(encoded) == 20
    e = encoded[0]
    assert e["chosen_ids"] and e["rejected_ids"]
    assert len(e["chosen_ids"]) == len(e["chosen_mask"])
    assert len(e["rejected_ids"]) == len(e["rejected_mask"])
    assert set(e["chosen_mask"]) <= {0, 1}
    assert sum(e["chosen_mask"]) > 0                       # completion tokens are active
    assert e["chosen_mask"][0] == 0                        # prompt prefix is masked


def test_split_is_deterministic_on_dpo_encoding(tmp_path, fixture_path):
    encoded = _encoded(tmp_path, fixture_path)
    train_a, holdout_a = split_pairs(encoded, holdout_frac=0.25, seed=7)
    train_b, holdout_b = split_pairs(encoded, holdout_frac=0.25, seed=7)
    assert len(holdout_a) == 5 and len(train_a) == 15
    assert holdout_a == holdout_b and train_a == train_b
    train_ids = {tuple(p["chosen_ids"]) for p in train_a}
    holdout_ids = {tuple(p["chosen_ids"]) for p in holdout_a}
    assert train_ids.isdisjoint(holdout_ids)


def test_get_pair_batch_is_pure_and_padded(tmp_path, fixture_path):
    encoded = _encoded(tmp_path, fixture_path)
    a = get_pair_batch(encoded, 4, 64, seed=1337, step=2, micro_step=0)
    b = get_pair_batch(encoded, 4, 64, seed=1337, step=2, micro_step=0)
    (cx_a, cy_a, cm_a), (rx_a, ry_a, rm_a) = a
    for t1, t2 in zip((*a[0], *a[1]), (*b[0], *b[1])):
        assert torch.equal(t1, t2)                         # pure in (seed, step, micro_step)
    assert cx_a.shape == cy_a.shape == cm_a.shape          # aligned x/y/mask
    assert cx_a.shape[0] == 4 and rx_a.shape[0] == 4
    assert cm_a.dtype == torch.float and cx_a.dtype == torch.long
    different = get_pair_batch(encoded, 4, 64, seed=1337, step=2, micro_step=1)
    assert not torch.equal(different[0][0], cx_a)           # micro_step changes the draw
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dpo_batch.py -q`
Expected: FAIL — `ImportError: cannot import name 'encode_pairs' from 'tinystories_v2.dpo'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/tinystories_v2/dpo.py` (after the loss primitives):

```python
def encode_pairs(tokenizer, pairs: list) -> list[dict]:
    """Precompute (ids, loss_mask) for the chosen and rejected completion of each
    pair via the Slot Prompt encoder: each is <|character|>…<|fable|>{body}<|end|>
    with the mask active over the body + <|end|> only (encode_example)."""
    encoded = []
    for pair in pairs:
        chosen = encode_example(tokenizer, pair.scaffold, pair.chosen)
        rejected = encode_example(tokenizer, pair.scaffold, pair.rejected)
        encoded.append({"chosen_ids": chosen.input_ids, "chosen_mask": chosen.loss_mask,
                        "rejected_ids": rejected.input_ids,
                        "rejected_mask": rejected.loss_mask})
    return encoded


def _pad_shifted(ids_list: list[list[int]], mask_list: list[list[int]],
                 context: int, device: str) -> tuple[torch.Tensor, ...]:
    """Right-pad (ids, loss_mask) rows into next-token (x, y, mask) tensors. Each
    row is truncated to context+1 ids, then shifted: x = ids[:-1], y = ids[1:],
    mask = loss_mask[1:] (active over body + <|end|>). Rows are padded to the
    batch's longest x with id 0 / mask 0; causal attention makes right-padding
    safe and padding never contributes to a completion log-prob."""
    rows = []
    for ids, m in zip(ids_list, mask_list):
        ids, m = ids[:context + 1], m[:context + 1]
        rows.append((ids[:-1], ids[1:], m[1:]))
    width = max(len(x) for x, _, _ in rows)
    xs, ys, ms = [], [], []
    for x, y, m in rows:
        pad = width - len(x)
        xs.append(x + [0] * pad)
        ys.append(y + [0] * pad)
        ms.append([float(v) for v in m] + [0.0] * pad)
    return (torch.tensor(xs, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device),
            torch.tensor(ms, dtype=torch.float, device=device))


def get_pair_batch(train: list[dict], micro_batch_size: int, context: int, *,
                   seed: int, step: int, micro_step: int,
                   device: str = "cpu") -> tuple[tuple, tuple]:
    """A (chosen_xyz, rejected_xyz) micro-batch sampled with replacement; a pure
    function of (seed, step, micro_step) so a resumed run replays it (resume
    contract). chosen and rejected are padded independently."""
    generator = torch.Generator()
    generator.manual_seed(((seed * 1_000_003 + step) * 1_009 + micro_step) % 2**63)
    picks = torch.randint(0, len(train), (micro_batch_size,),
                          generator=generator).tolist()
    chosen = _pad_shifted([train[i]["chosen_ids"] for i in picks],
                          [train[i]["chosen_mask"] for i in picks], context, device)
    rejected = _pad_shifted([train[i]["rejected_ids"] for i in picks],
                            [train[i]["rejected_mask"] for i in picks], context, device)
    return chosen, rejected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dpo_batch.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/dpo.py tests/test_dpo_batch.py
git commit -m "feat: DPO preference-pair encoding and deterministic batching (issue 08)"
```

---

### Task 3: Policy + frozen-reference init from the SFT checkpoint

Both models load from the `[init]` SFT checkpoint (fetched from Hub if absent), with architecture-match validation. The reference is frozen and eval-mode; the policy is trainable. The held-out margin metric closes over both.

**Files:**
- Modify: `src/tinystories_v2/dpo.py`
- Test: `tests/test_dpo_init.py`

**Interfaces:**
- Consumes: `checkpoint.latest_checkpoint`/`load_checkpoint`, `hub.fetch_from`, `model.FableLM`/`ModelConfig`, `sequence_logprobs`/`implicit_reward_margins`/`_pad_shifted` (Tasks 1–2), `conftest.make_init_checkpoint`.
- Produces:
  - `_load_sft_state(config: dict, device: str) -> dict` — the loaded SFT checkpoint `state` (raises `ValueError` on missing checkpoint or architecture mismatch).
  - `_build_model(config: dict, state: dict, device: str) -> FableLM` — a `FableLM` with the SFT weights loaded.
  - `evaluate_margin(policy: FableLM, reference: FableLM, holdout: list[dict], context: int, beta: float, *, device: str = "cpu", batch_size: int = 32) -> float` — mean held-out implicit-reward margin; `NaN` for an empty holdout.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dpo_init.py`:

```python
import math

import pytest
import torch

from tinystories_v2.dpo import _build_model, _load_sft_state, evaluate_margin
from tinystories_v2.model import FableLM

TOY_MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 64, "ffn_hidden": 192}


def _cfg(init_dir, model=None):
    return {"model": dict(model or TOY_MODEL), "init": {"local_dir": str(init_dir)}}


def test_builds_policy_and_reference_from_sft(tmp_path, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, "tok.json")
    state = _load_sft_state(_cfg(init), "cpu")
    policy = _build_model(_cfg(init), state, "cpu")
    reference = _build_model(_cfg(init), state, "cpu")
    # Both start equal to the SFT init...
    for k, v in policy.state_dict().items():
        assert torch.equal(reference.state_dict()[k], v)
    assert isinstance(policy, FableLM) and isinstance(reference, FableLM)


def test_reference_can_be_frozen(tmp_path, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, "tok.json")
    state = _load_sft_state(_cfg(init), "cpu")
    reference = _build_model(_cfg(init), state, "cpu").requires_grad_(False)
    assert all(not p.requires_grad for p in reference.parameters())


def test_missing_init_checkpoint_raises(tmp_path):
    with pytest.raises(ValueError, match="no SFT checkpoint"):
        _load_sft_state(_cfg(tmp_path / "empty"), "cpu")


def test_mismatched_init_architecture_raises(tmp_path, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, "tok.json")
    drifted = dict(TOY_MODEL, d_model=128)
    with pytest.raises(ValueError, match="SFT checkpoint"):
        _load_sft_state(_cfg(init, model=drifted), "cpu")


def test_evaluate_margin_is_zero_when_policy_equals_reference(tmp_path, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, "tok.json")
    state = _load_sft_state(_cfg(init), "cpu")
    policy = _build_model(_cfg(init), state, "cpu")
    reference = _build_model(_cfg(init), state, "cpu")
    # Two-token completions; masks active on the final token only.
    holdout = [{"chosen_ids": [1, 5, 9], "chosen_mask": [0, 0, 1],
                "rejected_ids": [1, 6, 8], "rejected_mask": [0, 0, 1]}]
    margin = evaluate_margin(policy, reference, holdout, context=64, beta=0.1)
    assert math.isclose(margin, 0.0, abs_tol=1e-6)          # identical models -> 0 margin


def test_evaluate_margin_nan_on_empty_holdout(tmp_path, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, "tok.json")
    state = _load_sft_state(_cfg(init), "cpu")
    policy = _build_model(_cfg(init), state, "cpu")
    reference = _build_model(_cfg(init), state, "cpu")
    assert math.isnan(evaluate_margin(policy, reference, [], context=64, beta=0.1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dpo_init.py -q`
Expected: FAIL — `ImportError: cannot import name '_load_sft_state' from 'tinystories_v2.dpo'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/tinystories_v2/dpo.py` (after the batching helpers):

```python
def _load_sft_state(config: dict, device: str) -> dict:
    """Load the SFT checkpoint declared in [init], fetching the artifact from the
    Hub first if the local checkpoint is absent (fresh VM), and validate its
    architecture matches [model]. Returns the loaded checkpoint state (contains
    'model'); both the policy and the frozen reference are built from it."""
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
    if ModelConfig(**state["config"]["model"]) != ModelConfig(**config["model"]):
        raise ValueError(
            f"[model] does not match the SFT checkpoint at {init_ckpt}; DPO must "
            f"fine-tune the SFT architecture")
    print(f"loaded SFT weights from {init_ckpt}")
    return state


def _build_model(config: dict, state: dict, device: str) -> FableLM:
    """Build a FableLM from [model] and load the SFT weights (strict)."""
    model = FableLM(ModelConfig(**config["model"])).to(device)
    model.load_state_dict(state["model"])
    return model


@torch.no_grad()
def evaluate_margin(policy: FableLM, reference: FableLM, holdout: list[dict],
                    context: int, beta: float, *, device: str = "cpu",
                    batch_size: int = 32) -> float:
    """Mean held-out implicit-reward margin. > 0 means the policy shifted toward
    the chosen completions relative to the frozen SFT reference. NaN for an empty
    holdout. Both models are read in eval mode with no grad."""
    if not holdout:
        return float("nan")
    was_training = policy.training
    policy.eval()
    reference.eval()
    margins = []
    for start in range(0, len(holdout), batch_size):
        chunk = holdout[start:start + batch_size]
        cx, cy, cm = _pad_shifted([p["chosen_ids"] for p in chunk],
                                  [p["chosen_mask"] for p in chunk], context, device)
        rx, ry, rm = _pad_shifted([p["rejected_ids"] for p in chunk],
                                  [p["rejected_mask"] for p in chunk], context, device)
        pc = sequence_logprobs(policy(cx), cy, cm)
        pr = sequence_logprobs(policy(rx), ry, rm)
        rc = sequence_logprobs(reference(cx), cy, cm)
        rr = sequence_logprobs(reference(rx), ry, rm)
        margins.append(implicit_reward_margins(pc, pr, rc, rr, beta))
    if was_training:
        policy.train()
    return torch.cat(margins).mean().item()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dpo_init.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/dpo.py tests/test_dpo_init.py
git commit -m "feat: DPO policy + frozen-reference init and held-out margin metric (issue 08)"
```

---

### Task 4: DPO training stage (`run`/`main`) with manifest and eval drop-in

The stage entrypoint: read config → build policy + frozen reference from SFT → train with grad-accum against the DPO loss → checkpoint-resume → write `manifest.json` with `heldout_margin`, `beta`, and the split recipe → Hub sync. Covers acceptance criteria 2, 3, and 5.

**Files:**
- Modify: `src/tinystories_v2/dpo.py`
- Test: `tests/test_dpo_stage.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3; `reward.load_pairs`/`split_pairs`; `pretrain.build_optimizer`/`lr_at`; `tracking.MetricsLogger`; `checkpoint.save_checkpoint`/`prune_checkpoints`; `hub.try_sync_to`; `eval.load_stage_model` (test only).
- Produces:
  - `run(config: dict, resume: bool = False) -> dict` — returns `{"step": int, "loss": float, "heldout_margin": float}`; writes `checkpoints/`, `metrics.jsonl`, `manifest.json` under `out_dir`. Checkpoint `state` keys: `step`, `pairs_seen`, `model` (policy `state_dict`), `optimizer`, `scaler`, `config`. `manifest.json` keys: `stage="dpo"`, `package_version`, `final_step`, `final_loss`, `heldout_margin`, `beta`, `pair_split` (`{seed, holdout_frac, n_pairs, n_train, n_holdout}`), `pairs_path`, `n_pairs`, `config`.
  - `main(argv: list[str] | None = None) -> None` — argparse over `--config` (required) and `--resume`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dpo_stage.py`:

```python
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from tinystories_v2.data import run as data_run
from tinystories_v2.dpo import run as dpo_run
from tinystories_v2.eval import load_stage_model
from tinystories_v2.judge import SlotCoverageFakeJudge, judge_with_order_swap
from tinystories_v2.model import FableLM
from tinystories_v2.pretrain import run as pretrain_run
from tinystories_v2.slot_prompt import SLOT_FIELDS
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run

TOY_MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 128, "ffn_hidden": 192}

_BLAND = ["A plain note with nothing much to say.",
          "Some words that go nowhere in particular."]


def _separable_pairs(rows):
    """Order-swap-consistent fake-Judge pairs where chosen mentions every slot
    value and rejected is bland — a learnable, separable preference signal."""
    judge = SlotCoverageFakeJudge()
    pairs = []
    for i, row in enumerate(rows):
        scaffold = Scaffold(**{f: row[f] for f in SLOT_FIELDS})
        chosen = (f"{scaffold.character}, a {scaffold.trait} one in "
                  f"{scaffold.setting}, met {scaffold.conflict}. "
                  f"{scaffold.resolution}. The moral: {scaffold.moral}.")
        pair = judge_with_order_swap(judge, scaffold, chosen, _BLAND[i % len(_BLAND)])
        assert pair is not None and pair.chosen == chosen
        pairs.append(pair)
    return pairs


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, fixture_path):
    base = tmp_path_factory.mktemp("dpo_stage_inputs")
    data_dir, tok_dir = base / "data", base / "tok"
    data_run({
        "out_dir": str(data_dir), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.1, "sft": 0.7,
                   "pref": 0.1, "eval": 0.1},
    })
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    with open(data_dir / "splits" / "sft.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    pairs = _separable_pairs(rows)
    pairs_path = base / "pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
    return {"pairs_path": str(pairs_path),
            "tokenizer": str(tok_dir / "tokenizer.json"), "n_pairs": len(pairs)}


def dpo_toy_config(out_dir, prepared, init_dir, model=None, **train_overrides) -> dict:
    train = {
        "steps": 60, "micro_batch_size": 8, "grad_accum": 1,
        "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
        "weight_decay": 0.0, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
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
        "dpo": {"beta": 0.1},
        "train": train,
        "wandb": {"enabled": False},
    }


def read_metrics(out_dir) -> list[dict]:
    lines = (Path(out_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_toy_dpo_shifts_policy_toward_chosen(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = dpo_toy_config(tmp_path / "out", prepared, init)
    summary = dpo_run(config)
    assert summary["heldout_margin"] > 0.0        # policy prefers chosen over the reference
    metrics = read_metrics(config["out_dir"])
    assert len(metrics) == 60
    assert {"step", "loss", "lr", "margin", "pairs_seen"} <= metrics[0].keys()
    assert metrics[-1]["loss"] < metrics[0]["loss"]   # DPO loss fell


def test_manifest_records_beta_margin_and_split(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = dpo_toy_config(tmp_path / "out", prepared, init, steps=6, checkpoint_every=3)
    dpo_run(config)
    manifest = json.loads(
        (Path(config["out_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "dpo"
    assert manifest["beta"] == 0.1
    assert isinstance(manifest["heldout_margin"], float)
    split = manifest["pair_split"]
    assert split["seed"] == 20260712 and split["holdout_frac"] == 0.25
    assert split["n_train"] + split["n_holdout"] == split["n_pairs"] == prepared["n_pairs"]
    assert manifest["pairs_path"] == prepared["pairs_path"]


def test_output_checkpoint_is_eval_drop_in(tmp_path, prepared, make_init_checkpoint):
    # Criterion 5: the DPO checkpoint loads through eval.load_stage_model exactly
    # like base/SFT — a plain FableLM, no DPO-specific eval code.
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = dpo_toy_config(tmp_path / "out", prepared, init, steps=6, checkpoint_every=6)
    dpo_run(config)
    model = load_stage_model({"name": "dpo", "local_dir": config["out_dir"]}, "cpu")
    assert isinstance(model, FableLM)


def test_init_from_a_real_checkpoint(tmp_path, prepared, fixture_path):
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
    config = dpo_toy_config(tmp_path / "dpo_out", prepared, pre_dir,
                            model=model64, steps=3, checkpoint_every=3)
    summary = dpo_run(config)
    assert summary["step"] == 3 and math.isfinite(summary["loss"])


def to_toml(config: dict) -> str:
    """Serialize the nested DPO config as TOML (stdlib has no writer)."""
    lines = [f'out_dir = "{config["out_dir"]}"']
    for section in ("model", "data", "init", "split", "dpo", "train", "wandb", "hub"):
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
    config = dpo_toy_config(tmp_path / "out", prepared, init, steps=2, checkpoint_every=2)
    config_file = tmp_path / "cfg.toml"
    config_file.write_text(to_toml(config), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.dpo", "--config", str(config_file)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (Path(config["out_dir"]) / "checkpoints" / "step_000002.pt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dpo_stage.py -q`
Expected: FAIL — `ImportError: cannot import name 'run' from 'tinystories_v2.dpo'`.

- [ ] **Step 3: Write minimal implementation**

Append `run` and `main` to `src/tinystories_v2/dpo.py`:

```python
def run(config: dict, resume: bool = False) -> dict:
    load_env()  # W&B/HF keys before wandb.init or hub sync; never printed
    train = config["train"]
    out_dir = Path(config["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    hub_target = config.get("hub", {}).get("target")
    beta = config["dpo"]["beta"]

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
    # Policy and frozen reference both start from the fixed SFT checkpoint. The
    # reference is re-derived here on every run (fresh or resume), never stored
    # in the DPO checkpoint, so it stays a pure function of [init].
    sft_state = _load_sft_state(config, device)
    policy = _build_model(config, sft_state, device)
    reference = _build_model(config, sft_state, device).requires_grad_(False)
    reference.eval()
    optimizer = build_optimizer(policy, train["peak_lr"],
                                (train["beta1"], train["beta2"]),
                                train["weight_decay"])

    start_step, pairs_seen = 0, 0
    if resume:
        if latest_checkpoint(ckpt_dir) is None and hub_target:
            try:
                fetch_from(hub_target, out_dir)  # fresh VM: pull previous session
            except Exception as err:  # noqa: BLE001 — first run: repo may not exist yet
                warnings.warn(
                    f"could not fetch a prior DPO run from {hub_target!r}; "
                    f"starting fresh: {err}", stacklevel=2)
        ckpt_path = latest_checkpoint(ckpt_dir)
        if ckpt_path is not None:
            state = load_checkpoint(ckpt_path)
            policy.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start_step, pairs_seen = state["step"], state["pairs_seen"]
            print(f"resumed from {ckpt_path.name} at step {start_step}")

    def checkpoint(step: int) -> None:
        save_checkpoint(ckpt_dir, step, {
            "step": step, "pairs_seen": pairs_seen,
            "model": policy.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "config": config,
        })
        prune_checkpoints(ckpt_dir, train.get("keep_last", 0))
        if hub_target:
            try_sync_to(hub_target, out_dir)

    logger = MetricsLogger(out_dir, config.get("wandb"))
    steps, accum = train["steps"], train["grad_accum"]
    micro_bs, context = train["micro_batch_size"], config["model"]["context"]
    loss_value, batch_margin = float("nan"), float("nan")
    policy.train()
    for step in range(start_step, steps):
        lr = lr_at(step, steps, train["peak_lr"],
                   train["warmup_frac"], train["min_lr_frac"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(accum):
            (cx, cy, cm), (rx, ry, rm) = get_pair_batch(
                train_pairs, micro_bs, context, seed=train["seed"],
                step=step, micro_step=micro_step, device=device)
            with autocast:
                pc = sequence_logprobs(policy(cx), cy, cm)
                pr = sequence_logprobs(policy(rx), ry, rm)
                with torch.no_grad():
                    rc = sequence_logprobs(reference(cx), cy, cm)
                    rr = sequence_logprobs(reference(rx), ry, rm)
                loss = dpo_loss(pc, pr, rc, rr, beta)
            scaler.scale(loss / accum).backward()
            pairs_seen += micro_bs
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), train["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        loss_value = loss.item()          # last micro-batch DPO loss
        batch_margin = implicit_reward_margins(
            pc.detach(), pr.detach(), rc, rr, beta).mean().item()
        done = step + 1
        if done % train["log_every"] == 0:
            logger.log({"loss": loss_value, "lr": lr, "margin": batch_margin,
                        "pairs_seen": pairs_seen}, step=done)
        if done % train["checkpoint_every"] == 0:
            checkpoint(done)
    if steps % train["checkpoint_every"] != 0:
        checkpoint(steps)
    logger.finish()

    heldout_margin = evaluate_margin(policy, reference, holdout_pairs,
                                     context, beta, device=device)

    manifest = {
        "stage": "dpo", "package_version": __version__,
        "final_step": steps, "final_loss": loss_value,
        "heldout_margin": heldout_margin, "beta": beta,
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
    print(f"held-out reward margin: {heldout_margin:.4f} (beta {beta})")
    return {"step": steps, "loss": loss_value, "heldout_margin": heldout_margin}


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

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dpo_stage.py -q`
Expected: PASS (5 passed). If `test_toy_dpo_shifts_policy_toward_chosen` is flaky (margin marginally ≤ 0), the fix is a stronger learning signal — raise the toy config `steps` to 80 in `dpo_toy_config`, not loosen the assertion. Note `test_cli_entrypoint_runs_standalone` invokes `python -m tinystories_v2.dpo` (module path), which works before the `pyproject` console-script entry is added in Task 6.

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/dpo.py tests/test_dpo_stage.py
git commit -m "feat: DPO training stage with margin metric and eval-drop-in checkpoint (issue 08)"
```

---

### Task 5: Kill-and-resume bitwise-identical contract

The checkpoint-resume contract end to end: SIGKILL a real subprocess mid-run and prove the resumed run reproduces the uninterrupted run's final weights and post-kill metrics exactly. This is acceptance criterion 4, and it also proves the reference is reconstructed identically (its weights never enter the checkpoint).

**Files:**
- Test: `tests/test_dpo_resume.py`
- (No source changes — this validates Task 4.)

**Interfaces:**
- Consumes: `dpo.run`, `data.run`, `tokenizer.run`, `judge.SlotCoverageFakeJudge`/`judge_with_order_swap`, `checkpoint.latest_checkpoint`/`load_checkpoint`, `conftest.make_init_checkpoint`/`fixture_path`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dpo_resume.py`:

```python
"""Kill-and-resume: the DPO checkpoint-resume contract, end to end.

Sized so the run is slow enough to SIGKILL mid-flight: d_model 128 / 4 layers /
ctx 128, 50 steps (each doing four forward passes — policy+reference over chosen
and rejected), checkpoint_every 5. Both runs share one init checkpoint and one
pairs.jsonl, so batches (a pure function of seed/step/micro_step), the held-out
split (a pure function of the split seed), and the frozen reference (a pure
function of the fixed SFT checkpoint) are identical across the two runs.
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
from tinystories_v2.dpo import run as dpo_run
from tinystories_v2.judge import SlotCoverageFakeJudge, judge_with_order_swap
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


def dpo_config(out_dir, pairs_path, tokenizer_path, init_dir) -> dict:
    return {
        "out_dir": str(out_dir),
        "model": dict(MODEL),
        "data": {"pairs_path": str(pairs_path), "tokenizer_path": str(tokenizer_path)},
        "init": {"local_dir": str(init_dir)},
        "split": {"holdout_frac": 0.2, "seed": 20260712},
        "dpo": {"beta": 0.1},
        "train": {"steps": STEPS, "micro_batch_size": 8, "grad_accum": 1,
                  "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                  "weight_decay": 0.0, "beta1": 0.9, "beta2": 0.95,
                  "grad_clip": 1.0, "precision": "fp32", "seed": 1337,
                  "checkpoint_every": CHECKPOINT_EVERY, "log_every": 1,
                  "keep_last": 0},
        "wandb": {"enabled": False},
    }


def to_toml(config: dict) -> str:
    lines = [f'out_dir = "{config["out_dir"]}"']
    for section in ("model", "data", "init", "split", "dpo", "train", "wandb"):
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


def test_killed_dpo_run_resumes_to_identical_final_state(
        tmp_path, fixture_path, make_init_checkpoint):
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
    reference = dpo_config(tmp_path / "reference", pairs_path, tokenizer_path, init_dir)
    dpo_run(reference)

    # Interrupted: run as a subprocess and SIGKILL once the kill-marker appears.
    interrupted = dpo_config(tmp_path / "interrupted", pairs_path,
                             tokenizer_path, init_dir)
    config_file = tmp_path / "interrupted.toml"
    config_file.write_text(to_toml(interrupted), encoding="utf-8")
    ckpt_dir = Path(interrupted["out_dir"]) / "checkpoints"
    kill_marker = ckpt_dir / f"step_{KILL_AFTER_STEP:06d}.pt"

    proc = subprocess.Popen(
        [sys.executable, "-m", "tinystories_v2.dpo", "--config", str(config_file)],
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

    dpo_run(interrupted, resume=True)

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

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `python -m pytest tests/test_dpo_resume.py -q`
Expected: PASS (the stage from Task 4 already satisfies the contract). If instead the subprocess finishes before the kill window (the toy run is too fast on the CI machine), lower `KILL_AFTER_STEP` or enlarge `MODEL` per the message in the test — do not weaken the equality assertions.

- [ ] **Step 3: Commit**

```bash
git add tests/test_dpo_resume.py
git commit -m "test: DPO kill-and-resume reproduces the uninterrupted run bitwise (issue 08)"
```

---

### Task 6: Configs, artifact schema, eval wiring, and console entrypoint

The declarative configs (toy + real), the artifact-contract doc, the commented eval-stage block that makes DPO a real third model, and the `ts2-dpo` entrypoint.

**Files:**
- Create: `configs/dpo_fixture.toml`, `configs/dpo_full.toml`, `docs/schemas/dpo-artifact-v1.md`
- Modify: `pyproject.toml`, `configs/eval_full.toml`
- Test: `tests/test_dpo_config.py`

**Interfaces:**
- Consumes: `config.load_config`.
- Produces: the two config files with sections `[model] [data] [init] [split] [dpo] [train] [wandb]` (and `[hub]` in full); `dpo-artifact-v1.md`; the `ts2-dpo` script entry.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dpo_config.py`:

```python
from pathlib import Path

from tinystories_v2.config import load_config

CONFIGS = Path(__file__).parent.parent / "configs"


def test_dpo_fixture_config_has_required_shape():
    cfg = load_config(CONFIGS / "dpo_fixture.toml")
    assert cfg["out_dir"]
    for section in ("model", "data", "init", "split", "dpo", "train", "wandb"):
        assert section in cfg, section
    assert cfg["dpo"]["beta"] > 0
    assert {"pairs_path", "tokenizer_path"} <= cfg["data"].keys()
    assert {"holdout_frac", "seed"} <= cfg["split"].keys()
    assert cfg["train"]["precision"] in {"fp32", "bf16", "fp16"}


def test_dpo_full_config_targets_hub_and_real_model():
    cfg = load_config(CONFIGS / "dpo_full.toml")
    assert cfg["model"]["vocab_size"] == 8192 and cfg["model"]["d_model"] == 512
    assert cfg["init"]["hub_source"].startswith("hf://")
    assert cfg["hub"]["target"].startswith("hf://")
    assert cfg["dpo"]["beta"] == 0.1
    assert cfg["wandb"]["enabled"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dpo_config.py -q`
Expected: FAIL — `FileNotFoundError: .../configs/dpo_fixture.toml`.

- [ ] **Step 3: Create the config files, schema, and entrypoint**

Create `configs/dpo_fixture.toml`:

```toml
# Toy CPU wiring smoke against fixture artifacts — local sanity runs and docs.
# Assumes an SFT-shaped init checkpoint and a preference-pair jsonl exist with a
# matching [model] block. Real runs use configs/dpo_full.toml. Stage behavior is
# guarded by tests/test_dpo_*.py.
out_dir = "artifacts/dpo_fixture"

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

[dpo]
beta = 0.1

[train]
steps = 60
micro_batch_size = 8
grad_accum = 1
peak_lr = 1e-3
warmup_frac = 0.1
min_lr_frac = 0.1
weight_decay = 0.0
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

Create `configs/dpo_full.toml`:

```toml
# Real DPO fallback run on Colab Pro. bf16 on L4 (preferred); on a T4 fallback
# set precision = "fp16". DPO fine-tunes the SFT policy directly against a frozen
# SFT reference on issue 04's preference pairs — the same artifact the Reward
# Model consumes. beta is the KL strength (0.1, Rafailov et al. 2023; distinct
# from GRPO's beta ~= 0.03). Prerequisites on Hub/disk: issue 04's pairs +
# tokenizer_full and the SFT checkpoint. [init].hub_source pulls the SFT artifact
# if the local copy is absent (fresh VM).
out_dir = "artifacts/dpo_full"

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

[dpo]
beta = 0.1

[train]
steps = 400
micro_batch_size = 8
grad_accum = 4
peak_lr = 5e-6
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
run_name = "dpo"

[hub]
target = "hf://congthanh991/tinystories-v2-dpo"
```

Create `docs/schemas/dpo-artifact-v1.md`:

```markdown
# DPO Artifact v1

Issue 08 pins the DPO fallback stage's `out_dir` artifact: the checkpoints,
metrics, and `manifest.json` metadata. The DPO checkpoint is a plain FableLM
policy — a drop-in third model for the eval suite (issue 07), loaded by
`eval.load_stage_model` exactly like the base and SFT checkpoints.

## Artifact layout

```
<out_dir>/
  checkpoints/step_XXXXXX.pt   full training state (atomic; resume contract)
  metrics.jsonl                one line per log_every steps
  manifest.json                stage metadata (below)
```

Each `checkpoints/step_XXXXXX.pt` holds `{step, pairs_seen, model, optimizer,
scaler, config}`, where `model` is the **policy** `state_dict` (a FableLM). The
frozen reference is **not** stored — it is re-derived from the `[init]` SFT
checkpoint on every run, so it is a pure function of `[init]` and resume is
bitwise-identical.

Each `metrics.jsonl` line: `{"step", "loss", "lr", "margin", "pairs_seen"}`,
where `loss` is the batch DPO loss and `margin` is the batch-mean implicit
reward margin `beta * [(logπ_c - logπ_r) - (logπ_ref_c - logπ_ref_r)]`.

## manifest.json

```json
{
  "stage": "dpo",
  "package_version": "0.1.0",
  "final_step": 400,
  "final_loss": 0.52,
  "heldout_margin": 0.31,
  "beta": 0.1,
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

`heldout_margin` is the mean implicit reward margin over the held-out pairs: a
value `> 0` means the trained policy prefers the chosen completions over the
rejected ones more strongly than the frozen SFT reference does (the policy
shifted the intended way). The `pair_split` recipe makes the held-out slice
reproducible: a seeded permutation of the encoded pairs with the last
`round(n_pairs * holdout_frac)` held out (shared with the Reward Model via
`reward.split_pairs`).

## Consumer contract

The DPO checkpoint is consumed by the eval suite (issue 07) as any other stage:
add an `[[stages]]` block pointing `local_dir` (and optionally `hub_source`) at
the DPO `out_dir`. It is scored, metriced, and win-rated identically to base and
SFT — no DPO-specific eval path. DPO consumes the identical preference-pair
artifact as the Reward Model (issue 05); there is no separate labeling path.
```

In `pyproject.toml`, add the `ts2-dpo` entry to `[project.scripts]` (after `ts2-reward`):

```toml
ts2-reward = "tinystories_v2.reward:main"
ts2-dpo = "tinystories_v2.dpo:main"
```

In `configs/eval_full.toml`, add a commented DPO stage block after the `rlaif` one so the real eval run can wire the DPO checkpoint as a third model:

```toml
# [[stages]]
# name = "rlaif"
# local_dir = "artifacts/grpo_full"
# hub_source = "hf://congthanh991/tinystories-v2-grpo"

# The DPO fallback (issue 08) is a drop-in third model — uncomment to compare it
# against base/SFT (and GRPO, if that landed) in the same win-rate tables.
# [[stages]]
# name = "dpo"
# local_dir = "artifacts/dpo_full"
# hub_source = "hf://congthanh991/tinystories-v2-dpo"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dpo_config.py tests/test_eval_stage.py -q`
Expected: PASS (the eval config still parses; the DPO configs have the required shape).

- [ ] **Step 5: Commit**

```bash
git add configs/dpo_fixture.toml configs/dpo_full.toml docs/schemas/dpo-artifact-v1.md pyproject.toml configs/eval_full.toml tests/test_dpo_config.py
git commit -m "feat: DPO configs, artifact schema, eval wiring, and ts2-dpo entrypoint (issue 08)"
```

---

### Task 7: One-command Colab bootstrap

The idempotent `download → ts2-dpo --resume` bootstrap for the real run, mirroring `scripts/reward_colab.py`.

**Files:**
- Create: `scripts/dpo_colab.py`
- Test: `tests/test_dpo_colab.py`

**Interfaces:**
- Consumes: `config.load_config`/`load_env`, `hub_download.download_file`, `dpo.run`.
- Produces: `prepare(dpo_config, *, download=None) -> Path` (returns the local pairs path; downloads tokenizer + pairs if absent); `main(argv=None)` with `--dpo-config` and `--skip-train`. Module constants `TOKENIZER_REPO`, `PAIRS_REPO`, `PAIRS_FILENAME`, `DEFAULT_DPO_CONFIG`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dpo_colab.py`:

```python
"""The DPO Colab bootstrap orchestrates download -> ts2-dpo --resume as one
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

from tinystories_v2.data import run as data_run
from tinystories_v2.judge import SlotCoverageFakeJudge, judge_with_order_swap
from tinystories_v2.slot_prompt import SLOT_FIELDS
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run

# scripts/ is not an importable package; load the bootstrap module by path.
_spec = importlib.util.spec_from_file_location(
    "dpo_colab", Path(__file__).parent.parent / "scripts" / "dpo_colab.py")
boot = importlib.util.module_from_spec(_spec)
sys.modules["dpo_colab"] = boot
_spec.loader.exec_module(boot)

_BLAND = ["A plain note with nothing much to say.",
          "Some words that go nowhere in particular."]


@pytest.fixture
def hub_and_config(tmp_path, fixture_path):
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
    dpo_cfg = tmp_path / "dpo.toml"
    dpo_cfg.write_text(
        f'out_dir = "{art / "dpo_full"}"\n\n'
        f'[data]\npairs_path = "{pairs_dst}"\n'
        f'tokenizer_path = "{tokenizer_dst}"\n', encoding="utf-8")

    sources = {
        (boot.TOKENIZER_REPO, "tokenizer.json"): hub / "tokenizer" / "tokenizer.json",
        (boot.PAIRS_REPO, boot.PAIRS_FILENAME): pairs_src,
    }
    return {"dpo_cfg": dpo_cfg, "tokenizer_dst": tokenizer_dst,
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
    pairs = boot.prepare(hub_and_config["dpo_cfg"],
                         download=_fake_download(hub_and_config["sources"], calls))
    assert {filename for _, filename in calls} == {"tokenizer.json", boot.PAIRS_FILENAME}
    assert pairs == hub_and_config["pairs_dst"]
    assert hub_and_config["tokenizer_dst"].exists() and pairs.exists()


def test_prepare_is_idempotent_on_a_warm_vm(hub_and_config):
    boot.prepare(hub_and_config["dpo_cfg"],
                 download=_fake_download(hub_and_config["sources"]))

    def boom(*args, **kwargs):
        raise AssertionError("download must not run when artifacts already exist")

    boot.prepare(hub_and_config["dpo_cfg"], download=boom)


def test_main_skip_train_prepares_without_training(hub_and_config, monkeypatch):
    monkeypatch.setattr(boot, "download_file", _fake_download(hub_and_config["sources"]))
    trained = []
    monkeypatch.setattr(boot.dpo, "run", lambda *a, **k: trained.append(True))
    boot.main(["--dpo-config", str(hub_and_config["dpo_cfg"]), "--skip-train"])
    assert hub_and_config["pairs_dst"].exists()
    assert trained == []


def test_main_trains_with_resume_after_prepare(hub_and_config, monkeypatch):
    monkeypatch.setattr(boot, "download_file", _fake_download(hub_and_config["sources"]))
    resume_flags = []

    def fake_run(config, resume=False):
        resume_flags.append(resume)
        return {"step": 1, "loss": 0.5, "heldout_margin": 0.2}

    monkeypatch.setattr(boot.dpo, "run", fake_run)
    boot.main(["--dpo-config", str(hub_and_config["dpo_cfg"])])
    assert hub_and_config["pairs_dst"].exists()
    assert resume_flags == [True]  # ts2-dpo invoked with resume=True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dpo_colab.py -q`
Expected: FAIL — `FileNotFoundError` / spec load error: `scripts/dpo_colab.py` does not exist.

- [ ] **Step 3: Write the bootstrap script**

Create `scripts/dpo_colab.py`:

```python
"""One-command real DPO fallback run for Colab (issue 08).

Turns a fresh L4 VM into a running DPO job with a single command. Idempotent:
safe to re-run after an L4 preemption — it skips already-present artifacts and
`ts2-dpo --resume` continues from the last Hub checkpoint.

Steps:
  1. load .env secrets (HF_TOKEN, WANDB_API_KEY) so Hub download/sync + W&B work
  2. download tokenizer.json + the preference pairs (issue 04's artifact) from
     the Hub (retry-wrapped) if the local copies are absent
  3. run the DPO stage (ts2-dpo, resume=True): fetches the SFT checkpoint via
     [init].hub_source (used for both the policy and the frozen reference),
     resumes any prior DPO checkpoint from the Hub, trains with the DPO loss, and
     checkpoints back to the Hub every checkpoint_every steps

Run on the VM (in-kernel, survives disconnects — never nohup-detach):
    python scripts/dpo_colab.py            # download + train
    python scripts/dpo_colab.py --skip-train   # download only
    colab exec -f scripts/dpo_colab.py

See docs/colab-notes.md for the CLI gotchas (push main first, .env via upload,
background + poll long commands, retries, always stop the VM). The preference
pairs live in issue 04's Hub repo (`PAIRS_REPO`); edit `PAIRS_REPO` /
`PAIRS_FILENAME` in this file if it moves.
"""

import argparse
from pathlib import Path

from tinystories_v2 import dpo
from tinystories_v2.config import load_config, load_env
from tinystories_v2.hub_download import download_file  # noqa: F401 — monkeypatched in tests

TOKENIZER_REPO = "congthanh991/tinystories-v2-tokenizer"
PAIRS_REPO = "congthanh991/tinystories-v2-pref-pairs"   # issue 04's preference artifact
PAIRS_FILENAME = "pairs.jsonl"
DEFAULT_DPO_CONFIG = "configs/dpo_full.toml"


def prepare(dpo_config, *, download=None) -> Path:
    """Ensure the tokenizer + preference pairs are present (download if missing).
    Returns the local pairs path. `download` is injectable for tests; it defaults
    to download_file resolved at call time. Idempotent: each step is guarded by an
    existence check, so re-running on a warm VM is a no-op up to training."""
    if download is None:
        download = download_file
    cfg = load_config(dpo_config)
    tokenizer_path = Path(cfg["data"]["tokenizer_path"])
    pairs_path = Path(cfg["data"]["pairs_path"])

    if not tokenizer_path.exists():
        print(f"[dpo_colab] downloading tokenizer -> {tokenizer_path}")
        download(TOKENIZER_REPO, "tokenizer.json", tokenizer_path.parent)
    if not pairs_path.exists():
        print(f"[dpo_colab] downloading preference pairs -> {pairs_path}")
        download(PAIRS_REPO, PAIRS_FILENAME, pairs_path.parent)
    return pairs_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dpo-config", default=DEFAULT_DPO_CONFIG, type=Path)
    parser.add_argument("--skip-train", action="store_true",
                        help="download the tokenizer + pairs only, then stop")
    args = parser.parse_args(argv)

    load_env()  # HF_TOKEN / WANDB_API_KEY reach Hub sync + wandb; never printed
    prepare(args.dpo_config)
    if args.skip_train:
        print("[dpo_colab] --skip-train: inputs ready; skipping training")
        return
    print("[dpo_colab] starting DPO training (ts2-dpo --resume)")
    summary = dpo.run(load_config(args.dpo_config), resume=True)
    print(f"[dpo_colab] done: step {summary['step']}, loss {summary['loss']:.4f}, "
          f"held-out margin {summary['heldout_margin']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dpo_colab.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/dpo_colab.py tests/test_dpo_colab.py
git commit -m "feat: one-command DPO Colab bootstrap (download -> ts2-dpo --resume) (issue 08)"
```

---

### Task 8: Thin Colab notebook

The documented parallel path for the real run (acceptance criterion 6). Must stay a thin wrapper: setup + the single bootstrap invocation, enforced by `tests/test_notebook.py`.

**Files:**
- Create: `notebooks/dpo_colab.ipynb`
- Modify: `tests/test_notebook.py`

**Interfaces:**
- Consumes: nothing at runtime (the notebook shells out to `scripts/dpo_colab.py`).
- Produces: `notebooks/dpo_colab.ipynb` whose code cells contain `scripts/dpo_colab.py`, no `def`/`class`/`import torch`/`for`/`while`, no `hf_` token literals, and no committed outputs.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_notebook.py`:

```python
DPO_NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "dpo_colab.ipynb"


def test_dpo_notebook_is_thin():
    cells = json.loads(DPO_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert 1 <= len(code_cells) <= 4
    source = "\n".join("".join(c["source"]) for c in code_cells)
    for forbidden in ("def ", "class ", "import torch", "for ", "while "):
        assert forbidden not in source, forbidden
    # Turnkey: the notebook invokes the one-command bootstrap (which itself
    # downloads the tokenizer + pairs, then runs ts2-dpo --resume) rather than
    # the stage directly.
    assert "scripts/dpo_colab.py" in source


def test_dpo_notebook_has_no_secrets_or_outputs():
    text = DPO_NOTEBOOK.read_text(encoding="utf-8")
    assert "hf_" not in text
    cells = json.loads(text)["cells"]
    assert all(not c.get("outputs") for c in cells)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_notebook.py -k dpo -q`
Expected: FAIL — `FileNotFoundError: .../notebooks/dpo_colab.ipynb`.

- [ ] **Step 3: Create the notebook**

Create `notebooks/dpo_colab.ipynb` (valid nbformat 4 JSON; note the `<` in `hf_` checks — the token string `hf_` must not appear anywhere, so do not paste a token):

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# DPO fallback on Colab Pro (L4 preferred, T4 fallback)\n",
    "\n",
    "One command per docs/DESIGN.md: clone → install → secrets → `scripts/dpo_colab.py`.\n",
    "\n",
    "That bootstrap downloads the tokenizer + preference pairs (issue 04's artifact, the *same*\n",
    "pairs the Reward Model consumes) from the Hub, then runs the DPO stage (ts2-dpo with\n",
    "--resume) — which initializes both the policy and the frozen reference from the SFT\n",
    "checkpoint via `[init].hub_source`, trains with the hand-written DPO loss, and checkpoints\n",
    "back to the Hub. It records the held-out reward margin in the artifact's manifest.json. It\n",
    "is idempotent: after an L4 preemption, just re-run the last cell to resume from the latest\n",
    "Hub checkpoint. The output checkpoint is a drop-in third model for the eval suite (issue\n",
    "07). All logic lives in the package/script; edit `configs/dpo_full.toml`, not here. See\n",
    "`docs/colab-notes.md` for the run procedure.\n",
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
    "!python scripts/dpo_colab.py"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {"provenance": []},
  "kernelspec": {"display_name": "Python 3", "name": "python3"},
  "language_info": {"name": "python"}
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_notebook.py -k dpo -q`
Expected: PASS (2 passed). Verify the JSON is valid: `python -c "import json; json.load(open('notebooks/dpo_colab.ipynb'))"`.

- [ ] **Step 5: Commit**

```bash
git add notebooks/dpo_colab.ipynb tests/test_notebook.py
git commit -m "feat: thin DPO Colab notebook wrapping the bootstrap (issue 08)"
```

---

### Task 9: Full-suite green + issue status update

Run the whole suite, confirm the editable install exposes `ts2-dpo`, then record the issue as code-complete.

**Files:**
- Modify: `PROGRESS.md`
- Modify: `.scratch/tinystories-v2-pipeline/issues/08-dpo-fallback.md` (status line + check the acceptance boxes)

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS — all prior tests plus the new `test_dpo_*` files green.

- [ ] **Step 2: Confirm the console entrypoint resolves**

Run: `python -m pip install -e . -q && ts2-dpo --help`
Expected: the argparse help for the DPO stage prints (the first docstring line as the description), confirming the `pyproject` entrypoint from Task 6.

- [ ] **Step 3: Update the issue file's acceptance checklist and status**

In `.scratch/tinystories-v2-pipeline/issues/08-dpo-fallback.md`, change `Status: ready-for-agent` to `Status: code-complete` and tick each acceptance box:
- `[x]` DPO loss direct test → `tests/test_dpo_loss.py`
- `[x]` toy DPO run shifts the policy (margin increases) on CPU → `tests/test_dpo_stage.py::test_toy_dpo_shifts_policy_toward_chosen`
- `[x]` consumes the identical preference-pair artifact as issue 05 → `reward.load_pairs`/`split_pairs` reused; same schema; `docs/schemas/dpo-artifact-v1.md`
- `[x]` kill-and-resume + W&B metrics → `tests/test_dpo_resume.py`, `MetricsLogger`
- `[x]` output checkpoint is a drop-in third eval model → `tests/test_dpo_stage.py::test_output_checkpoint_is_eval_drop_in` + `configs/eval_full.toml`
- `[x]` thin Colab notebook → `notebooks/dpo_colab.ipynb`

- [ ] **Step 4: Update `PROGRESS.md`**

In the "Now" section, add a bullet mirroring the issue-05 entry's style:

```markdown
- ✅ **Issue 08 (DPO fallback stage) code complete** — `ts2-dpo` stage
  (hand-written DPO loss on the SFT policy against a frozen SFT reference,
  deterministic held-out split shared with issue 05, checkpoint-resume),
  `configs/dpo_{fixture,full}.toml`, the one-command `scripts/dpo_colab.py`
  bootstrap + `dpo_colab.ipynb`, and the `dpo-artifact-v1` schema all landed
  with tests green. Consumes issue 04's preference pairs (identical artifact to
  issue 05 — no separate labeling path); the output checkpoint is a drop-in
  third model for the eval suite (issue 07). The real run additionally needs
  issue 03's SFT checkpoint and issue 04's labeled pairs.
```

In the "Issue board" table, change issue 08's status cell to `✅ code complete (real run needs 03 ckpt + 04 labels)`. Add a dated entry to the "Log" section summarizing the same. Update the "Now" note about "highest-leverage grabs" if issue 08 is no longer among the ready code-work items.

- [ ] **Step 5: Commit**

```bash
git add PROGRESS.md .scratch/tinystories-v2-pipeline/issues/08-dpo-fallback.md
git commit -m "docs: mark issue 08 (DPO fallback stage) code complete"
```

---

## Self-Review

**1. Spec coverage** (issue 08 acceptance criteria → task):
- "DPO loss has a direct test: hand-computed loss on a tiny batch with known log-probs" → Task 1 (`test_dpo_loss.py::test_dpo_loss_matches_hand_computation`). ✓
- "Toy DPO run … shifts the policy toward chosen (margin increases), on CPU" → Task 4 (`test_toy_dpo_shifts_policy_toward_chosen`, `heldout_margin > 0` + falling loss). ✓
- "Consumes the identical preference-pair artifact as issue 05 — no separate labeling path" → Task 2 imports `reward.load_pairs`/`split_pairs`; same `preference-pair-v1` schema; documented in Task 6's schema. ✓
- "Kill-and-resume works; metrics stream to W&B when enabled" → Task 5 (`test_dpo_resume.py`) + Task 4 wires `MetricsLogger(out_dir, config.get("wandb"))`. ✓
- "Output checkpoint is a drop-in third model for the eval suite (issue 07)" → Task 4 (`test_output_checkpoint_is_eval_drop_in` via `eval.load_stage_model`) + Task 6 (commented eval stage block). ✓
- "Thin Colab notebook exists for the real run" → Task 8. ✓
- "Build it as a sibling of the other stages — no special-case wiring" → `dpo.py` mirrors `reward.py`/`sft.py`; reuses `build_optimizer`/`lr_at`, checkpoint/config/tracking/hub verbatim. ✓
- Blocked-by 02 (model) + 10 (schema): uses `FableLM` and `preferences.validate_preference_pair` (via `reward.load_pairs`). ✓

**2. Placeholder scan:** every code step contains complete code; every run step has an exact command and expected output. No "TBD"/"add error handling"/"similar to". The one intentional cross-reference (loss/batch primitives reused by later tasks) is defined in Tasks 1–2 with full signatures in the Interfaces blocks. ✓

**3. Type consistency:** `sequence_logprobs(logits, y, mask) -> [B]`, `dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta) -> scalar`, `implicit_reward_margins(...) -> [B]`, `encode_pairs(...) -> list[dict]` with keys `chosen_ids/chosen_mask/rejected_ids/rejected_mask`, `get_pair_batch(...) -> ((cx,cy,cm),(rx,ry,rm))`, `_load_sft_state -> state`, `_build_model -> FableLM`, `evaluate_margin(...) -> float`, `run(config, resume) -> {"step","loss","heldout_margin"}`. The checkpoint state (`step, pairs_seen, model, optimizer, scaler, config`) and manifest keys (`stage, package_version, final_step, final_loss, heldout_margin, beta, pair_split, pairs_path, n_pairs, config`) are used identically in Task 4's `run`, Task 4/5 tests, and Task 6's schema. Config section names (`model, data, init, split, dpo, train, wandb, hub`) match across the two configs, the `to_toml` helpers, and `test_dpo_config.py`. ✓

---

**Plan complete and saved to `docs/superpowers/plans/2026-07-12-08-dpo-fallback-stage.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

**Which approach?**
