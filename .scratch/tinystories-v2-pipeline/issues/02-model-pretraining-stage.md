# 02 — Model + Pretraining stage with checkpoint-resume contract

Status: complete — real run done 2026-07-12 (Colab L4 bf16, 3800 steps / 498M tokens, final loss 1.279); checkpoints, tokenizer, packed data, and splits on private HF Hub

## Parent

`.scratch/tinystories-v2-pipeline/PRD.md`

## What to build

The hand-written model and the Pretraining stage, demoable as: pack the
fixture, train a toy config on CPU for a few hundred steps, kill the process,
resume from the latest checkpoint, and generate text from the result.

**Model** (ADR-0002, ADR-0005): plain PyTorch module, Llama-style decoder-only
— pre-norm RMSNorm, RoPE, SwiGLU, no biases, tied embeddings — fully
config-driven (the toy test config and the real ~27M config differ only in
numbers).

**Pretraining stage**: packs the pretrain split into a binary of token IDs
(Fable text only, `<|end|>`-separated), then trains with mixed precision as a
config knob — bf16 autocast on L4/Ada (preferred), fp16 AMP + GradScaler on
T4 (Turing has no bf16) — gradient accumulation, AdamW, warmup + cosine
schedule, grad clipping, and W&B logging (design-doc defaults; all in config).

**Checkpoint-resume contract**: the stage periodically persists full training
state (model, optimizer, scaler if used, progress) and resumes from the
latest checkpoint with one flag — Colab Pro sessions are longer but still die
(idle timeout, preemption).
Storage targets a local directory with a thin sync layer to private HF Hub
repos, so tests never touch the network.

**Generation utility**: batched, seedable temperature/top-p sampling from any
checkpoint — this is the shared sampling path for later stages (preference
data, eval, demo).

Also: the thin Colab notebook wrapper (clone → install → run stage with the
real config). Real-GPU throughput validation (design estimates: L4 in bf16
roughly 2–3× the T4's ~30–60k tokens/sec) is a human follow-up on Colab, not
part of this issue's acceptance.

## Acceptance criteria

- [x] Model invariant tests pass at the forward boundary: position-t logits unaffected by future tokens (causality), parameter count within budget for the real config, forward works in fp32 and under autocast in both bf16 and fp16
- [x] Toy Pretraining run on the fixture decreases training loss, and a test asserts it through the stage entrypoint
- [x] Kill-and-resume test: interrupted toy run resumed from latest checkpoint continues from the recorded step with matching training state
- [x] Packed-data artifact has documented dtype/shape and a test verifies round-trip against the tokenizer
- [x] Hub sync layer is exercised in tests against a local path (no network); pushing/pulling real Hub repos works via config
- [x] Metrics (loss, LR, tokens seen) stream to W&B when enabled and degrade gracefully to local JSONL when not
- [x] Generation utility produces seeded-reproducible samples from a toy checkpoint
- [x] Thin Colab notebook exists and contains no logic beyond setup + stage invocation

## Blocked by

- `01-walking-skeleton-data-tokenizer.md`
