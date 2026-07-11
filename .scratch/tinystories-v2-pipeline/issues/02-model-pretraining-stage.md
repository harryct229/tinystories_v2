# 02 — Model + Pretraining stage with checkpoint-resume contract

Status: ready-for-agent

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
(Fable text only, `<|end|>`-separated), then trains with fp16 AMP + gradient
scaling (T4 has no bf16), gradient accumulation, AdamW, warmup + cosine
schedule, grad clipping, and W&B logging (design-doc defaults; all in config).

**Checkpoint-resume contract**: the stage periodically persists full training
state (model, optimizer, scaler, progress) and resumes from the latest
checkpoint with one flag — free-Colab disconnects are the normal case.
Storage targets a local directory with a thin sync layer to private HF Hub
repos, so tests never touch the network.

**Generation utility**: batched, seedable temperature/top-p sampling from any
checkpoint — this is the shared sampling path for later stages (preference
data, eval, demo).

Also: the thin Colab notebook wrapper (clone → install → run stage with the
real config). Real-GPU throughput validation (~30–60k tokens/sec estimate) is
a human follow-up on Colab, not part of this issue's acceptance.

## Acceptance criteria

- [ ] Model invariant tests pass at the forward boundary: position-t logits unaffected by future tokens (causality), parameter count within budget for the real config, forward works in fp32 and under autocast
- [ ] Toy Pretraining run on the fixture decreases training loss, and a test asserts it through the stage entrypoint
- [ ] Kill-and-resume test: interrupted toy run resumed from latest checkpoint continues from the recorded step with matching training state
- [ ] Packed-data artifact has documented dtype/shape and a test verifies round-trip against the tokenizer
- [ ] Hub sync layer is exercised in tests against a local path (no network); pushing/pulling real Hub repos works via config
- [ ] Metrics (loss, LR, tokens seen) stream to W&B when enabled and degrade gracefully to local JSONL when not
- [ ] Generation utility produces seeded-reproducible samples from a toy checkpoint
- [ ] Thin Colab notebook exists and contains no logic beyond setup + stage invocation

## Blocked by

- `01-walking-skeleton-data-tokenizer.md`
