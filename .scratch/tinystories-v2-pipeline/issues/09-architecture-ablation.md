# 09 — Architecture ablation at 5M scale

Status: ready-for-agent

## Parent

`.scratch/tinystories-v2-pipeline/PRD.md` (user story 29 — promoted from
out-of-scope when the team upgraded to Colab Pro)

## What to build

Empirical evidence for the report's layer-justification section: train ~5M-
param variants of the model that differ in exactly one component, on the same
Pretraining data slice with the same token budget, and compare.

**Variants** (config-selected, per the design): the Llama-style baseline
(ADR-0002) versus (a) learned absolute positional embeddings instead of RoPE,
and (b) GELU MLP instead of SwiGLU (parameter-matched — widen the GELU hidden
dim so variant sizes stay comparable). This requires the model module to
support these component swaps via config; that flexibility is part of this
issue, not issue 02.

**Comparison**: held-out perplexity at matched token counts (loss curves via
W&B), plus generation samples from each variant on fixture Scaffolds. Runs
are cheap (~1–2 L4-hours for all variants) and reuse the Pretraining stage
entrypoint unchanged — an ablation is just a different model config.

## Acceptance criteria

- [ ] Model config can select positional-encoding type (RoPE / learned) and MLP type (SwiGLU / GELU); invariant tests (causality, param count within tolerance across variants) pass for every variant
- [ ] Ablation variants train through the existing Pretraining stage entrypoint with no stage-code changes — config only
- [ ] Toy runs of all variants on the fixture decrease loss, asserted by a parameterized test on CPU
- [ ] A small report artifact (table: variant, params, final val loss/perplexity at matched tokens) is produced by an eval helper, testable on toy runs
- [ ] Thin Colab notebook (or config set) exists to launch the real 5M ablation runs
- [ ] Variant configs are parameter-matched within a documented tolerance so comparisons are fair

## Blocked by

- `02-model-pretraining-stage.md`
