# 07 — Evaluation suite: cross-family win-rates, reference-free metrics, sample sheet

Status: ready-for-agent

## Parent

`.scratch/tinystories-v2-pipeline/PRD.md`

## What to build

The stage that produces the report's evidence. Demoable as soon as SFT exists
(base vs SFT); the RLAIF column is added when issue 06 lands — do not block
on it.

**Win-rates**: pairwise comparisons between stage checkpoints (base vs SFT vs
RLAIF) on the same held-out eval Scaffolds, scored by a cross-family eval
judge — Llama-3.1-8B-Instruct (fp16 on L4, 4-bit on T4) — implemented as
another Judge-interface implementation (issue 10), never the Qwen Judge that
produced the reward
signal (self-preference bias, per the dataset paper). Order-swapped double
judging applies here too.

**Reference-free metrics**: Self-BLEU, Distinct-n, Flesch Reading Ease over
generated fables per stage, plus held-out perplexity for base/SFT checkpoints
— computed via issue 11's metrics library, directly comparable to the dataset
paper's tables.

**Qualitative sample sheet**: a fixed set of eval Scaffolds rendered by every
stage checkpoint side by side, exported in a report-pastable format.

## Acceptance criteria

- [ ] Eval stage runs on CPU with the fake Judge and toy checkpoints, producing a results artifact with win-rate tables (with counts), metric tables, and the sample sheet
- [ ] Eval judge is config-selected via the Judge interface; a test guards that the eval-judge identity is recorded in the results artifact (so "who judged" is never ambiguous in the report)
- [ ] Comparisons use identical Scaffolds and sampling settings across checkpoints, asserted by a test
- [ ] Works with only base+SFT present; RLAIF column appears when a third checkpoint is configured
- [ ] Thin Colab notebook exists for the real eval run

## Blocked by

- `03-sft-stage.md`
- `10-judge-seam-and-pair-schema.md`
- `11-reference-free-metrics.md`
