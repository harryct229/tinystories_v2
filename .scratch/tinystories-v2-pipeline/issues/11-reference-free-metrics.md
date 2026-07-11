# 11 — Reference-free metrics library

Status: ready-for-agent

## Parent

`.scratch/tinystories-v2-pipeline/PRD.md` (extracted from issue 07 to unblock
parallel work)

## What to build

The metric implementations the report compares against the dataset paper's
tables, as a pure library with no model, GPU, or network dependencies —
consumed by the eval suite (issue 07) and by GRPO's diversity monitoring
(issue 06).

- **Self-BLEU** (diversity: lower = more diverse) over a set of fables
- **Distinct-n** (lexical richness; n=1 at minimum, matching the paper)
- **Flesch Reading Ease** (age fit)
- **Perplexity helper**: takes a model and tokenized held-out text, returns
  perplexity — usable with any checkpoint, including toy test models

## Acceptance criteria

- [ ] Each text metric is tested against hand-computed values on tiny inputs
- [ ] Edge cases covered: empty set, single fable, identical fables (Self-BLEU extreme), degenerate short texts
- [ ] Perplexity helper matches a hand-rolled loss computation on the fixture with a toy model, on CPU
- [ ] Metrics accept plain lists of strings — no coupling to stage artifacts or model classes (except the perplexity helper's model argument)
- [ ] Deterministic given the same inputs (any sampling, e.g. for Self-BLEU cost control, is seeded and config-driven)

## Blocked by

None — can start immediately (issue 01 is complete).
