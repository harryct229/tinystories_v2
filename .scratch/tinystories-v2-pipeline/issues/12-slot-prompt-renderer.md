# 12 — Slot Prompt renderer + SFT dataset builder

Status: ready-for-agent

## Parent

`.scratch/tinystories-v2-pipeline/PRD.md` (extracted from issue 03 to unblock
parallel work — this pins the format contract issues 03, 04, 06, and 07 rely
on)

## What to build

The Slot Prompt format as code: rendering, parsing, and the SFT training-
example builder. Needs only issue 01's outputs (extracted slots, tokenizer
with reserved special tokens).

**Renderer**: extracted slots → the compact special-token sequence decided in
the design (the exact token order is the contract every later stage relies
on):

```
<|character|>…<|trait|>…<|setting|>…<|conflict|>…<|resolution|>…<|moral|>…<|fable|>{fable text}<|end|>
```

**Parser**: the inverse — given a token sequence, recover the slots and fable
body. Used by eval and by guards in later stages.

**SFT dataset builder**: turns the sft split into training examples with the
loss mask covering exactly the Slot Prompt segment (loss only on the Fable
body and `<|end|>`), producing an artifact per the stage convention.

## Acceptance criteria

- [ ] A test asserts the rendered token sequence matches the schema exactly: special tokens encode as single IDs, order is fixed, and the loss-mask boundary sits exactly at `<|fable|>`
- [ ] Renderer→parser round-trip recovers slots and fable body on fixture data
- [ ] Dataset builder produces a schema-documented artifact from the fixture's sft split via the stage convention, deterministically across two runs
- [ ] Malformed input handling is defined and tested (missing slot, unexpected token order)

## Blocked by

None — can start immediately (issue 01 is complete).
