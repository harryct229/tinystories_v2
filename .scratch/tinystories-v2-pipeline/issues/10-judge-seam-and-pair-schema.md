# 10 — Judge seam: interface, fake and real Judges, preference-pair schema

Status: ready-for-agent

## Parent

`.scratch/tinystories-v2-pipeline/PRD.md` (extracted from issue 04 to unblock
parallel work — this is the PRD's second test seam as its own slice)

## What to build

The Judge abstraction and the data contract that issues 04 (labeling),
05 (Reward Model), 07 (eval judge), and 08 (DPO) all consume. Everything here
operates on strings — no model training code required, so it runs fully in
parallel with issue 02.

**Judge interface**: a small client abstraction taking (Scaffold, fable A,
fable B) → verdict.

**Fake Judges for tests**: at least (a) a deterministic consistent fake
(e.g., prefers the fable realizing more slots) and (b) an intentionally
position-biased fake, so consistency filtering is testable.

**Real Judges**: Qwen3-8B in fp16 via transformers (fits the L4's 24 GB;
config fallback: Qwen3-4B-Instruct-2507 for T4), prompted with the dataset
paper's adherence-weighted rubric (Scaffold adherence highest, then moral
clarity; age 4–7 as constraint). Keep the interface model-agnostic — the
cross-family eval judge (issue 07) will be a third implementation.

**Order-swap consistency filtering**: reusable logic that judges every pair
twice with A/B order swapped and keeps only consistent verdicts
(position-bias filter, per the design).

**Preference-pair schema**: the pinned record format for judged pairs
(scaffold, chosen, rejected, verdict metadata: judge identity, order-swap
outcome) with a validation helper. Downstream issues build against this
schema using fake-Judge pairs on the fixture before real labels exist.

## Acceptance criteria

- [ ] Judge interface + fakes run on CPU on fixture fables with no GPU, network, or model download
- [ ] Order-swap consistency filter is tested through the interface: the position-biased fake yields discarded pairs, the consistent fake yields kept pairs
- [ ] Rubric prompt includes the four axes with adherence weighted highest and is covered by a rendering test (no model download in tests)
- [ ] Real Judge implementations are config-selected (model id, precision), sharing one code path
- [ ] Preference-pair schema is documented, has a validation helper, and a fixture-based test produces schema-valid pairs via the fake Judge
- [ ] Verdict metadata records judge identity so downstream artifacts are never ambiguous about who judged

## Blocked by

None — can start immediately (issue 01 is complete).
