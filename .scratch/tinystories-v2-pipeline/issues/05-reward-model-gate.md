# 05 — Reward Model stage + accuracy gate

Status: complete — real run done 2026-07-13: held-out pair accuracy 0.739 clears the ~68% gate; model at hf://congthanh991/tinystories-v2-reward (GRPO green-lit)

## Parent

`.scratch/tinystories-v2-pipeline/PRD.md`

## What to build

The stage that distills Judge preferences into a scalar reward signal, and
the gate that protects GRPO from a bad one.

**Reward Model** (per CONTEXT.md): the SFT model with its LM head replaced by
a scalar head, trained with Bradley-Terry loss (hand-written, ADR-0005) on
preference pairs conforming to issue 10's schema, reusing the
checkpoint-resume and W&B conventions. Holds out a slice of pairs for
accuracy measurement. Development and tests use fake-Judge pairs on the
fixture with toy checkpoints — this issue is code-complete without real
labels; the production run additionally needs issue 03's SFT checkpoint and
issue 04's labeled pairs.

**Accuracy gate**: the stage records held-out pair accuracy in the Reward
Model artifact's metadata, and downstream RL refuses to start when accuracy
is below the configured gate (~68% default, per the design) — below the gate
the fix is better labels, not RL.

## Acceptance criteria

- [ ] Toy Reward Model trained through the stage entrypoint on synthetically separable fake-Judge pairs (fixture data) achieves held-out accuracy well above chance, asserted by a test
- [ ] Reward Model scores are usable downstream: a scoring call takes (Slot Prompt, fable) and returns a scalar, batched, on CPU in tests
- [ ] Held-out pair accuracy and the pair-split recipe are recorded in the artifact metadata
- [ ] Gate behavior is tested: a below-gate artifact causes the GRPO entrypoint (or a shared gate check) to refuse with a clear message; an above-gate artifact passes
- [ ] Training resumes after a kill; metrics stream to W&B when enabled
- [ ] Thin Colab notebook exists for the real Reward Model run

## Blocked by

- `02-model-pretraining-stage.md`
- `10-judge-seam-and-pair-schema.md`

(Production run also waits on `03-sft-stage.md` and
`04-judge-seam-preference-labeling.md`, but the code and tests do not.)
