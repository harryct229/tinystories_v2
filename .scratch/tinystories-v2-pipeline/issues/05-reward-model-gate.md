# 05 — Reward Model stage + accuracy gate

Status: ready-for-agent

## Parent

`.scratch/tinystories-v2-pipeline/PRD.md`

## What to build

The stage that distills Judge preferences into a scalar reward signal, and
the gate that protects GRPO from a bad one.

**Reward Model** (per CONTEXT.md): the SFT model with its LM head replaced by
a scalar head, trained on the preference-pair artifact with Bradley-Terry
loss (hand-written, ADR-0005), reusing the checkpoint-resume and W&B
conventions. Holds out a slice of pairs for accuracy measurement.

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

- `04-judge-seam-preference-labeling.md`
