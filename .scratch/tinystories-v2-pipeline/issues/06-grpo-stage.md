# 06 — GRPO stage: RL against the Reward Model

Status: complete — real run done 2026-07-14: 300 steps, final reward 5.226, KL 0.039; model at hf://congthanh991/tinystories-v2-grpo

## Parent

`.scratch/tinystories-v2-pipeline/PRD.md`

## What to build

The RLAIF training loop (ADR-0004, ADR-0006), hand-written (ADR-0005):
for each batch of Slot Prompts from unseen Scaffolds, sample G rollouts per
prompt from the policy, score them with the Reward Model, compute group-
relative advantages (group-mean baseline, no value network), and update the
policy with PPO-style clipping plus a KL penalty against the frozen SFT
reference. Hyperparameters (G, clip ε, KL β, LR, steps) from config with
design-doc defaults.

Instrumentation is a first-class requirement: per-step mean reward, KL to
reference, and rollout diversity (Self-BLEU via issue 11's metrics library)
stream to W&B so reward hacking and diversity collapse are visible early.
Checkpoint-resume covers policy, optimizer, and RL progress.

The stage starts by enforcing issue 05's accuracy gate.

## Acceptance criteria

- [ ] Toy GRPO run through the stage entrypoint against a rigged reward function (e.g., rewards presence of a target token) measurably raises mean reward, asserted by a test on CPU
- [ ] Group-relative advantage computation has a direct test at the stage seam: known rewards in, known normalized advantages out
- [ ] KL penalty demonstrably constrains the policy in a toy run (KL stays bounded; disabling it is config-only)
- [ ] Refuses to start when the Reward Model artifact is below the accuracy gate
- [ ] Kill-and-resume works mid-RL-run; reward/KL/Self-BLEU curves stream to W&B
- [ ] Whole chain (fake Judge pairs → toy Reward Model → toy GRPO) runs in the test suite without GPU or network
- [ ] Thin Colab notebook exists for the real GRPO run

## Blocked by

- `05-reward-model-gate.md`
- `11-reference-free-metrics.md`
