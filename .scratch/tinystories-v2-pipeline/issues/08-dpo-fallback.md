# 08 — DPO fallback stage

Status: code-complete

## Parent

`.scratch/tinystories-v2-pipeline/PRD.md`

## What to build

The pre-committed stage-3 fallback (ADR-0004, design doc): a DPO training
stage consuming the same preference-pair artifact as the Reward Model, fine-
tuning the SFT model directly on (chosen, rejected) pairs with the frozen SFT
reference — hand-written loss (ADR-0005), reusing checkpoint-resume, config,
and W&B conventions. Development and tests use fake-Judge pairs (issue 10's
schema) with toy checkpoints; the production run additionally needs issue
03's SFT checkpoint and issue 04's labeled pairs.

This exists so the W5 fallback decision is cheap: if GRPO is unstable or the
Reward Model can't clear its gate by the schedule checkpoint, the team ships
DPO as the aligned model and reports the GRPO attempt honestly. It also gives
the report a GRPO-vs-DPO comparison for free when both land. Build it as a
sibling of the other stages — no special-case wiring.

## Acceptance criteria

- [x] DPO loss has a direct test: hand-computed loss on a tiny batch with known log-probs matches the implementation — `tests/test_dpo_loss.py`
- [x] Toy DPO run through the stage entrypoint on fake-Judge preference pairs shifts the policy toward chosen completions (chosen-vs-rejected reward margin increases), on CPU — `tests/test_dpo_stage.py::test_toy_dpo_shifts_policy_toward_chosen` (held-out margin > 0)
- [x] Consumes the identical preference-pair artifact as issue 05 — no separate labeling path — reuses `reward.load_pairs`/`split_pairs`; same `preference-pair-v1` schema
- [x] Kill-and-resume works; metrics stream to W&B when enabled — `tests/test_dpo_resume.py` (bitwise, incl. pre-kill checkpoint immutability); `MetricsLogger` wired
- [x] Output checkpoint is a drop-in third model for the eval suite (issue 07) — `tests/test_dpo_stage.py::test_output_checkpoint_is_eval_drop_in`; commented `[[stages]]` block in `configs/eval_full.toml`
- [x] Thin Colab notebook exists for the real run — `notebooks/dpo_colab.ipynb` (+ `scripts/dpo_colab.py` bootstrap)

## Blocked by

- `02-model-pretraining-stage.md`
- `10-judge-seam-and-pair-schema.md`

(Production run also waits on `03-sft-stage.md` and
`04-judge-seam-preference-labeling.md`, but the code and tests do not.)
