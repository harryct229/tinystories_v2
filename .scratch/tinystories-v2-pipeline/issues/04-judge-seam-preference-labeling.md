# 04 — Preference labeling stage

Status: code-complete — real labeling run ready (issue 03's SFT checkpoint is on the Hub)

## Parent

`.scratch/tinystories-v2-pipeline/PRD.md`

## What to build

The offline stage that produces Reward Model training data, built on the
Judge seam and preference-pair schema from issue 10.

**Preference labeling stage**: for each Scaffold in the pref split, sample N
completions from the SFT model (via issue 02's generation utility), form 2–3
pairs per Scaffold, and label each pair with the Judge via issue 10's
order-swap consistency filtering. The job is a resumable offline batch:
progress persists incrementally so labeling accumulates across Colab sessions
into one growing, schema-valid preference-pair artifact synced to the Hub.

## Acceptance criteria

- [x] Whole stage runs on CPU with the fake Judge on fixture data, producing an artifact that passes issue 10's schema validation
- [x] Stage applies order-swap consistency filtering; the discarded-pair rate is recorded in artifact metadata
- [x] Labeling job kill-and-resume test: no duplicate and no lost pairs after interruption
- [x] Judge implementation (real or fake) is config-selected through issue 10's interface
- [x] Sampling parameters (N per Scaffold, temperature, top-p) come from config with design-doc defaults
- [x] Thin Colab notebook exists for the real labeling run

## Blocked by

- `03-sft-stage.md`
- `10-judge-seam-and-pair-schema.md`
