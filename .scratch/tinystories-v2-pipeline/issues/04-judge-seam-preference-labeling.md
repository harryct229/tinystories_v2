# 04 — Judge seam + preference labeling stage

Status: ready-for-agent

## Parent

`.scratch/tinystories-v2-pipeline/PRD.md`

## What to build

The Judge interface (the PRD's second test seam) and the offline stage that
produces Reward Model training data.

**Judge interface**: a small client abstraction taking (Scaffold, fable A,
fable B) → verdict. Implementations: (a) the real Judge — Qwen3-4B-
Instruct-2507 in fp16 via transformers, prompted with the dataset paper's
adherence-weighted rubric (Scaffold adherence highest, then moral clarity;
age 4–7 as constraint); (b) a deterministic fake for tests (e.g., prefers the
fable realizing more slots). The eval judge in issue 07 will be a third
implementation, so keep the interface model-agnostic.

**Preference labeling stage**: for each Scaffold in the pref split, sample N
completions from the SFT model (via issue 02's generation utility), form 2–3
pairs per Scaffold, and label each pair with the Judge — judging every pair
twice with A/B order swapped and keeping only consistent verdicts (position-
bias filter, per the design). The job is a resumable offline batch: progress
persists incrementally so labeling accumulates across Colab sessions into one
growing preference-pair artifact synced to the Hub.

## Acceptance criteria

- [ ] Whole stage runs on CPU with the fake Judge on fixture data, producing a preference-pair artifact with a documented schema (scaffold, chosen, rejected, verdict metadata)
- [ ] Order-swap consistency filter is tested through the Judge interface: an intentionally position-biased fake yields discarded pairs, a consistent fake yields kept pairs
- [ ] Labeling job kill-and-resume test: no duplicate and no lost pairs after interruption
- [ ] Real Judge implementation is config-selected; its rubric prompt includes the four axes with adherence weighted highest and is covered by a rendering test (no model download in tests)
- [ ] Sampling parameters (N per Scaffold, temperature, top-p) come from config with design-doc defaults
- [ ] Thin Colab notebook exists for the real labeling run

## Blocked by

- `03-sft-stage.md`
