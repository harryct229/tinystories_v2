# Progress

Team-facing snapshot of where the project stands. Update the table and log
whenever an issue changes state — the issue files in
`.scratch/tinystories-v2-pipeline/issues/` (their `Status:` lines) are the
source of truth; this file is the at-a-glance view.

_Last updated: 2026-07-12_

## Now

- ✅ **Pretraining (issue 02) done — real run complete.** ~29.9M-param FableLM,
  3800 steps / 498M tokens on Colab L4 (bf16), final loss **1.279** (from 7.71).
  Model + all data artifacts on private HF Hub under `congthanh991` (see Log).
  Note: no W&B dashboard this run (the `[track]` extra wasn't installed on the
  VM) — metrics live in `metrics.jsonl`, synced to the Hub checkpoint repo.
- ✅ **Issue 03 (SFT + demo) done — real SFT run complete.** ~29.9M FableLM
  fine-tuned 800 steps / 50,572 examples on Colab L4 (bf16), final masked loss
  **1.083** (from ~1.29 at the pretrained init). Model on the private HF Hub
  (`congthanh991/tinystories-v2-sft`); `ts2-demo` generates coherent
  Scaffold-conditioned fables (W&B dashboard live this run). Unblocks **issue 04**
  (preference labeling); with issue 11 done, **issue 07** (evaluation) is ready too.
- ✅ **Issue 05 (Reward Model stage + accuracy gate) code complete** — `ts2-reward`
  stage (Bradley-Terry loss on the SFT backbone + a scalar head, deterministic
  held-out split, checkpoint-resume), the shared `gate.check_reward_gate`
  (~68% default), `configs/reward_{fixture,full}.toml`, the one-command
  `scripts/reward_colab.py` bootstrap + `reward_colab.ipynb`, and the
  `reward-model-artifact-v1` schema all landed with tests green. The real run
  consumes issue 03's SFT checkpoint and issue 04's labeled pairs. Unblocks
  **issue 06** (GRPO), which enforces the gate at startup.
- ✅ **Issue 11 (reference-free metrics) complete** — pooled and paper-aligned
  per-Fable Distinct-n, seeded Self-BLEU, Flesch Reading Ease, and a lazy-Torch
  held-out perplexity helper are merged with CPU-only deterministic tests.
  This unblocks **issue 07** and removes one of issue 06's two blockers.
- ✅ **Issue 04 (preference labeling) code complete** — `ts2-pref-data`
  samples N completions per pref Scaffold, labels round-robin pairs through
  the order-swap filter, and accumulates a kill-safe, Hub-synced
  `pairs.jsonl`. The real labeling run is ready — issue 03's SFT checkpoint
  is on the Hub.
- 🟢 Highest-leverage grabs now: the ready code-work issues **05, 07, 08, 09**.

## Issue board

| # | Issue | Blocked by | Status |
|---|-------|-----------|--------|
| 01 | Walking skeleton (scaffold, fixture, data-prep, tokenizer) | — | ✅ complete |
| 10 | Judge seam + preference-pair schema | — | ✅ complete |
| 02 | Model + Pretraining stage | 01 | ✅ complete — real run done, model on Hub |
| 11 | Reference-free metrics library | — | ✅ complete |
| 12 | Slot Prompt renderer + SFT dataset builder | — | ✅ complete |
| 05 | Reward Model stage + accuracy gate | 02 ✅, 10 ✅ | ✅ code complete (real run needs 03 ckpt + 04 labels) |
| 08 | DPO fallback stage | 02 ✅, 10 ✅ | 🟢 ready (code work) |
| 09 | Architecture ablation at 5M scale | 02 ✅ | 🟢 ready |
| 03 | SFT stage + demo script | 02 ✅, 12 ✅ | ✅ complete — real SFT run done, model on Hub |
| 04 | Preference labeling stage | 03 ✅, 10 ✅ | ✅ code complete (real run ready — 03's SFT checkpoint on Hub) |
| 07 | Evaluation suite | 03 ✅, 10 ✅, 11 ✅ | 🟢 ready (code work) |
| 06 | GRPO stage | 05 ✅code, 11 ✅ | 🟢 ready (code work) |

Production-run gates (beyond code): 03's SFT checkpoint is now on the Hub; 05
and 08 still need 04's real labels; 06 additionally needs 05's Reward Model to
clear the accuracy gate (~68% held-out pair accuracy).

## Milestones vs plan (`docs/DESIGN.md`)

| Week | Planned | Actual |
|------|---------|--------|
| W1 | repo skeleton, tokenizer, splits, packed data | ✅ done 2026-07-11 (day 1) |
| W2 | Pretraining runs | ✅ done 2026-07-12 — final loss 1.279, well ahead of schedule |
| W3 | SFT | issue 12 ✅ + SFT (03) done 2026-07-12 — real run complete, final loss 1.083, model on Hub |
| W4–5 | Judge labeling, Reward Model + gate | seam (10) already done; labeling stage (04) code complete and ready to run |
| W5–6 | GRPO (fallback decision point mid-W5) | — |
| W7–8 | eval suite, 5M ablation, report | reference-free metrics (11) ✅; eval suite (07) ready |

## Log

- **2026-07-12** — Issue 05 (Reward Model stage + accuracy gate) code complete:
  `reward_model.py` (FableLM backbone + scalar head, last-real-token scoring,
  hand-written Bradley-Terry loss), `reward.py` stage (`ts2-reward`, deterministic
  train/holdout split, checkpoint-resume, held-out accuracy + split recipe in the
  manifest), `gate.py` (`check_reward_gate`, ~68% default), configs, the
  `scripts/reward_colab.py` bootstrap (shared `hub_download` helper, extracted
  from `sft_colab.py`), thin `reward_colab.ipynb`, and
  `docs/schemas/reward-model-artifact-v1.md`. Added a behavior-preserving
  `FableLM.hidden_states` seam. Tests: scoring/padding-invariance, batching,
  separable fake-Judge pairs beat chance on held-out, kill-and-resume bitwise,
  gate refusal, and bootstrap orchestration. Unblocks issue 06 (GRPO).
- **2026-07-12** — Issue 04 (preference labeling) code complete:
  `pref_data.py` stage (`ts2-pref-data`) with per-Scaffold seeded sampling,
  round-robin pairing, order-swap consistency filtering, a per-Scaffold
  append+fsync/atomic-commit resume protocol (SIGKILL test proves byte-identical
  recovery), single-file Hub fetches for the pref split and tokenizer,
  `configs/pref_data_{fixture,full}.toml`, and a thin
  `pref_data_colab.ipynb`. The real run is ready — issue 03's SFT checkpoint
  (step_000800, masked loss 1.083) is on the Hub.
- **2026-07-12** — **Issue 03 SFT run complete.** Fine-tuned the pretrained
  ~29.9M FableLM for 800 steps on all 50,572 sft-split examples (Colab L4,
  bf16, LR 1e-4 cosine, `grad_accum` 8), final masked loss 1.083 (from ~1.29
  at init). Smoke-run validated the real-scale wiring first, then the full run;
  checkpoints synced to `hf://congthanh991/tinystories-v2-sft` (step_000800.pt),
  with a live W&B dashboard (Colab preinstalls `wandb`). `ts2-demo` produces
  coherent Scaffold-conditioned fables — the base model can't follow a Slot
  Prompt, the SFT model does. Ran via the one-command bootstrap
  `scripts/sft_colab.py`; the L4 was preempted just after the final Hub sync
  (harmless — `--resume` would have recovered). Unblocks issue 04.
- **2026-07-12** — Issue 11 (reference-free metrics library) complete and
  merged to main: shared word tokenization, pooled `distinct_n`,
  paper-comparable `mean_distinct_n`, seeded Self-BLEU, Flesch Reading Ease,
  and held-out perplexity with lazy Torch imports. Whole-branch review added
  the per-Fable aggregation contract and float32 loss accumulation for fp16
  checkpoint safety, then hardened all identified coverage gaps. Unblocks
  issue 07 and leaves issue 06 blocked only on issue 05.
- **2026-07-12** — Issue 03 (SFT stage + demo) code complete: `sft.py` stage
  (masked-loss fine-tune from a Pretraining checkpoint, checkpoint-resume,
  `ts2-sft`), `demo.py` (`ts2-demo`), `configs/sft_{fixture,full}.toml`,
  `notebooks/sft_colab.ipynb`, and tests (batching, stage, kill-resume,
  `<|end|>` format-learning, demo, notebook). 145 tests green. Whole-branch
  review applied two fixes: token-weighted grad-accum masked loss, and
  `--resume` tolerating a missing SFT Hub repo (mirrors 02's `8b77654`).
  Unblocks issue 04.
- **2026-07-12** — **Issue 02 Pretraining run complete.** ~29.9M FableLM, 3800
  steps / 498M tokens on Colab L4 (bf16), final loss 1.279 (from 7.71), LR
  cosine-decayed to its 6e-5 floor; seeded generation produces coherent
  fable-domain text. Artifacts on private HF Hub (`congthanh991`):
  `tinystories-v2-pretrain` (checkpoints/manifest/metrics),
  `tinystories-v2-tokenizer`, `tinystories-v2-packed` (746 MB uint16 binary),
  `tinystories-v2-data` (four splits). Fixed + pushed `8b77654`: `--resume`
  now starts fresh instead of crashing when the checkpoint repo doesn't exist.
  Ops lesson: run training in-kernel (not `nohup`, which lets idle VMs get
  reaped); L4s preempt ~hourly but `--resume` recovers from the last 400-step
  Hub checkpoint. Issue 03 (SFT) picks up from here.
- **2026-07-12** — Issue 12 (Slot Prompt renderer + SFT dataset builder)
  complete and merged to main: render/encode/parse in `slot_prompt.py`, the
  `sft_data` stage + `sft-example-v1` schema, 123 tests green. Unblocks issue
  03 (SFT).
- **2026-07-12** — Issue 10 (Judge seam) complete. Issue 02 code complete;
  real Pretraining run started on a VM. Issues 05, 08, 09 unblocked for code
  work.
- **2026-07-11** — Design grilled and documented (DESIGN.md, CONTEXT.md,
  ADRs 0001–0006). PRD published; issues 01–09 created, then restructured
  into parallel waves (new issues 10–12). Plan revised for Colab Pro. Issue
  01 (walking skeleton) completed.

## Open questions

- Training runs on Colab (L4, driven by the `colab` CLI), not a dedicated VM.
  L4 bf16 held ~66k+ tokens/sec; the DESIGN.md compute/budget section is
  broadly accurate. W&B needs no `[track]` extra — Colab preinstalls `wandb`,
  and the SFT run logged a live dashboard once `WANDB_API_KEY` was in `.env`
  (the pretrain run used Hub-synced `metrics.jsonl` only).
- L4 sessions get preempted ~hourly on this account — a real SFT/RM/GRPO run
  must checkpoint frequently and rely on `--resume` (already the contract).
- Colab run gotchas (CWD/package shadowing, long-exec disconnects, retrying
  `ConnectionResetError`, idle-VM reaping, Hub-as-source-of-truth) are written
  up in `docs/colab-notes.md` — read it before the next RM/GRPO run.
