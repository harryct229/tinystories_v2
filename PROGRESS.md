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
- 🔄 **Issue 03 (SFT stage + demo) in progress** — being implemented now.
  Consumes the pretrained checkpoint + issue 12's `sft_data` builder. Gates the
  whole wave-3 chain (04, 07, and downstream).

## Issue board

| # | Issue | Blocked by | Status |
|---|-------|-----------|--------|
| 01 | Walking skeleton (scaffold, fixture, data-prep, tokenizer) | — | ✅ complete |
| 10 | Judge seam + preference-pair schema | — | ✅ complete |
| 02 | Model + Pretraining stage | 01 | ✅ complete — real run done, model on Hub |
| 11 | Reference-free metrics library | — | 🟢 ready |
| 12 | Slot Prompt renderer + SFT dataset builder | — | ✅ complete |
| 05 | Reward Model stage + accuracy gate | 02 ✅, 10 ✅ | 🟢 ready (code work) |
| 08 | DPO fallback stage | 02 ✅, 10 ✅ | 🟢 ready (code work) |
| 09 | Architecture ablation at 5M scale | 02 ✅ | 🟢 ready |
| 03 | SFT stage + demo script | 02 ✅, 12 ✅ | 🔄 in progress |
| 04 | Preference labeling stage | 03, 10 ✅ | 🔴 blocked |
| 07 | Evaluation suite | 03, 10 ✅, 11 ⏳ | 🔴 blocked |
| 06 | GRPO stage | 05, 11 ⏳ | 🔴 blocked |

Production-run gates (beyond code): 05 and 08 need 03's SFT checkpoint and
04's real labels; 06 additionally needs 05's Reward Model to clear the
accuracy gate (~68% held-out pair accuracy).

## Milestones vs plan (`docs/DESIGN.md`)

| Week | Planned | Actual |
|------|---------|--------|
| W1 | repo skeleton, tokenizer, splits, packed data | ✅ done 2026-07-11 (day 1) |
| W2 | Pretraining runs | ✅ done 2026-07-12 — final loss 1.279, well ahead of schedule |
| W3 | SFT | issue 12 ✅; SFT (03) 🔄 in progress 2026-07-12 |
| W4–5 | Judge labeling, Reward Model + gate | seam (10) already done; labeling waits on SFT |
| W5–6 | GRPO (fallback decision point mid-W5) | — |
| W7–8 | eval suite, 5M ablation, report | — |

## Log

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
  broadly accurate. Enable the `[track]` extra for a W&B dashboard on the next
  real run if the team wants one (this run used Hub-synced `metrics.jsonl`).
- L4 sessions get preempted ~hourly on this account — a real SFT/RM/GRPO run
  must checkpoint frequently and rely on `--resume` (already the contract).
