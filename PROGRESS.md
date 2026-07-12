# Progress

Team-facing snapshot of where the project stands. Update the table and log
whenever an issue changes state — the issue files in
`.scratch/tinystories-v2-pipeline/issues/` (their `Status:` lines) are the
source of truth; this file is the at-a-glance view.

_Last updated: 2026-07-12_

## Now

- 🔄 **Real Pretraining run in progress on a VM** (issue 02 — code complete,
  all acceptance criteria met). Watch the W&B loss curve; checkpoints sync to
  the Hub.
- ✅ **Issue 03 (SFT stage + demo) code complete** — `ts2-sft` stage,
  `ts2-demo` script, and the thin `sft_colab.ipynb` all landed with tests
  green. A real SFT run still waits on issue 02's Pretraining checkpoint (in
  flight on the VM). Unblocks **issue 04** (preference labeling); issue 07
  (eval) is still blocked on issue 11.
- 🟢 Highest-leverage grabs now: **issue 04** (unblocked by 03), and the
  ready code-work issues **05, 08, 09, 11**.

## Issue board

| # | Issue | Blocked by | Status |
|---|-------|-----------|--------|
| 01 | Walking skeleton (scaffold, fixture, data-prep, tokenizer) | — | ✅ complete |
| 10 | Judge seam + preference-pair schema | — | ✅ complete |
| 02 | Model + Pretraining stage | 01 | 🔄 in progress — code done, real run on VM |
| 11 | Reference-free metrics library | — | 🟢 ready |
| 12 | Slot Prompt renderer + SFT dataset builder | — | ✅ complete |
| 05 | Reward Model stage + accuracy gate | 02 ✅code, 10 ✅ | 🟢 ready (code work) |
| 08 | DPO fallback stage | 02 ✅code, 10 ✅ | 🟢 ready (code work) |
| 09 | Architecture ablation at 5M scale | 02 ✅code | 🟢 ready |
| 03 | SFT stage + demo script | 02 ✅code, 12 ✅ | ✅ code complete (real run gated on 02's checkpoint) |
| 04 | Preference labeling stage | 03 ✅code, 10 ✅ | 🟢 ready (code work) |
| 07 | Evaluation suite | 03, 10 ✅, 11 ⏳ | 🔴 blocked |
| 06 | GRPO stage | 05, 11 ⏳ | 🔴 blocked |

Production-run gates (beyond code): 05 and 08 need 03's SFT checkpoint and
04's real labels; 06 additionally needs 05's Reward Model to clear the
accuracy gate (~68% held-out pair accuracy).

## Milestones vs plan (`docs/DESIGN.md`)

| Week | Planned | Actual |
|------|---------|--------|
| W1 | repo skeleton, tokenizer, splits, packed data | ✅ done 2026-07-11 (day 1) |
| W2 | Pretraining runs | 🔄 started 2026-07-12 — ahead of schedule |
| W3 | SFT | issue 12 ✅ + SFT (03) code-complete 2026-07-12; real run waits on 02's checkpoint |
| W4–5 | Judge labeling, Reward Model + gate | seam (10) already done; labeling waits on SFT |
| W5–6 | GRPO (fallback decision point mid-W5) | — |
| W7–8 | eval suite, 5M ablation, report | — |

## Log

- **2026-07-12** — Issue 03 (SFT stage + demo) code complete: `sft.py` stage
  (masked-loss fine-tune from a Pretraining checkpoint, checkpoint-resume,
  `ts2-sft`), `demo.py` (`ts2-demo`), `configs/sft_{fixture,full}.toml`,
  `notebooks/sft_colab.ipynb`, and tests (batching, stage, kill-resume,
  `<|end|>` format-learning, demo, notebook). 143 tests green. Real SFT run
  still gated on issue 02's Pretraining checkpoint (in flight on the VM).
  Unblocks issue 04.
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

- VM replaces Colab Pro as the primary training environment? If yes,
  `docs/DESIGN.md`'s compute/budget section and the precision defaults
  (bf16 vs fp16) should be updated to match the VM's GPU.
