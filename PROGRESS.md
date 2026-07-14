# Progress

Team-facing snapshot of where the project stands. Update the table and log
whenever an issue changes state — the issue files in
`.scratch/tinystories-v2-pipeline/issues/` (their `Status:` lines) are the
source of truth; this file is the at-a-glance view.

_Last updated: 2026-07-13_

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
- ✅ **Issue 05 (Reward Model + gate) DONE — real run complete, GATE PASSED.**
  400 steps of Bradley-Terry training on Colab L4 (bf16, LR 1e-5) over issue
  04's pairs (8,086 train / 898 holdout): final loss 0.146, **held-out pair
  accuracy 73.9% ≥ the ~68% gate** — **GRPO (issue 06) is green-lit**; the
  DPO fallback (08) is not gate-required. Model on the private Hub
  (`congthanh991/tinystories-v2-reward`). Caveat for the report: accuracy is
  measured on margin-kept pairs (|margin| > 0.5), the discriminable subset.
- ✅ **Issue 11 (reference-free metrics) complete** — pooled and paper-aligned
  per-Fable Distinct-n, seeded Self-BLEU, Flesch Reading Ease, and a lazy-Torch
  held-out perplexity helper are merged with CPU-only deterministic tests.
  This unblocks **issue 07** and removes one of issue 06's two blockers.
- ✅ **Issue 04 (preference labeling) DONE — real labeling run complete.**
  All 4,053 pref Scaffolds labeled on Colab L4: **8,984 schema-valid pairs**
  (73.9% keep, 26.1% margin-discard, 0 degenerate, 0 judge errors) on the
  private Hub (`congthanh991/tinystories-v2-pref-pairs`). The judge was
  redesigned mid-run: Qwen3-8B's greedy A/B verdicts saturated at 'A'
  (position prior p≈1.0, 100% order-swap discards), so labeling now uses
  **logit-margin debiasing** (`transformers_margin`, tau=0.5 calibrated 8/8
  against thinking-mode ground truth). Unblocks the **real runs of 05 (RM)
  and 08 (DPO)**.
- ✅ **Issue 08 (DPO fallback) DONE — real run complete.** 400 DPO steps
  (β=0.1, bf16, LR 5e-6) on Colab L4 over issue 04's pairs (same 8,086/898
  split as the RM): final loss 0.185, **held-out reward margin 0.515** (the
  policy prefers chosen over rejected). Model on the private Hub
  (`congthanh991/tinystories-v2-dpo`) — a drop-in third model for the eval
  suite (07) alongside SFT and, later, GRPO.
- ✅ **Issue 06 (GRPO) DONE — real run complete.** 300 GRPO steps on Colab L4
  (8 prompts × G=8 rollouts × 384 tokens, clip 0.2, KL β=0.03, LR 2e-6):
  **final mean reward 5.226, final KL 0.039** — the leash held, no diversity
  collapse flagged. Model on the private Hub (`congthanh991/tinystories-v2-grpo`,
  W&B run `grpo`). The full InstructGPT recipe is now real end-to-end:
  pretrain → SFT → labels → RM (gate passed) → GRPO. Ops notes: two run-killers
  fixed in flight — the runner watchdog now watches artifact mtimes (GRPO
  prints nothing per-step), and Hub Xet downloads (which hung on 359MB blobs)
  are disabled on VMs (`HF_HUB_DISABLE_XET=1`).
- 🟢 Highest-leverage grabs now: **the real eval run (07)** — all four models
  on Hub — and **09** (ablation).

## Issue board

| # | Issue | Blocked by | Status |
|---|-------|-----------|--------|
| 01 | Walking skeleton (scaffold, fixture, data-prep, tokenizer) | — | ✅ complete |
| 10 | Judge seam + preference-pair schema | — | ✅ complete |
| 02 | Model + Pretraining stage | 01 | ✅ complete — real run done, model on Hub |
| 11 | Reference-free metrics library | — | ✅ complete |
| 12 | Slot Prompt renderer + SFT dataset builder | — | ✅ complete |
| 05 | Reward Model stage + accuracy gate | 02 ✅, 10 ✅ | ✅ complete — RM on Hub, held-out acc 73.9% (gate ~68% passed) |
| 08 | DPO fallback stage | 02 ✅, 10 ✅ | ✅ complete — DPO model on Hub (held-out margin 0.515) |
| 09 | Architecture ablation at 5M scale | 02 ✅ | 🟢 ready |
| 03 | SFT stage + demo script | 02 ✅, 12 ✅ | ✅ complete — real SFT run done, model on Hub |
| 04 | Preference labeling stage | 03 ✅, 10 ✅ | ✅ complete — 8,984 real pairs on Hub (margin judge) |
| 07 | Evaluation suite | 03 ✅, 10 ✅, 11 ✅ | ✅ code complete (real run needs stage checkpoints on Hub) |
| 06 | GRPO stage | 05 ✅, 11 ✅ | ✅ complete — RLAIF model on Hub (reward 5.23, KL 0.039) |

Production-run gates (beyond code): all cleared for 06 — the SFT checkpoint,
8,984 labeled pairs, and a gate-passing Reward Model (73.9% ≥ ~68%) are on
the Hub. 07/08 real runs are likewise fully unblocked.

## Milestones vs plan (`docs/DESIGN.md`)

| Week | Planned | Actual |
|------|---------|--------|
| W1 | repo skeleton, tokenizer, splits, packed data | ✅ done 2026-07-11 (day 1) |
| W2 | Pretraining runs | ✅ done 2026-07-12 — final loss 1.279, well ahead of schedule |
| W3 | SFT | issue 12 ✅ + SFT (03) done 2026-07-12 — real run complete, final loss 1.083, model on Hub |
| W4–5 | Judge labeling, Reward Model + gate | ✅ done 2026-07-13 — 8,984 pairs + RM on Hub, gate passed (73.9% ≥ 68%), a week+ ahead |
| W5–6 | GRPO (fallback decision point mid-W5) | ✅ done 2026-07-14 — RLAIF model on Hub (reward 5.23, KL 0.039); DPO (08) banked too — ~2 weeks ahead |
| W7–8 | eval suite, 5M ablation, report | reference-free metrics (11) ✅; eval suite (07) ready |

## Log

- **2026-07-14** — **Issue 06 real GRPO run complete — the RLAIF model
  exists.** 300 steps (8 prompts × G=8 × 384 tokens, clip 0.2, KL β=0.03,
  LR 2e-6, ~60s/step on L4): final mean reward **5.226**, final KL **0.039**;
  artifact on the Hub (`tinystories-v2-grpo`: step_000300.pt, manifest with
  gate provenance; W&B run `grpo`). Two infrastructure faults diagnosed and
  fixed during the run: (1) the supervised-runner watchdog killed healthy
  training because GRPO prints nothing per-step — staleness is now judged on
  metrics/checkpoint mtimes; (2) Hub Xet-backed downloads hung indefinitely
  on the 359MB SFT checkpoint — VMs now set `HF_HUB_DISABLE_XET=1` (20s
  download vs 25-min hang). Cost ≈ 7 L4-h incl. the debugging. Next: the
  four-model eval run (07).
- **2026-07-13** — Issue 06 (GRPO stage) code complete: `grpo.py` stage
  (`ts2-grpo`) — a hand-written group-relative PPO loss (group-mean baseline,
  no value network, ADR-0006) with a KL leash to a frozen SFT reference
  (ADR-0004/0005), enforcing issue 05's accuracy gate before any model or
  `out_dir` is created. Per step: sample a batch of Slot Prompts from the
  pref split, draw G rollouts per prompt from the policy, score them with the
  frozen Reward Model (issue 05), form group-relative advantages, and update
  the policy with the clipped surrogate + KL penalty; reward/KL/Self-BLEU
  stream to W&B so reward hacking and diversity collapse are visible early.
  Kill-and-resume is bitwise-identical (frozen reference and Reward Model are
  re-derived from their source artifacts, never checkpointed). Output is a
  plain FableLM checkpoint — a drop-in third model (`rlaif`) for the eval
  suite (07), loaded by `eval.load_stage_model` exactly like base/SFT.
  Landed: `configs/grpo_{fixture,full}.toml`,
  `docs/schemas/grpo-artifact-v1.md`, the one-command `scripts/grpo_colab.py`
  bootstrap (download tokenizer + pref split, then `ts2-grpo --resume`), a
  thin `grpo_colab.ipynb`, and the `rlaif` `[[stages]]` block activated in
  `configs/eval_full.toml`. Full suite green (368 passed): group-relative
  advantages, clipped policy loss, KL penalty, gate refusal, rigged-reward
  mean-reward-rises test through the real entrypoint, the fake-Judge→toy-RM→
  toy-GRPO whole chain, kill-and-resume, bootstrap orchestration, and
  notebook thinness — all CPU-only, no GPU/network. Built subagent-driven
  from `docs/superpowers/plans/2026-07-13-06-grpo-stage.md`. The real run
  additionally needs the SFT checkpoint (issue 03), the gate-passing Reward
  Model (issue 05), and the pref split (issue 01) — all three already on the
  Hub, so the real run is fully unblocked.
- **2026-07-13** — **Issue 08 real DPO run complete.** 400 steps (β=0.1,
  bf16, LR 5e-6) over the 8,984 pairs (same 8,086/898 deterministic split as
  the RM, seed 20260712): final DPO loss 0.185, held-out reward margin
  **0.515**. Artifact on the Hub (`tinystories-v2-dpo`: step_000400.pt,
  manifest; W&B run `dpo`). Single ~50-min L4 session via the
  supervised-runner pattern, no preemption. The DPO checkpoint joins SFT as
  an eval-suite (07) model; GRPO (06) remains the mainline RLAIF path.
- **2026-07-13** — **Issue 05 real Reward Model run complete — GATE PASSED.**
  400 BT steps on an L4 (bf16, LR 1e-5) over the 8,984 pairs (8,086/898
  deterministic split, seed 20260712): final loss 0.146, held-out pair
  accuracy **0.739 ≥ ~0.68 gate**. Artifact on the Hub
  (`tinystories-v2-reward`: step_000400.pt, manifest, metrics; W&B run
  `reward`). Single ~35-min session via the supervised-runner pattern.
  GRPO (06) is green-lit; DPO (08) remains optional insurance. Report
  caveat: accuracy is measured on margin-kept (discriminable) pairs.
- **2026-07-13** — Issue 08 (DPO fallback stage) code complete: `dpo.py`
  stage (`ts2-dpo`) — hand-written DPO loss `-log σ(β·[(logπ_c−logπ_r) −
  (logπ_ref_c−logπ_ref_r)])` (β=0.1) over completion log-probs, fine-tuning the
  SFT policy against a frozen SFT reference re-derived from the fixed `[init]`
  checkpoint (never stored in the DPO checkpoint, so resume stays bitwise). Data
  path reuses `reward.load_pairs`/`split_pairs` (identical preference-pair
  artifact as issue 05 — no separate labeling), and the output checkpoint is a
  plain FableLM that `eval.load_stage_model` loads as a drop-in third model
  (issue 07). `configs/dpo_{fixture,full}.toml`, `docs/schemas/dpo-artifact-v1.md`,
  the one-command `scripts/dpo_colab.py` bootstrap, and a thin `dpo_colab.ipynb`
  landed with tests green (332 passed): direct loss test, toy separable-pair run
  raising held-out margin > 0 with falling loss, kill-and-resume (bitwise +
  pre-kill checkpoint immutability), eval drop-in load, bootstrap orchestration,
  and notebook thinness. Built subagent-driven from
  `docs/superpowers/plans/2026-07-12-08-dpo-fallback-stage.md`. Real run
  additionally needs issue 03's SFT checkpoint and issue 04's labeled pairs.
- **2026-07-13** — **Issue 04 real labeling run complete.** All 4,053 pref
  Scaffolds → **8,984 schema-valid preference pairs** on the Hub
  (`tinystories-v2-pref-pairs`): 73.9% kept, 26.1% margin-discarded, 0
  degenerate, 0 judge errors, single judge_id
  `transformers-margin:Qwen/Qwen3-8B;precision=fp16;tau=0.5`. The first
  attempt discarded 100% of pairs — Qwen3-8B's greedy A/B verdict saturates
  at 'A' (position prior p≈1.0) on same-model completions, and probes showed
  prompt restructuring, the 4B fallback, and temperature-mixing all fail. Fix:
  **logit-margin debiasing** (read A/B first-token logits in both orders; the
  position prior cancels in the half-difference), validated 8/8 against
  thinking-mode ground truth and calibrated tau=0.5 (~74% keep). Also fixed
  in flight: `fetch_file_from` falls back to dataset repo type (the data repo
  is dataset-type). Ops: ~6 preemption cycles survived via the supervised
  runner (bounded retries + stall watchdog) and Hub-synced resume; ~10 L4-h
  labeling + ~3 L4-h diagnosis/calibration. Real runs of 05 (RM) and 08 (DPO)
  are now fully unblocked.
- **2026-07-12** — Issue 07 (evaluation suite) code complete: `eval.py` stage
  (`ts2-eval`) — order-swapped cross-family win-rates over stage checkpoints,
  issue 11's reference-free metrics + held-out perplexity per stage, and a
  report-pastable sample sheet, writing `results.json` (schema:
  `docs/schemas/eval-results-v1.md`) + `report.md`. Generation feeds every
  stage identical Scaffolds/seeds/sampling (asserted); the eval Judge is
  config-selected via the issue 10 interface (Llama-3.1-8B-Instruct for real
  runs, never the Qwen reward Judge) and its identity is recorded in the
  artifact. Works with base+SFT alone; the RLAIF column appears when a third
  `[[stages]]` block is configured. `configs/eval_{fixture,full}.toml`, the
  one-command `scripts/eval_colab.py` bootstrap, and a thin `eval_colab.ipynb`
  landed with tests green (win-rate/metric/report/generate units, a CPU stage
  test on toy checkpoints with the fake Judge, bootstrap orchestration, and
  notebook thinness).
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
