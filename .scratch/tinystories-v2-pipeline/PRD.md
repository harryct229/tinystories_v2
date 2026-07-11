# PRD: tinystories_v2 — Fable LM trained from scratch with RLAIF

Status: ready-for-agent

## Problem Statement

We are a university team (2–4 students, 4–8 weeks) taking a Generative AI
course. We must deliver a graded project that demonstrates we understand
modern LM training end-to-end: the course explicitly grades our ability to
justify **why** each architecture layer and training-stage choice was made.
We want to train a small language model **from scratch** that generates moral
Fables conditioned on a six-slot Scaffold (character, trait, setting,
conflict, resolution, moral), and align it with **Reinforcement Learning
where a local model, not humans, provides the feedback** (RLAIF). Our only
compute is free-tier Google Colab (T4, ~3–4h sessions that disconnect without
warning), so naive long-running training scripts will lose work and blow the
deadline. Nothing exists yet: the repo contains only the dataset paper,
design docs, and an HF API key.

## Solution

A Python package plus thin Colab notebooks implementing the three-stage
pipeline resolved in `docs/DESIGN.md` and the ADRs:

1. **Pretraining** — a hand-written ~27M-param Llama-style decoder-only
   transformer (RMSNorm, RoPE, SwiGLU, tied embeddings) trained on ~500M
   tokens of TF1-EN-3M with a custom 8k BPE tokenizer.
2. **SFT** — fine-tuning on compact Slot Prompt → Fable pairs so the model
   generates a fable from any Scaffold.
3. **RLAIF** — the full InstructGPT recipe: the Judge (Qwen3-4B-Instruct-2507)
   labels pairwise preferences offline; a Reward Model (SFT model + scalar
   head, Bradley-Terry loss) distills them; GRPO optimizes the policy against
   the Reward Model with a KL leash to the SFT reference.

Every long-running stage checkpoints to private HF Hub repos and resumes
cleanly; metrics stream to a shared W&B project; evaluation uses a
cross-family judge (Llama-3.1-8B-Instruct) plus the dataset paper's
reference-free metrics, giving the report a measurable before/after at every
stage.

## User Stories

1. As a student team member, I want a custom 8k BPE tokenizer trained on the corpus with the Slot Prompt special tokens reserved, so that embedding parameters stay ~16% of the model budget instead of consuming all of it.
2. As a student team member, I want a data-prep stage that downloads TF1-EN-3M once and produces disjoint pretrain/SFT/preference/eval splits, so that no evaluation Fable ever leaks into training and the report's splits are defensible.
3. As a student team member, I want the pretrain split packed into a binary of token IDs stored on HF Hub, so that a fresh Colab session starts training in minutes instead of re-tokenizing 1.1M fables.
4. As a student team member, I want a hand-written Llama-style model whose every component maps to a citable justification, so that the report's "why these layers" section is grounded in code we wrote and understand.
5. As a student team member, I want a Pretraining loop with fp16 AMP, gradient accumulation, cosine schedule, and W&B logging, so that a ~27M model reaches ~500M tokens within 1–2 free T4 sessions.
6. As a student team member, I want every training stage to checkpoint model+optimizer+progress to HF Hub at short intervals and resume from the latest checkpoint with one flag, so that a Colab disconnect costs minutes, not a session.
7. As a student team member, I want a script that extracts the six Scaffold slots from the dataset's verbose prompt field, so that SFT and evaluation use the compact Slot Prompt format instead of ~170 tokens of boilerplate.
8. As a student team member, I want an SFT stage that masks loss on Slot Prompt tokens, so that the model learns to write Fables conditioned on Scaffolds rather than to parrot prompts.
9. As a student team member, I want a generation utility (temperature/top-p, batched, seedable), so that sampling from any checkpoint for demos, preference data, or eval is one call.
10. As a student team member, I want a preference-data stage that samples N completions per Scaffold from the SFT model and forms candidate pairs, so that the Judge has something to compare.
11. As a student team member, I want the Judge to produce pairwise A/B verdicts using the paper's adherence-weighted rubric, judging each pair twice with order swapped and keeping only consistent verdicts, so that Reward Model labels aren't polluted by position bias or score-calibration noise.
12. As a student team member, I want Judge labeling to run as a resumable offline batch job that appends to a Hub dataset, so that ~10–12k pair labels accumulate across 1–2 sessions without losing progress.
13. As a student team member, I want a Reward Model stage (SFT weights + scalar head, Bradley-Terry loss) that reports held-out pair accuracy, so that we know the reward signal beats chance before spending GPU-hours on RL.
14. As a student team member, I want the pipeline to refuse to start GRPO when Reward Model held-out accuracy is below the ~68% gate, so that we don't optimize a policy against noise.
15. As a student team member, I want a GRPO stage (G rollouts per Slot Prompt, group-mean baseline, PPO-style clipping, KL penalty to the frozen SFT reference), so that stage 3 is genuinely reinforcement learning as the course brief demands.
16. As a student team member, I want GRPO to log reward, KL divergence, and diversity (Self-BLEU) per step, so that reward hacking and diversity collapse are visible while the run is still cheap to stop.
17. As a student team member, I want a DPO fallback that trains on the same Judge preference pairs, so that if GRPO is unstable at the schedule checkpoint we still ship an aligned model and an honest comparison.
18. As a student team member, I want an evaluation stage that computes win-rates of base vs SFT vs RLAIF on the same held-out Scaffolds using a cross-family judge, so that we don't grade the policy with the same model family that wrote its reward signal.
19. As a student team member, I want the paper's reference-free metrics (Self-BLEU, Distinct-n, Flesch Reading Ease) plus held-out perplexity computed per stage, so that the report compares directly against the dataset paper's published tables.
20. As a student team member, I want a fixed qualitative sample sheet (same Scaffolds rendered by all three stages), so that the report shows, not just tells, what each stage improved.
21. As a teammate, I want all real code in an installable package with thin Colab notebooks (clone → install → run script with config), so that we review each other's work as normal diffs instead of notebook JSON.
22. As a teammate, I want every stage driven by a declarative config with sensible defaults from the design doc, so that runs are reproducible and hyperparameter changes are diffable.
23. As a teammate, I want secrets (HF token, W&B key) read from the environment/.env and never committed or printed, so that a leaked notebook or repo doesn't leak credentials.
24. As a teammate, I want artifacts in private HF Hub repos (tokenizer, packed data, preference pairs, checkpoints, final models), so that anyone on the team can reproduce or continue any run from any machine.
25. As a teammate, I want the full test suite to run on a laptop CPU in minutes with no GPU, network, or 4B-parameter Judge, so that we can develop locally and only spend Colab hours on real runs.
26. As a grader, I want the model's layer choices, stage design, and their justifications recorded in the report backed by ADRs and the design doc, so that the academic requirement — justified choices — is met with citations.
27. As a grader, I want honest reporting of failures and fallbacks (e.g., GRPO instability → DPO), so that the project demonstrates engineering judgment rather than cherry-picking.
28. As a student presenting a demo, I want a small script that takes six slot values and prints the RLAIF model's Fable, so that the final presentation has a live, tangible artifact.

## Implementation Decisions

All decisions below were resolved in a design grilling on 2026-07-11 and are
recorded in `docs/DESIGN.md`, `CONTEXT.md` (vocabulary), and ADRs 0001–0006.
Key ones for the implementing agent:

- **Three-stage pipeline** (ADR-0001): Pretraining → SFT → RLAIF. The RLAIF
  stage is the full InstructGPT recipe (ADR-0004) with GRPO instead of PPO
  (ADR-0006). DPO exists only as a pre-committed fallback.
- **Model** (ADR-0002): hand-written PyTorch Llama-style decoder-only module;
  d_model 512, 8 layers, 8 heads, context 512, pre-norm RMSNorm, RoPE, SwiGLU
  (hidden 1408), no biases, tied embeddings, dropout 0. ~27M params.
- **Tokenizer** (ADR-0003): custom byte-level BPE, vocab 8192, trained on a
  corpus sample; Slot Prompt special tokens reserved at creation time.
- **Slot Prompt format**: compact special-token encoding of a Scaffold —
  `<|character|>…<|trait|>…<|setting|>…<|conflict|>…<|resolution|>…<|moral|>…<|fable|>`
  terminated by `<|end|>`. Slots are regex-extracted from the dataset's fixed
  verbose prompt template (the dataset has no slot columns; verify the regex
  against real records, not the paper).
- **Hand-written training loops** (ADR-0005): pretraining, SFT, Bradley-Terry
  RM loss, and GRPO are all implemented in-repo. Libraries only at the edges:
  `datasets`/`tokenizers`/`huggingface_hub` for data and artifacts,
  `transformers` for Judge and eval-judge inference.
- **Stages as config→artifacts entrypoints**: each stage (tokenizer,
  data-prep, pretrain, sft, judge-label, rm, grpo, eval) is an independently
  invocable entrypoint that reads a declarative config and produces versioned
  artifacts. Stages communicate only through artifacts, never in-memory state.
- **Judge behind an interface**: Judge access goes through a small client
  interface (real impl: Qwen3-4B-Instruct-2507 fp16 via transformers) so the
  eval judge (Llama-3.1-8B-Instruct 4-bit) and test fakes are drop-in.
- **Checkpoint-resume is a contract, not a feature**: every long-running
  stage periodically persists full training state to HF Hub and can resume
  from the latest state with a flag. Free-tier Colab disconnects are the
  normal case, not the exception.
- **Data splits are disjoint by fable**: ~1.1M-fable pretrain slice (~500M
  tokens), ~50k SFT, ~4k preference Scaffolds, ~5k+1k eval. Split membership
  is deterministic and recorded with the artifacts.
- **Precision**: fp16 AMP with gradient scaling (T4 is Turing — no bf16).
- **Tracking**: shared W&B project; metrics must survive session death.
- **Storage**: private HF Hub repos for every artifact; Google Drive at most
  as scratch cache. Secrets from environment/.env (gitignored).

## Testing Decisions

Two seams, confirmed with the user:

1. **Stage entrypoints (primary seam).** Tests invoke the real stage
   entrypoint with a toy config and a ~100-fable fixture and assert on the
   artifact contract: expected files exist, shapes and dtypes correct,
   training loss decreased, resume-from-checkpoint reproduces state, RM
   held-out accuracy beats chance on synthetically separable pairs, GRPO
   raises mean reward on a rigged reward function. Real code paths at toy
   scale (tiny model config, seconds on CPU) — no mocking of our own code.
2. **Judge interface (secondary seam).** A deterministic fake Judge (e.g.,
   prefers the fable that mentions the moral slot) lets the whole
   labeling → RM → GRPO chain run in tests with no GPU, network, or large
   model. The consistency-filtering and order-swap logic is tested through
   this same interface.

Good tests here assert external behavior only: artifact contracts, metric
directions, and invariants — never layer internals or private attributes. The
model's forward boundary gets a small set of behavioral invariant tests
(causality: position-t logits unaffected by future tokens; parameter count in
budget; fp32/fp16 output closeness), since the hand-written model is itself a
public interface of the package (ADR-0005). No prior test art exists in this
repo — this PRD establishes the convention. Everything must pass on laptop
CPU; GPU-scale correctness is verified by the real Colab runs, not the suite.

## Out of Scope

- **Human evaluation** — the user chose cross-family LLM judge + reference-free metrics only; a blind human eval was explicitly not selected.
- **Architecture ablations** (5M ladder, RoPE-vs-learned, GPT-2 comparison) and a **PPO-vs-GRPO comparison** — stretch material only if the schedule allows (>8-week option was not selected).
- **Slot-adherence checker script** — offered during evaluation design, not selected.
- **Multilingual fables, other datasets, or dataset regeneration** — TF1-EN-3M English only.
- **Deployment** beyond a demo generation script (no API, no app).
- **Colab Pro features, non-T4 GPUs, distributed training.**
- **GitHub Issues migration** — tracker is local markdown until the team creates a shared remote.

## Further Notes

- Respect the vocabulary in `CONTEXT.md`: Fable, Scaffold, Slot Prompt,
  Pretraining, SFT, RLAIF (never "RLHF"), Judge, Reward Model.
- Week-1 validation tasks called out in the design: measure real T4
  tokens/sec against the ~30–60k estimate; confirm Qwen3-4B-Instruct-2507
  loads in fp16 on T4; write the slot-extraction regex against real dataset
  records.
- Milestones and the GRPO→DPO fallback decision point are in
  `docs/DESIGN.md`; each stage must yield report material even if later
  stages slip.
- The repo has no commits yet — the implementing agent's first act should be
  an initial commit of the existing docs before scaffolding code.
