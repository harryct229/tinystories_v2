# Three-stage pipeline: Pretraining → SFT → RLAIF

The project brief said "train from scratch with RLHF", but RL cannot train a
randomly-initialized model — it needs a policy that already produces coherent
text. We decided on the full three-stage pipeline: (1) Pretraining a ~25M-param
decoder-only LM on TF1-EN-3M fable text, (2) SFT on the dataset's prompt→fable
pairs so the model follows the six-slot Scaffold, (3) RLAIF using a local Qwen
model as the Judge instead of human annotators. This mirrors the canonical
InstructGPT recipe, gives each stage a measurable before/after for the report,
and correctly renames the feedback stage RLAIF (the feedback source is a model,
not humans).
