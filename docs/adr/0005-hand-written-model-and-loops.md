# Hand-written model and training loops; libraries only at the edges

We implement the 25M-param model as a plain PyTorch nn.Module and hand-write
all four training procedures (pretraining, SFT, Bradley-Terry RM loss, GRPO)
instead of using HF Trainer / TRL's RewardTrainer / GRPOTrainer. Libraries are
used only where they are not the point of the course: `datasets`/`tokenizers`/
`huggingface_hub` for data and artifacts, `transformers` for Judge and eval-
judge inference. Rationale: the course grades justification of architecture
and training choices — code we wrote is code we can defend — and at 25M scale
GRPO is ~200 lines, small enough that TRL's abstractions cost more (wrapping a
custom model as PreTrainedModel, API churn, opaque Colab debugging) than they
save. Accepted cost: we own every bug.
