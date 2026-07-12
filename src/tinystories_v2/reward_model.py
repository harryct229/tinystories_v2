"""Reward Model: the SFT backbone with a scalar head, plus the scoring and
Bradley-Terry primitives the reward stage (issue 05) and GRPO (issue 06) share.

The Reward Model reuses FableLM's transformer backbone (ADR-0005) and replaces
the tied LM head with one linear scalar head. A sequence's reward is the scalar
read from the hidden state at its last real (non-pad) token; right-padding plus
causal attention make that position independent of padding, so a Fable scores
identically alone or inside a padded batch.
"""

import torch
from tokenizers import Tokenizer
from torch import nn

from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.slot_prompt import render_example
from tinystories_v2.slots import Scaffold


class RewardModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.backbone = FableLM(config)
        self.score_head = nn.Linear(config.d_model, 1, bias=False)
        nn.init.normal_(self.score_head.weight, mean=0.0, std=0.02)

    def load_backbone_state_dict(self, backbone_state: dict) -> None:
        """Load an SFT/Pretraining state['model'] into the backbone (strict). The
        scalar head keeps its fresh init — it has no pretrained counterpart."""
        self.backbone.load_state_dict(backbone_state)

    def forward(self, idx: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Scalar reward per sequence. idx: [B, T] right-padded token ids;
        lengths: [B] real-token count per row. Returns [B]. Pools the hidden
        state at each row's last real token (lengths-1)."""
        hidden = self.backbone.hidden_states(idx)          # [B, T, d_model]
        last = (lengths - 1).clamp(min=0)
        rows = torch.arange(hidden.size(0), device=hidden.device)
        pooled = hidden[rows, last]                         # [B, d_model]
        return self.score_head(pooled).squeeze(-1)         # [B]


def pad_sequences(sequences: list[list[int]], context: int,
                  device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad token-id lists to a rectangular batch. Each sequence is first
    truncated to `context` (the backbone rejects longer inputs). Returns
    (idx [B, W] long, lengths [B] long)."""
    seqs = [s[:context] for s in sequences]
    lengths = [len(s) for s in seqs]
    width = max(lengths)
    padded = [s + [0] * (width - len(s)) for s in seqs]
    return (torch.tensor(padded, dtype=torch.long, device=device),
            torch.tensor(lengths, dtype=torch.long, device=device))


@torch.no_grad()
def score_sequences(model: RewardModel, sequences: list[list[int]], *,
                    device: str = "cpu", batch_size: int = 64) -> torch.Tensor:
    """Batched no-grad scoring of token-id sequences. Returns [len(sequences)]."""
    model = model.to(device).eval()
    context = model.config.context
    chunks = []
    for start in range(0, len(sequences), batch_size):
        idx, lengths = pad_sequences(sequences[start:start + batch_size],
                                     context, device)
        chunks.append(model(idx, lengths))
    return torch.cat(chunks) if chunks else torch.empty(0, device=device)


def score_fables(model: RewardModel, tokenizer: Tokenizer,
                 items: list[tuple[Scaffold, str]], *,
                 device: str = "cpu") -> list[float]:
    """Score each (Slot Prompt Scaffold, Fable body) on its full rendered
    sequence. The downstream scoring call (GRPO, eval, demos)."""
    sequences = [tokenizer.encode(render_example(scaffold, fable)).ids
                 for scaffold, fable in items]
    return score_sequences(model, sequences, device=device).tolist()


def bradley_terry_loss(chosen_scores: torch.Tensor,
                       rejected_scores: torch.Tensor) -> torch.Tensor:
    """-log σ(r_chosen - r_rejected), averaged. Minimized when the model scores
    every chosen Fable above its rejected partner (ADR-0005, hand-written)."""
    return -torch.nn.functional.logsigmoid(chosen_scores - rejected_scores).mean()


def pair_accuracy(chosen_scores: torch.Tensor,
                  rejected_scores: torch.Tensor) -> torch.Tensor:
    """Fraction of pairs with r_chosen > r_rejected (chance = 0.5). Computed in
    double precision so exact fractions (e.g. 1/3) match Python float literals
    bit-for-bit — float32's .mean() rounds to a value that differs from the
    float64 literal at the last bit."""
    return (chosen_scores > rejected_scores).double().mean()
