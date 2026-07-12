"""DPO fallback stage: fine-tune the SFT policy directly on Judge preference
pairs against a frozen SFT reference, with a hand-written DPO loss (issue 08).

Invoke standalone:
    ts2-dpo --config configs/dpo_fixture.toml [--resume]
    (or: python -m tinystories_v2.dpo --config ...)

The pre-committed stage-3 fallback (ADR-0004): if GRPO is unstable or the Reward
Model can't clear its gate by the schedule checkpoint, ship DPO as the aligned
model. It consumes the *identical* preference-pair artifact as the Reward Model
(issue 05) and produces a plain FableLM checkpoint that is a drop-in third model
for the eval suite (issue 07). Reuses issue 02's checkpoint-resume contract,
optimizer conventions (build_optimizer), LR schedule (lr_at), precision knob,
W&B logging, and Hub sync verbatim (ADR-0005: libraries only at the edges).

Both the policy and the frozen reference initialize from the SFT checkpoint in
[init]; the reference is always re-derived from that fixed checkpoint (never a
resumed policy), so it is a pure function of [init] and resume stays bitwise.

Artifacts in <out_dir> (schema: docs/schemas/dpo-artifact-v1.md):
    checkpoints/step_XXXXXX.pt   full training state (atomic; resume contract)
    metrics.jsonl                one line per log_every steps: loss, lr, margin, pairs_seen
    manifest.json                stage, version, final step/loss, heldout_margin,
                                 beta, pair_split recipe, pairs_path, config
"""

import argparse
import json
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from tinystories_v2 import __version__
from tinystories_v2.checkpoint import (
    latest_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint,
)
from tinystories_v2.config import load_config, load_env
from tinystories_v2.hub import fetch_from, try_sync_to
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.pretrain import build_optimizer, lr_at
from tinystories_v2.reward import load_pairs, split_pairs
from tinystories_v2.slot_prompt import encode_example
from tinystories_v2.tracking import MetricsLogger


def sequence_logprobs(logits: torch.Tensor, y: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """Sum of per-token target log-probs over active (mask==1) positions.

    logits [B, T, V] are next-token scores for inputs x = ids[:-1]; y [B, T] are
    the shifted targets ids[1:]; mask [B, T] is 1 over the fable body + <|end|>
    and 0 over the prompt prefix and right-padding. Returns [B]: the completion
    log-probability log p(completion | prompt) the model assigns to each row."""
    logp = F.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)   # [B, T]
    return (token_logp * mask).sum(dim=-1)                       # [B]


def implicit_reward_margins(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
                            ref_chosen: torch.Tensor, ref_rejected: torch.Tensor,
                            beta: float) -> torch.Tensor:
    """Per-pair DPO implicit-reward margin (Rafailov et al. 2023):
    beta * [ (logπ_c - logπ_ref_c) - (logπ_r - logπ_ref_r) ]. Positive means the
    policy prefers chosen over rejected more than the frozen reference does. [B]."""
    return beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor,
             beta: float) -> torch.Tensor:
    """-log σ(beta * [(logπ_c - logπ_r) - (logπ_ref_c - logπ_ref_r)]), averaged
    (ADR-0005, hand-written; no TRL DPOTrainer). Minimized when the policy raises
    the chosen-minus-rejected completion log-ratio above the frozen reference's."""
    logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    return -F.logsigmoid(beta * logits).mean()
