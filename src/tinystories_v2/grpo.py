"""GRPO stage: reinforcement learning of the policy against the frozen Reward
Model, with a hand-written group-relative PPO loss (issues 06; ADR-0004,
ADR-0006, ADR-0005).

Invoke standalone:
    ts2-grpo --config configs/grpo_fixture.toml [--resume]
    (or: python -m tinystories_v2.grpo --config ...)

Stage 3's genuine RL (the course brief's requirement). Per step: sample a batch
of Slot Prompts from the pref split, draw G rollouts each from the policy, score
them with the frozen Reward Model (issue 05), form group-relative advantages
(group-mean baseline, no value network — ADR-0006), and update the policy with a
PPO-style clipped surrogate plus a KL penalty to the frozen SFT reference.
Reuses issue 02's checkpoint-resume contract, optimizer conventions
(build_optimizer), LR schedule (lr_at), precision knob, W&B logging, and Hub
sync verbatim (ADR-0005: libraries only at the edges).

The stage refuses to start unless the Reward Model clears its accuracy gate
(issue 05's gate). Reward is behind an injectable seam (run(reward_fn=...)) so
the whole chain runs on CPU in tests and a rigged reward drives the mean-reward
test through the real entrypoint. The output checkpoint is a plain FableLM
policy — a drop-in RLAIF model for the eval suite (issue 07).

Artifacts in <out_dir> (schema: docs/schemas/grpo-artifact-v1.md):
    checkpoints/step_XXXXXX.pt   full training state (atomic; resume contract)
    metrics.jsonl                one line per log_every steps: loss, lr,
                                 reward_mean, kl, self_bleu, policy_loss,
                                 rollouts_seen
    manifest.json                stage, version, final step/loss, final
                                 reward_mean/kl, reward_gate recipe, grpo
                                 hyperparameters, pref_split, config

Determinism contract: the Scaffold batch is a pure function of (seed, step),
each rollout's sampling of (seed, step, prompt_index); the frozen reference is a
pure function of [init] and the Reward Model of [reward]; optimizer + scaler
state round-trip, so an interrupted-and-resumed run reproduces the uninterrupted
run exactly (fp32 CPU; asserted by tests/test_grpo_resume.py).
"""

import argparse
import json
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from tinystories_v2 import __version__
from tinystories_v2.checkpoint import (
    latest_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint,
)
from tinystories_v2.config import load_config, load_env
from tinystories_v2.gate import DEFAULT_ACCURACY_GATE, check_reward_gate
from tinystories_v2.generate import sample
from tinystories_v2.hub import fetch_file_from, fetch_from, try_sync_to
from tinystories_v2.metrics import self_bleu, tokenize_words
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.pretrain import build_optimizer, lr_at
from tinystories_v2.reward_model import RewardModel, score_fables
from tinystories_v2.slot_prompt import END_TOKEN, SLOT_FIELDS, render_prompt
from tinystories_v2.slots import Scaffold
from tinystories_v2.tracking import MetricsLogger


# --- loss library (ADR-0005, ADR-0006): pure tensor functions -----------------

def token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-token target log-probs. logits [B, T, V] are next-token scores for
    inputs x = seq[:-1]; targets [B, T] are the shifted tokens seq[1:]. Returns
    [B, T]: log p(targets[b, t]) under the model at position t. Unlike DPO's
    summed sequence_logprobs, GRPO needs the per-token grid for the PPO ratio
    and the per-token KL."""
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def group_relative_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Group-relative advantages (ADR-0006: group-mean baseline, no value
    network). rewards [P, G] are the G rollout rewards for each of P Slot
    Prompts. Each row is mean-centred (the baseline) and divided by its
    population std + eps (scale normalization, DeepSeek-R1 practice). A group
    with no reward spread yields ~0 advantages — no learning signal that step.
    Returns [P, G]."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, unbiased=False, keepdim=True)
    return (rewards - mean) / (std + eps)


def clipped_policy_loss(logprobs: torch.Tensor, old_logprobs: torch.Tensor,
                        advantages: torch.Tensor, mask: torch.Tensor,
                        clip_eps: float) -> torch.Tensor:
    """PPO-style clipped surrogate (negated for minimization), masked-mean over
    active completion tokens. logprobs/old_logprobs/mask are [B, T]; advantages
    [B] is per-rollout, broadcast across tokens. ratio = exp(logπ - logπ_old);
    surrogate = min(ratio·A, clip(ratio, 1±ε)·A). old_logprobs are the sampling
    policy's (detached), so within a step the first update has ratio 1 and the
    clip binds only across ppo_epochs > 1."""
    ratio = torch.exp(logprobs - old_logprobs)
    adv = advantages.unsqueeze(-1)                                   # [B, 1]
    surrogate = torch.min(ratio * adv,
                          torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    return -(surrogate * mask).sum() / mask.sum().clamp(min=1.0)


def kl_penalty(logprobs: torch.Tensor, ref_logprobs: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """Per-token KL(policy‖reference) via the k3 unbiased estimator
    exp(Δ) - Δ - 1 with Δ = logπ_ref - logπ (Schulman; non-negative, low
    variance), masked-mean over completion tokens. The leash that keeps GRPO
    from drifting off the SFT manifold; β scales it in grpo_loss."""
    delta = ref_logprobs - logprobs
    per_token = torch.exp(delta) - delta - 1.0
    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)


def grpo_loss(logprobs: torch.Tensor, old_logprobs: torch.Tensor,
              ref_logprobs: torch.Tensor, advantages: torch.Tensor,
              mask: torch.Tensor, clip_eps: float,
              kl_beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Total GRPO objective = clipped policy surrogate + β·KL-to-reference.
    Returns (total, policy_loss, kl) so the stage can log the parts. Setting
    kl_beta = 0 disables the leash (config-only, criterion 3)."""
    policy = clipped_policy_loss(logprobs, old_logprobs, advantages, mask, clip_eps)
    kl = kl_penalty(logprobs, ref_logprobs, mask)
    return policy + kl_beta * kl, policy, kl
