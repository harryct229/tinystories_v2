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


# --- rollouts, batching, Scaffold loading -------------------------------------

def load_scaffolds(path, tokenizer: Tokenizer, context: int) -> list[Scaffold]:
    """Read pref-split rows into Scaffolds whose Slot Prompt fits within the
    context with room to generate. Rows whose render_prompt tokenizes to >=
    context tokens are dropped (the sampler could not extend them). Mirrors
    pref_data's per-row length guard."""
    scaffolds = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            scaffold = Scaffold(**{fld: row[fld] for fld in SLOT_FIELDS})
            if len(tokenizer.encode(render_prompt(scaffold)).ids) < context:
                scaffolds.append(scaffold)
    return scaffolds


def get_scaffold_batch(scaffolds: list[Scaffold], prompts_per_step: int, *,
                       seed: int, step: int) -> list[Scaffold]:
    """A with-replacement batch of Slot Prompts for one step; a pure function of
    (seed, step) so a resumed run draws the identical prompts (resume contract)."""
    generator = torch.Generator()
    generator.manual_seed((seed * 1_000_003 + step) % 2**63)
    picks = torch.randint(0, len(scaffolds), (prompts_per_step,),
                          generator=generator).tolist()
    return [scaffolds[i] for i in picks]


def sample_rollouts(policy: FableLM, tokenizer: Tokenizer, scaffold: Scaffold, *,
                    group_size: int, max_new_tokens: int, temperature: float,
                    top_p: float, seed: int,
                    device: str = "cpu") -> tuple[list[list[int]], int, list[str]]:
    """Draw group_size rollouts from the policy for one Slot Prompt. Returns
    (sequences, prompt_len, fable_texts): each sequence is prompt ids + generated
    ids (truncated at <|end|> when emitted); fable_texts are the decoded fable
    bodies (prompt + specials excluded). Seeded via generate.sample, so a resumed
    step reproduces the identical rollouts. Leaves the policy in eval mode — the
    caller restores train() (dropout is 0, so this is belt-and-braces)."""
    prompt_ids = tokenizer.encode(render_prompt(scaffold)).ids
    sequences = sample(
        policy, prompt_ids, num_samples=group_size, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p, seed=seed,
        end_id=tokenizer.token_to_id(END_TOKEN), device=device)
    fables = [tokenizer.decode(seq[len(prompt_ids):]).strip() for seq in sequences]
    return sequences, len(prompt_ids), fables


def rollout_batch(sequences: list[list[int]], prompt_lens: list[int],
                  context: int, device: str) -> tuple[torch.Tensor, ...]:
    """Right-pad rollouts into next-token (x, y, mask) tensors. Each sequence is
    truncated to context+1 ids, then shifted: x = seq[:-1], y = seq[1:]. The mask
    is active over completion targets only — y index t predicts seq[t+1], which
    is generated once t+1 >= prompt_len, i.e. t >= prompt_len-1 — and 0 over the
    prompt prefix and right-padding (causal attention makes right-padding safe)."""
    rows = []
    for seq, plen in zip(sequences, prompt_lens):
        seq = seq[:context + 1]
        x, y = seq[:-1], seq[1:]
        mask = [1.0 if t >= plen - 1 else 0.0 for t in range(len(y))]
        rows.append((x, y, mask))
    width = max(len(x) for x, _, _ in rows)
    xs, ys, ms = [], [], []
    for x, y, m in rows:
        pad = width - len(x)
        xs.append(x + [0] * pad)
        ys.append(y + [0] * pad)
        ms.append(m + [0.0] * pad)
    return (torch.tensor(xs, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device),
            torch.tensor(ms, dtype=torch.float, device=device))


def safe_self_bleu(fables: list[str]) -> float:
    """Self-BLEU over rollouts with a diversity-collapse guard: rollouts with no
    words are dropped, and fewer than two usable rollouts yields NaN (undefined,
    not an error) so the metric never wedges the training loop."""
    usable = [f for f in fables if f.strip() and tokenize_words(f)]
    if len(usable) < 2:
        return float("nan")
    try:
        return self_bleu(usable)
    except ValueError:
        return float("nan")


# --- reward seam: frozen Reward Model as the scoring function -----------------

def _load_reward_model(reward_cfg: dict, device: str) -> RewardModel:
    """Load the frozen Reward Model (issue 05 artifact) that scores rollouts.
    Fetches the artifact from [reward].hub_source if the local checkpoint is
    absent (fresh VM). The model is eval() and requires_grad_(False) — it is a
    fixed reward, never updated by GRPO."""
    local_dir = Path(reward_cfg["local_dir"])
    ckpt_dir = local_dir / "checkpoints"
    if latest_checkpoint(ckpt_dir) is None and reward_cfg.get("hub_source"):
        fetch_from(reward_cfg["hub_source"], local_dir)  # fresh VM: pull the RM
    ckpt = latest_checkpoint(ckpt_dir)
    if ckpt is None:
        raise ValueError(
            f"no Reward Model checkpoint under {ckpt_dir}; point [reward].local_dir "
            f"(and optionally [reward].hub_source) at the issue-05 RM artifact")
    state = load_checkpoint(ckpt)
    model = RewardModel(ModelConfig(**state["config"]["model"])).to(device)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    print(f"loaded frozen Reward Model from {ckpt}")
    return model


def make_reward_scorer(reward_model: RewardModel, tokenizer: Tokenizer, device: str):
    """Wrap the Reward Model as the GRPO reward function: score(scaffold, fables)
    -> [float] over each rendered (Slot Prompt, fable). Empty/whitespace rollouts
    score 0.0 (render_example rejects an empty body), so a policy that emits
    <|end|> immediately gets no reward rather than crashing the loop."""
    def score(scaffold: Scaffold, fables: list[str]) -> list[float]:
        scores = [0.0] * len(fables)
        usable = [(i, f) for i, f in enumerate(fables) if f.strip()]
        if usable:
            values = score_fables(reward_model, tokenizer,
                                  [(scaffold, f) for _, f in usable], device=device)
            for (i, _), value in zip(usable, values):
                scores[i] = value
        return scores
    return score


# --- SFT init: policy + frozen reference (both from [init], like dpo.py) -------

def _load_sft_state(config: dict, device: str) -> dict:
    """Load the SFT checkpoint declared in [init], fetching from the Hub first if
    the local checkpoint is absent (fresh VM), and validate its architecture
    matches [model]. The returned state (contains 'model') builds both the policy
    and the frozen reference."""
    init = config["init"]
    init_dir = Path(init["local_dir"])
    init_ckpt_dir = init_dir / "checkpoints"
    if latest_checkpoint(init_ckpt_dir) is None and init.get("hub_source"):
        fetch_from(init["hub_source"], init_dir)  # fresh Colab VM: pull SFT
    init_ckpt = latest_checkpoint(init_ckpt_dir)
    if init_ckpt is None:
        raise ValueError(
            f"no SFT checkpoint under {init_ckpt_dir}; point [init].local_dir "
            f"(and optionally [init].hub_source) at the SFT artifact")
    state = load_checkpoint(init_ckpt)
    if ModelConfig(**state["config"]["model"]) != ModelConfig(**config["model"]):
        raise ValueError(
            f"[model] does not match the SFT checkpoint at {init_ckpt}; GRPO must "
            f"optimize the SFT architecture")
    print(f"loaded SFT weights from {init_ckpt}")
    return state


def _build_model(config: dict, state: dict, device: str) -> FableLM:
    """Build a FableLM from [model] and load the SFT weights (strict)."""
    model = FableLM(ModelConfig(**config["model"])).to(device)
    model.load_state_dict(state["model"])
    return model
