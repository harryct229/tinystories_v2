import math

import torch

from tinystories_v2.grpo import (
    clipped_policy_loss, grpo_loss, group_relative_advantages, kl_penalty,
    token_logprobs,
)


def test_token_logprobs_gathers_target_log_softmax():
    logits = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]])   # [1, 3, 2]
    targets = torch.tensor([[1, 0, 1]])
    logp = torch.log_softmax(logits, dim=-1)
    expected = torch.tensor([[logp[0, 0, 1], logp[0, 1, 0], logp[0, 2, 1]]])
    got = token_logprobs(logits, targets)
    assert got.shape == (1, 3)
    assert torch.allclose(got, expected)


def test_group_relative_advantages_known_in_known_out():
    # One group of four rewards. mean 1.5, population std sqrt(1.25)=1.1180.
    rewards = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    adv = group_relative_advantages(rewards, eps=0.0)
    std = math.sqrt(1.25)
    expected = torch.tensor([[-1.5, -0.5, 0.5, 1.5]]) / std
    assert torch.allclose(adv, expected, atol=1e-6)
    assert math.isclose(adv.mean().item(), 0.0, abs_tol=1e-6)   # mean-centred


def test_group_relative_advantages_constant_group_is_zero():
    # A group with no reward spread has no learning signal: eps -> all zeros.
    rewards = torch.full((2, 5), 4.0)
    adv = group_relative_advantages(rewards, eps=1e-6)
    assert torch.allclose(adv, torch.zeros(2, 5), atol=1e-4)


def test_clipped_policy_loss_binds_upper_clip_on_positive_advantage():
    # ratio = exp(logp - old) = exp(0.5) ~= 1.6487 > 1+eps=1.2 -> clip binds.
    old = torch.zeros(1, 1)
    logp = torch.full((1, 1), 0.5)
    adv = torch.tensor([2.0])
    mask = torch.ones(1, 1)
    clip_eps = 0.2
    # positive advantage: surrogate = min(ratio*A, clip(ratio)*A) = 1.2 * 2.0.
    expected = -(1.2 * 2.0)
    got = clipped_policy_loss(logp, old, adv, mask, clip_eps)
    assert math.isclose(got.item(), expected, rel_tol=1e-6)


def test_clipped_policy_loss_masks_inactive_positions():
    old = torch.zeros(1, 2)
    logp = torch.tensor([[0.5, 9.0]])         # second position would explode if counted
    adv = torch.tensor([1.0])
    mask = torch.tensor([[1.0, 0.0]])         # only the first token is active
    got = clipped_policy_loss(logp, old, adv, mask, 0.2)
    assert math.isclose(got.item(), -(1.2 * 1.0), rel_tol=1e-6)


def test_kl_penalty_is_zero_at_equality_and_positive_otherwise():
    logp = torch.tensor([[-1.0, -2.0]])
    mask = torch.ones(1, 2)
    assert math.isclose(kl_penalty(logp, logp, mask).item(), 0.0, abs_tol=1e-7)
    ref = torch.tensor([[-0.5, -2.5]])
    assert kl_penalty(logp, ref, mask).item() > 0.0     # k3 estimator is non-negative


def test_grpo_loss_composes_policy_and_kl():
    logp = torch.tensor([[0.5]])
    old = torch.zeros(1, 1)
    ref = torch.tensor([[0.2]])
    adv = torch.tensor([1.0])
    mask = torch.ones(1, 1)
    total, policy, kl = grpo_loss(logp, old, ref, adv, mask, clip_eps=0.2, kl_beta=0.03)
    assert math.isclose(total.item(), (policy + 0.03 * kl).item(), rel_tol=1e-6)
    assert kl.item() >= 0.0
