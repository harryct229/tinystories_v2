import math

import torch

from tinystories_v2.dpo import dpo_loss, implicit_reward_margins, sequence_logprobs


def test_dpo_loss_matches_hand_computation():
    # policy/reference summed completion log-probs for a 2-pair batch.
    pc = torch.tensor([-2.0, -1.0])   # policy chosen
    pr = torch.tensor([-3.0, -4.0])   # policy rejected
    rc = torch.tensor([-2.5, -2.0])   # reference chosen
    rr = torch.tensor([-2.5, -3.0])   # reference rejected
    beta = 0.1
    # logits = (pc - pr) - (rc - rr) = (1.0, 3.0) - (0.0, 1.0) = (1.0, 2.0)
    logits = torch.tensor([1.0, 2.0])
    expected = (-torch.nn.functional.logsigmoid(beta * logits)).mean().item()
    assert math.isclose(dpo_loss(pc, pr, rc, rr, beta).item(), expected, rel_tol=1e-6)


def test_zero_margin_gives_log2_loss_and_zero_margin():
    # policy == reference -> logits 0 -> loss = -log sigma(0) = log 2, margin 0.
    z = torch.zeros(4)
    assert math.isclose(dpo_loss(z, z, z, z, 0.1).item(), math.log(2), rel_tol=1e-6)
    assert torch.allclose(implicit_reward_margins(z, z, z, z, 0.1), torch.zeros(4))


def test_margin_is_beta_scaled_and_signed():
    pc, pr = torch.tensor([0.0]), torch.tensor([-1.0])   # policy prefers chosen
    rc, rr = torch.tensor([0.0]), torch.tensor([0.0])    # reference indifferent
    # margin = 0.1 * ((0 - 0) - (-1 - 0)) = 0.1 * 1.0 = 0.1
    assert math.isclose(implicit_reward_margins(pc, pr, rc, rr, 0.1)[0].item(), 0.1, rel_tol=1e-6)


def test_sequence_logprobs_sums_active_positions_only():
    # 1 row, T=3, V=2. First position is prompt (masked); last two are completion.
    logits = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]])   # [1, 3, 2]
    y = torch.tensor([[1, 0, 1]])
    mask = torch.tensor([[0.0, 1.0, 1.0]])
    logp = torch.log_softmax(logits, dim=-1)
    expected = (logp[0, 1, 0] + logp[0, 2, 1]).item()
    got = sequence_logprobs(logits, y, mask)
    assert got.shape == (1,)
    assert math.isclose(got[0].item(), expected, rel_tol=1e-6)
