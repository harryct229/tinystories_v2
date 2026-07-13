import math

import pytest
import torch

from tinystories_v2.dpo import _build_model, _load_sft_state, evaluate_margin
from tinystories_v2.model import FableLM

TOY_MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 64, "ffn_hidden": 192}


def _cfg(init_dir, model=None):
    return {"model": dict(model or TOY_MODEL), "init": {"local_dir": str(init_dir)}}


def test_builds_policy_and_reference_from_sft(tmp_path, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, "tok.json")
    state = _load_sft_state(_cfg(init), "cpu")
    policy = _build_model(_cfg(init), state, "cpu")
    reference = _build_model(_cfg(init), state, "cpu")
    # Both start equal to the SFT init...
    for k, v in policy.state_dict().items():
        assert torch.equal(reference.state_dict()[k], v)
    assert isinstance(policy, FableLM) and isinstance(reference, FableLM)


def test_reference_can_be_frozen(tmp_path, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, "tok.json")
    state = _load_sft_state(_cfg(init), "cpu")
    reference = _build_model(_cfg(init), state, "cpu").requires_grad_(False)
    assert all(not p.requires_grad for p in reference.parameters())


def test_missing_init_checkpoint_raises(tmp_path):
    with pytest.raises(ValueError, match="no SFT checkpoint"):
        _load_sft_state(_cfg(tmp_path / "empty"), "cpu")


def test_mismatched_init_architecture_raises(tmp_path, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, "tok.json")
    drifted = dict(TOY_MODEL, d_model=128)
    with pytest.raises(ValueError, match="SFT checkpoint"):
        _load_sft_state(_cfg(init, model=drifted), "cpu")


def test_evaluate_margin_is_zero_when_policy_equals_reference(tmp_path, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, "tok.json")
    state = _load_sft_state(_cfg(init), "cpu")
    policy = _build_model(_cfg(init), state, "cpu")
    reference = _build_model(_cfg(init), state, "cpu")
    # Two-token completions; masks active on the final token only.
    holdout = [{"chosen_ids": [1, 5, 9], "chosen_mask": [0, 0, 1],
                "rejected_ids": [1, 6, 8], "rejected_mask": [0, 0, 1]}]
    margin = evaluate_margin(policy, reference, holdout, context=64, beta=0.1)
    assert math.isclose(margin, 0.0, abs_tol=1e-6)          # identical models -> 0 margin


def test_evaluate_margin_nan_on_empty_holdout(tmp_path, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, "tok.json")
    state = _load_sft_state(_cfg(init), "cpu")
    policy = _build_model(_cfg(init), state, "cpu")
    reference = _build_model(_cfg(init), state, "cpu")
    assert math.isnan(evaluate_margin(policy, reference, [], context=64, beta=0.1))
