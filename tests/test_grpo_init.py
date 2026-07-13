import pytest
import torch

from tinystories_v2.grpo import _build_model, _load_sft_state
from tinystories_v2.model import FableLM

TOY_MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 128, "ffn_hidden": 192}


def _cfg(init_dir, model=None):
    return {"model": dict(model or TOY_MODEL), "init": {"local_dir": str(init_dir)}}


def test_builds_policy_and_reference_from_the_same_sft(tmp_path, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, "tok.json")
    state = _load_sft_state(_cfg(init), "cpu")
    policy = _build_model(_cfg(init), state, "cpu")
    reference = _build_model(_cfg(init), state, "cpu")
    assert isinstance(policy, FableLM) and isinstance(reference, FableLM)
    for key, value in policy.state_dict().items():
        assert torch.equal(reference.state_dict()[key], value)      # identical init


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
