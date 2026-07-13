import torch
from tokenizers import Tokenizer

import pytest

from tinystories_v2.checkpoint import save_checkpoint
from tinystories_v2.grpo import _load_reward_model, make_reward_scorer
from tinystories_v2.model import ModelConfig
from tinystories_v2.reward_model import RewardModel
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run

TOY_MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 128, "ffn_hidden": 192}


def _rm_artifact(dir_path, model_cfg=TOY_MODEL):
    """Save a RewardModel checkpoint the way reward.run does (state['model'] is
    the full RewardModel state_dict; state['config']['model'] is the arch)."""
    torch.manual_seed(0)
    rm = RewardModel(ModelConfig(**model_cfg))
    save_checkpoint(dir_path / "checkpoints", 0, {
        "step": 0, "model": rm.state_dict(),
        "config": {"model": dict(model_cfg)},
    })
    return rm


def _tokenizer(tmp_path, fixture_path) -> Tokenizer:
    tok_dir = tmp_path / "tok"
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    return Tokenizer.from_file(str(tok_dir / "tokenizer.json"))


def test_load_reward_model_restores_weights_frozen(tmp_path):
    saved = _rm_artifact(tmp_path / "rm")
    loaded = _load_reward_model({"local_dir": str(tmp_path / "rm")}, "cpu")
    assert isinstance(loaded, RewardModel)
    assert all(not p.requires_grad for p in loaded.parameters())     # frozen
    for key, tensor in saved.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], tensor)         # exact weights


def test_load_reward_model_missing_checkpoint_raises(tmp_path):
    with pytest.raises(ValueError, match="no Reward Model checkpoint"):
        _load_reward_model({"local_dir": str(tmp_path / "empty")}, "cpu")


def test_reward_scorer_scores_fables_and_tolerates_empties(tmp_path, fixture_path):
    saved = _rm_artifact(tmp_path / "rm")
    tokenizer = _tokenizer(tmp_path, fixture_path)
    scorer = make_reward_scorer(saved, tokenizer, "cpu")
    scaffold = Scaffold("fox", "sly", "a wood", "a gate", "it waited", "patience wins")
    scores = scorer(scaffold, ["a real fable body", "", "   "])
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)
    assert scores[1] == 0.0 and scores[2] == 0.0                     # empties -> 0.0
