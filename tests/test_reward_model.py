import torch
from tokenizers import Tokenizer

from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.reward_model import (
    RewardModel, bradley_terry_loss, pad_sequences, pair_accuracy,
    score_fables, score_sequences,
)
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run

CONFIG = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2,
                     context=32, ffn_hidden=64)


def _reward_model():
    torch.manual_seed(0)
    return RewardModel(CONFIG)


def test_forward_returns_one_scalar_per_sequence():
    model = _reward_model()
    idx = torch.randint(0, CONFIG.vocab_size, (5, 12))
    lengths = torch.tensor([12, 10, 8, 6, 4])
    scores = model(idx, lengths)
    assert scores.shape == (5,)
    assert scores.dtype == torch.float32


def test_pad_sequences_shapes_and_lengths():
    idx, lengths = pad_sequences([[1, 2, 3], [4, 5], [6]], context=32, device="cpu")
    assert idx.shape == (3, 3)  # padded to the longest (3)
    assert idx[1].tolist() == [4, 5, 0]  # right-padded with 0
    assert lengths.tolist() == [3, 2, 1]


def test_pad_sequences_truncates_to_context():
    idx, lengths = pad_sequences([list(range(50))], context=8, device="cpu")
    assert idx.shape == (1, 8)
    assert lengths.tolist() == [8]


def test_score_is_padding_invariant():
    # A sequence scores identically alone and inside a longer padded batch:
    # last-real-token pooling + causal attention ignore right-padding.
    model = _reward_model()
    short = [3, 7, 1, 9]
    long = [2, 2, 5, 8, 4, 6, 1]
    alone = score_sequences(model, [short], device="cpu")
    batched = score_sequences(model, [short, long], device="cpu")
    assert torch.allclose(alone[0], batched[0], atol=1e-6)


def test_score_sequences_is_batched():
    model = _reward_model()
    scores = score_sequences(model, [[1, 2], [3, 4, 5], [6]], device="cpu")
    assert scores.shape == (3,)


def test_bradley_terry_loss_and_accuracy():
    chosen = torch.tensor([2.0, 1.0, 0.5])
    rejected = torch.tensor([0.0, 3.0, 0.5])
    # Manual BT: -mean(log σ(chosen - rejected)).
    expected = -torch.nn.functional.logsigmoid(chosen - rejected).mean()
    assert torch.allclose(bradley_terry_loss(chosen, rejected), expected)
    # Accuracy counts strictly-greater: pair 0 wins, pair 1 loses, pair 2 ties (not >).
    assert pair_accuracy(chosen, rejected).item() == 1 / 3


def test_load_backbone_state_dict_strict(tmp_path, fixture_path):
    # A RewardModel loads a FableLM state dict into its backbone strictly; the
    # scalar head keeps its fresh init.
    torch.manual_seed(1)
    sft_like = FableLM(CONFIG)
    model = _reward_model()
    model.load_backbone_state_dict(sft_like.state_dict())
    for key, tensor in sft_like.state_dict().items():
        assert torch.equal(model.backbone.state_dict()[key], tensor), key


def test_score_fables_returns_one_float_each(tmp_path, fixture_path):
    tok_dir = tmp_path / "tok"
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    model = RewardModel(ModelConfig(vocab_size=512, d_model=32, n_layers=2,
                                    n_heads=2, context=128, ffn_hidden=64))
    scaffold = Scaffold("fox", "sly", "a wood", "a gate", "it waited", "patience wins")
    scores = score_fables(model, tokenizer,
                          [(scaffold, "The sly fox waited."),
                           (scaffold, "A different tale entirely.")],
                          device="cpu")
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)
