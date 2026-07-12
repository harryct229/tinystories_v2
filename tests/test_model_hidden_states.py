import torch

from tinystories_v2.model import FableLM, ModelConfig

CONFIG = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2,
                     context=16, ffn_hidden=64)


def _model():
    torch.manual_seed(0)
    return FableLM(CONFIG).eval()


def test_hidden_states_shape():
    model = _model()
    idx = torch.randint(0, CONFIG.vocab_size, (3, 10))
    hidden = model.hidden_states(idx)
    assert hidden.shape == (3, 10, CONFIG.d_model)


def test_forward_equals_lm_head_of_hidden_states():
    model = _model()
    idx = torch.randint(0, CONFIG.vocab_size, (2, 8))
    with torch.no_grad():
        expected = model.lm_head(model.hidden_states(idx))
        assert torch.equal(model(idx), expected)


def test_hidden_states_respects_context_limit():
    model = _model()
    idx = torch.randint(0, CONFIG.vocab_size, (1, CONFIG.context + 1))
    import pytest
    with pytest.raises(ValueError, match="exceeds context"):
        model.hidden_states(idx)
