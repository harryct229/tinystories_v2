import dataclasses

import pytest
import torch

from tinystories_v2.model import FableLM, ModelConfig

TOY = ModelConfig(
    vocab_size=512, d_model=64, n_layers=2, n_heads=2, context=64, ffn_hidden=192
)
# The real run's architecture numbers (DESIGN.md). Duplicated in
# configs/pretrain_full.toml; test_pretrain_stage.py ties the two together.
REAL = ModelConfig(
    vocab_size=8192, d_model=512, n_layers=8, n_heads=8, context=512, ffn_hidden=1408
)
PARAM_BUDGET = 32_000_000


def expected_params(c: ModelConfig) -> int:
    per_layer = (
        4 * c.d_model * c.d_model        # q, k, v, o projections (no biases)
        + 3 * c.d_model * c.ffn_hidden   # SwiGLU gate, up, down
        + 2 * c.d_model                  # two RMSNorm weights
    )
    return c.vocab_size * c.d_model + c.n_layers * per_layer + c.d_model  # + final norm


@pytest.fixture(scope="module")
def toy_model():
    torch.manual_seed(0)
    return FableLM(TOY).eval()


def test_forward_shape_fp32(toy_model):
    idx = torch.randint(TOY.vocab_size, (2, 16))
    logits = toy_model(idx)
    assert logits.shape == (2, 16, TOY.vocab_size)
    assert logits.dtype == torch.float32


def test_causality_future_tokens_do_not_affect_past_logits(toy_model):
    torch.manual_seed(1)
    a = torch.randint(TOY.vocab_size, (2, 32))
    b = a.clone()
    b[:, 20:] = (b[:, 20:] + 7) % TOY.vocab_size  # perturb only the future
    with torch.no_grad():
        la, lb = toy_model(a), toy_model(b)
    assert torch.allclose(la[:, :20], lb[:, :20], atol=1e-5)
    assert not torch.allclose(la[:, 20:], lb[:, 20:], atol=1e-5)


def test_tied_embeddings_share_storage(toy_model):
    assert toy_model.lm_head.weight is toy_model.tok_emb.weight


def test_no_biases_anywhere(toy_model):
    for name, _ in toy_model.named_parameters():
        assert not name.endswith("bias"), name


def test_param_count_matches_analytic_formula(toy_model):
    assert toy_model.num_params() == expected_params(TOY)


def test_real_config_param_count_within_budget():
    assert expected_params(REAL) == 29_893_120
    torch.manual_seed(0)
    model = FableLM(REAL)
    assert model.num_params() == 29_893_120 < PARAM_BUDGET


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_under_autocast(toy_model, dtype):
    idx = torch.randint(TOY.vocab_size, (2, 16))
    with torch.autocast(device_type="cpu", dtype=dtype), torch.no_grad():
        logits = toy_model(idx)
    assert logits.shape == (2, 16, TOY.vocab_size)
    assert torch.isfinite(logits.float()).all()


def test_config_is_frozen_and_buildable_from_dict():
    cfg = ModelConfig(**{
        "vocab_size": 512, "d_model": 64, "n_layers": 2, "n_heads": 2,
        "context": 64, "ffn_hidden": 192,
    })
    assert cfg.rope_theta == 10000.0 and cfg.norm_eps == 1e-5
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.d_model = 1


def test_d_model_must_divide_by_heads():
    with pytest.raises(ValueError, match="n_heads"):
        FableLM(ModelConfig(vocab_size=64, d_model=65, n_layers=1, n_heads=2,
                            context=8, ffn_hidden=32))
