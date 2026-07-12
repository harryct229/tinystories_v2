import dataclasses

import pytest
import torch

from tinystories_v2.model import GELUMLP, FableLM, ModelConfig, SwiGLU

TOY = ModelConfig(
    vocab_size=512,
    d_model=64,
    n_layers=2,
    n_heads=2,
    context=64,
    ffn_hidden=192,
)
TOY_VARIANTS = {
    "rope_swiglu": TOY,
    "learned_swiglu": dataclasses.replace(
        TOY, position_encoding="learned"
    ),
    "rope_gelu": dataclasses.replace(
        TOY, mlp_type="gelu", ffn_hidden=288
    ),
}

# The production model from DESIGN.md/configs/pretrain_full.toml. Defaults must
# preserve the architecture and state-dict contract of existing checkpoints.
REAL = ModelConfig(
    vocab_size=8192,
    d_model=512,
    n_layers=8,
    n_heads=8,
    context=512,
    ffn_hidden=1408,
)
PARAM_BUDGET = 32_000_000

ABLATION_VARIANTS = {
    "rope_swiglu": ModelConfig(
        vocab_size=8192,
        d_model=256,
        n_layers=4,
        n_heads=4,
        context=512,
        ffn_hidden=704,
        position_encoding="rope",
        mlp_type="swiglu",
    ),
    "learned_swiglu": ModelConfig(
        vocab_size=8192,
        d_model=256,
        n_layers=4,
        n_heads=4,
        context=512,
        ffn_hidden=704,
        position_encoding="learned",
        mlp_type="swiglu",
    ),
    "rope_gelu": ModelConfig(
        vocab_size=8192,
        d_model=256,
        n_layers=4,
        n_heads=4,
        context=512,
        ffn_hidden=1056,
        position_encoding="rope",
        mlp_type="gelu",
    ),
}


def expected_params(config: ModelConfig) -> int:
    mlp_projections = {"swiglu": 3, "gelu": 2}[config.mlp_type]
    per_layer = (
        4 * config.d_model * config.d_model
        + mlp_projections * config.d_model * config.ffn_hidden
        + 2 * config.d_model
    )
    learned_positions = (
        config.context * config.d_model
        if config.position_encoding == "learned"
        else 0
    )
    return (
        config.vocab_size * config.d_model
        + learned_positions
        + config.n_layers * per_layer
        + config.d_model
    )


@pytest.fixture(scope="module")
def toy_model():
    torch.manual_seed(0)
    return FableLM(TOY).eval()


def test_forward_shape_fp32(toy_model):
    idx = torch.randint(TOY.vocab_size, (2, 16))
    logits = toy_model(idx)
    assert logits.shape == (2, 16, TOY.vocab_size)
    assert logits.dtype == torch.float32


@pytest.mark.parametrize(
    "config", TOY_VARIANTS.values(), ids=TOY_VARIANTS.keys()
)
def test_causality_future_tokens_do_not_affect_past_logits(config):
    torch.manual_seed(1)
    model = FableLM(config).eval()
    a = torch.randint(config.vocab_size, (2, 32))
    b = a.clone()
    b[:, 20:] = (b[:, 20:] + 7) % config.vocab_size
    with torch.no_grad():
        logits_a, logits_b = model(a), model(b)
    assert torch.allclose(logits_a[:, :20], logits_b[:, :20], atol=1e-5)
    assert not torch.allclose(logits_a[:, 20:], logits_b[:, 20:], atol=1e-5)


@pytest.mark.parametrize(
    "name,config", TOY_VARIANTS.items(), ids=TOY_VARIANTS.keys()
)
def test_variant_selects_only_requested_components(name, config):
    model = FableLM(config)
    if name == "learned_swiglu":
        assert isinstance(model.pos_emb, torch.nn.Embedding)
        assert model.rope_cos is None and model.rope_sin is None
    else:
        assert model.pos_emb is None
        assert model.rope_cos is not None and model.rope_sin is not None
    expected_mlp = GELUMLP if name == "rope_gelu" else SwiGLU
    assert all(isinstance(block.mlp, expected_mlp) for block in model.blocks)


@pytest.mark.parametrize(
    "config", TOY_VARIANTS.values(), ids=TOY_VARIANTS.keys()
)
def test_tied_embeddings_and_no_biases_for_every_variant(config):
    model = FableLM(config)
    assert model.lm_head.weight is model.tok_emb.weight
    for name, _ in model.named_parameters():
        assert not name.endswith("bias"), name


@pytest.mark.parametrize(
    "config", TOY_VARIANTS.values(), ids=TOY_VARIANTS.keys()
)
def test_param_count_matches_variant_analytic_formula(config):
    assert FableLM(config).num_params() == expected_params(config)


def test_real_config_param_count_within_budget():
    assert expected_params(REAL) == 29_893_120
    torch.manual_seed(0)
    model = FableLM(REAL)
    assert model.num_params() == 29_893_120 < PARAM_BUDGET


def test_ablation_variants_are_within_three_percent_of_baseline():
    counts = {
        name: FableLM(config).num_params()
        for name, config in ABLATION_VARIANTS.items()
    }
    assert counts == {
        "rope_swiglu": 5_310_720,
        "learned_swiglu": 5_441_792,
        "rope_gelu": 5_310_720,
    }
    baseline = counts["rope_swiglu"]
    assert all(abs(count - baseline) / baseline <= 0.03 for count in counts.values())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_under_autocast(toy_model, dtype):
    idx = torch.randint(TOY.vocab_size, (2, 16))
    with torch.autocast(device_type="cpu", dtype=dtype), torch.no_grad():
        logits = toy_model(idx)
    assert logits.shape == (2, 16, TOY.vocab_size)
    assert torch.isfinite(logits.float()).all()


def test_config_is_frozen_backward_compatible_and_buildable_from_dict():
    config = ModelConfig(
        **{
            "vocab_size": 512,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 2,
            "context": 64,
            "ffn_hidden": 192,
        }
    )
    assert config.rope_theta == 10000.0 and config.norm_eps == 1e-5
    assert config.position_encoding == "rope"
    assert config.mlp_type == "swiglu"
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.d_model = 1


def test_d_model_must_divide_by_heads():
    with pytest.raises(ValueError, match="n_heads"):
        FableLM(
            ModelConfig(
                vocab_size=64,
                d_model=65,
                n_layers=1,
                n_heads=2,
                context=8,
                ffn_hidden=32,
            )
        )


@pytest.mark.parametrize(
    "field,value",
    [("position_encoding", "sinusoidal"), ("mlp_type", "relu")],
)
def test_unknown_component_selector_is_rejected(field, value):
    values = dataclasses.asdict(TOY)
    values[field] = value
    with pytest.raises(ValueError, match=field):
        FableLM(ModelConfig(**values))
