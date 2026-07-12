import tomllib
from pathlib import Path

from tinystories_v2.model import FableLM, ModelConfig

REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATHS = {
    "rope_swiglu": REPO_ROOT / "configs" / "ablation_5m_rope_swiglu.toml",
    "learned_swiglu": (
        REPO_ROOT / "configs" / "ablation_5m_learned_swiglu.toml"
    ),
    "rope_gelu": REPO_ROOT / "configs" / "ablation_5m_rope_gelu.toml",
}
EXPECTED_MODELS = {
    "rope_swiglu": {
        "vocab_size": 8192,
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 4,
        "context": 512,
        "ffn_hidden": 704,
        "position_encoding": "rope",
        "mlp_type": "swiglu",
    },
    "learned_swiglu": {
        "vocab_size": 8192,
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 4,
        "context": 512,
        "ffn_hidden": 704,
        "position_encoding": "learned",
        "mlp_type": "swiglu",
    },
    "rope_gelu": {
        "vocab_size": 8192,
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 4,
        "context": 512,
        "ffn_hidden": 1056,
        "position_encoding": "rope",
        "mlp_type": "gelu",
    },
}


def load_configs() -> dict[str, dict]:
    return {
        name: tomllib.loads(path.read_text(encoding="utf-8"))
        for name, path in CONFIG_PATHS.items()
    }


def test_real_ablation_model_configs_and_exact_parameter_counts():
    configs = load_configs()
    assert {name: config["model"] for name, config in configs.items()} == (
        EXPECTED_MODELS
    )
    counts = {
        name: FableLM(ModelConfig(**config["model"])).num_params()
        for name, config in configs.items()
    }
    assert counts == {
        "rope_swiglu": 5_310_720,
        "learned_swiglu": 5_441_792,
        "rope_gelu": 5_310_720,
    }
    baseline = counts["rope_swiglu"]
    assert all(abs(count - baseline) / baseline <= 0.03 for count in counts.values())


def test_each_variant_changes_only_its_named_component():
    baseline = EXPECTED_MODELS["rope_swiglu"]
    learned = EXPECTED_MODELS["learned_swiglu"]
    gelu = EXPECTED_MODELS["rope_gelu"]
    assert {key for key in baseline if baseline[key] != learned[key]} == {
        "position_encoding"
    }
    # ffn_hidden is widened only to compensate for GELU's two projections.
    assert {key for key in baseline if baseline[key] != gelu[key]} == {
        "ffn_hidden",
        "mlp_type",
    }


def test_variants_share_data_schedule_seed_and_exact_token_budget():
    configs = load_configs()
    baseline = configs["rope_swiglu"]
    assert all(
        config["data"] == baseline["data"] for config in configs.values()
    )
    assert all(
        config["train"] == baseline["train"] for config in configs.values()
    )
    train = baseline["train"]
    context = baseline["model"]["context"]
    tokens = (
        train["steps"]
        * train["micro_batch_size"]
        * train["grad_accum"]
        * context
    )
    assert train["seed"] == 1337
    assert tokens == 498_073_600


def test_tracking_and_artifact_destinations_are_unique():
    configs = load_configs()
    assert all(config["wandb"]["enabled"] for config in configs.values())
    assert len(
        {config["wandb"]["run_name"] for config in configs.values()}
    ) == 3
    assert len({config["hub"]["target"] for config in configs.values()}) == 3
    assert len({config["out_dir"] for config in configs.values()}) == 3
