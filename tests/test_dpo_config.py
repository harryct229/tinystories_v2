from pathlib import Path

from tinystories_v2.config import load_config

CONFIGS = Path(__file__).parent.parent / "configs"


def test_dpo_fixture_config_has_required_shape():
    cfg = load_config(CONFIGS / "dpo_fixture.toml")
    assert cfg["out_dir"]
    for section in ("model", "data", "init", "split", "dpo", "train", "wandb"):
        assert section in cfg, section
    assert cfg["dpo"]["beta"] > 0
    assert {"pairs_path", "tokenizer_path"} <= cfg["data"].keys()
    assert {"holdout_frac", "seed"} <= cfg["split"].keys()
    assert cfg["train"]["precision"] in {"fp32", "bf16", "fp16"}


def test_dpo_full_config_targets_hub_and_real_model():
    cfg = load_config(CONFIGS / "dpo_full.toml")
    assert cfg["model"]["vocab_size"] == 8192 and cfg["model"]["d_model"] == 512
    assert cfg["init"]["hub_source"].startswith("hf://")
    assert cfg["hub"]["target"].startswith("hf://")
    assert cfg["dpo"]["beta"] == 0.1
    assert cfg["wandb"]["enabled"] is True
