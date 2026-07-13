from pathlib import Path

from tinystories_v2.config import load_config

CONFIGS = Path(__file__).parent.parent / "configs"


def test_grpo_fixture_config_has_required_shape():
    cfg = load_config(CONFIGS / "grpo_fixture.toml")
    assert cfg["out_dir"]
    for section in ("model", "data", "init", "reward", "grpo", "sampling", "train", "wandb"):
        assert section in cfg, section
    assert {"group_size", "clip_eps", "kl_beta", "ppo_epochs", "adv_eps"} <= cfg["grpo"].keys()
    assert {"pref_split", "tokenizer_path"} <= cfg["data"].keys()
    assert {"max_new_tokens", "temperature", "top_p"} <= cfg["sampling"].keys()
    assert cfg["train"]["precision"] in {"fp32", "bf16", "fp16"}
    assert "local_dir" in cfg["reward"]


def test_grpo_full_config_targets_hub_and_real_model():
    cfg = load_config(CONFIGS / "grpo_full.toml")
    assert cfg["model"]["vocab_size"] == 8192 and cfg["model"]["d_model"] == 512
    assert cfg["init"]["hub_source"].startswith("hf://")
    assert cfg["reward"]["hub_source"].startswith("hf://")
    assert cfg["hub"]["target"].startswith("hf://")
    assert cfg["grpo"]["group_size"] == 8
    assert cfg["grpo"]["clip_eps"] == 0.2
    assert cfg["grpo"]["kl_beta"] == 0.03
    assert cfg["wandb"]["enabled"] is True
