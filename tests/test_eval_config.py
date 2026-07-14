"""Pin configs/eval_full.toml's shape: catches config corruption (e.g. a sed
edit clobbering [sampling].max_new_tokens into a stray duplicate — happened
2026-07-14) before it burns a full Colab session on a late KeyError."""

from pathlib import Path

from tinystories_v2.config import load_config

CONFIGS = Path(__file__).parent.parent / "configs"


def test_eval_full_config_has_required_shape():
    cfg = load_config(CONFIGS / "eval_full.toml")
    assert {"max_new_tokens", "temperature", "top_p", "seed"} <= cfg["sampling"].keys()
    assert isinstance(cfg["sampling"]["max_new_tokens"], int)
    assert cfg["sampling"]["max_new_tokens"] > 0
    assert {"model_id", "precision", "device"} <= cfg["judge"].keys()
    assert len(cfg["stages"]) == 4
    assert {s["name"] for s in cfg["stages"]} == {"base", "sft", "rlaif", "dpo"}
    assert all(s["hub_source"].startswith("hf://") for s in cfg["stages"])


def test_eval_full_config_uses_margin_judge():
    # The greedy Llama judge decided 0/1200 real comparisons (position-
    # saturated verdicts + frequent unparseable output) — margin judging
    # replaced it. Locks the regression in.
    cfg = load_config(CONFIGS / "eval_full.toml")
    assert cfg["judge"]["kind"] == "transformers_margin"
    assert cfg["judge"]["margin_threshold"] > 0
