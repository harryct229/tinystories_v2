"""Generation is identical across stages (same Scaffolds/seeds/sampling) — the
apples-to-apples criterion — and stage-model loading errors are explicit."""

import pytest

from tinystories_v2.eval import generate_all_stages, load_stage_model
from tinystories_v2.slots import Scaffold

SCAFFOLDS = [Scaffold("fox", "sly", "a wood", "a gate", "it shared", "share"),
             Scaffold("owl", "wise", "a barn", "a storm", "it waited", "wait")]
SEEDS = [11, 22]
SAMPLING = {"max_new_tokens": 8, "temperature": 0.8, "top_p": 0.95, "seed": 1337}


def test_generate_all_stages_feeds_identical_inputs_to_every_stage():
    calls = []

    def spy(model, tokenizer, scaffolds, seeds, sampling, *, device="cpu"):
        calls.append({"scaffolds": scaffolds, "seeds": seeds, "sampling": sampling})
        # A stage-distinct but Scaffold-aligned canned completion.
        return [f"{model}:{s.character}" for s in scaffolds]

    stage_models = {"base": "M_BASE", "sft": "M_SFT"}
    out = generate_all_stages(stage_models, tokenizer=None, scaffolds=SCAFFOLDS,
                              seeds=SEEDS, sampling=SAMPLING, generate_fn=spy)

    assert list(out) == ["base", "sft"]
    assert out["base"] == ["M_BASE:fox", "M_BASE:owl"]
    assert out["sft"] == ["M_SFT:fox", "M_SFT:owl"]
    # Both stages saw the SAME Scaffolds, seeds, and sampling object.
    assert calls[0]["scaffolds"] is calls[1]["scaffolds"] is SCAFFOLDS
    assert calls[0]["seeds"] is calls[1]["seeds"] is SEEDS
    assert calls[0]["sampling"] is calls[1]["sampling"] is SAMPLING


def test_load_stage_model_raises_without_a_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="no checkpoint for stage 'base'"):
        load_stage_model({"name": "base", "local_dir": str(tmp_path / "missing")},
                         device="cpu")
