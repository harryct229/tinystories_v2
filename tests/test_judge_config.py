from pathlib import Path

import pytest

from tinystories_v2.config import load_config
from tinystories_v2.judge import (
    PositionBiasedFakeJudge,
    SlotCoverageFakeJudge,
    TransformersJudge,
    build_judge,
)

CONFIG_DIR = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize(
    ("filename", "model_id", "enable_thinking"),
    [
        ("judge_l4.toml", "Qwen/Qwen3-8B", False),
        (
            "judge_t4.toml",
            "Qwen/Qwen3-4B-Instruct-2507",
            None,
        ),
    ],
)
def test_real_configs_select_one_lazy_transformers_path(
    filename,
    model_id,
    enable_thinking,
):
    config = load_config(CONFIG_DIR / filename)["judge"]
    judge = build_judge(config)

    assert type(judge) is TransformersJudge
    assert judge.model_id == model_id
    assert judge.precision == "fp16"
    assert judge.device == "cuda"
    assert judge.enable_thinking is enable_thinking
    assert model_id in judge.judge_id
    assert "precision=fp16" in judge.judge_id
    assert "rubric=fable-pairwise-v1" in judge.judge_id


@pytest.mark.parametrize(
    ("kind", "expected_type"),
    [
        ("fake_slot_coverage", SlotCoverageFakeJudge),
        ("fake_position_biased", PositionBiasedFakeJudge),
    ],
)
def test_factory_selects_offline_fakes(kind, expected_type):
    assert type(build_judge({"kind": kind})) is expected_type


def test_factory_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown Judge kind"):
        build_judge({"kind": "remote_api"})


def test_factory_rejects_unknown_precision_without_loading_model():
    with pytest.raises(ValueError, match="precision"):
        build_judge(
            {
                "kind": "transformers",
                "model_id": "Qwen/Qwen3-8B",
                "precision": "int8",
                "device": "cuda",
            }
        )
