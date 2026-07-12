"""Position-debiased margin judging (issue 04 follow-up): the real Qwen3-8B
run showed greedy A/B verdicts saturate at 'A' (p~1.0) for same-model
completions, so order-swap discarded 100% of pairs. The margin judge reads the
A/B first-token logits for both presentation orders; the position prior
cancels in (s_first - s_swapped) / 2 and only the content differential
remains. All tests are CPU-only with stub backends — no model download."""

from types import SimpleNamespace

import pytest
import torch

from tinystories_v2.judge import (
    JudgeOutputError,
    MarginTransformersJudge,
    build_judge,
    judge_with_margin,
)
from tinystories_v2.pref_data import label_scaffold
from tinystories_v2.preferences import validate_preference_pair
from tinystories_v2.slots import Scaffold


@pytest.fixture
def scaffold() -> Scaffold:
    return Scaffold(character="fox", trait="greedy", setting="a dense forest",
                    conflict="loses food to a trick",
                    resolution="the trickster is exposed",
                    moral="honesty is the best policy")


class _StubMarginJudge:
    judge_id = "stub-margin:tau=1.0"
    margin_threshold = 1.0

    def __init__(self, value: float) -> None:
        self.value = value

    def margin(self, scaffold, fable_a, fable_b) -> float:
        return self.value


def test_positive_margin_keeps_first_fable(scaffold):
    pair = judge_with_margin(_StubMarginJudge(2.5), scaffold,
                             "Alpha fable.", "Beta fable.")
    assert pair.chosen == "Alpha fable."
    assert pair.rejected == "Beta fable."
    assert pair.verdict.first_pass == "A"
    assert pair.verdict.swapped_pass == "B"
    assert pair.verdict.consistent is True
    assert validate_preference_pair(pair.to_dict()) == pair


def test_negative_margin_keeps_second_fable(scaffold):
    pair = judge_with_margin(_StubMarginJudge(-2.5), scaffold,
                             "Alpha fable.", "Beta fable.")
    assert pair.chosen == "Beta fable."
    assert pair.verdict.first_pass == "B"
    assert pair.verdict.swapped_pass == "A"


def test_sub_threshold_margin_discards_the_pair(scaffold):
    assert judge_with_margin(_StubMarginJudge(0.5), scaffold,
                             "Alpha fable.", "Beta fable.") is None
    # The boundary itself is "no reliable signal" too.
    assert judge_with_margin(_StubMarginJudge(1.0), scaffold,
                             "Alpha fable.", "Beta fable.") is None
    assert judge_with_margin(_StubMarginJudge(-1.0), scaffold,
                             "Alpha fable.", "Beta fable.") is None


def test_label_scaffold_dispatches_margin_judges(scaffold):
    completions = ["Alpha text.", "Beta text.", "Gamma text.", "Delta text."]
    pairs, counters = label_scaffold(_StubMarginJudge(3.0), scaffold,
                                     completions, 3)
    assert counters == {"kept": 3, "discarded_inconsistent": 0,
                        "skipped_degenerate": 0, "judge_error": 0}
    # pair_indices(4, 3) == [(0, 3), (1, 2), (0, 2)]; positive margin keeps
    # each pair's first completion.
    assert [p.chosen for p in pairs] == ["Alpha text.", "Beta text.",
                                         "Alpha text."]

    _, counters = label_scaffold(_StubMarginJudge(0.2), scaffold,
                                 completions, 3)
    assert counters["discarded_inconsistent"] == 3
    assert counters["kept"] == 0


def test_constructor_rejects_thinking_and_bad_threshold():
    with pytest.raises(ValueError, match="thinking"):
        MarginTransformersJudge(model_id="m", precision="fp16", device="cpu",
                                margin_threshold=1.0, enable_thinking=True)
    with pytest.raises(ValueError, match="margin_threshold"):
        MarginTransformersJudge(model_id="m", precision="fp16", device="cpu",
                                margin_threshold=-0.1)


def test_judge_id_records_method_model_and_threshold():
    judge = MarginTransformersJudge(model_id="Qwen/Qwen3-8B", precision="fp16",
                                    device="cuda", margin_threshold=1.5)
    assert judge.judge_id.startswith("transformers-margin:Qwen/Qwen3-8B")
    assert "tau=1.5" in judge.judge_id
    assert "rubric=" in judge.judge_id


class _FakeBatch(dict):
    def to(self, device):
        return self


def _make_backend(gaps: list[float], a_ids=(65,), b_ids=(66,)):
    """Fake (torch, tokenizer, model): logit('A') - logit('B') at the last
    position is gaps[i] on the i-th forward call."""
    calls = {"n": 0}

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return {"A": list(a_ids), "B": list(b_ids)}[text]

        def apply_chat_template(self, messages, **kwargs):
            return _FakeBatch(input_ids=torch.tensor([[1, 2, 3]]))

    class Model:
        def __call__(self, **inputs):
            gap = gaps[calls["n"]]
            calls["n"] += 1
            logits = torch.zeros(1, inputs["input_ids"].shape[-1], 100)
            logits[0, -1, 65] = gap
            logits[0, -1, 66] = 0.0
            return SimpleNamespace(logits=logits)

    return (torch, Tok(), Model())


def test_margin_is_half_the_gap_difference_between_orders(scaffold):
    judge = MarginTransformersJudge(model_id="m", precision="fp16",
                                    device="cpu", margin_threshold=0.5)
    # Position prior +2.0 in both orders, content differential +-1.0: the
    # prior must cancel, leaving exactly the content term.
    judge._backend = _make_backend([3.0, 1.0])
    assert judge.margin(scaffold, "Alpha fable.", "Beta fable.") == \
        pytest.approx(1.0)

    judge._backend = _make_backend([2.0, 2.0])  # pure position bias
    assert judge.margin(scaffold, "Alpha fable.", "Beta fable.") == \
        pytest.approx(0.0)


def test_margin_requires_single_token_verdict_labels(scaffold):
    judge = MarginTransformersJudge(model_id="m", precision="fp16",
                                    device="cpu", margin_threshold=0.5)
    judge._backend = _make_backend([1.0, 1.0], a_ids=(65, 99))
    with pytest.raises(JudgeOutputError, match="single token"):
        judge.margin(scaffold, "Alpha fable.", "Beta fable.")


def test_build_judge_constructs_margin_judge_from_config():
    judge = build_judge({"kind": "transformers_margin", "model_id": "m",
                         "precision": "fp16", "device": "cpu",
                         "margin_threshold": 1.5})
    assert isinstance(judge, MarginTransformersJudge)
    assert judge.margin_threshold == 1.5
    assert judge._backend is None  # no weights loaded at build time


def test_build_judge_requires_numeric_margin_threshold():
    with pytest.raises(ValueError, match="margin_threshold"):
        build_judge({"kind": "transformers_margin", "model_id": "m",
                     "precision": "fp16"})
