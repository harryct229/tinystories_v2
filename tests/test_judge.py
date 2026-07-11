from dataclasses import asdict

import pytest

from tinystories_v2.judge import (
    Judge,
    JudgeOutputError,
    PositionBiasedFakeJudge,
    SlotCoverageFakeJudge,
    Verdict,
    judge_with_order_swap,
    parse_verdict,
    render_rubric_prompt,
)
from tinystories_v2.preferences import validate_preference_pair
from tinystories_v2.slots import extract_slots


def fixture_case(fixture_records):
    source = fixture_records[0]
    scaffold = extract_slots(source["prompt"])
    explicit_slots = " ".join(asdict(scaffold).values())
    candidate_a = f"{source['fable']}\n\n{explicit_slots}"
    candidate_b = fixture_records[1]["fable"]
    return scaffold, candidate_a, candidate_b


def test_fakes_implement_judge_interface_on_fixture_fables(fixture_records):
    scaffold, candidate_a, candidate_b = fixture_case(fixture_records)
    consistent = SlotCoverageFakeJudge()
    biased = PositionBiasedFakeJudge()

    assert isinstance(consistent, Judge)
    assert isinstance(biased, Judge)
    assert consistent.compare(scaffold, candidate_a, candidate_b) is Verdict.A
    assert consistent.compare(scaffold, candidate_b, candidate_a) is Verdict.B
    assert biased.compare(scaffold, candidate_a, candidate_b) is Verdict.A
    assert biased.compare(scaffold, candidate_b, candidate_a) is Verdict.A


def test_equal_coverage_tie_break_is_position_independent(fixture_records):
    scaffold = extract_slots(fixture_records[0]["prompt"])
    judge = SlotCoverageFakeJudge()
    first = judge.compare(
        scaffold,
        "An unrelated amber Fable.",
        "An unrelated blue Fable.",
    )
    swapped = judge.compare(
        scaffold,
        "An unrelated blue Fable.",
        "An unrelated amber Fable.",
    )
    assert swapped is first.opposite


def test_consistent_fake_produces_schema_valid_pair(fixture_records):
    scaffold, candidate_a, candidate_b = fixture_case(fixture_records)
    pair = judge_with_order_swap(
        SlotCoverageFakeJudge(),
        scaffold,
        candidate_a,
        candidate_b,
    )

    assert pair is not None
    assert pair.chosen == candidate_a
    assert pair.rejected == candidate_b
    assert pair.verdict.judge_id == "fake:slot-coverage-v1"
    assert pair.verdict.first_pass == "A"
    assert pair.verdict.swapped_pass == "B"
    assert validate_preference_pair(pair.to_dict()) == pair


def test_consistent_fake_preserves_original_b_winner(fixture_records):
    scaffold, stronger_fable, weaker_fable = fixture_case(fixture_records)
    original_a = weaker_fable
    original_b = stronger_fable

    pair = judge_with_order_swap(
        SlotCoverageFakeJudge(),
        scaffold,
        original_a,
        original_b,
    )

    assert pair is not None
    assert pair.chosen == original_b
    assert pair.rejected == original_a
    assert pair.verdict.first_pass == "B"
    assert pair.verdict.swapped_pass == "A"


def test_position_biased_fake_is_discarded(fixture_records):
    scaffold, candidate_a, candidate_b = fixture_case(fixture_records)
    assert (
        judge_with_order_swap(
            PositionBiasedFakeJudge(),
            scaffold,
            candidate_a,
            candidate_b,
        )
        is None
    )


def test_rubric_renders_all_axes_priority_and_age_constraint(fixture_records):
    scaffold = extract_slots(fixture_records[0]["prompt"])
    prompt = render_rubric_prompt(
        scaffold,
        "Candidate A uses simple words and states the moral.",
        "Candidate B has a different ending.",
    )

    for axis in (
        "Grammar & Style",
        "Creativity",
        "Moral Clarity",
        "Prompt Adherence",
    ):
        assert axis in prompt
    assert "Prompt Adherence (HIGHEST WEIGHT)" in prompt
    assert "Moral Clarity (SECOND PRIORITY)" in prompt
    assert "ages 4–7" in prompt
    assert "HARD CONSTRAINT" in prompt
    for slot_value in asdict(scaffold).values():
        assert slot_value in prompt
    assert "Candidate A uses simple words" in prompt
    assert "Candidate B has a different ending" in prompt
    assert "Return exactly one capital letter: A or B." in prompt


@pytest.mark.parametrize(
    ("raw_output", "expected"),
    [
        ("A", Verdict.A),
        ("\nB\n", Verdict.B),
        ("Verdict: A.", Verdict.A),
    ],
)
def test_parse_verdict_accepts_one_unambiguous_label(raw_output, expected):
    assert parse_verdict(raw_output) is expected


@pytest.mark.parametrize("raw_output", ["tie", "A because it follows the moral"])
def test_parse_verdict_rejects_ambiguous_output(raw_output):
    with pytest.raises(JudgeOutputError, match="single A/B verdict"):
        parse_verdict(raw_output)
