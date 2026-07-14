"""Order-swapped win-rate tallies over aligned per-stage fable lists (issue 07)."""

from tinystories_v2.eval import (
    all_pairwise_win_rates,
    stage_win,
    win_rate_table,
)
from tinystories_v2.judge import PositionBiasedFakeJudge, SlotCoverageFakeJudge
from tinystories_v2.slots import Scaffold

SCAFFOLD = Scaffold("fox", "sly", "a wood", "a locked gate",
                    "the fox shared", "sharing brings friends")
# SlotCoverageFakeJudge prefers whichever candidate mentions more slot values.
RICH = ("The sly fox in a wood met a locked gate; the fox shared, and sharing "
        "brings friends.")
BLAND = "A plain note with nothing much to say."


def test_stage_win_picks_the_slot_rich_side_consistently():
    assert stage_win(SlotCoverageFakeJudge(), SCAFFOLD, RICH, BLAND) == "a"
    assert stage_win(SlotCoverageFakeJudge(), SCAFFOLD, BLAND, RICH) == "b"


def test_stage_win_is_a_tie_when_order_swap_is_inconsistent():
    # PositionBiasedFakeJudge always answers "A": it prefers position, not
    # content, so the two presentation orders disagree -> a tie.
    assert stage_win(PositionBiasedFakeJudge(), SCAFFOLD, RICH, BLAND) == "tie"


def test_win_rate_table_counts_wins_and_skips_degenerate_pairs():
    scaffolds = [SCAFFOLD, SCAFFOLD, SCAFFOLD]
    fables_a = [RICH, RICH, "identical"]
    fables_b = [BLAND, BLAND, "IDENTICAL"]  # third pair is degenerate (casefold-equal)
    table = win_rate_table(SlotCoverageFakeJudge(), scaffolds,
                           "sft", fables_a, "base", fables_b)
    assert table == {"stage_a": "sft", "stage_b": "base", "wins_a": 2,
                     "wins_b": 0, "ties": 0, "skipped": 1, "judge_error": 0,
                     "n": 3}


def test_win_rate_table_rejects_misaligned_lists():
    import pytest
    with pytest.raises(ValueError, match="align"):
        win_rate_table(SlotCoverageFakeJudge(), [SCAFFOLD], "a", [RICH],
                       "b", [BLAND, RICH])


def test_win_rate_table_counts_unparseable_verdicts_as_judge_error():
    from tinystories_v2.judge import JudgeOutputError

    class _RaisingJudge:
        judge_id = "fake:raises-v1"

        def compare(self, scaffold, fable_a, fable_b):
            raise JudgeOutputError("unparseable verdict")

    scaffolds = [SCAFFOLD, SCAFFOLD]
    table = win_rate_table(_RaisingJudge(), scaffolds,
                           "sft", [RICH, RICH], "base", [BLAND, BLAND])
    assert table["judge_error"] == 2
    assert table["wins_a"] == table["wins_b"] == table["ties"] == table["skipped"] == 0
    assert table["n"] == 2


def test_all_pairwise_win_rates_covers_each_unordered_stage_pair():
    scaffolds = [SCAFFOLD]
    stage_fables = {"base": [BLAND], "sft": [RICH], "rlaif": [RICH]}
    tables = all_pairwise_win_rates(SlotCoverageFakeJudge(), scaffolds, stage_fables)
    pairs = {(t["stage_a"], t["stage_b"]) for t in tables}
    assert pairs == {("base", "sft"), ("base", "rlaif"), ("sft", "rlaif")}


class _StubMarginJudge:
    """Margin-capable judge: stage_win must dispatch on margin(), not
    compare() (greedy verdicts saturate by position — the real Llama run
    decided 0 of 1,200 comparisons)."""

    judge_id = "stub-margin:tau=0.5"
    margin_threshold = 0.5

    def __init__(self, value: float) -> None:
        self.value = value

    def margin(self, scaffold, fable_a, fable_b) -> float:
        return self.value

    def compare(self, scaffold, fable_a, fable_b):
        raise AssertionError("margin judges must not fall back to compare()")


def test_stage_win_dispatches_margin_judges():
    from tinystories_v2.eval import stage_win
    from tinystories_v2.slots import Scaffold
    scaffold = Scaffold(character="fox", trait="greedy",
                        setting="a dense forest",
                        conflict="loses food to a trick",
                        resolution="the trickster is exposed",
                        moral="honesty is the best policy")
    assert stage_win(_StubMarginJudge(2.0), scaffold, "Alpha.", "Beta.") == "a"
    assert stage_win(_StubMarginJudge(-2.0), scaffold, "Alpha.", "Beta.") == "b"
    assert stage_win(_StubMarginJudge(0.1), scaffold, "Alpha.", "Beta.") == "tie"
