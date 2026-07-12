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
                     "wins_b": 0, "ties": 0, "skipped": 1, "n": 3}


def test_win_rate_table_rejects_misaligned_lists():
    import pytest
    with pytest.raises(ValueError, match="align"):
        win_rate_table(SlotCoverageFakeJudge(), [SCAFFOLD], "a", [RICH],
                       "b", [BLAND, RICH])


def test_all_pairwise_win_rates_covers_each_unordered_stage_pair():
    scaffolds = [SCAFFOLD]
    stage_fables = {"base": [BLAND], "sft": [RICH], "rlaif": [RICH]}
    tables = all_pairwise_win_rates(SlotCoverageFakeJudge(), scaffolds, stage_fables)
    pairs = {(t["stage_a"], t["stage_b"]) for t in tables}
    assert pairs == {("base", "sft"), ("base", "rlaif"), ("sft", "rlaif")}
