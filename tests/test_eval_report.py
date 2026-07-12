"""Markdown rendering of the eval report and sample sheet (issue 07)."""

from tinystories_v2.eval import render_report, sample_sheet_md
from tinystories_v2.slots import Scaffold

SCAFFOLD = Scaffold("fox", "sly", "a wood", "a locked gate",
                    "the fox shared", "sharing brings friends")


def test_sample_sheet_shows_each_stage_side_by_side():
    scaffolds = [SCAFFOLD, SCAFFOLD]
    stage_fables = {"base": ["base body one", "base body two"],
                    "sft": ["sft body one", "sft body two"]}
    sheet = sample_sheet_md(scaffolds, stage_fables, k=1)
    assert "Scaffold 1" in sheet
    assert "Scaffold 2" not in sheet          # k=1 truncates
    assert "sly" in sheet and "the fox shared" in sheet  # slot values rendered
    assert "### base" in sheet and "### sft" in sheet
    assert "base body one" in sheet and "sft body one" in sheet


def test_sample_sheet_marks_empty_bodies():
    sheet = sample_sheet_md([SCAFFOLD], {"base": [""]}, k=1)
    assert "_(empty)_" in sheet


def test_render_report_has_winrate_and_metric_tables_with_counts():
    results = {
        "eval_judge_id": "fake:slot-coverage-v1",
        "n_scaffolds": 3,
        "win_rates": [{"stage_a": "base", "stage_b": "sft", "wins_a": 1,
                       "wins_b": 2, "ties": 0, "skipped": 0, "judge_error": 0,
                       "n": 3}],
        "metrics": {
            "base": {"mean_distinct_1": 0.5, "distinct_2": 0.9,
                     "self_bleu": 0.1, "mean_flesch_reading_ease": 80.0,
                     "perplexity": 42.0, "n_usable": 3},
            "sft": {"mean_distinct_1": 0.6, "distinct_2": None,
                    "self_bleu": None, "mean_flesch_reading_ease": 78.9,
                    "perplexity": 30.0, "n_usable": 1},
        },
    }
    report = render_report(results, "SAMPLE-SHEET-BODY")
    assert "fake:slot-coverage-v1" in report          # eval judge identity
    assert "| base | sft | 1 | 2 | 0 | 0 | 0 | 3 |" in report   # win-rate counts
    assert "| base |" in report and "42.000" in report      # metric row
    assert "n/a" in report                            # None -> n/a
    assert report.rstrip().endswith("SAMPLE-SHEET-BODY")     # sheet embedded last
