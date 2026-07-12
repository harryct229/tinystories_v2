"""Per-stage reference-free metric aggregation for the eval stage (issue 07)."""

from tinystories_v2.eval import reference_free_metrics


def test_reference_free_metrics_reports_the_paper_family():
    fables = [
        "The fox shared the grapes and learned that sharing brings friends.",
        "A wise owl watched the moon and taught the mouse to be patient.",
        "The greedy dog dropped his bone chasing a shadow in the pond.",
    ]
    m = reference_free_metrics(fables)
    assert set(m) == {"mean_distinct_1", "distinct_2", "self_bleu",
                      "mean_flesch_reading_ease", "n_usable"}
    assert m["n_usable"] == 3
    assert 0.0 < m["mean_distinct_1"] <= 1.0
    assert isinstance(m["distinct_2"], float)
    assert isinstance(m["self_bleu"], float)
    assert isinstance(m["mean_flesch_reading_ease"], float)


def test_reference_free_metrics_drops_wordless_fables():
    # An empty / whitespace body contributes no words and is dropped; with only
    # one usable fable, Self-BLEU is undefined (None) but Distinct-1 is defined.
    m = reference_free_metrics(["Hello there, small friend.", "", "   "])
    assert m["n_usable"] == 1
    assert m["self_bleu"] is None
    assert m["distinct_2"] is None  # single 4-word fable, but guarded either way
    assert isinstance(m["mean_distinct_1"], float)


def test_reference_free_metrics_all_none_when_no_usable_fables():
    m = reference_free_metrics(["", "   ", "\n"])
    assert m["n_usable"] == 0
    assert m["mean_distinct_1"] is None
    assert m["self_bleu"] is None
    assert m["mean_flesch_reading_ease"] is None


def test_reference_free_metrics_forwards_self_bleu_subsampling():
    fables = [f"story number {i} about a small brave animal today" for i in range(10)]
    full = reference_free_metrics(fables)
    subsampled = reference_free_metrics(fables, self_bleu_sample_size=4, self_bleu_seed=7)
    assert isinstance(full["self_bleu"], float)
    assert isinstance(subsampled["self_bleu"], float)
