import pytest

from tinystories_v2.metrics import distinct_n, tokenize_words


def test_tokenize_words_casefolds_and_keeps_apostrophes():
    assert tokenize_words("Don't STOP, little fox!") == [
        "don't",
        "stop",
        "little",
        "fox",
    ]


def test_distinct_1_hand_computed():
    # Pooled unigrams: the, cat, sat, the, dog, sat -> 4 distinct / 6 total.
    assert distinct_n(["The cat sat.", "The dog sat."], n=1) == pytest.approx(
        4 / 6
    )


def test_distinct_2_hand_computed():
    # Bigrams: (the,cat) (cat,sat) (the,dog) (dog,sat) -> 4 distinct / 4.
    assert distinct_n(["The cat sat.", "The dog sat."], n=2) == pytest.approx(
        1.0
    )


def test_distinct_1_identical_fables_halves():
    assert distinct_n(["a b", "a b"], n=1) == pytest.approx(0.5)


def test_distinct_works_on_a_single_fable():
    assert distinct_n(["a b a"], n=1) == pytest.approx(2 / 3)


def test_short_fables_contribute_zero_ngrams():
    # "x" is shorter than n=2, so only "a b c" contributes bigrams.
    assert distinct_n(["a b c", "x"], n=2) == pytest.approx(1.0)


def test_distinct_rejects_empty_set():
    with pytest.raises(ValueError, match="at least one"):
        distinct_n([])


def test_distinct_rejects_fable_without_words():
    with pytest.raises(ValueError, match="no words"):
        distinct_n(["a real fable", "!!!"])


def test_distinct_rejects_all_fables_shorter_than_n():
    with pytest.raises(ValueError, match="no 3-grams"):
        distinct_n(["a b", "c d"], n=3)


def test_distinct_rejects_n_below_one():
    with pytest.raises(ValueError, match="at least 1"):
        distinct_n(["a b"], n=0)
