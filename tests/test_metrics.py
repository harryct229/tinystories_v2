import math
import random

import pytest

from tinystories_v2.metrics import distinct_n, self_bleu, tokenize_words


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


def test_self_bleu_identical_fables_is_maximally_redundant():
    assert self_bleu(
        ["The fox ran home.", "The fox ran home."]
    ) == pytest.approx(1.0)


def test_self_bleu_disjoint_fables_is_zero():
    assert self_bleu(["aa bb cc dd", "ee ff gg hh"]) == 0.0


def test_self_bleu_partial_overlap_hand_computed():
    # Each fable vs the other: p1 = 2/3, p2 = 1/2, equal lengths so no
    # brevity penalty; BLEU-2 = sqrt(2/3 * 1/2) = sqrt(1/3) for both.
    assert self_bleu(
        ["the cat sat", "the cat ran"], max_n=2
    ) == pytest.approx(math.sqrt(1 / 3))


def test_self_bleu_single_token_fables():
    # Degenerate one-word fables score on unigrams only: each "a" finds
    # the other "a" among its references (BLEU 1), "b" finds nothing (0).
    assert self_bleu(["a", "b", "a"]) == pytest.approx(2 / 3)


def test_self_bleu_rejects_fewer_than_two_fables():
    with pytest.raises(ValueError, match="at least two"):
        self_bleu(["only one fable"])


def test_self_bleu_rejects_fable_without_words():
    with pytest.raises(ValueError, match="no words"):
        self_bleu(["a real fable", "!!!"])


def test_self_bleu_sampling_is_seeded_and_deterministic():
    fables = [f"fable number {i} tells of animal {i}" for i in range(6)]
    sampled_directly = random.Random(0).sample(fables, 3)
    result = self_bleu(fables, sample_size=3, seed=0)
    assert result == self_bleu(sampled_directly)
    assert result == self_bleu(fables, sample_size=3, seed=0)


def test_self_bleu_rejects_sample_size_below_two():
    with pytest.raises(ValueError, match="sample_size"):
        self_bleu(["a b", "c d", "e f"], sample_size=1)
