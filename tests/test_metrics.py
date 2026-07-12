import math
import random

import pytest
import tinystories_v2.metrics as metrics

from tinystories_v2.metrics import (
    distinct_n,
    flesch_reading_ease,
    mean_flesch_reading_ease,
    self_bleu,
    tokenize_words,
)


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


def test_mean_distinct_1_averages_per_fable_ratios_not_pooled_counts():
    fables = ["a a a a", "b c"]

    # Per-Fable ratios are 1/4 and 2/2, whose mean is 5/8. Pooling instead
    # gives 3 unique unigrams / 6 total = 1/2.
    assert metrics.mean_distinct_n(fables) == pytest.approx(5 / 8)
    assert distinct_n(fables) == pytest.approx(1 / 2)


def test_mean_distinct_rejects_empty_set():
    with pytest.raises(ValueError, match="at least one"):
        metrics.mean_distinct_n([])


def test_mean_distinct_rejects_fable_without_words():
    with pytest.raises(ValueError, match="no words"):
        metrics.mean_distinct_n(["a real fable", "!!!"])


def test_mean_distinct_rejects_fable_shorter_than_n():
    with pytest.raises(ValueError, match=r"index 1.*shorter than n=2"):
        metrics.mean_distinct_n(["a b c", "x"], n=2)


def test_mean_distinct_rejects_n_below_one():
    with pytest.raises(ValueError, match="at least 1"):
        metrics.mean_distinct_n(["a b"], n=0)


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


def test_self_bleu_sampling_is_seeded_and_selects_scored_subset():
    fables = [
        "fox ran safely home",
        "fox ran safely home",
        "dog barked beside river",
        "dog barked beside river",
        "stars shimmer over ocean",
        "queen slept under mountain",
    ]
    seed_0_sample = random.Random(0).sample(fables, 2)
    seed_15_sample = random.Random(15).sample(fables, 2)

    seed_0_result = self_bleu(fables, sample_size=2, seed=0)
    seed_15_result = self_bleu(fables, sample_size=2, seed=15)

    assert seed_0_result == self_bleu(seed_0_sample) == 0.0
    assert seed_15_result == self_bleu(seed_15_sample) == pytest.approx(1.0)
    assert seed_0_result == self_bleu(fables, sample_size=2, seed=0)


def test_self_bleu_rejects_sample_size_below_two():
    with pytest.raises(ValueError, match="sample_size"):
        self_bleu(["a b", "c d", "e f"], sample_size=1)


def test_flesch_single_sentence_hand_computed():
    # 6 words, 6 syllables, 1 sentence:
    # 206.835 - 1.015*(6/1) - 84.6*(6/6) = 116.145
    assert flesch_reading_ease("The cat sat on the mat.") == pytest.approx(
        116.145
    )


def test_flesch_multi_sentence_and_syllable_rules_hand_computed():
    # Syllables: the=1 happy=2 fox=1 ran=1 it=1 was=1 little=2 ("le"
    # ending keeps its final vowel run) -> 9 syllables, 7 words,
    # 2 sentences:
    # 206.835 - 1.015*(7/2) - 84.6*(9/7) = 94.51107142857143
    assert flesch_reading_ease(
        "The happy fox ran. It was little."
    ) == pytest.approx(94.51107142857143)


def test_flesch_text_without_terminal_punctuation_is_one_sentence():
    # hello=2 world=1 -> 3 syllables, 2 words, 1 sentence:
    # 206.835 - 1.015*2 - 84.6*1.5 = 77.905
    assert flesch_reading_ease("hello world") == pytest.approx(77.905)


def test_flesch_removes_silent_final_e_vowel_run():
    # cake has two vowel runs but a silent final e, so it has one syllable.
    assert flesch_reading_ease("Cake.") == pytest.approx(121.22)


def test_flesch_zero_vowel_word_has_one_syllable_minimum():
    # nth has no [aeiouy] run but is clamped to one syllable.
    assert flesch_reading_ease("Nth.") == pytest.approx(121.22)


def test_flesch_rejects_text_without_words():
    with pytest.raises(ValueError, match="no words"):
        flesch_reading_ease("?!.")


def test_mean_flesch_is_mean_of_per_fable_scores():
    assert mean_flesch_reading_ease(
        ["The cat sat on the mat.", "hello world"]
    ) == pytest.approx((116.145 + 77.905) / 2)


def test_mean_flesch_rejects_empty_set():
    with pytest.raises(ValueError, match="at least one"):
        mean_flesch_reading_ease([])
