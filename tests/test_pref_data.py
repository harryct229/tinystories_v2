"""Unit seams of the preference-labeling stage: pair scheduling, per-Scaffold
seeding, completion sampling, order-swap pair labeling, and the kill-safe
progress store."""

import pytest

from tinystories_v2.pref_data import pair_indices, scaffold_seed


def test_pair_indices_covers_all_completions_before_repeating():
    # Round-robin: the first two pairs are disjoint, so N=4 completions all
    # appear before any completion is reused.
    assert pair_indices(4, 3) == [(0, 3), (1, 2), (0, 2)]


def test_pair_indices_full_schedule_is_every_unique_pair():
    schedule = pair_indices(4, 6)
    assert len(set(schedule)) == 6
    assert all(a < b for a, b in schedule)


def test_pair_indices_handles_odd_completion_counts():
    assert pair_indices(3, 3) == [(1, 2), (0, 2), (0, 1)]


def test_pair_indices_rejects_bad_counts():
    with pytest.raises(ValueError):
        pair_indices(4, 7)   # only C(4,2) = 6 pairs exist
    with pytest.raises(ValueError):
        pair_indices(4, 0)
    with pytest.raises(ValueError):
        pair_indices(1, 1)


def test_scaffold_seed_is_deterministic_and_input_sensitive():
    assert scaffold_seed(1337, "abc") == scaffold_seed(1337, "abc")
    assert scaffold_seed(1337, "abc") != scaffold_seed(1337, "abd")
    assert scaffold_seed(1, "abc") != scaffold_seed(2, "abc")
    assert 0 <= scaffold_seed(1337, "abc") < 2**63
