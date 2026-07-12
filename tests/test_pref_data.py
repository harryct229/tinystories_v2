"""Unit seams of the preference-labeling stage: pair scheduling, per-Scaffold
seeding, completion sampling, order-swap pair labeling, and the kill-safe
progress store."""

import pytest
import torch
from tokenizers import Tokenizer

from tinystories_v2.judge import PositionBiasedFakeJudge, SlotCoverageFakeJudge
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.pref_data import (
    label_scaffold,
    pair_indices,
    sample_completions,
    scaffold_seed,
)
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run


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


TOY_MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 256, "ffn_hidden": 192}


@pytest.fixture(scope="module")
def toy_tokenizer(tmp_path_factory, fixture_path) -> Tokenizer:
    out = tmp_path_factory.mktemp("tok")
    tokenizer_run({"out_dir": str(out), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    return Tokenizer.from_file(str(out / "tokenizer.json"))


@pytest.fixture(scope="module")
def toy_model() -> FableLM:
    torch.manual_seed(0)
    return FableLM(ModelConfig(**TOY_MODEL))


@pytest.fixture(scope="module")
def toy_scaffold() -> Scaffold:
    return Scaffold(character="fox", trait="greedy", setting="a dense forest",
                    conflict="loses food to a trick",
                    resolution="the trickster is exposed",
                    moral="honesty is the best policy")


def test_sample_completions_is_seed_deterministic(toy_model, toy_tokenizer,
                                                  toy_scaffold):
    kwargs = dict(num_completions=4, max_new_tokens=16, temperature=1.0,
                  top_p=0.95, device="cpu")
    first = sample_completions(toy_model, toy_tokenizer, toy_scaffold,
                               seed=7, **kwargs)
    second = sample_completions(toy_model, toy_tokenizer, toy_scaffold,
                                seed=7, **kwargs)
    assert first == second
    assert len(first) == 4
    assert all(isinstance(text, str) for text in first)


def test_label_scaffold_keeps_consistent_pairs_and_counts_degenerate(
        toy_scaffold):
    completions = [
        "The greedy fox in a dense forest learned honesty is the best policy.",
        "A bird flew.",
        "A bird flew.",   # duplicate of index 1 -> the (1, 2) pair is degenerate
        "Fish swam in a dense forest.",
    ]
    pairs, counters = label_scaffold(
        SlotCoverageFakeJudge(), toy_scaffold, completions, 3)
    # pair_indices(4, 3) == [(0, 3), (1, 2), (0, 2)]; completion 0 has the
    # highest slot coverage, so it wins both non-degenerate pairs.
    assert counters == {"kept": 2, "discarded_inconsistent": 0,
                        "skipped_degenerate": 1}
    assert [pair.chosen for pair in pairs] == [completions[0], completions[0]]
    assert all(pair.verdict.consistent for pair in pairs)


def test_label_scaffold_discards_all_position_biased_verdicts(toy_scaffold):
    pairs, counters = label_scaffold(
        PositionBiasedFakeJudge(), toy_scaffold,
        ["Alpha text.", "Beta text.", "Gamma text.", "Delta text."], 3)
    assert pairs == []
    assert counters == {"kept": 0, "discarded_inconsistent": 3,
                        "skipped_degenerate": 0}


def test_label_scaffold_skips_empty_completions(toy_scaffold):
    pairs, counters = label_scaffold(
        SlotCoverageFakeJudge(), toy_scaffold,
        ["", "Beta text.", "  ", "Delta text."], 3)
    # Schedule (0,3), (1,2), (0,2): every pair touches an empty completion.
    assert pairs == []
    assert counters == {"kept": 0, "discarded_inconsistent": 0,
                        "skipped_degenerate": 3}
