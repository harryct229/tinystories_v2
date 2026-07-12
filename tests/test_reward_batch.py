import json

import torch
from tokenizers import Tokenizer

from tinystories_v2.preferences import PreferencePair, VerdictMetadata
from tinystories_v2.reward import encode_pairs, get_pair_batch, load_pairs, split_pairs
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run


def _pair(chosen: str, rejected: str) -> PreferencePair:
    scaffold = Scaffold("fox", "sly", "a wood", "a gate", "it waited", "patience wins")
    return PreferencePair(
        scaffold=scaffold, chosen=chosen, rejected=rejected,
        verdict=VerdictMetadata(judge_id="fake:slot-coverage-v1",
                                first_pass="A", swapped_pass="B", consistent=True))


def test_load_pairs_validates_schema(tmp_path):
    path = tmp_path / "pairs.jsonl"
    records = [_pair("A good fable.", "bad").to_dict(),
               _pair("Another good one.", "meh").to_dict()]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n  \n",  # blank ignored
                    encoding="utf-8")
    pairs = load_pairs(path)
    assert len(pairs) == 2
    assert pairs[0].chosen == "A good fable."


def test_load_pairs_rejects_bad_schema(tmp_path):
    import pytest

    from tinystories_v2.preferences import PreferencePairValidationError
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps({"schema_version": 1, "bogus": True}) + "\n",
                    encoding="utf-8")
    with pytest.raises(PreferencePairValidationError):
        load_pairs(path)


def _encoded(tmp_path, fixture_path):
    tok_dir = tmp_path / "tok"
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    pairs = [_pair(f"Chosen fable number {i}.", "A plain note.") for i in range(20)]
    return encode_pairs(tokenizer, pairs)


def test_encode_pairs_produces_id_lists(tmp_path, fixture_path):
    encoded = _encoded(tmp_path, fixture_path)
    assert len(encoded) == 20
    assert encoded[0]["chosen_ids"] and encoded[0]["rejected_ids"]
    assert all(isinstance(i, int) for i in encoded[0]["chosen_ids"])


def test_split_is_deterministic_and_disjoint(tmp_path, fixture_path):
    encoded = _encoded(tmp_path, fixture_path)
    train_a, holdout_a = split_pairs(encoded, holdout_frac=0.25, seed=7)
    train_b, holdout_b = split_pairs(encoded, holdout_frac=0.25, seed=7)
    assert len(holdout_a) == 5 and len(train_a) == 15
    assert holdout_a == holdout_b and train_a == train_b       # pure function of seed
    other_seed = split_pairs(encoded, holdout_frac=0.25, seed=99)[1]
    assert other_seed != holdout_a                             # seed actually shuffles
    # Train and holdout are disjoint (compare by chosen_ids identity).
    train_ids = [tuple(p["chosen_ids"]) for p in train_a]
    holdout_ids = [tuple(p["chosen_ids"]) for p in holdout_a]
    assert set(train_ids).isdisjoint(holdout_ids)


def test_get_pair_batch_is_pure_and_padded(tmp_path, fixture_path):
    encoded = _encoded(tmp_path, fixture_path)
    a = get_pair_batch(encoded, 4, 64, seed=1337, step=2, micro_step=0)
    b = get_pair_batch(encoded, 4, 64, seed=1337, step=2, micro_step=0)
    for t1, t2 in zip(a, b):
        assert torch.equal(t1, t2)                             # pure in (seed, step, micro_step)
    c_idx, c_len, r_idx, r_len = a
    assert c_idx.shape[0] == c_len.shape[0] == 4
    assert r_idx.shape[0] == r_len.shape[0] == 4
    assert c_idx.shape[1] == int(c_len.max())                  # padded to longest real length
    different = get_pair_batch(encoded, 4, 64, seed=1337, step=2, micro_step=1)
    assert not all(torch.equal(t1, t2) for t1, t2 in zip(a, different))
