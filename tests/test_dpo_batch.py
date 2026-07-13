import torch
from tokenizers import Tokenizer

from tinystories_v2.dpo import encode_pairs, get_pair_batch
from tinystories_v2.preferences import PreferencePair, VerdictMetadata
from tinystories_v2.reward import split_pairs
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run


def _pair(chosen: str, rejected: str) -> PreferencePair:
    scaffold = Scaffold("fox", "sly", "a wood", "a gate", "it waited", "patience wins")
    return PreferencePair(
        scaffold=scaffold, chosen=chosen, rejected=rejected,
        verdict=VerdictMetadata(judge_id="fake:slot-coverage-v1",
                                first_pass="A", swapped_pass="B", consistent=True))


def _encoded(tmp_path, fixture_path):
    tok_dir = tmp_path / "tok"
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    pairs = [_pair(f"Chosen fable number {i}.", "A plain note.") for i in range(20)]
    return encode_pairs(tokenizer, pairs)


def test_encode_pairs_produces_ids_and_masks(tmp_path, fixture_path):
    encoded = _encoded(tmp_path, fixture_path)
    assert len(encoded) == 20
    e = encoded[0]
    assert e["chosen_ids"] and e["rejected_ids"]
    assert len(e["chosen_ids"]) == len(e["chosen_mask"])
    assert len(e["rejected_ids"]) == len(e["rejected_mask"])
    assert set(e["chosen_mask"]) <= {0, 1}
    assert sum(e["chosen_mask"]) > 0                       # completion tokens are active
    assert e["chosen_mask"][0] == 0                        # prompt prefix is masked


def test_split_is_deterministic_on_dpo_encoding(tmp_path, fixture_path):
    encoded = _encoded(tmp_path, fixture_path)
    train_a, holdout_a = split_pairs(encoded, holdout_frac=0.25, seed=7)
    train_b, holdout_b = split_pairs(encoded, holdout_frac=0.25, seed=7)
    assert len(holdout_a) == 5 and len(train_a) == 15
    assert holdout_a == holdout_b and train_a == train_b
    train_ids = {tuple(p["chosen_ids"]) for p in train_a}
    holdout_ids = {tuple(p["chosen_ids"]) for p in holdout_a}
    assert train_ids.isdisjoint(holdout_ids)


def test_get_pair_batch_is_pure_and_padded(tmp_path, fixture_path):
    encoded = _encoded(tmp_path, fixture_path)
    a = get_pair_batch(encoded, 4, 64, seed=1337, step=2, micro_step=0)
    b = get_pair_batch(encoded, 4, 64, seed=1337, step=2, micro_step=0)
    (cx_a, cy_a, cm_a), (rx_a, ry_a, rm_a) = a
    for t1, t2 in zip((*a[0], *a[1]), (*b[0], *b[1])):
        assert torch.equal(t1, t2)                         # pure in (seed, step, micro_step)
    assert cx_a.shape == cy_a.shape == cm_a.shape          # aligned x/y/mask
    assert cx_a.shape[0] == 4 and rx_a.shape[0] == 4
    assert cm_a.dtype == torch.float and cx_a.dtype == torch.long
    different = get_pair_batch(encoded, 4, 64, seed=1337, step=2, micro_step=1)
    assert not torch.equal(different[0][0], cx_a)           # micro_step changes the draw
