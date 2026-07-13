import json
import math

import torch
from tokenizers import Tokenizer

from tinystories_v2.grpo import (
    get_scaffold_batch, load_scaffolds, rollout_batch, safe_self_bleu,
    sample_rollouts,
)
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.slot_prompt import SLOT_FIELDS, render_prompt
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run

TOY_MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 128, "ffn_hidden": 192}


def _scaffold(i: int = 0) -> Scaffold:
    return Scaffold(f"fox{i}", "sly", "a wood", "a gate", "it waited", "patience wins")


def _tokenizer(tmp_path, fixture_path) -> Tokenizer:
    tok_dir = tmp_path / "tok"
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    return Tokenizer.from_file(str(tok_dir / "tokenizer.json"))


def test_load_scaffolds_reads_rows_and_filters_long_prompts(tmp_path, fixture_path):
    tokenizer = _tokenizer(tmp_path, fixture_path)
    split = tmp_path / "pref.jsonl"
    keep = _scaffold(1)
    with split.open("w", encoding="utf-8") as f:
        row = {"prompt_hash": "h1", **{fld: getattr(keep, fld) for fld in SLOT_FIELDS}}
        f.write(json.dumps(row) + "\n")
        # A Scaffold whose prompt exceeds the tiny context is dropped.
        long_setting = "x " * 200
        big = {"prompt_hash": "h2", "character": "a", "trait": "b",
               "setting": long_setting, "conflict": "c", "resolution": "d", "moral": "e"}
        f.write(json.dumps(big) + "\n")
    scaffolds = load_scaffolds(split, tokenizer, context=34)
    assert [s.character for s in scaffolds] == ["fox1"]   # only the short prompt kept


def test_get_scaffold_batch_is_pure_in_seed_and_step(tmp_path, fixture_path):
    scaffolds = [_scaffold(i) for i in range(12)]
    a = get_scaffold_batch(scaffolds, 4, seed=1337, step=3)
    b = get_scaffold_batch(scaffolds, 4, seed=1337, step=3)
    assert [s.character for s in a] == [s.character for s in b]     # reproducible
    assert len(a) == 4
    other = get_scaffold_batch(scaffolds, 4, seed=1337, step=4)
    assert [s.character for s in a] != [s.character for s in other]  # step changes it


def test_sample_rollouts_returns_sequences_prompt_len_and_fables(tmp_path, fixture_path):
    tokenizer = _tokenizer(tmp_path, fixture_path)
    torch.manual_seed(0)
    model = FableLM(ModelConfig(**TOY_MODEL))
    scaffold = _scaffold()
    prompt_len = len(tokenizer.encode(render_prompt(scaffold)).ids)
    sequences, plen, fables = sample_rollouts(
        model, tokenizer, scaffold, group_size=3, max_new_tokens=8,
        temperature=1.0, top_p=1.0, seed=7)
    assert plen == prompt_len
    assert len(sequences) == len(fables) == 3
    for seq in sequences:
        assert seq[:plen] == tokenizer.encode(render_prompt(scaffold)).ids  # prompt intact
        assert len(seq) > plen                                              # generated something


def test_sample_rollouts_is_seeded(tmp_path, fixture_path):
    tokenizer = _tokenizer(tmp_path, fixture_path)
    torch.manual_seed(0)
    model = FableLM(ModelConfig(**TOY_MODEL))
    scaffold = _scaffold()
    first, _, _ = sample_rollouts(model, tokenizer, scaffold, group_size=2,
                                  max_new_tokens=8, temperature=1.0, top_p=1.0, seed=7)
    second, _, _ = sample_rollouts(model, tokenizer, scaffold, group_size=2,
                                   max_new_tokens=8, temperature=1.0, top_p=1.0, seed=7)
    assert first == second                                    # same seed -> same rollouts


def test_rollout_batch_masks_completion_only_and_pads():
    # Two rollouts, prompt_len 2 each, different total lengths.
    sequences = [[5, 6, 7, 8], [5, 6, 9]]     # completions: [7,8] and [9]
    x, y, mask = rollout_batch(sequences, [2, 2], context=128, device="cpu")
    assert x.shape == y.shape == mask.shape
    assert x.shape[0] == 2 and x.shape[1] == 3           # padded to longest x (len 3)
    # Row 0: y = [6,7,8]; completion targets are seq[2:]=7,8 -> mask positions 1,2.
    assert mask[0].tolist() == [0.0, 1.0, 1.0]
    # Row 1: y = [6,9,pad]; completion target seq[2]=9 -> mask position 1; pad is 0.
    assert mask[1].tolist() == [0.0, 1.0, 0.0]
    assert x.dtype == torch.long and mask.dtype == torch.float


def test_safe_self_bleu_guards_degenerate_rollouts():
    assert math.isnan(safe_self_bleu(["only one"]))            # < 2 usable
    assert math.isnan(safe_self_bleu(["", "   "]))            # no words
    value = safe_self_bleu(["the fox ran home", "the fox ran home", "a bird sang"])
    assert 0.0 <= value <= 1.0
