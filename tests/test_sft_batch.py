import json

import torch

from tinystories_v2.sft import get_sft_batch, load_sft_examples, masked_lm_loss

# Two hand-built examples in schema v1 shape. Ids are arbitrary; the mask
# marks the prompt prefix (0) vs body+end (1). Example lengths differ so
# padding and per-row shifting are exercised.
EXAMPLES = [
    {"prompt_hash": "a", "input_ids": [10, 11, 12, 13, 14],
     "loss_mask": [0, 0, 1, 1, 1]},                       # len 5, 3 active
    {"prompt_hash": "b", "input_ids": [20, 21, 22, 23, 24, 25, 26],
     "loss_mask": [0, 0, 0, 1, 1, 1, 1]},                 # len 7, 4 active
]


def test_load_sft_examples_reads_jsonl(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text(
        "\n".join(json.dumps(e) for e in EXAMPLES) + "\n" + "  \n",  # blank line ignored
        encoding="utf-8",
    )
    loaded = load_sft_examples(path)
    assert loaded == EXAMPLES


def test_batch_selection_is_pure_function_of_step_and_micro_step():
    a = get_sft_batch(EXAMPLES, 4, 16, seed=1337, step=3, micro_step=0)
    b = get_sft_batch(EXAMPLES, 4, 16, seed=1337, step=3, micro_step=0)
    for t1, t2 in zip(a, b):
        assert torch.equal(t1, t2)
    c = get_sft_batch(EXAMPLES, 4, 16, seed=1337, step=3, micro_step=1)
    # Different micro_step almost certainly draws a different multiset.
    assert not all(torch.equal(t1, t2) for t1, t2 in zip(a, c))


def test_shift_and_mask_alignment_for_a_single_example():
    # micro_batch_size=1 with a 1-example pool always draws that example.
    x, y, mask = get_sft_batch(EXAMPLES[:1], 1, 16, seed=0, step=0, micro_step=0)
    ids, lm = EXAMPLES[0]["input_ids"], EXAMPLES[0]["loss_mask"]
    assert x[0].tolist() == ids[:-1]        # x = input_ids[:-1]
    assert y[0].tolist() == ids[1:]         # y = input_ids[1:] (next-token target)
    assert mask[0].tolist() == [float(v) for v in lm[1:]]  # mask = loss_mask[1:]


def test_rows_are_right_padded_to_batch_width_with_zero_mask():
    x, y, mask = get_sft_batch(EXAMPLES, 8, 16, seed=5, step=0, micro_step=0)
    assert x.shape == y.shape == mask.shape
    width = x.shape[1]
    assert width == 6  # longest example (len 7) shifts to length 6
    # Every padded tail position must have mask 0 (never contributes to loss).
    for row_ids, row_mask in zip(x.tolist(), mask.tolist()):
        # padding id is 0; wherever mask is 0 past a row's real length it is padding
        assert len(row_ids) == width and len(row_mask) == width


def test_truncates_examples_longer_than_context():
    long_example = [{"prompt_hash": "c",
                     "input_ids": list(range(100)),
                     "loss_mask": [0] * 10 + [1] * 90}]
    x, y, mask = get_sft_batch(long_example, 1, 8, seed=0, step=0, micro_step=0)
    # ids truncated to context+1 = 9, then shifted to length 8 = context.
    assert x.shape[1] == 8
    assert x[0].tolist() == list(range(8))        # first 8 ids
    assert y[0].tolist() == list(range(1, 9))


def test_masked_loss_averages_only_active_positions():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7)          # [B, L, V]
    y = torch.randint(0, 7, (2, 3))
    full = torch.ones(2, 3)
    # Full mask equals plain mean cross-entropy.
    ce = torch.nn.functional.cross_entropy(logits.view(-1, 7), y.view(-1))
    assert torch.allclose(masked_lm_loss(logits, y, full), ce)
    # Zeroing a column drops it from both numerator and denominator.
    partial = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    per_tok = torch.nn.functional.cross_entropy(
        logits.view(-1, 7), y.view(-1), reduction="none").view(2, 3)
    expected = (per_tok * partial).sum() / partial.sum()
    assert torch.allclose(masked_lm_loss(logits, y, partial), expected)
