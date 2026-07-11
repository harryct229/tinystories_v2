import pytest

from tinystories_v2.slots import Scaffold, SLOT_SPECIAL_TOKENS
from tinystories_v2.slot_prompt import (
    SLOT_FIELDS,
    END_TOKEN,
    FABLE_TOKEN,
    SlotPromptError,
    encode_example,
    render_example,
    render_prompt,
)
from tinystories_v2.tokenizer import iter_corpus, train_tokenizer

SCAFFOLD = Scaffold(
    character="fox",
    trait="greedy",
    setting="a dense forest",
    conflict="loses their food",
    resolution="the trickster is exposed",
    moral="honesty is the best policy",
)


def test_render_prompt_is_exact_slot_sequence_ending_at_fable():
    assert render_prompt(SCAFFOLD) == (
        "<|character|>fox<|trait|>greedy<|setting|>a dense forest"
        "<|conflict|>loses their food<|resolution|>the trickster is exposed"
        "<|moral|>honesty is the best policy<|fable|>"
    )


def test_render_example_appends_fable_body_and_end():
    text = render_example(SCAFFOLD, "One day, a fox schemed.")
    assert text == render_prompt(SCAFFOLD) + "One day, a fox schemed.<|end|>"


def test_render_empty_slot_raises():
    bad = Scaffold(
        character="fox", trait="   ", setting="s",
        conflict="c", resolution="r", moral="m",
    )
    with pytest.raises(SlotPromptError, match="trait"):
        render_prompt(bad)


def test_render_empty_fable_raises():
    with pytest.raises(SlotPromptError, match="fable"):
        render_example(SCAFFOLD, "   ")


def test_render_constants_derive_from_special_token_contract():
    # Guard against drift: the render constants must stay consistent with the
    # single source of truth for token order (slots.SLOT_SPECIAL_TOKENS).
    assert SLOT_FIELDS == tuple(token[2:-2] for token in SLOT_SPECIAL_TOKENS[:6])
    assert FABLE_TOKEN == SLOT_SPECIAL_TOKENS[6]
    assert END_TOKEN == SLOT_SPECIAL_TOKENS[7]


@pytest.fixture(scope="module")
def tokenizer(fixture_path):
    # Toy vocab; the artifact contract (specials -> single IDs) is identical to
    # the real 8192 tokenizer, and 512 trains in well under a second.
    texts = iter_corpus([str(fixture_path)], "fable")
    return train_tokenizer(texts, vocab_size=512)


def test_special_tokens_encode_to_single_ids_in_order(tokenizer):
    ex = encode_example(tokenizer, SCAFFOLD, "One day, a fox schemed and lost.")
    spec_ids = {t: tokenizer.token_to_id(t) for t in SLOT_SPECIAL_TOKENS}
    seen = [t for i in ex.input_ids for t in SLOT_SPECIAL_TOKENS if spec_ids[t] == i]
    assert seen == list(SLOT_SPECIAL_TOKENS)  # each special once, in order


def test_loss_mask_boundary_is_exactly_at_fable_token(tokenizer):
    ex = encode_example(tokenizer, SCAFFOLD, "One day, a fox schemed and lost.")
    boundary = ex.input_ids.index(tokenizer.token_to_id(FABLE_TOKEN))
    assert ex.n_prompt_tokens == boundary + 1
    assert ex.loss_mask == (
        [0] * (boundary + 1) + [1] * (len(ex.input_ids) - boundary - 1)
    )
    assert ex.loss_mask[boundary] == 0      # <|fable|> itself is masked
    assert ex.loss_mask[boundary + 1] == 1  # first fable-body token is active
    assert ex.loss_mask[-1] == 1            # <|end|> is active


def test_mask_and_ids_are_same_length(tokenizer):
    ex = encode_example(tokenizer, SCAFFOLD, "A short fable body.")
    assert len(ex.loss_mask) == len(ex.input_ids)


def test_example_to_dict_has_schema_fields(tokenizer):
    ex = encode_example(tokenizer, SCAFFOLD, "A short fable body.")
    assert set(ex.to_dict()) == {"input_ids", "loss_mask", "n_prompt_tokens"}
