import pytest

from tinystories_v2.slots import Scaffold, SLOT_SPECIAL_TOKENS
from tinystories_v2.slot_prompt import (
    SLOT_FIELDS,
    END_TOKEN,
    FABLE_TOKEN,
    SlotPromptError,
    render_example,
    render_prompt,
)

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
