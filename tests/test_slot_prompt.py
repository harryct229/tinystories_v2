import pytest

from tinystories_v2.slots import Scaffold
from tinystories_v2.slot_prompt import (
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
