"""Slot Prompt format: render a Scaffold to the reserved special-token
sequence, encode SFT training examples with a masked-loss boundary, and parse
the inverse.

The token order is the format contract every stage from SFT onward relies on
(issues 03, 04, 06, 07). The sequence is exactly:

    <|character|>C<|trait|>T<|setting|>S<|conflict|>Cf<|resolution|>R
    <|moral|>M<|fable|>{fable body}<|end|>

Loss is masked over the conditioning prefix (through and including
`<|fable|>`) and active over the fable body and the trailing `<|end|>`.
"""

from tinystories_v2.slots import Scaffold

# The six conditioning slots in render order — the first six SLOT_SPECIAL_TOKENS
# without their <| |> delimiters. The trailing two specials (<|fable|>, <|end|>)
# frame the fable body, not a slot.
SLOT_FIELDS = ("character", "trait", "setting", "conflict", "resolution", "moral")

FABLE_TOKEN = "<|fable|>"
END_TOKEN = "<|end|>"


class SlotPromptError(ValueError):
    """Raised when a Scaffold cannot be rendered or a token sequence cannot be
    parsed as a Slot Prompt (empty slot, missing marker, wrong order)."""


def _slot_values(scaffold: Scaffold) -> list[str]:
    values = []
    for field in SLOT_FIELDS:
        value = getattr(scaffold, field)
        if not value or not value.strip():
            raise SlotPromptError(f"{field} slot is empty")
        values.append(value)
    return values


def render_prompt(scaffold: Scaffold) -> str:
    """The conditioning prefix ending at <|fable|> (no fable body). Feed this to
    the model and let it complete the fable."""
    values = _slot_values(scaffold)
    parts = [f"<|{field}|>{value}" for field, value in zip(SLOT_FIELDS, values)]
    return "".join(parts) + FABLE_TOKEN


def render_example(scaffold: Scaffold, fable: str) -> str:
    """The full training text: prompt prefix + fable body + <|end|>."""
    if not fable or not fable.strip():
        raise SlotPromptError("fable body is empty")
    return render_prompt(scaffold) + fable + END_TOKEN
