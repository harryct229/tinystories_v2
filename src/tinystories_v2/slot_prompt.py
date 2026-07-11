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

from collections.abc import Sequence
from dataclasses import dataclass

from tokenizers import Tokenizer

from tinystories_v2.slots import SLOT_SPECIAL_TOKENS, Scaffold

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


@dataclass(frozen=True)
class SlotPromptExample:
    input_ids: list[int]
    loss_mask: list[int]   # 0 over the prompt prefix, 1 over body + <|end|>
    n_prompt_tokens: int   # leading masked tokens (through <|fable|>)

    def to_dict(self) -> dict:
        return {
            "input_ids": self.input_ids,
            "loss_mask": self.loss_mask,
            "n_prompt_tokens": self.n_prompt_tokens,
        }


def encode_example(
    tokenizer: Tokenizer, scaffold: Scaffold, fable: str
) -> SlotPromptExample:
    """Tokenize a training example and compute its loss mask: masked (0) through
    and including <|fable|>, active (1) over the fable body and <|end|>."""
    text = render_example(scaffold, fable)
    ids = tokenizer.encode(text).ids
    boundary = ids.index(tokenizer.token_to_id(FABLE_TOKEN))  # <|fable|> once
    n_prompt_tokens = boundary + 1
    loss_mask = [0] * n_prompt_tokens + [1] * (len(ids) - n_prompt_tokens)
    return SlotPromptExample(
        input_ids=ids, loss_mask=loss_mask, n_prompt_tokens=n_prompt_tokens
    )


@dataclass(frozen=True)
class ParsedSlotPrompt:
    scaffold: Scaffold
    fable: str


def parse_example(tokenizer: Tokenizer, ids: Sequence[int]) -> ParsedSlotPrompt:
    """Recover the Scaffold and fable body from a complete example sequence.
    Raises SlotPromptError unless all eight special tokens appear exactly once,
    in canonical order, with <|character|> first and <|end|> last."""
    ids = list(ids)
    special_ids = {tok: tokenizer.token_to_id(tok) for tok in SLOT_SPECIAL_TOKENS}
    for tok, sid in special_ids.items():
        if ids.count(sid) != 1:
            raise SlotPromptError(f"{tok} must appear exactly once")
    positions = [ids.index(special_ids[tok]) for tok in SLOT_SPECIAL_TOKENS]
    if positions != sorted(positions):
        raise SlotPromptError("special tokens are out of order")
    if ids[0] != special_ids["<|character|>"] or ids[-1] != special_ids[END_TOKEN]:
        raise SlotPromptError(
            "sequence must start with <|character|> and end with <|end|>"
        )
    # positions align 1:1 with SLOT_SPECIAL_TOKENS; decode each inter-marker span.
    n = len(SLOT_FIELDS)
    slot_values = {
        field: tokenizer.decode(ids[start + 1 : end])
        for field, start, end in zip(SLOT_FIELDS, positions[:n], positions[1 : n + 1])
    }
    fable = tokenizer.decode(ids[positions[n] + 1 : positions[n + 1]])
    return ParsedSlotPrompt(scaffold=Scaffold(**slot_values), fable=fable)
