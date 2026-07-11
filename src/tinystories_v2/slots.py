"""Scaffold slot extraction from TF1-EN-3M's verbose prompt field.

The dataset has no slot columns: the six Scaffold slots are embedded in a
fixed natural-language template. Written against real records, which differ
from the paper: the trait is folded into "Main Character: a <trait>
<character>" and the setting ends with "where our story unfolds" boilerplate.
"""

import re
from dataclasses import dataclass

# Reserved at tokenizer creation (ADR-0003). Order is fixed — downstream
# stages assume it; never reorder or append in the middle.
SLOT_SPECIAL_TOKENS = (
    "<|character|>",
    "<|trait|>",
    "<|setting|>",
    "<|conflict|>",
    "<|resolution|>",
    "<|moral|>",
    "<|fable|>",
    "<|end|>",
)


class SlotExtractionError(ValueError):
    pass


@dataclass(frozen=True)
class Scaffold:
    character: str
    trait: str
    setting: str
    conflict: str
    resolution: str
    moral: str


# Anchored on the template's field labels; tolerant of newline-vs-space
# separators and -/– list dashes. Trait is the first word after the article.
_PROMPT_RE = re.compile(
    r"Main Character:\s*(?:[Aa]n?\s+)?(?P<trait>\S+)\s+(?P<character>.+?)"
    r"\s*[-–]\s*Setting:\s*(?P<setting>.+?)"
    r"\s*[-–]\s*Challenge:\s*(?P<conflict>.+?)"
    r"\s*[-–]\s*Outcome:\s*(?P<resolution>.+?)"
    r"\s*[-–]\s*Teaching:\s*(?P<moral>.+?)"
    r"\s*The fable should",
    re.DOTALL,
)

_SETTING_BOILERPLATE = re.compile(r"\s+where our story unfolds$")


def extract_slots(prompt: str) -> Scaffold:
    match = _PROMPT_RE.search(prompt)
    if match is None:
        raise SlotExtractionError(f"prompt does not match template: {prompt[:120]!r}")
    return Scaffold(
        character=match["character"].strip(),
        trait=match["trait"].strip(),
        setting=_SETTING_BOILERPLATE.sub("", match["setting"].strip()),
        conflict=match["conflict"].strip(),
        resolution=match["resolution"].strip(),
        moral=match["moral"].strip(),
    )
