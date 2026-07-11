"""Reference-free text metrics over plain lists of Fable strings.

Implements the dataset paper's reference-free table (Self-BLEU, Distinct-n,
Flesch Reading Ease) as a pure standard-library module: no model, GPU, or
network dependencies, and deterministic for identical inputs. Consumed by
the eval suite (issue 07) and GRPO diversity monitoring (issue 06).

Word convention shared by every metric: casefolded runs of letters/digits
with internal apostrophes kept ("Don't" -> "don't").
"""

import re
from collections.abc import Sequence

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def tokenize_words(text: str) -> list[str]:
    """Split text into the casefolded word tokens all metrics count."""
    return _WORD_RE.findall(text.casefold())


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _tokenize_fables(fables: Sequence[str]) -> list[list[str]]:
    tokenized = []
    for index, fable in enumerate(fables):
        tokens = tokenize_words(fable)
        if not tokens:
            raise ValueError(f"fable at index {index} contains no words")
        tokenized.append(tokens)
    return tokenized


def distinct_n(fables: Sequence[str], n: int = 1) -> float:
    """Unique / total n-grams pooled over the set (paper reports n = 1).

    Higher = richer vocabulary. Fables shorter than n tokens contribute
    no n-grams; the set must still yield at least one n-gram overall.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if not fables:
        raise ValueError("distinct_n needs at least one fable")
    pooled: list[tuple[str, ...]] = []
    for tokens in _tokenize_fables(fables):
        pooled.extend(_ngrams(tokens, n))
    if not pooled:
        raise ValueError(f"no {n}-grams: every fable is shorter than n={n}")
    return len(set(pooled)) / len(pooled)
