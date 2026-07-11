"""Reference-free text metrics over plain lists of Fable strings.

Implements the dataset paper's reference-free table (Self-BLEU, Distinct-n,
Flesch Reading Ease) as a pure standard-library module: no model, GPU, or
network dependencies, and deterministic for identical inputs. Consumed by
the eval suite (issue 07) and GRPO diversity monitoring (issue 06).

Word convention shared by every metric: casefolded runs of letters/digits
with internal apostrophes kept ("Don't" -> "don't").
"""

import math
import random
import re
from collections import Counter
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


def _modified_precision(
    hypothesis: list[str],
    references: list[list[str]],
    n: int,
) -> tuple[int, int]:
    hyp_counts = Counter(_ngrams(hypothesis, n))
    max_ref_counts: Counter[tuple[str, ...]] = Counter()
    for reference in references:
        for gram, count in Counter(_ngrams(reference, n)).items():
            max_ref_counts[gram] = max(max_ref_counts[gram], count)
    clipped = sum(
        min(count, max_ref_counts[gram])
        for gram, count in hyp_counts.items()
    )
    return clipped, sum(hyp_counts.values())


def _bleu(
    hypothesis: list[str],
    references: list[list[str]],
    max_n: int,
) -> float:
    hyp_len = len(hypothesis)
    log_precisions = []
    for n in range(1, min(max_n, hyp_len) + 1):
        clipped, total = _modified_precision(hypothesis, references, n)
        if clipped == 0:
            return 0.0
        log_precisions.append(math.log(clipped / total))
    ref_len = min(
        (abs(len(reference) - hyp_len), len(reference))
        for reference in references
    )[1]
    brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return brevity * math.exp(sum(log_precisions) / len(log_precisions))


def self_bleu(
    fables: Sequence[str],
    max_n: int = 4,
    sample_size: int | None = None,
    seed: int = 0,
) -> float:
    """Mean BLEU of each fable against all others (lower = more diverse).

    Matches the paper's Self-BLEU usage: every fable is scored as a
    hypothesis with the remaining fables as references, and the scores
    are averaged. BLEU here is the geometric mean of clipped n-gram
    precisions over orders 1..min(max_n, hypothesis length) with a
    brevity penalty against the closest reference length; any zero
    precision scores that fable 0 (no smoothing).

    For cost control on large sets, pass sample_size to score a
    deterministic random.Random(seed) subsample instead of the full set.
    Consuming stages wire sample_size/seed to their configs.
    """
    if max_n < 1:
        raise ValueError("max_n must be at least 1")
    if sample_size is not None and sample_size < 2:
        raise ValueError("sample_size must be at least 2")
    chosen = list(fables)
    if sample_size is not None and sample_size < len(chosen):
        chosen = random.Random(seed).sample(chosen, sample_size)
    if len(chosen) < 2:
        raise ValueError("self_bleu needs at least two fables")
    tokenized = _tokenize_fables(chosen)
    scores = [
        _bleu(hypothesis, tokenized[:i] + tokenized[i + 1 :], max_n)
        for i, hypothesis in enumerate(tokenized)
    ]
    return sum(scores) / len(scores)


_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_VOWEL_RUN_RE = re.compile(r"[aeiouy]+")


def _count_syllables(word: str) -> int:
    """Heuristic: vowel runs, dropping a silent final 'e' (but not 'le')."""
    runs = _VOWEL_RUN_RE.findall(word)
    count = len(runs)
    if count > 1 and word.endswith("e") and not word.endswith("le"):
        count -= 1
    return max(count, 1)


def flesch_reading_ease(text: str) -> float:
    """206.835 - 1.015*(words/sentences) - 84.6*(syllables/words).

    Sentences are [.!?]-delimited segments containing at least one word;
    text with words but no terminal punctuation counts as one sentence.
    Higher = easier reading; the paper's Fables average 78.9 (ages 4-7).
    Syllable counts use the documented vowel-run heuristic, so absolute
    values are comparable within this library, not across other tools.
    """
    words = tokenize_words(text)
    if not words:
        raise ValueError("text contains no words")
    sentences = sum(
        1
        for segment in _SENTENCE_SPLIT_RE.split(text)
        if tokenize_words(segment)
    )
    syllables = sum(_count_syllables(word) for word in words)
    return (
        206.835
        - 1.015 * (len(words) / sentences)
        - 84.6 * (syllables / len(words))
    )


def mean_flesch_reading_ease(fables: Sequence[str]) -> float:
    """Mean per-fable Flesch Reading Ease, matching the paper's tables."""
    if not fables:
        raise ValueError("mean_flesch_reading_ease needs at least one fable")
    return sum(flesch_reading_ease(fable) for fable in fables) / len(fables)
