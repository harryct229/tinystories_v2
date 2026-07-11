"""Pairwise Judge seam for preference labeling and downstream tests."""

import hashlib
import json
import re
from dataclasses import asdict, astuple, dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from tinystories_v2.preferences import (
    PreferencePair,
    VerdictMetadata,
    validate_preference_pair,
)
from tinystories_v2.slots import Scaffold


class Verdict(StrEnum):
    A = "A"
    B = "B"

    @property
    def opposite(self) -> "Verdict":
        return Verdict.B if self is Verdict.A else Verdict.A


RUBRIC_VERSION = "fable-pairwise-v1"


class JudgeOutputError(ValueError):
    """Raised when a real Judge does not return one parseable verdict."""


_VERDICT_RE = re.compile(
    r"\s*(?:verdict\s*:\s*)?([AB])\s*[.]?\s*",
    re.IGNORECASE,
)


def render_rubric_prompt(
    scaffold: Scaffold,
    fable_a: str,
    fable_b: str,
) -> str:
    """Render the pairwise rubric without invoking or importing a model."""

    payload = json.dumps(
        {
            "scaffold": asdict(scaffold),
            "candidate_a": fable_a,
            "candidate_b": fable_b,
        },
        ensure_ascii=False,
        indent=2,
    )
    return (
        "You are the Judge selecting the better moral Fable for children "
        "ages 4–7.\n\n"
        "Compare the candidates on exactly these four axes:\n"
        "1. Prompt Adherence (HIGHEST WEIGHT): faithful realization of all "
        "six Scaffold slots and the requested Fable form.\n"
        "2. Moral Clarity (SECOND PRIORITY): an explicit, relevant ethical "
        "lesson connected to the ending.\n"
        "3. Grammar & Style: correct, fluent, concrete, age-appropriate "
        "language.\n"
        "4. Creativity: an engaging and original narrative realization.\n\n"
        "Age suitability is a HARD CONSTRAINT: reject content whose "
        "vocabulary, syntax, themes, or detail are unsuitable for ages 4–7.\n"
        "Candidate labels are arbitrary. Judge content, never presentation "
        "position. Do not return a tie.\n\n"
        f"INPUT:\n{payload}\n\n"
        "Return exactly one capital letter: A or B."
    )


def parse_verdict(raw_output: str) -> Verdict:
    match = _VERDICT_RE.fullmatch(raw_output)
    if match is None:
        raise JudgeOutputError(
            f"Judge must return a single A/B verdict, got {raw_output[:120]!r}"
        )
    return Verdict(match.group(1).upper())


@runtime_checkable
class Judge(Protocol):
    @property
    def judge_id(self) -> str:
        raise NotImplementedError

    def compare(
        self,
        scaffold: Scaffold,
        fable_a: str,
        fable_b: str,
    ) -> Verdict:
        raise NotImplementedError


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _validate_candidates(fable_a: str, fable_b: str) -> None:
    if not fable_a.strip() or not fable_b.strip():
        raise ValueError("both candidate Fables must be non-empty")
    if _normalize(fable_a) == _normalize(fable_b):
        raise ValueError("candidate Fables must differ")


def _coverage_score(scaffold: Scaffold, fable: str) -> tuple[int, bytes]:
    normalized_fable = _normalize(fable)
    coverage = sum(
        _normalize(slot_value) in normalized_fable
        for slot_value in astuple(scaffold)
    )
    stable_tie_break = hashlib.sha256(
        normalized_fable.encode("utf-8")
    ).digest()
    return coverage, stable_tie_break


@dataclass(frozen=True)
class SlotCoverageFakeJudge:
    judge_id: str = "fake:slot-coverage-v1"

    def compare(
        self,
        scaffold: Scaffold,
        fable_a: str,
        fable_b: str,
    ) -> Verdict:
        _validate_candidates(fable_a, fable_b)
        score_a = _coverage_score(scaffold, fable_a)
        score_b = _coverage_score(scaffold, fable_b)
        return Verdict.A if score_a > score_b else Verdict.B


@dataclass(frozen=True)
class PositionBiasedFakeJudge:
    judge_id: str = "fake:position-a-v1"

    def compare(
        self,
        scaffold: Scaffold,
        fable_a: str,
        fable_b: str,
    ) -> Verdict:
        return Verdict.A


def judge_with_order_swap(
    judge: Judge,
    scaffold: Scaffold,
    fable_a: str,
    fable_b: str,
) -> PreferencePair | None:
    """Judge both presentations and retain only one consistent preference."""

    _validate_candidates(fable_a, fable_b)
    first_pass = judge.compare(scaffold, fable_a, fable_b)
    swapped_pass = judge.compare(scaffold, fable_b, fable_a)
    if first_pass is swapped_pass:
        return None

    if first_pass is Verdict.A:
        chosen, rejected = fable_a, fable_b
    else:
        chosen, rejected = fable_b, fable_a

    pair = PreferencePair(
        scaffold=scaffold,
        chosen=chosen,
        rejected=rejected,
        verdict=VerdictMetadata(
            judge_id=judge.judge_id,
            first_pass=first_pass.value,
            swapped_pass=swapped_pass.value,
            consistent=True,
        ),
    )
    return validate_preference_pair(pair.to_dict())
