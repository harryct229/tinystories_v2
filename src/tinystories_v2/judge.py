"""Pairwise Judge seam for preference labeling and downstream tests."""

import hashlib
from dataclasses import astuple, dataclass
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
