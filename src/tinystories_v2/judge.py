"""Pairwise Judge seam for preference labeling and downstream tests."""

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, astuple, dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

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


def normalize_text(text: str) -> str:
    """Whitespace-collapsed casefold: the seam's notion of candidate equality,
    shared with the labeling stage's degeneracy check."""
    return " ".join(text.casefold().split())


def _validate_candidates(fable_a: str, fable_b: str) -> None:
    if not fable_a.strip() or not fable_b.strip():
        raise ValueError("both candidate Fables must be non-empty")
    if normalize_text(fable_a) == normalize_text(fable_b):
        raise ValueError("candidate Fables must differ")


def _coverage_score(scaffold: Scaffold, fable: str) -> tuple[int, bytes]:
    normalized_fable = normalize_text(fable)
    coverage = sum(
        normalize_text(slot_value) in normalized_fable
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


class TransformersJudge:
    """Pairwise local-model Judge backed by one lazy Transformers code path."""

    def __init__(
        self,
        model_id: str,
        precision: str,
        device: str,
        enable_thinking: bool | None = None,
        max_new_tokens: int = 4,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if precision not in {"fp16", "bf16"}:
            raise ValueError("precision must be 'fp16' or 'bf16'")
        if not device.strip():
            raise ValueError("device must be a non-empty string")
        if enable_thinking is not None and type(enable_thinking) is not bool:
            raise ValueError("enable_thinking must be bool or omitted")
        if (
            type(max_new_tokens) is not int
            or max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")

        self.model_id = model_id
        self.precision = precision
        self.device = device
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self._backend: tuple[Any, Any, Any] | None = None

    @property
    def judge_id(self) -> str:
        if self.enable_thinking is None:
            thinking_mode = "default"
        else:
            thinking_mode = str(self.enable_thinking).lower()
        return (
            f"transformers:{self.model_id};precision={self.precision};"
            f"thinking={thinking_mode};rubric={RUBRIC_VERSION}"
        )

    def _load_backend(self) -> tuple[Any, Any, Any]:
        if self._backend is not None:
            return self._backend
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "real Judge dependencies are missing; "
                "install with: uv pip install -e '.[judge]'"
            ) from exc

        dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        )
        model = model.to(self.device)
        model.eval()
        self._backend = (torch, tokenizer, model)
        return self._backend

    def compare(
        self,
        scaffold: Scaffold,
        fable_a: str,
        fable_b: str,
    ) -> Verdict:
        _validate_candidates(fable_a, fable_b)
        torch, tokenizer, model = self._load_backend()
        messages = [
            {
                "role": "system",
                "content": (
                    "Follow the pairwise Fable rubric and return only its "
                    "requested verdict label."
                ),
            },
            {
                "role": "user",
                "content": render_rubric_prompt(
                    scaffold,
                    fable_a,
                    fable_b,
                ),
            },
        ]
        template_options: dict[str, Any] = {}
        if self.enable_thinking is not None:
            template_options["enable_thinking"] = self.enable_thinking
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_options,
        ).to(self.device)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        raw_output = tokenizer.decode(
            generated[0][prompt_length:],
            skip_special_tokens=True,
        )
        return parse_verdict(raw_output)


def _required_config_string(
    config: Mapping[str, Any],
    key: str,
) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Judge config {key!r} must be a non-empty string")
    return value.strip()


def build_judge(config: Mapping[str, Any]) -> Judge:
    """Construct a fake or real Judge without loading real model weights."""

    kind = config.get("kind")
    if kind == "fake_slot_coverage":
        return SlotCoverageFakeJudge()
    if kind == "fake_position_biased":
        return PositionBiasedFakeJudge()
    if kind == "transformers":
        device = config.get("device", "cuda")
        if not isinstance(device, str) or not device.strip():
            raise ValueError(
                "Judge config 'device' must be a non-empty string"
            )
        return TransformersJudge(
            model_id=_required_config_string(config, "model_id"),
            precision=_required_config_string(config, "precision"),
            device=device.strip(),
            enable_thinking=config.get("enable_thinking"),
            max_new_tokens=config.get("max_new_tokens", 4),
        )
    raise ValueError(f"unknown Judge kind: {kind!r}")
