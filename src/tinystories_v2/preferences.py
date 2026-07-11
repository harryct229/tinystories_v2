"""Versioned preference-pair records shared by RLAIF stages.

The JSON-compatible v1 contract is documented in
docs/schemas/preference-pair-v1.md. Only order-swap-consistent Judge
comparisons become PreferencePair values.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from tinystories_v2.slots import Scaffold

SCHEMA_VERSION = 1
SCAFFOLD_FIELDS = (
    "character",
    "trait",
    "setting",
    "conflict",
    "resolution",
    "moral",
)
TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "scaffold", "chosen", "rejected", "verdict"}
)
VERDICT_FIELDS = frozenset(
    {"judge_id", "first_pass", "swapped_pass", "consistent"}
)
ALLOWED_VERDICTS = frozenset({"A", "B"})


class PreferencePairValidationError(ValueError):
    """Raised when a value does not conform to preference-pair schema v1."""


@dataclass(frozen=True)
class VerdictMetadata:
    judge_id: str
    first_pass: str
    swapped_pass: str
    consistent: bool


@dataclass(frozen=True)
class PreferencePair:
    scaffold: Scaffold
    chosen: str
    rejected: str
    verdict: VerdictMetadata
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scaffold": asdict(self.scaffold),
            "chosen": self.chosen,
            "rejected": self.rejected,
            "verdict": asdict(self.verdict),
        }


def _require_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PreferencePairValidationError(f"{path} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    path: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(str(key) for key in actual - expected)
        raise PreferencePairValidationError(
            f"{path} keys mismatch: missing={missing}, extra={extra}"
        )


def _require_non_empty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreferencePairValidationError(f"{path} must be a non-empty string")
    return value


def validate_preference_pair(record: Mapping[str, object]) -> PreferencePair:
    """Parse and validate one JSON-decoded preference-pair v1 record."""

    top_level = _require_mapping(record, "record")
    _require_exact_keys(top_level, TOP_LEVEL_FIELDS, "record")

    version = top_level["schema_version"]
    if type(version) is not int or version != SCHEMA_VERSION:
        raise PreferencePairValidationError(
            f"schema_version must be exactly {SCHEMA_VERSION}"
        )

    scaffold_record = _require_mapping(top_level["scaffold"], "scaffold")
    _require_exact_keys(
        scaffold_record,
        frozenset(SCAFFOLD_FIELDS),
        "scaffold",
    )
    scaffold_values = {
        field: _require_non_empty_string(
            scaffold_record[field],
            f"scaffold.{field}",
        )
        for field in SCAFFOLD_FIELDS
    }
    scaffold = Scaffold(**scaffold_values)

    chosen = _require_non_empty_string(top_level["chosen"], "chosen")
    rejected = _require_non_empty_string(top_level["rejected"], "rejected")
    if chosen == rejected:
        raise PreferencePairValidationError("chosen and rejected must differ")

    verdict_record = _require_mapping(top_level["verdict"], "verdict")
    _require_exact_keys(verdict_record, VERDICT_FIELDS, "verdict")
    judge_id = _require_non_empty_string(
        verdict_record["judge_id"],
        "verdict.judge_id",
    )
    first_pass = _require_non_empty_string(
        verdict_record["first_pass"],
        "verdict.first_pass",
    )
    swapped_pass = _require_non_empty_string(
        verdict_record["swapped_pass"],
        "verdict.swapped_pass",
    )
    if first_pass not in ALLOWED_VERDICTS:
        raise PreferencePairValidationError(
            "verdict.first_pass must be 'A' or 'B'"
        )
    if swapped_pass not in ALLOWED_VERDICTS:
        raise PreferencePairValidationError(
            "verdict.swapped_pass must be 'A' or 'B'"
        )
    if verdict_record["consistent"] is not True:
        raise PreferencePairValidationError(
            "verdict.consistent must be true for a retained pair"
        )
    if first_pass == swapped_pass:
        raise PreferencePairValidationError(
            "first_pass and swapped_pass must be opposite for a retained pair"
        )

    return PreferencePair(
        scaffold=scaffold,
        chosen=chosen,
        rejected=rejected,
        verdict=VerdictMetadata(
            judge_id=judge_id,
            first_pass=first_pass,
            swapped_pass=swapped_pass,
            consistent=True,
        ),
    )
