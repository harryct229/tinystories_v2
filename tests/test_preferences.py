import copy
import json

import pytest

from tinystories_v2.preferences import (
    PreferencePair,
    PreferencePairValidationError,
    VerdictMetadata,
    validate_preference_pair,
)
from tinystories_v2.slots import Scaffold


def make_pair() -> PreferencePair:
    return PreferencePair(
        scaffold=Scaffold(
            character="fox",
            trait="greedy",
            setting="a dense forest",
            conflict="loses food to a trick",
            resolution="the trickster is exposed",
            moral="honesty is the best policy",
        ),
        chosen=(
            "A greedy fox admitted the trick, returned the food, and learned "
            "that honesty is the best policy."
        ),
        rejected="A fox walked through a forest and went home.",
        verdict=VerdictMetadata(
            judge_id="fake:slot-coverage-v1",
            first_pass="A",
            swapped_pass="B",
            consistent=True,
        ),
    )


def test_schema_round_trips_through_json():
    pair = make_pair()
    payload = json.loads(json.dumps(pair.to_dict()))
    assert validate_preference_pair(payload) == pair
    assert set(payload) == {
        "schema_version",
        "scaffold",
        "chosen",
        "rejected",
        "verdict",
    }


def test_schema_rejects_missing_judge_identity():
    payload = copy.deepcopy(make_pair().to_dict())
    payload["verdict"]["judge_id"] = ""
    with pytest.raises(PreferencePairValidationError, match="judge_id"):
        validate_preference_pair(payload)


def test_schema_rejects_same_winner_in_both_presentations():
    payload = copy.deepcopy(make_pair().to_dict())
    payload["verdict"]["swapped_pass"] = "A"
    with pytest.raises(PreferencePairValidationError, match="opposite"):
        validate_preference_pair(payload)


def test_schema_rejects_identical_completions():
    payload = copy.deepcopy(make_pair().to_dict())
    payload["rejected"] = payload["chosen"]
    with pytest.raises(PreferencePairValidationError, match="must differ"):
        validate_preference_pair(payload)
