import pytest

from tinystories_v2.slots import Scaffold, SlotExtractionError, extract_slots

# Mirrors the template observed in real TF1-EN-3M records (paper section 3,
# corrected against reality: trait is folded into the Main Character line and
# the setting carries a "where our story unfolds" suffix).
SYNTHETIC_PROMPT = (
    "Create a fable based on the following elements. Weave them naturally into a story: \n"
    "- Main Character: a greedy fox \n"
    "- Setting: a dense forest where our story unfolds \n"
    "- Challenge: loses their food to someone's trick \n"
    "- Outcome: the trickster is exposed \n"
    "- Teaching: honesty is the best policy \n"
    "The fable should: \n"
    "- Be appropriate for age group B (4-7 years)"
)


def test_extract_from_verbose_template():
    assert extract_slots(SYNTHETIC_PROMPT) == Scaffold(
        character="fox",
        trait="greedy",
        setting="a dense forest",
        conflict="loses their food to someone's trick",
        resolution="the trickster is exposed",
        moral="honesty is the best policy",
    )


def test_extract_known_real_record(fixture_records):
    # Expected values read from the live dataset (train row 0, previewed
    # 2026-07-11). If this prompt_hash is missing from the fixture (dataset
    # re-upload), open the fixture's first record, read its prompt, and
    # replace hash + expected values with that record's actual content.
    by_hash = {r["prompt_hash"]: r for r in fixture_records}
    record = by_hash["71df0b5fc187f6e393954bc32cccac0cf9f856e31df8276ea6557c9b1710294e"]
    assert extract_slots(record["prompt"]) == Scaffold(
        character="firefly",
        trait="persuasive",
        setting="a canyon",
        conflict="betrayal by a friend",
        resolution="a lesson is documented for future generations",
        moral="timely help earns lasting loyalty",
    )


def test_extract_every_fixture_record(fixture_records):
    for record in fixture_records:
        scaffold = extract_slots(record["prompt"])
        for field in ("character", "trait", "setting", "conflict", "resolution", "moral"):
            assert getattr(scaffold, field), f"{field} empty for {record['prompt_hash']}"
        assert "where our story unfolds" not in scaffold.setting
        assert "fable should" not in scaffold.moral


def test_non_template_prompt_raises():
    with pytest.raises(SlotExtractionError):
        extract_slots("Write me a story about a dog.")
