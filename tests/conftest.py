import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "tf1_sample.jsonl"


@pytest.fixture(scope="session")
def fixture_records(fixture_path) -> list[dict]:
    with fixture_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
