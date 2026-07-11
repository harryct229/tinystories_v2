import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import pytest

from tinystories_v2.data import SPLIT_NAMES, run
from tinystories_v2.slots import SlotExtractionError


def make_config(out_dir: Path, source_path: Path) -> dict:
    return {
        "out_dir": str(out_dir),
        "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(source_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.5, "sft": 0.2, "pref": 0.1, "eval": 0.2},
    }


def read_membership(out_dir: Path) -> dict:
    return json.loads((out_dir / "membership.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory, fixture_path):
    out = tmp_path_factory.mktemp("data_prep")
    run(make_config(out, fixture_path))
    return out


def test_produces_all_four_split_artifacts(artifact_dir):
    for name in SPLIT_NAMES:
        assert (artifact_dir / "splits" / f"{name}.jsonl").exists()
    assert (artifact_dir / "membership.json").exists()
    counts = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))["counts"]
    for name in SPLIT_NAMES:
        assert counts[name] > 0, f"{name} split is empty"


def test_splits_disjoint_by_fable(artifact_dir):
    membership = read_membership(artifact_dir)
    for a, b in combinations(SPLIT_NAMES, 2):
        assert not set(membership[a]) & set(membership[b]), f"{a} and {b} overlap"


def test_membership_matches_split_files(artifact_dir):
    membership = read_membership(artifact_dir)
    for name in SPLIT_NAMES:
        lines = (artifact_dir / "splits" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["prompt_hash"] for line in lines] == membership[name]


def test_split_records_carry_scaffold_and_fable(artifact_dir):
    first = (artifact_dir / "splits" / "pretrain.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert set(json.loads(first)) == {
        "prompt_hash", "character", "trait", "setting",
        "conflict", "resolution", "moral", "fable",
    }


def test_two_runs_are_byte_identical(tmp_path, fixture_path):
    for name in ("run1", "run2"):
        run(make_config(tmp_path / name, fixture_path))
    for rel in ["membership.json"] + [f"splits/{n}.jsonl" for n in SPLIT_NAMES]:
        assert (tmp_path / "run1" / rel).read_bytes() == (tmp_path / "run2" / rel).read_bytes(), rel


def test_extraction_failure_budget(tmp_path, fixture_records):
    corrupt = tmp_path / "corrupt.jsonl"
    rows = [
        dict(fixture_records[0]),
        {"prompt_hash": "0" * 64, "prompt": "not the template", "fable": "x"},
    ]
    corrupt.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    all_pretrain = {"seed": "fixture-v1", "pretrain": 1.0, "sft": 0.0, "pref": 0.0, "eval": 0.0}

    strict = make_config(tmp_path / "strict", corrupt)
    strict["splits"] = dict(all_pretrain)
    with pytest.raises(SlotExtractionError):
        run(strict)
    assert not list((tmp_path / "strict" / "splits").glob("*.jsonl")), (
        "aborted run must not leave partial split files behind"
    )

    lenient = make_config(tmp_path / "lenient", corrupt)
    lenient["splits"] = dict(all_pretrain)
    lenient["max_extraction_failures"] = 1
    run(lenient)
    manifest = json.loads((tmp_path / "lenient" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["skipped_extraction_failures"] == 1
    assert manifest["counts"]["pretrain"] == 1


def test_cli_entrypoint_runs_standalone(tmp_path, fixture_path):
    out = tmp_path / "cli_out"
    config_file = tmp_path / "cfg.toml"
    config_file.write_text(
        f'out_dir = "{out}"\nmax_extraction_failures = 0\n\n'
        f'[source]\nkind = "jsonl"\npath = "{fixture_path}"\n\n'
        '[splits]\nseed = "fixture-v1"\npretrain = 0.5\nsft = 0.2\npref = 0.1\neval = 0.2\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.data", "--config", str(config_file)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "membership.json").exists()
