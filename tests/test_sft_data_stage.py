import json
import subprocess
import sys

import pytest
from tokenizers import Tokenizer

from tinystories_v2.data import run as data_run
from tinystories_v2.sft_data import run as sft_run
from tinystories_v2.slot_prompt import parse_example
from tinystories_v2.tokenizer import run as tokenizer_run


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, fixture_path):
    """Run the data-prep and tokenizer stages so the SFT builder has real
    upstream artifacts to read (stages couple only through artifacts)."""
    base = tmp_path_factory.mktemp("sft_inputs")
    data_dir = base / "data"
    tok_dir = base / "tok"
    data_run({
        "out_dir": str(data_dir),
        "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        # sft weighted high so the split has plenty of examples from ~120 Fables.
        "splits": {"seed": "fixture-v1", "pretrain": 0.4, "sft": 0.4, "pref": 0.1, "eval": 0.1},
    })
    tokenizer_run({
        "out_dir": str(tok_dir),
        "corpus": [str(fixture_path)],
        "text_field": "fable",
        "vocab_size": 512,
    })
    return {
        "sft_split": str(data_dir / "splits" / "sft.jsonl"),
        "tokenizer": str(tok_dir / "tokenizer.json"),
    }


def make_config(out_dir, prepared) -> dict:
    return {
        "out_dir": str(out_dir),
        "tokenizer": prepared["tokenizer"],
        "sft_split": prepared["sft_split"],
        "max_examples": 0,
    }


def sft_hashes(prepared) -> set:
    with open(prepared["sft_split"], encoding="utf-8") as f:
        return {json.loads(line)["prompt_hash"] for line in f if line.strip()}


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory, prepared):
    out = tmp_path_factory.mktemp("sft_data")
    sft_run(make_config(out, prepared))
    return out


def read_examples(artifact_dir) -> list[dict]:
    lines = (artifact_dir / "examples.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_artifact_contract(artifact_dir, prepared):
    assert (artifact_dir / "examples.jsonl").exists()
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "sft_data"
    assert manifest["schema_version"] == 1
    assert manifest["count"] == len(sft_hashes(prepared))


def test_example_records_have_schema_and_aligned_mask(artifact_dir):
    records = read_examples(artifact_dir)
    assert records
    for rec in records:
        assert set(rec) == {"prompt_hash", "input_ids", "loss_mask", "n_prompt_tokens"}
        assert len(rec["input_ids"]) == len(rec["loss_mask"])
        n = rec["n_prompt_tokens"]
        assert rec["loss_mask"][:n] == [0] * n
        assert set(rec["loss_mask"][n:]) <= {1}
        assert rec["loss_mask"][n:]  # at least one active (fable body) token


def test_examples_parse_back_to_a_scaffold(artifact_dir, prepared):
    tokenizer = Tokenizer.from_file(prepared["tokenizer"])
    parsed = parse_example(tokenizer, read_examples(artifact_dir)[0]["input_ids"])
    assert parsed.fable.strip()


def test_reads_only_the_configured_sft_split(artifact_dir, prepared):
    emitted = {rec["prompt_hash"] for rec in read_examples(artifact_dir)}
    assert emitted == sft_hashes(prepared)


def test_two_runs_are_byte_identical(tmp_path, prepared):
    for name in ("run1", "run2"):
        sft_run(make_config(tmp_path / name, prepared))
    assert (tmp_path / "run1" / "examples.jsonl").read_bytes() == (
        tmp_path / "run2" / "examples.jsonl"
    ).read_bytes()


def test_max_examples_caps_output(tmp_path, prepared):
    out = tmp_path / "capped"
    config = make_config(out, prepared)
    config["max_examples"] = 2
    sft_run(config)
    assert len(read_examples(out)) == 2


def test_cli_entrypoint_runs_standalone(tmp_path, prepared):
    out = tmp_path / "cli_out"
    config_file = tmp_path / "cfg.toml"
    config_file.write_text(
        f'out_dir = "{out}"\ntokenizer = "{prepared["tokenizer"]}"\n'
        f'sft_split = "{prepared["sft_split"]}"\nmax_examples = 0\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.sft_data", "--config", str(config_file)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "examples.jsonl").exists()
