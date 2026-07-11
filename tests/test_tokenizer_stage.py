import json
import subprocess
import sys

import pytest
from tokenizers import Tokenizer

from tinystories_v2.slots import SLOT_SPECIAL_TOKENS
from tinystories_v2.tokenizer import run

VOCAB_SIZE = 512  # 8192 needs the real corpus; the artifact contract is identical


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory, fixture_path):
    out = tmp_path_factory.mktemp("tokenizer")
    run({
        "out_dir": str(out),
        "corpus": [str(fixture_path)],
        "text_field": "fable",
        "vocab_size": VOCAB_SIZE,
    })
    return out


@pytest.fixture(scope="module")
def tokenizer(artifact_dir):
    return Tokenizer.from_file(str(artifact_dir / "tokenizer.json"))


def test_artifact_contract(artifact_dir):
    assert (artifact_dir / "tokenizer.json").exists()
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "tokenizer"
    assert manifest["vocab_size"] == VOCAB_SIZE
    assert manifest["special_tokens"] == list(SLOT_SPECIAL_TOKENS)


def test_vocab_size_matches_config(tokenizer):
    assert tokenizer.get_vocab_size() == VOCAB_SIZE


def test_slot_special_tokens_encode_to_single_ids(tokenizer):
    id_lists = [tokenizer.encode(token).ids for token in SLOT_SPECIAL_TOKENS]
    assert all(len(ids) == 1 for ids in id_lists), id_lists
    assert len({ids[0] for ids in id_lists}) == len(SLOT_SPECIAL_TOKENS)


def test_roundtrips_fixture_fables_losslessly(tokenizer, fixture_records):
    for record in fixture_records:
        text = record["fable"]
        assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_roundtrips_slot_prompt_text(tokenizer):
    text = (
        "<|character|>fox<|trait|>greedy<|setting|>a dense forest"
        "<|conflict|>loses their food<|resolution|>the trickster is exposed"
        "<|moral|>honesty is the best policy<|fable|>One day, a fox...<|end|>"
    )
    decoded = tokenizer.decode(tokenizer.encode(text).ids, skip_special_tokens=False)
    assert decoded == text


def test_vocab_size_too_small_raises(tmp_path, fixture_path):
    with pytest.raises(ValueError, match="vocab_size"):
        run({
            "out_dir": str(tmp_path / "too_small"),
            "corpus": [str(fixture_path)],
            "text_field": "fable",
            "vocab_size": 100,
        })


def test_cli_entrypoint_runs_standalone(tmp_path, fixture_path):
    out = tmp_path / "cli_out"
    config_file = tmp_path / "cfg.toml"
    config_file.write_text(
        f'out_dir = "{out}"\ncorpus = ["{fixture_path}"]\n'
        f'text_field = "fable"\nvocab_size = {VOCAB_SIZE}\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.tokenizer", "--config", str(config_file)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "tokenizer.json").exists()
