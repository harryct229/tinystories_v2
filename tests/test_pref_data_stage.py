"""Preference-labeling stage behavior on CPU fixture artifacts with fake
Judges: a schema-valid growing artifact, config-selected Judge, determinism,
and resume semantics."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tinystories_v2.config import load_config
from tinystories_v2.data import run as data_run
from tinystories_v2.pref_data import run as pref_run
from tinystories_v2.preferences import validate_preference_pair
from tinystories_v2.tokenizer import run as tokenizer_run

MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
         "n_heads": 2, "context": 256, "ffn_hidden": 192}


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, fixture_path):
    """Run the data-prep and tokenizer stages once so labeling has real
    upstream artifacts (stages couple only through artifacts)."""
    base = tmp_path_factory.mktemp("pref_inputs")
    data_dir, tok_dir = base / "data", base / "tok"
    data_run({
        "out_dir": str(data_dir), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        # pref weighted high so the split has plenty of Scaffolds from ~120 Fables.
        "splits": {"seed": "fixture-v1", "pretrain": 0.3, "sft": 0.2,
                   "pref": 0.4, "eval": 0.1},
    })
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    return {"pref_split": str(data_dir / "splits" / "pref.jsonl"),
            "tokenizer": str(tok_dir / "tokenizer.json"),
            "data_root": str(data_dir), "tok_root": str(tok_dir)}


@pytest.fixture
def init_dir(tmp_path, prepared, make_init_checkpoint):
    return make_init_checkpoint(tmp_path / "init", MODEL,
                                prepared["tokenizer"])


def make_config(out_dir, prepared, init_dir, *,
                judge_kind="fake_slot_coverage", max_scaffolds=6,
                temperature=1.0) -> dict:
    return {
        "out_dir": str(out_dir),
        "max_scaffolds": max_scaffolds,
        "data": {"pref_split": prepared["pref_split"]},
        "checkpoint": {"local_dir": str(init_dir)},
        "sampling": {"num_completions": 4, "pairs_per_scaffold": 3,
                     "temperature": temperature, "top_p": 0.95,
                     "max_new_tokens": 32, "seed": 1337},
        "judge": {"kind": judge_kind},
    }


def to_toml(config: dict) -> str:
    lines = [f'out_dir = "{config["out_dir"]}"',
             f"max_scaffolds = {config['max_scaffolds']}"]
    for section in ("data", "checkpoint", "sampling", "judge"):
        lines.append(f"[{section}]")
        for key, value in config[section].items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"


def read_pairs(out_dir) -> list[dict]:
    text = (Path(out_dir) / "pairs.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_manifest(out_dir) -> dict:
    return json.loads(
        (Path(out_dir) / "manifest.json").read_text(encoding="utf-8"))


def test_artifact_is_schema_valid_with_rates_in_manifest(tmp_path, prepared,
                                                         init_dir):
    out = tmp_path / "out"
    result = pref_run(make_config(out, prepared, init_dir))
    records = read_pairs(out)
    assert records, "the consistent fake Judge must retain pairs"
    for record in records:
        validate_preference_pair(record)   # raises on any schema violation
    manifest = read_manifest(out)
    assert manifest["stage"] == "pref_data"
    assert manifest["schema_version"] == 1
    assert manifest["judge_id"] == "fake:slot-coverage-v1"
    assert manifest["scaffolds_done"] == result["scaffolds"] == 6
    counters = manifest["counters"]
    assert counters["kept"] == len(records) == result["pairs"]
    kept = counters.get("kept", 0)
    discarded = counters.get("discarded_inconsistent", 0)
    assert manifest["discard_rate"] == pytest.approx(
        discarded / max(kept + discarded, 1))


def test_two_fresh_runs_are_byte_identical(tmp_path, prepared, init_dir):
    for name in ("run1", "run2"):
        pref_run(make_config(tmp_path / name, prepared, init_dir))
    assert (tmp_path / "run1" / "pairs.jsonl").read_bytes() == \
        (tmp_path / "run2" / "pairs.jsonl").read_bytes()


def test_judge_is_config_selected_and_biased_judge_discards_all(
        tmp_path, prepared, init_dir):
    out = tmp_path / "out"
    pref_run(make_config(out, prepared, init_dir,
                         judge_kind="fake_position_biased"))
    assert read_pairs(out) == []
    manifest = read_manifest(out)
    assert manifest["judge_id"] == "fake:position-a-v1"
    assert manifest["counters"].get("kept", 0) == 0
    assert manifest["counters"]["discarded_inconsistent"] > 0
    assert manifest["discard_rate"] == 1.0


def test_greedy_sampling_yields_only_degenerate_pairs(tmp_path, prepared,
                                                      init_dir):
    # temperature 0.0 makes all N completions identical (argmax), so every
    # pair is degenerate: nothing reaches the Judge, nothing is kept.
    out = tmp_path / "out"
    pref_run(make_config(out, prepared, init_dir, temperature=0.0))
    manifest = read_manifest(out)
    assert manifest["counters"].get("kept", 0) == 0
    assert manifest["counters"].get("discarded_inconsistent", 0) == 0
    assert manifest["counters"]["skipped_degenerate"] > 0


def test_max_scaffolds_caps_progress(tmp_path, prepared, init_dir):
    out = tmp_path / "out"
    result = pref_run(make_config(out, prepared, init_dir, max_scaffolds=2))
    assert result["scaffolds"] == 2
    assert read_manifest(out)["scaffolds_done"] == 2


def test_resume_after_cap_matches_uninterrupted_run(tmp_path, prepared,
                                                    init_dir):
    reference = make_config(tmp_path / "ref", prepared, init_dir)
    pref_run(reference)

    partial = make_config(tmp_path / "partial", prepared, init_dir,
                          max_scaffolds=2)
    pref_run(partial)
    partial["max_scaffolds"] = 6
    pref_run(partial, resume=True)

    assert (tmp_path / "partial" / "pairs.jsonl").read_bytes() == \
        (tmp_path / "ref" / "pairs.jsonl").read_bytes()
    assert read_manifest(tmp_path / "partial")["counters"] == \
        read_manifest(tmp_path / "ref")["counters"]


def test_resume_discards_uncommitted_trailing_garbage(tmp_path, prepared,
                                                      init_dir):
    reference = make_config(tmp_path / "ref", prepared, init_dir)
    pref_run(reference)

    partial = make_config(tmp_path / "partial", prepared, init_dir,
                          max_scaffolds=2)
    pref_run(partial)
    with (tmp_path / "partial" / "pairs.jsonl").open(
            "a", encoding="utf-8") as f:
        f.write('{"crashed mid-')
    partial["max_scaffolds"] = 6
    pref_run(partial, resume=True)

    assert (tmp_path / "partial" / "pairs.jsonl").read_bytes() == \
        (tmp_path / "ref" / "pairs.jsonl").read_bytes()


def test_fresh_run_refuses_an_existing_labeling_dir(tmp_path, prepared,
                                                    init_dir):
    config = make_config(tmp_path / "out", prepared, init_dir,
                         max_scaffolds=2)
    pref_run(config)
    with pytest.raises(ValueError, match="resume"):
        pref_run(config)


def test_empty_pref_split_is_an_error(tmp_path, prepared, init_dir):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    config = make_config(tmp_path / "out", prepared, init_dir)
    config["data"] = {"pref_split": str(empty)}
    with pytest.raises(ValueError, match="no Scaffolds"):
        pref_run(config)


def test_cli_entrypoint_runs_standalone(tmp_path, prepared, init_dir):
    out = tmp_path / "cli_out"
    config_file = tmp_path / "cfg.toml"
    config_file.write_text(
        to_toml(make_config(out, prepared, init_dir, max_scaffolds=2)),
        encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.pref_data",
         "--config", str(config_file)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (out / "pairs.jsonl").exists()
    assert (out / "manifest.json").exists()


CONFIG_DIR = Path(__file__).parents[1] / "configs"


def test_hub_target_mirrors_artifact_and_fresh_vm_resume_completes(
        tmp_path, prepared, init_dir):
    reference = make_config(tmp_path / "ref", prepared, init_dir)
    pref_run(reference)

    mirror = tmp_path / "mirror"
    out = tmp_path / "out"
    config = make_config(out, prepared, init_dir, max_scaffolds=2)
    config["hub"] = {"target": str(mirror)}
    pref_run(config)
    for name in ("pairs.jsonl", "progress.json", "manifest.json"):
        assert (mirror / name).exists(), name

    shutil.rmtree(out)  # a fresh Colab VM has no local artifact
    config["max_scaffolds"] = 6
    pref_run(config, resume=True)  # pulls prior session from the mirror
    assert (out / "pairs.jsonl").read_bytes() == \
        (tmp_path / "ref" / "pairs.jsonl").read_bytes()
    assert (mirror / "pairs.jsonl").read_bytes() == \
        (tmp_path / "ref" / "pairs.jsonl").read_bytes()


def test_missing_split_and_tokenizer_fetched_from_hub_sources(
        tmp_path, prepared, init_dir):
    reference = make_config(tmp_path / "ref", prepared, init_dir)
    pref_run(reference)

    config = make_config(tmp_path / "out", prepared, init_dir)
    config["data"] = {
        "pref_split": str(tmp_path / "fetched" / "splits" / "pref.jsonl"),
        "hub_source": prepared["data_root"],
        "tokenizer": str(tmp_path / "fetched" / "tokenizer.json"),
        "tokenizer_hub_source": prepared["tok_root"],
    }
    pref_run(config)
    assert (tmp_path / "out" / "pairs.jsonl").read_bytes() == \
        (tmp_path / "ref" / "pairs.jsonl").read_bytes()


def test_missing_checkpoint_fetched_from_hub_source(tmp_path, prepared,
                                                    init_dir):
    config = make_config(tmp_path / "out", prepared, init_dir,
                         max_scaffolds=2)
    config["checkpoint"] = {"local_dir": str(tmp_path / "ckpt_local"),
                            "hub_source": str(init_dir)}
    result = pref_run(config)
    assert result["scaffolds"] == 2


def test_full_config_pins_design_doc_defaults():
    config = load_config(CONFIG_DIR / "pref_data_full.toml")
    assert config["sampling"]["num_completions"] == 4
    assert config["sampling"]["temperature"] == 1.0
    assert config["sampling"]["top_p"] == 0.95
    assert config["sampling"]["pairs_per_scaffold"] == 3
    assert config["judge"]["kind"] == "transformers"
    assert config["judge"]["model_id"] == "Qwen/Qwen3-8B"
    assert config["judge"]["precision"] == "fp16"
    assert config["hub"]["target"].startswith("hf://")
    assert config["checkpoint"]["hub_source"].startswith("hf://")
    assert config["data"]["hub_source"].startswith("hf://")
    assert config["data"]["tokenizer_hub_source"].startswith("hf://")


def test_fixture_config_selects_the_fake_judge():
    config = load_config(CONFIG_DIR / "pref_data_fixture.toml")
    assert config["judge"]["kind"] == "fake_slot_coverage"
    assert config["sampling"]["num_completions"] == 4
