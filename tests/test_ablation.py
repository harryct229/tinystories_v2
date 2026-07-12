import dataclasses
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from tinystories_v2.ablation import run, validate_comparability
from tinystories_v2.checkpoint import save_checkpoint
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.pack import load_packed
from tinystories_v2.tokenizer import run as run_tokenizer


@pytest.fixture()
def toy_artifacts(tmp_path, fixture_path):
    tokenizer_dir = tmp_path / "tokenizer"
    run_tokenizer(
        {
            "out_dir": str(tokenizer_dir),
            "corpus": [str(fixture_path)],
            "text_field": "fable",
            "vocab_size": 512,
        }
    )
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    base = ModelConfig(
        vocab_size=512,
        d_model=64,
        n_layers=2,
        n_heads=2,
        context=64,
        ffn_hidden=192,
    )
    configs = {
        "rope_swiglu": base,
        "learned_swiglu": dataclasses.replace(
            base, position_encoding="learned"
        ),
        "rope_gelu": dataclasses.replace(
            base, mlp_type="gelu", ffn_hidden=288
        ),
    }
    variants = []
    for index, (name, model_config) in enumerate(configs.items()):
        torch.manual_seed(index)
        model = FableLM(model_config)
        checkpoint_dir = tmp_path / name / "checkpoints"
        save_checkpoint(
            checkpoint_dir,
            2,
            {
                "step": 2,
                "tokens_seen": 256,
                "model": model.state_dict(),
                "optimizer": {},
                "scaler": {},
                "config": {
                    "model": dataclasses.asdict(model_config),
                    "data": {"tokenizer_path": str(tokenizer_path)},
                },
            },
        )
        variants.append(
            {"name": name, "checkpoint": str(checkpoint_dir)}
        )
    return {
        "tokenizer_path": tokenizer_path,
        "variants": variants,
    }


def report_config(tmp_path, fixture_path, toy_artifacts):
    return {
        "out_dir": str(tmp_path / "report"),
        "baseline": "rope_swiglu",
        "param_tolerance": 0.03,
        "expected_tokens": 256,
        "eval": {
            "split_path": str(fixture_path),
            "tokenizer_path": str(toy_artifacts["tokenizer_path"]),
            "packed_path": str(tmp_path / "eval" / "eval.bin"),
            "max_tokens": 128,
            "batch_size": 2,
            "device": "cpu",
            "sample_count": 1,
            "max_new_tokens": 2,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 7,
        },
        "variants": toy_artifacts["variants"],
    }


def write_report_config(path: Path, config: dict) -> None:
    lines = [
        f'out_dir = "{config["out_dir"]}"',
        f'baseline = "{config["baseline"]}"',
        f'param_tolerance = {config["param_tolerance"]}',
        f'expected_tokens = {config["expected_tokens"]}',
        "",
        "[eval]",
    ]
    for key, value in config["eval"].items():
        if isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        else:
            lines.append(f"{key} = {str(value).lower()}")
    for variant in config["variants"]:
        lines.extend(
            [
                "",
                "[[variants]]",
                f'name = "{variant["name"]}"',
                f'checkpoint = "{variant["checkpoint"]}"',
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_validate_comparability_accepts_matched_rows():
    rows = [
        {"variant": "base", "params": 100, "tokens_seen": 1_000},
        {"variant": "position", "params": 102, "tokens_seen": 1_000},
        {"variant": "mlp", "params": 100, "tokens_seen": 1_000},
    ]
    assert validate_comparability(
        rows, baseline="base", param_tolerance=0.03
    ) == 1_000


def test_validate_comparability_rejects_mismatched_training_tokens():
    rows = [
        {"variant": "base", "params": 100, "tokens_seen": 1_000},
        {"variant": "position", "params": 102, "tokens_seen": 900},
    ]
    with pytest.raises(ValueError, match="matched training tokens"):
        validate_comparability(
            rows, baseline="base", param_tolerance=0.03
        )


def test_validate_comparability_rejects_parameter_drift():
    rows = [
        {"variant": "base", "params": 100, "tokens_seen": 1_000},
        {"variant": "oversized", "params": 104, "tokens_seen": 1_000},
    ]
    with pytest.raises(ValueError, match="parameter tolerance"):
        validate_comparability(
            rows, baseline="base", param_tolerance=0.03
        )


def test_validate_comparability_rejects_duplicate_variant_names():
    rows = [
        {"variant": "base", "params": 100, "tokens_seen": 1_000},
        {"variant": "base", "params": 100, "tokens_seen": 1_000},
    ]
    with pytest.raises(ValueError, match="variant names must be unique"):
        validate_comparability(
            rows, baseline="base", param_tolerance=0.03
        )


def test_validate_comparability_rejects_missing_baseline():
    rows = [
        {"variant": "position", "params": 100, "tokens_seen": 1_000},
    ]
    with pytest.raises(ValueError, match="baseline variant 'base' is missing"):
        validate_comparability(
            rows, baseline="base", param_tolerance=0.03
        )


def test_run_rejects_equally_incomplete_checkpoints(
    tmp_path, fixture_path, toy_artifacts
):
    config = report_config(tmp_path, fixture_path, toy_artifacts)
    config["expected_tokens"] = 512
    with pytest.raises(ValueError, match="completed token budget"):
        run(config)


def test_run_rebuilds_stale_eval_cache_from_configured_inputs(
    tmp_path, fixture_path, toy_artifacts
):
    config = report_config(tmp_path, fixture_path, toy_artifacts)
    packed_path = Path(config["eval"]["packed_path"])
    packed_path.parent.mkdir(parents=True)
    packed_path.write_bytes(b"\x00\x00\x00\x00")
    Path(str(packed_path) + ".json").write_text(
        '{"source": "stale"}\n', encoding="utf-8"
    )

    report = run(config)

    tokenizer = Tokenizer.from_file(config["eval"]["tokenizer_path"])
    first_record = json.loads(
        fixture_path.read_text(encoding="utf-8").splitlines()[0]
    )
    expected_prefix = tokenizer.encode(first_record["fable"]).ids[:4]
    rebuilt = load_packed(packed_path)
    assert report["eval_tokens"] == 128
    assert len(rebuilt) > 2
    assert rebuilt[: len(expected_prefix)].tolist() == expected_prefix


def test_run_rejects_undersized_rebuilt_eval_split(
    tmp_path, fixture_path, toy_artifacts
):
    record = json.loads(
        fixture_path.read_text(encoding="utf-8").splitlines()[0]
    )
    record["fable"] = "Tiny."
    split_path = tmp_path / "undersized.jsonl"
    split_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    config = report_config(tmp_path, fixture_path, toy_artifacts)
    config["eval"]["split_path"] = str(split_path)

    with pytest.raises(ValueError, match="configured max_tokens"):
        run(config)


def test_run_writes_comparison_table_and_fixed_scaffold_samples(
    tmp_path, fixture_path, toy_artifacts
):
    config = report_config(tmp_path, fixture_path, toy_artifacts)

    report = run(config)

    assert report["baseline"] == "rope_swiglu"
    assert report["expected_tokens"] == 256
    assert report["matched_tokens"] == 256
    assert report["eval_tokens"] == 128
    assert [row["variant"] for row in report["rows"]] == [
        "rope_swiglu",
        "learned_swiglu",
        "rope_gelu",
    ]
    assert {row["variant"]: row["params"] for row in report["rows"]} == {
        "rope_swiglu": 139_584,
        "learned_swiglu": 143_680,
        "rope_gelu": 139_584,
    }
    for row in report["rows"]:
        assert row["tokens_seen"] == 256
        assert math.isfinite(row["val_loss"])
        assert math.isfinite(row["perplexity"])
        assert row["val_loss"] == pytest.approx(math.log(row["perplexity"]))

    assert len(report["samples"]) == 1
    assert report["samples"][0]["seed"] == (
        "In a canyon, a persuasive firefly"
    )
    generations = report["samples"][0]["generations"]
    assert set(generations) == {
        "rope_swiglu",
        "learned_swiglu",
        "rope_gelu",
    }
    assert all(isinstance(text, str) for text in generations.values())

    out_dir = Path(config["out_dir"])
    assert json.loads(
        (out_dir / "report.json").read_text(encoding="utf-8")
    ) == report
    markdown = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "| Variant | Position | MLP | FFN hidden | Params |" in markdown
    assert "Matched training tokens: **256**" in markdown
    assert "## Scaffold 1" in markdown
    assert all(name in markdown for name in generations)


def test_module_cli_writes_report(tmp_path, fixture_path, toy_artifacts):
    config = report_config(tmp_path, fixture_path, toy_artifacts)
    config["out_dir"] = str(tmp_path / "cli-report")
    config_path = tmp_path / "report.toml"
    write_report_config(config_path, config)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tinystories_v2.ablation",
            "--config",
            str(config_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (Path(config["out_dir"]) / "report.json").exists()
    assert (Path(config["out_dir"]) / "report.md").exists()
