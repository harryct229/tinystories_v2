"""End-to-end eval stage on CPU with the fake Judge and toy checkpoints:
produces the results artifact (win-rate tables with counts, metric tables,
sample sheet), records the eval-Judge identity, and adds the RLAIF column only
when a third stage is configured."""

import json
import subprocess
import sys
from pathlib import Path

from tinystories_v2 import eval as eval_stage
from tinystories_v2.data import run as data_run
from tinystories_v2.tokenizer import run as tokenizer_run

TOY_MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 128, "ffn_hidden": 192}


def _prepare_inputs(base, fixture_path, make_init_checkpoint, stage_names):
    """A tokenizer, an eval split, and one toy checkpoint per stage name."""
    data_dir, tok_dir = base / "data", base / "tok"
    data_run({
        "out_dir": str(data_dir), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.1, "sft": 0.6,
                   "pref": 0.1, "eval": 0.2},
    })
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer_path = tok_dir / "tokenizer.json"
    stages = []
    for name in stage_names:
        stage_dir = base / name
        make_init_checkpoint(stage_dir, TOY_MODEL, tokenizer_path)
        stages.append({"name": name, "local_dir": str(stage_dir)})
    return {
        "out_dir": str(base / "out"),
        "max_eval_scaffolds": 16,
        "sample_sheet_k": 3,
        "data": {"eval_split": str(data_dir / "splits" / "eval.jsonl"),
                 "tokenizer": str(tokenizer_path)},
        "stages": stages,
        "sampling": {"max_new_tokens": 24, "temperature": 0.8, "top_p": 0.95,
                     "seed": 1337},
        "metrics": {"self_bleu_sample_size": 0, "self_bleu_seed": 0},
        "judge": {"kind": "fake_slot_coverage"},
    }


def test_eval_stage_produces_the_results_artifact(tmp_path, fixture_path,
                                                  make_init_checkpoint):
    config = _prepare_inputs(tmp_path, fixture_path, make_init_checkpoint,
                             ["base", "sft"])
    results = eval_stage.run(config)

    out = Path(config["out_dir"])
    assert (out / "results.json").exists() and (out / "report.md").exists()
    saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert saved["stage"] == "eval"
    # Criterion: the eval-Judge identity is recorded in the artifact.
    assert saved["eval_judge_id"] == "fake:slot-coverage-v1"
    assert results["eval_judge_id"] == saved["eval_judge_id"]
    # Win-rate table for the single base-vs-sft pair, with counts summing to n.
    assert saved["stages"] == ["base", "sft"]
    assert len(saved["win_rates"]) == 1
    w = saved["win_rates"][0]
    assert (w["stage_a"], w["stage_b"]) == ("base", "sft")
    assert w["wins_a"] + w["wins_b"] + w["ties"] + w["skipped"] + w["judge_error"] == w["n"]
    assert w["n"] == saved["n_scaffolds"]
    # Metric table per stage, with perplexity attached.
    assert set(saved["metrics"]) == {"base", "sft"}
    for m in saved["metrics"].values():
        assert set(m) >= {"mean_distinct_1", "distinct_2", "self_bleu",
                          "mean_flesch_reading_ease", "n_usable", "perplexity"}
        assert isinstance(m["perplexity"], float)
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## Win-rates" in report and "Qualitative sample sheet" in report


def test_rlaif_column_appears_only_with_a_third_stage(tmp_path, fixture_path,
                                                      make_init_checkpoint):
    two = _prepare_inputs(tmp_path / "two", fixture_path, make_init_checkpoint,
                          ["base", "sft"])
    assert set(eval_stage.run(two)["metrics"]) == {"base", "sft"}

    three = _prepare_inputs(tmp_path / "three", fixture_path, make_init_checkpoint,
                            ["base", "sft", "rlaif"])
    r = eval_stage.run(three)
    assert set(r["metrics"]) == {"base", "sft", "rlaif"}
    # C(3,2) = 3 pairwise comparisons.
    assert len(r["win_rates"]) == 3


def test_identical_scaffolds_and_sampling_across_stages(tmp_path, fixture_path,
                                                        make_init_checkpoint):
    # Criterion asserted here at the stage level: every stage's generate_fn call
    # receives the same Scaffolds and sampling settings.
    config = _prepare_inputs(tmp_path, fixture_path, make_init_checkpoint,
                             ["base", "sft"])
    seen = []

    def spy(model, tokenizer, scaffolds, seeds, sampling, *, device="cpu"):
        seen.append((tuple(scaffolds), tuple(seeds), tuple(sorted(sampling.items()))))
        return ["a small fable body about a brave animal" for _ in scaffolds]

    eval_stage.run(config, generate_fn=spy)
    assert len(seen) == 2 and seen[0] == seen[1]  # base and sft got identical inputs


def to_toml(config: dict) -> str:
    """Serialize an eval config to TOML for tests (stdlib has no writer).
    Handles the top-level scalars, [[stages]] array-of-tables, and the nested
    [data]/[sampling]/[metrics]/[judge]/[hub] sections used by this stage."""
    def scalar(value) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, str):
            return f'"{value}"'
        return str(value)

    lines = []
    for key, value in config.items():
        if not isinstance(value, (dict, list)):
            lines.append(f"{key} = {scalar(value)}")
    for stage in config["stages"]:
        lines.append("[[stages]]")
        for key, value in stage.items():
            lines.append(f"{key} = {scalar(value)}")
    for section in ("data", "sampling", "metrics", "judge", "hub"):
        if section not in config:
            continue
        lines.append(f"[{section}]")
        for key, value in config[section].items():
            lines.append(f"{key} = {scalar(value)}")
    return "\n".join(lines) + "\n"


def test_cli_entrypoint_runs_standalone(tmp_path, fixture_path, make_init_checkpoint):
    config = _prepare_inputs(tmp_path, fixture_path, make_init_checkpoint,
                             ["base", "sft"])
    config_file = tmp_path / "eval.toml"
    config_file.write_text(to_toml(config), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.eval", "--config", str(config_file)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (Path(config["out_dir"]) / "results.json").exists()
