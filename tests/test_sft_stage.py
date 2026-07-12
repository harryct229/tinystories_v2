import json
import subprocess
import sys
from pathlib import Path

import pytest

from tinystories_v2.data import run as data_run
from tinystories_v2.pretrain import run as pretrain_run
from tinystories_v2.sft import run as sft_run
from tinystories_v2.sft_data import run as sft_data_run
from tinystories_v2.tokenizer import run as tokenizer_run

TOY_MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 128, "ffn_hidden": 192}


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, fixture_path):
    """Run data-prep + tokenizer + sft_data so the SFT stage has a real
    examples.jsonl and split-hash sets to read (stages couple via artifacts)."""
    base = tmp_path_factory.mktemp("sft_stage_inputs")
    data_dir, tok_dir, sd_dir = base / "data", base / "tok", base / "sd"
    data_run({
        "out_dir": str(data_dir),
        "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.4, "sft": 0.4,
                   "pref": 0.1, "eval": 0.1},
    })
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer = str(tok_dir / "tokenizer.json")
    sft_data_run({"out_dir": str(sd_dir), "tokenizer": tokenizer,
                  "sft_split": str(data_dir / "splits" / "sft.jsonl"),
                  "max_examples": 0})

    def split_hashes(name):
        with open(data_dir / "splits" / f"{name}.jsonl", encoding="utf-8") as f:
            return {json.loads(line)["prompt_hash"] for line in f if line.strip()}

    return {
        "examples_path": str(sd_dir / "examples.jsonl"),
        "tokenizer": tokenizer,
        "sft_hashes": split_hashes("sft"),
        "pretrain_hashes": split_hashes("pretrain"),
        "eval_hashes": split_hashes("eval"),
    }


def sft_toy_config(out_dir, prepared, init_dir, model=None, **train_overrides) -> dict:
    train = {
        "steps": 30, "micro_batch_size": 8, "grad_accum": 1,
        "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
        "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
        "precision": "fp32", "seed": 1337,
        "checkpoint_every": 10, "log_every": 1, "keep_last": 0,
    }
    train.update(train_overrides)
    return {
        "out_dir": str(out_dir),
        "model": dict(model or TOY_MODEL),
        "data": {"examples_path": prepared["examples_path"],
                 "tokenizer_path": prepared["tokenizer"]},
        "init": {"local_dir": str(init_dir)},
        "train": train,
        "wandb": {"enabled": False},
    }


def read_metrics(out_dir) -> list[dict]:
    lines = (Path(out_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_toy_sft_decreases_masked_loss(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = sft_toy_config(tmp_path / "out", prepared, init)
    summary = sft_run(config)
    metrics = read_metrics(config["out_dir"])
    assert len(metrics) == 30
    assert metrics[-1]["loss"] < metrics[0]["loss"] - 0.5  # verified drop ~0.85
    assert summary["step"] == 30
    assert {"step", "loss", "lr", "tokens_seen"} <= metrics[0].keys()


def test_stage_artifacts_and_manifest(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = sft_toy_config(tmp_path / "out", prepared, init, steps=4,
                            checkpoint_every=2)
    sft_run(config)
    out = Path(config["out_dir"])
    ckpts = sorted(p.name for p in (out / "checkpoints").glob("step_*.pt"))
    assert ckpts == ["step_000002.pt", "step_000004.pt"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "sft"
    assert manifest["final_step"] == 4
    assert manifest["examples_path"] == prepared["examples_path"]
    assert manifest["n_examples"] > 0


def test_init_from_a_real_pretraining_checkpoint(tmp_path, prepared, fixture_path):
    # Produce a real pretrain checkpoint (context 64), then SFT from it. Exercises
    # the init load + architecture-match validation against a genuine artifact.
    model64 = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
               "n_heads": 2, "context": 64, "ffn_hidden": 192}
    pre_dir = tmp_path / "pretrain"
    pretrain_run({
        "out_dir": str(pre_dir),
        "model": dict(model64),
        "data": {"split_path": str(fixture_path),
                 "tokenizer_path": prepared["tokenizer"],
                 "packed_path": str(tmp_path / "packed" / "pretrain.bin")},
        "train": {"steps": 2, "micro_batch_size": 4, "grad_accum": 1,
                  "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                  "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
                  "grad_clip": 1.0, "precision": "fp32", "seed": 1,
                  "checkpoint_every": 2, "log_every": 1, "keep_last": 0},
        "wandb": {"enabled": False},
    })
    config = sft_toy_config(tmp_path / "sft_out", prepared, pre_dir,
                            model=model64, steps=3, checkpoint_every=3)
    summary = sft_run(config)
    import math
    assert summary["step"] == 3 and math.isfinite(summary["loss"])


def test_mismatched_init_architecture_raises(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    drifted = dict(TOY_MODEL, d_model=128)  # differs from the init checkpoint
    config = sft_toy_config(tmp_path / "out", prepared, init, model=drifted,
                            steps=2, checkpoint_every=2)
    with pytest.raises(ValueError, match="Pretraining checkpoint"):
        sft_run(config)


def test_missing_init_checkpoint_raises(tmp_path, prepared):
    config = sft_toy_config(tmp_path / "out", prepared, tmp_path / "empty_init",
                            steps=2, checkpoint_every=2)
    with pytest.raises(ValueError, match="no Pretraining checkpoint"):
        sft_run(config)


def test_stage_trains_only_on_the_sft_split(tmp_path, prepared, make_init_checkpoint):
    # Split-leakage guard: every prompt_hash the stage trains on comes from the
    # sft split; none leak in from pretrain or eval.
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = sft_toy_config(tmp_path / "out", prepared, init, steps=2,
                            checkpoint_every=2)
    sft_run(config)
    manifest = json.loads(
        (Path(config["out_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    from tinystories_v2.sft import load_sft_examples
    trained = {rec["prompt_hash"] for rec in load_sft_examples(manifest["examples_path"])}
    assert trained  # non-empty
    assert trained <= prepared["sft_hashes"]
    assert trained.isdisjoint(prepared["pretrain_hashes"] | prepared["eval_hashes"])


def to_toml(config: dict) -> str:
    """Serialize the nested SFT config as TOML (stdlib has no writer)."""
    lines = [f'out_dir = "{config["out_dir"]}"']
    for section in ("model", "data", "init", "train", "wandb", "hub"):
        if section not in config:
            continue
        lines.append(f"[{section}]")
        for key, value in config[section].items():
            if isinstance(value, bool):
                lines.append(f"{key} = {str(value).lower()}")
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"


def test_cli_entrypoint_runs_standalone(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = sft_toy_config(tmp_path / "out", prepared, init, steps=2,
                            checkpoint_every=2)
    config_file = tmp_path / "cfg.toml"
    config_file.write_text(to_toml(config), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.sft", "--config", str(config_file)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (Path(config["out_dir"]) / "checkpoints" / "step_000002.pt").exists()


def test_grad_accum_trains_and_decreases_loss(tmp_path, prepared, make_init_checkpoint):
    # Exercises the production grad_accum>1 path (sft_full uses grad_accum=8),
    # which the other stage tests (grad_accum=1) never reach.
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = sft_toy_config(tmp_path / "out", prepared, init, grad_accum=2)
    summary = sft_run(config)
    metrics = read_metrics(config["out_dir"])
    assert len(metrics) == 30
    assert metrics[-1]["loss"] < metrics[0]["loss"] - 0.5  # accum path still trains
    import math
    assert summary["step"] == 30 and math.isfinite(summary["loss"])


def test_resume_with_missing_hub_target_starts_fresh(
        tmp_path, prepared, make_init_checkpoint, monkeypatch):
    # First real --resume on a fresh VM: the SFT Hub repo doesn't exist yet, so
    # the resume-time fetch raises. The stage must warn and train from init,
    # not crash.
    import tinystories_v2.sft as sft_mod

    def _raise(*args, **kwargs):
        raise RuntimeError("repo not found")

    monkeypatch.setattr(sft_mod, "fetch_from", _raise)
    monkeypatch.setattr(sft_mod, "try_sync_to", lambda *a, **k: None)
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = sft_toy_config(tmp_path / "out", prepared, init, steps=2,
                            checkpoint_every=2)
    config["hub"] = {"target": "hf://example/does-not-exist"}
    with pytest.warns(UserWarning, match="starting fresh"):
        summary = sft_run(config, resume=True)
    assert summary["step"] == 2  # trained from init despite the failed fetch
