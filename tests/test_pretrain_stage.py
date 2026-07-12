import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from tinystories_v2.model import ModelConfig
from tinystories_v2.pretrain import lr_at, run
from tinystories_v2.tokenizer import run as run_tokenizer

REPO_ROOT = Path(__file__).parent.parent


def toy_config(tmp_path: Path, fixture_path: Path, tokenizer_path: Path, **train_overrides) -> dict:
    train = {
        "steps": 30, "micro_batch_size": 8, "grad_accum": 1,
        "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
        "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
        "precision": "fp32", "seed": 1337,
        "checkpoint_every": 10, "log_every": 1, "keep_last": 0,
    }
    train.update(train_overrides)
    return {
        "out_dir": str(tmp_path / "out"),
        "model": {"vocab_size": 512, "d_model": 64, "n_layers": 2,
                  "n_heads": 2, "context": 64, "ffn_hidden": 192},
        "data": {"split_path": str(fixture_path),
                 "tokenizer_path": str(tokenizer_path),
                 "packed_path": str(tmp_path / "packed" / "pretrain.bin")},
        "train": train,
        "wandb": {"enabled": False},
    }


@pytest.fixture(scope="module")
def tokenizer_path(tmp_path_factory, fixture_path):
    out = tmp_path_factory.mktemp("tok")
    run_tokenizer({"out_dir": str(out), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    return out / "tokenizer.json"


def read_metrics(out_dir: Path) -> list[dict]:
    lines = (out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_toy_run_decreases_loss_through_stage_entrypoint(tmp_path, fixture_path, tokenizer_path):
    config = toy_config(tmp_path, fixture_path, tokenizer_path)
    summary = run(config)
    metrics = read_metrics(Path(config["out_dir"]))
    assert len(metrics) == 30
    first, last = metrics[0], metrics[-1]
    assert last["loss"] < first["loss"] - 0.5  # random init starts near ln(512) ~ 6.2
    assert summary["step"] == 30
    # loss, LR, tokens seen all present per line (W&B-off degrade path)
    assert {"step", "loss", "lr", "tokens_seen"} <= first.keys()
    assert last["tokens_seen"] == 30 * 8 * 64  # steps * micro_batch * context


def test_stage_artifacts_and_manifest(tmp_path, fixture_path, tokenizer_path):
    config = toy_config(tmp_path, fixture_path, tokenizer_path, steps=4,
                        checkpoint_every=2)
    run(config)
    out = Path(config["out_dir"])
    assert Path(config["data"]["packed_path"]).exists()
    ckpts = sorted(p.name for p in (out / "checkpoints").glob("step_*.pt"))
    assert ckpts == ["step_000002.pt", "step_000004.pt"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "pretrain"
    assert manifest["final_step"] == 4
    assert manifest["config"]["train"]["steps"] == 4


def test_packing_skipped_when_binary_exists(tmp_path, fixture_path, tokenizer_path):
    config = toy_config(tmp_path, fixture_path, tokenizer_path, steps=2,
                        checkpoint_every=2)
    run(config)
    before = Path(config["data"]["packed_path"]).stat().st_mtime_ns
    config["out_dir"] = str(tmp_path / "out2")
    run(config)
    assert Path(config["data"]["packed_path"]).stat().st_mtime_ns == before


def test_packed_reuse_with_mismatched_vocab_raises(tmp_path, fixture_path, tokenizer_path):
    config = toy_config(tmp_path, fixture_path, tokenizer_path, steps=2, checkpoint_every=2)
    run(config)
    config["out_dir"] = str(tmp_path / "out2")
    config["model"]["vocab_size"] = 1024  # drifts from the packed binary's 512
    with pytest.raises(ValueError, match="vocab"):
        run(config)


def test_packed_binary_without_manifest_is_repacked(tmp_path, fixture_path, tokenizer_path):
    config = toy_config(tmp_path, fixture_path, tokenizer_path, steps=2, checkpoint_every=2)
    run(config)
    packed = Path(config["data"]["packed_path"])
    Path(str(packed) + ".json").unlink()
    config["out_dir"] = str(tmp_path / "out2")
    run(config)  # must not raise; manifest restored
    assert Path(str(packed) + ".json").exists()


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_mixed_precision_loop_runs_on_cpu(tmp_path, fixture_path, tokenizer_path, precision):
    config = toy_config(tmp_path, fixture_path, tokenizer_path, steps=2,
                        checkpoint_every=2, precision=precision)
    summary = run(config)
    assert summary["step"] == 2
    assert math.isfinite(summary["loss"])


def test_lr_schedule_warmup_peak_cosine_floor():
    peak = 6e-4
    kwargs = {"total_steps": 1000, "peak_lr": peak,
              "warmup_frac": 0.1, "min_lr_frac": 0.1}
    assert lr_at(0, **kwargs) == pytest.approx(peak / 100)   # first step, warming up
    assert lr_at(100, **kwargs) == pytest.approx(peak)       # end of warmup
    assert lr_at(1000, **kwargs) == pytest.approx(peak * 0.1)  # cosine floor
    mid = lr_at(550, **kwargs)
    assert peak * 0.1 < mid < peak
    assert mid == pytest.approx(
        peak * 0.1 + (peak - peak * 0.1) * 0.5 * (1 + math.cos(math.pi * 0.5))
    )


def test_hub_sync_after_checkpoint(tmp_path, fixture_path, tokenizer_path):
    config = toy_config(tmp_path, fixture_path, tokenizer_path, steps=2,
                        checkpoint_every=2)
    config["hub"] = {"target": str(tmp_path / "mirror")}
    run(config)
    assert (tmp_path / "mirror" / "checkpoints" / "step_000002.pt").exists()
    assert (tmp_path / "mirror" / "metrics.jsonl").exists()


def test_resume_with_missing_hub_repo_starts_fresh(tmp_path, fixture_path,
                                                   tokenizer_path, monkeypatch):
    # A first-ever --resume against a hub target that doesn't exist: the fetch
    # raises (here: simulated), and the stage must warn and start from step 0
    # rather than crash. No network — fetch_from is monkeypatched.
    def boom(target, local_dir):
        raise OSError("repo not found")

    monkeypatch.setattr("tinystories_v2.pretrain.fetch_from", boom)
    config = toy_config(tmp_path, fixture_path, tokenizer_path, steps=2,
                        checkpoint_every=2)
    config["hub"] = {"target": str(tmp_path / "mirror")}  # local: sync stays offline
    with pytest.warns(UserWarning, match="resume fetch"):
        summary = run(config, resume=True)
    assert summary["step"] == 2
    assert (Path(config["out_dir"]) / "checkpoints" / "step_000002.pt").exists()


def test_full_config_parses_and_matches_budgeted_model():
    config = tomllib.loads(
        (REPO_ROOT / "configs" / "pretrain_full.toml").read_text(encoding="utf-8")
    )
    assert ModelConfig(**config["model"]) == ModelConfig(
        vocab_size=8192, d_model=512, n_layers=8, n_heads=8,
        context=512, ffn_hidden=1408,
    )
    train = config["train"]
    assert train["peak_lr"] == 6e-4 and train["precision"] == "bf16"
    assert train["micro_batch_size"] == 32 and train["grad_accum"] == 8


def test_cli_entrypoint_runs_standalone(tmp_path, fixture_path, tokenizer_path):
    config = toy_config(tmp_path, fixture_path, tokenizer_path, steps=2,
                        checkpoint_every=2, log_every=1)
    config_file = tmp_path / "cfg.toml"
    config_file.write_text(to_toml(config), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.pretrain", "--config", str(config_file)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (Path(config["out_dir"]) / "checkpoints" / "step_000002.pt").exists()


def to_toml(config: dict) -> str:
    """Serialize the nested toy config as TOML (stdlib has no writer)."""
    lines = [f'out_dir = "{config["out_dir"]}"']
    for section in ("model", "data", "train", "wandb", "hub"):
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
