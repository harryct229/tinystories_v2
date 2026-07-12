import json
import subprocess
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from tinystories_v2.checkpoint import latest_checkpoint, load_checkpoint
from tinystories_v2.data import run as data_run
from tinystories_v2.judge import SlotCoverageFakeJudge, judge_with_order_swap
from tinystories_v2.model import ModelConfig
from tinystories_v2.pretrain import run as pretrain_run
from tinystories_v2.reward import run as reward_run
from tinystories_v2.reward_model import RewardModel, score_fables
from tinystories_v2.slot_prompt import SLOT_FIELDS
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run

TOY_MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 128, "ffn_hidden": 192}

# Bland rejected bodies mention no slot value, so SlotCoverageFakeJudge always
# prefers the slot-rich chosen body: the pairs are synthetically separable.
_BLAND = ["A plain note with nothing much to say.",
          "Some words that go nowhere in particular."]


def _separable_pairs(rows):
    """Build order-swap-consistent fake-Judge pairs where chosen mentions every
    slot value and rejected is bland — a learnable, separable signal."""
    judge = SlotCoverageFakeJudge()
    pairs = []
    for i, row in enumerate(rows):
        scaffold = Scaffold(**{f: row[f] for f in SLOT_FIELDS})
        chosen = (f"{scaffold.character}, a {scaffold.trait} one in "
                  f"{scaffold.setting}, met {scaffold.conflict}. "
                  f"{scaffold.resolution}. The moral: {scaffold.moral}.")
        rejected = _BLAND[i % len(_BLAND)]
        pair = judge_with_order_swap(judge, scaffold, chosen, rejected)
        assert pair is not None and pair.chosen == chosen  # judge picked the rich body
        pairs.append(pair)
    return pairs


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, fixture_path):
    """Prepare a tokenizer and a separable fake-Judge pairs.jsonl from the
    fixture's sft split (stages couple via artifacts)."""
    base = tmp_path_factory.mktemp("reward_stage_inputs")
    data_dir, tok_dir = base / "data", base / "tok"
    data_run({
        "out_dir": str(data_dir), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.1, "sft": 0.7,
                   "pref": 0.1, "eval": 0.1},
    })
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer = str(tok_dir / "tokenizer.json")
    with open(data_dir / "splits" / "sft.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    pairs = _separable_pairs(rows)
    pairs_path = base / "pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
    return {"pairs_path": str(pairs_path), "tokenizer": tokenizer, "n_pairs": len(pairs)}


def reward_toy_config(out_dir, prepared, init_dir, model=None, **train_overrides) -> dict:
    train = {
        "steps": 60, "micro_batch_size": 8, "grad_accum": 1,
        "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
        "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
        "precision": "fp32", "seed": 1337,
        "checkpoint_every": 20, "log_every": 1, "keep_last": 0,
    }
    train.update(train_overrides)
    return {
        "out_dir": str(out_dir),
        "model": dict(model or TOY_MODEL),
        "data": {"pairs_path": prepared["pairs_path"],
                 "tokenizer_path": prepared["tokenizer"]},
        "init": {"local_dir": str(init_dir)},
        "split": {"holdout_frac": 0.25, "seed": 20260712},
        "train": train,
        "wandb": {"enabled": False},
    }


def read_metrics(out_dir) -> list[dict]:
    lines = (Path(out_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_toy_reward_model_beats_chance_on_heldout(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = reward_toy_config(tmp_path / "out", prepared, init)
    summary = reward_run(config)
    assert summary["heldout_accuracy"] > 0.8   # well above chance (0.5) on separable pairs
    metrics = read_metrics(config["out_dir"])
    assert len(metrics) == 60
    assert {"step", "loss", "lr", "accuracy", "pairs_seen"} <= metrics[0].keys()
    assert metrics[-1]["loss"] < metrics[0]["loss"]   # Bradley-Terry loss fell


def test_manifest_records_accuracy_and_split_recipe(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = reward_toy_config(tmp_path / "out", prepared, init, steps=6,
                              checkpoint_every=3)
    reward_run(config)
    manifest = json.loads(
        (Path(config["out_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "reward_model"
    assert isinstance(manifest["heldout_accuracy"], float)
    split = manifest["pair_split"]
    assert split["seed"] == 20260712 and split["holdout_frac"] == 0.25
    assert split["n_pairs"] == prepared["n_pairs"]
    assert split["n_train"] + split["n_holdout"] == split["n_pairs"]
    assert split["n_holdout"] == round(prepared["n_pairs"] * 0.25)
    assert manifest["pairs_path"] == prepared["pairs_path"]


def test_scores_are_usable_downstream(tmp_path, prepared, make_init_checkpoint):
    # Criterion 2: a scoring call takes (Slot Prompt Scaffold, Fable) -> scalar,
    # batched, on CPU, from a trained artifact.
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    config = reward_toy_config(tmp_path / "out", prepared, init, steps=6,
                              checkpoint_every=6)
    reward_run(config)
    state = load_checkpoint(latest_checkpoint(Path(config["out_dir"]) / "checkpoints"))
    model = RewardModel(ModelConfig(**state["config"]["model"]))
    model.load_state_dict(state["model"])
    tokenizer = Tokenizer.from_file(prepared["tokenizer"])
    scaffold = Scaffold("fox", "sly", "a wood", "a gate", "it waited", "patience wins")
    scores = score_fables(model, tokenizer,
                          [(scaffold, "The sly fox waited by the gate."),
                           (scaffold, "A plain note with nothing much to say.")],
                          device="cpu")
    assert len(scores) == 2 and all(isinstance(s, float) for s in scores)


def test_split_recipe_is_reproducible(tmp_path, prepared, make_init_checkpoint):
    # Same split seed -> identical held-out accuracy across two fresh runs.
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    a = reward_run(reward_toy_config(tmp_path / "a", prepared, init, steps=6,
                                     checkpoint_every=6))
    b = reward_run(reward_toy_config(tmp_path / "b", prepared, init, steps=6,
                                     checkpoint_every=6))
    assert a["heldout_accuracy"] == b["heldout_accuracy"]


def test_init_from_a_real_checkpoint(tmp_path, prepared, fixture_path):
    # Init the backbone from a genuine checkpoint (a Pretraining checkpoint is
    # structurally identical to an SFT one for the backbone). Exercises the load
    # + architecture-match validation against a real artifact.
    model64 = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
               "n_heads": 2, "context": 64, "ffn_hidden": 192}
    pre_dir = tmp_path / "pretrain"
    pretrain_run({
        "out_dir": str(pre_dir), "model": dict(model64),
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
    config = reward_toy_config(tmp_path / "reward_out", prepared, pre_dir,
                               model=model64, steps=3, checkpoint_every=3)
    summary = reward_run(config)
    import math
    assert summary["step"] == 3 and math.isfinite(summary["loss"])


def test_mismatched_init_architecture_raises(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    drifted = dict(TOY_MODEL, d_model=128)  # differs from the init checkpoint
    config = reward_toy_config(tmp_path / "out", prepared, init, model=drifted,
                              steps=2, checkpoint_every=2)
    with pytest.raises(ValueError, match="SFT checkpoint"):
        reward_run(config)


def test_missing_init_checkpoint_raises(tmp_path, prepared):
    config = reward_toy_config(tmp_path / "out", prepared, tmp_path / "empty_init",
                              steps=2, checkpoint_every=2)
    with pytest.raises(ValueError, match="no SFT checkpoint"):
        reward_run(config)


def to_toml(config: dict) -> str:
    """Serialize the nested reward config as TOML (stdlib has no writer)."""
    lines = [f'out_dir = "{config["out_dir"]}"']
    for section in ("model", "data", "init", "split", "train", "wandb", "hub"):
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
    config = reward_toy_config(tmp_path / "out", prepared, init, steps=2,
                              checkpoint_every=2)
    config_file = tmp_path / "cfg.toml"
    config_file.write_text(to_toml(config), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.reward", "--config", str(config_file)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (Path(config["out_dir"]) / "checkpoints" / "step_000002.pt").exists()
