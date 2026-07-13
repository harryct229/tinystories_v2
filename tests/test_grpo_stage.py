import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from tinystories_v2.data import run as data_run
from tinystories_v2.eval import load_stage_model
from tinystories_v2.gate import RewardGateError
from tinystories_v2.grpo import run as grpo_run
from tinystories_v2.judge import SlotCoverageFakeJudge, judge_with_order_swap
from tinystories_v2.model import FableLM
from tinystories_v2.reward import run as reward_run
from tinystories_v2.slot_prompt import SLOT_FIELDS
from tinystories_v2.slots import Scaffold
from tinystories_v2.tokenizer import run as tokenizer_run

TOY_MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
             "n_heads": 2, "context": 128, "ffn_hidden": 192}
TARGET_TOKEN = "a"     # the rigged reward counts this character in each fable


def _write_reward_manifest(reward_dir: Path, accuracy: float) -> None:
    reward_dir.mkdir(parents=True, exist_ok=True)
    (reward_dir / "manifest.json").write_text(
        json.dumps({"stage": "reward_model", "heldout_accuracy": accuracy}),
        encoding="utf-8")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, fixture_path):
    """Tokenizer + pref split, shared across the stage tests."""
    base = tmp_path_factory.mktemp("grpo_stage_inputs")
    data_dir, tok_dir = base / "data", base / "tok"
    data_run({
        "out_dir": str(data_dir), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.1, "sft": 0.6,
                   "pref": 0.2, "eval": 0.1},
    })
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    return {"pref_split": str(data_dir / "splits" / "pref.jsonl"),
            "tokenizer": str(tok_dir / "tokenizer.json"),
            "sft_rows": str(data_dir / "splits" / "sft.jsonl")}


def grpo_toy_config(out_dir, prepared, init_dir, reward_dir, *, kl_beta=0.0,
                    gate=0.5, **overrides) -> dict:
    grpo = {"group_size": 6, "clip_eps": 0.2, "kl_beta": kl_beta,
            "ppo_epochs": 2, "adv_eps": 1e-6}
    grpo.update(overrides.pop("grpo", {}))
    train = {"steps": 30, "prompts_per_step": 4, "peak_lr": 1e-3,
             "warmup_frac": 0.1, "min_lr_frac": 0.1, "weight_decay": 0.0,
             "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0, "precision": "fp32",
             "seed": 1337, "checkpoint_every": 10, "log_every": 1, "keep_last": 0}
    train.update(overrides.pop("train", {}))
    return {
        "out_dir": str(out_dir), "model": dict(TOY_MODEL),
        "data": {"pref_split": prepared["pref_split"],
                 "tokenizer_path": prepared["tokenizer"]},
        "init": {"local_dir": str(init_dir)},
        "reward": {"local_dir": str(reward_dir), "gate": gate},
        "grpo": grpo, "sampling": {"max_new_tokens": 16, "temperature": 1.0, "top_p": 1.0},
        "train": train, "wandb": {"enabled": False},
    }


def _rigged_reward(scaffold, fables):
    """Rewards presence of the target token — the acceptance-criterion example."""
    return [float(f.count(TARGET_TOKEN)) for f in fables]


def _train_toy_rm(reward_dir, init_dir, prepared) -> None:
    """Train a real toy Reward Model into reward_dir (manifest + checkpoint) from
    fake-Judge separable pairs, so the RM-backed scorer path (reward_fn=None) has
    something to load. Shared by the whole-chain and CLI tests."""
    with open(prepared["sft_rows"], encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    judge = SlotCoverageFakeJudge()
    bland = ["A plain note with nothing much to say.",
             "Some words that go nowhere in particular."]
    pairs_path = Path(reward_dir).parent / "pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            scaffold = Scaffold(**{fld: row[fld] for fld in SLOT_FIELDS})
            chosen = (f"{scaffold.character}, a {scaffold.trait} one in "
                      f"{scaffold.setting}, met {scaffold.conflict}. "
                      f"{scaffold.resolution}. The moral: {scaffold.moral}.")
            pair = judge_with_order_swap(judge, scaffold, chosen, bland[i % len(bland)])
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
    reward_run({
        "out_dir": str(reward_dir), "model": dict(TOY_MODEL),
        "data": {"pairs_path": str(pairs_path), "tokenizer_path": prepared["tokenizer"]},
        "init": {"local_dir": str(init_dir)},
        "split": {"holdout_frac": 0.25, "seed": 20260712},
        "train": {"steps": 40, "micro_batch_size": 8, "grad_accum": 1,
                  "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                  "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
                  "precision": "fp32", "seed": 1337, "checkpoint_every": 40,
                  "log_every": 20, "keep_last": 0},
        "wandb": {"enabled": False},
    })


def read_metrics(out_dir) -> list[dict]:
    lines = (Path(out_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_rigged_reward_raises_mean_reward(tmp_path, prepared, make_init_checkpoint):
    # Criterion 1: through the real entrypoint, a rigged reward measurably lifts
    # mean reward. kl_beta 0 so nothing fights the reward; gate satisfied by a
    # passing manifest; scoring replaced by the rigged reward_fn.
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    reward_dir = tmp_path / "rm"
    _write_reward_manifest(reward_dir, 0.75)
    config = grpo_toy_config(tmp_path / "out", prepared, init, reward_dir,
                             kl_beta=0.0, train={"steps": 40, "seed": 1337,
                             "prompts_per_step": 4, "peak_lr": 1e-3,
                             "warmup_frac": 0.1, "min_lr_frac": 0.1,
                             "weight_decay": 0.0, "beta1": 0.9, "beta2": 0.95,
                             "grad_clip": 1.0, "precision": "fp32",
                             "checkpoint_every": 40, "log_every": 1, "keep_last": 0})
    grpo_run(config, reward_fn=_rigged_reward)
    rewards = [m["reward_mean"] for m in read_metrics(config["out_dir"])]
    first = sum(rewards[:8]) / 8
    last = sum(rewards[-8:]) / 8
    assert last > first        # policy shifted toward the rewarded token


def test_kl_penalty_constrains_the_policy(tmp_path, prepared, make_init_checkpoint):
    # Criterion 3: a large kl_beta keeps the final KL below the unpenalized run's,
    # and disabling the leash is config-only (kl_beta = 0).
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    reward_dir = tmp_path / "rm"
    _write_reward_manifest(reward_dir, 0.75)
    free = grpo_toy_config(tmp_path / "free", prepared, init, reward_dir, kl_beta=0.0)
    leashed = grpo_toy_config(tmp_path / "leashed", prepared, init, reward_dir,
                              kl_beta=1.0)
    grpo_run(free, reward_fn=_rigged_reward)
    grpo_run(leashed, reward_fn=_rigged_reward)
    free_kl = read_metrics(free["out_dir"])[-1]["kl"]
    leashed_kl = read_metrics(leashed["out_dir"])[-1]["kl"]
    assert leashed_kl <= free_kl        # the leash constrains divergence


def test_refuses_to_start_below_the_gate(tmp_path, prepared, make_init_checkpoint):
    # Criterion 4: a Reward Model below the gate raises before any training.
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    reward_dir = tmp_path / "rm"
    _write_reward_manifest(reward_dir, 0.55)          # below the 0.68 default gate
    config = grpo_toy_config(tmp_path / "out", prepared, init, reward_dir)
    config["reward"].pop("gate")                        # use the default 0.68 gate
    with pytest.raises(RewardGateError, match="below the gate"):
        grpo_run(config, reward_fn=_rigged_reward)
    assert not (Path(config["out_dir"]) / "checkpoints").exists()  # never trained


def test_manifest_records_recipe_and_gate(tmp_path, prepared, make_init_checkpoint):
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    reward_dir = tmp_path / "rm"
    _write_reward_manifest(reward_dir, 0.75)
    config = grpo_toy_config(tmp_path / "out", prepared, init, reward_dir,
                             train={"steps": 4, "prompts_per_step": 2,
                             "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                             "weight_decay": 0.0, "beta1": 0.9, "beta2": 0.95,
                             "grad_clip": 1.0, "precision": "fp32", "seed": 1337,
                             "checkpoint_every": 2, "log_every": 1, "keep_last": 0})
    grpo_run(config, reward_fn=_rigged_reward)
    manifest = json.loads(
        (Path(config["out_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "grpo"
    assert manifest["grpo"]["group_size"] == 6 and manifest["grpo"]["clip_eps"] == 0.2
    assert manifest["reward_gate"]["accuracy"] == 0.75
    assert manifest["reward_gate"]["gate"] == 0.5
    assert isinstance(manifest["final_reward_mean"], float)
    assert manifest["pref_split"] == prepared["pref_split"]


def test_resuming_a_completed_run_does_not_clobber_the_manifest(
        tmp_path, prepared, make_init_checkpoint):
    # Whole-branch review fix: run() used to unconditionally re-derive
    # final_reward_mean/final_loss/final_kl from locals seeded to NaN, then
    # rewrite manifest.json and push to the Hub even when start_step >= steps
    # (nothing trained this call). Resuming a finished job must be a no-op that
    # preserves the real published manifest instead of degrading it to NaN.
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    reward_dir = tmp_path / "rm"
    _write_reward_manifest(reward_dir, 0.75)
    config = grpo_toy_config(tmp_path / "out", prepared, init, reward_dir,
                             train={"steps": 4, "prompts_per_step": 2,
                             "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                             "weight_decay": 0.0, "beta1": 0.9, "beta2": 0.95,
                             "grad_clip": 1.0, "precision": "fp32", "seed": 1337,
                             "checkpoint_every": 4, "log_every": 1, "keep_last": 0})
    grpo_run(config, reward_fn=_rigged_reward)
    manifest_path = Path(config["out_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_reward_mean = manifest["final_reward_mean"]
    assert not math.isnan(first_reward_mean)

    # Resume a run that is already complete (start_step == steps): the guard
    # must skip the tail checkpoint/manifest/Hub-sync finalization rather than
    # overwriting the good manifest with NaN placeholders.
    grpo_run(config, resume=True, reward_fn=_rigged_reward)
    manifest_after_resume = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_after_resume["final_reward_mean"] == first_reward_mean
    assert not math.isnan(manifest_after_resume["final_reward_mean"])


def test_output_checkpoint_is_eval_drop_in(tmp_path, prepared, make_init_checkpoint):
    # Criterion: the GRPO checkpoint loads through eval.load_stage_model exactly
    # like base/SFT — a plain FableLM, no GRPO-specific eval code.
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    reward_dir = tmp_path / "rm"
    _write_reward_manifest(reward_dir, 0.75)
    config = grpo_toy_config(tmp_path / "out", prepared, init, reward_dir,
                             train={"steps": 4, "prompts_per_step": 2,
                             "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                             "weight_decay": 0.0, "beta1": 0.9, "beta2": 0.95,
                             "grad_clip": 1.0, "precision": "fp32", "seed": 1337,
                             "checkpoint_every": 4, "log_every": 1, "keep_last": 0})
    grpo_run(config, reward_fn=_rigged_reward)
    model = load_stage_model({"name": "rlaif", "local_dir": config["out_dir"]}, "cpu")
    assert isinstance(model, FableLM)


def test_whole_chain_fake_judge_to_toy_rm_to_grpo(tmp_path, prepared, make_init_checkpoint):
    # Criterion 6: fake Judge pairs -> toy Reward Model -> toy GRPO with the REAL
    # RM-backed scorer (reward_fn=None), no GPU or network. Gate lowered to 0.5 so
    # the toy RM's separable-pair accuracy passes.
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    reward_dir = tmp_path / "rm"
    _train_toy_rm(reward_dir, init, prepared)
    config = grpo_toy_config(tmp_path / "out", prepared, init, reward_dir, gate=0.5,
                             train={"steps": 4, "prompts_per_step": 2, "peak_lr": 1e-3,
                             "warmup_frac": 0.1, "min_lr_frac": 0.1, "weight_decay": 0.0,
                             "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
                             "precision": "fp32", "seed": 1337, "checkpoint_every": 4,
                             "log_every": 1, "keep_last": 0})
    summary = grpo_run(config, reward_fn=None)     # the real frozen Reward Model scores
    assert summary["step"] == 4
    assert (Path(config["out_dir"]) / "checkpoints" / "step_000004.pt").exists()


def to_toml(config: dict) -> str:
    """Serialize the nested GRPO config as TOML (stdlib has no writer)."""
    lines = [f'out_dir = "{config["out_dir"]}"']
    for section in ("model", "data", "init", "reward", "grpo", "sampling",
                    "train", "wandb", "hub"):
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
    # The CLI has no injected reward, so [reward] must hold a real RM (manifest +
    # checkpoint) that the default scorer loads; gate 0.5 lets the toy RM pass.
    init = make_init_checkpoint(tmp_path / "init", TOY_MODEL, prepared["tokenizer"])
    reward_dir = tmp_path / "rm"
    _train_toy_rm(reward_dir, init, prepared)
    config = grpo_toy_config(tmp_path / "out", prepared, init, reward_dir, gate=0.5,
                             train={"steps": 2, "prompts_per_step": 2, "peak_lr": 1e-3,
                             "warmup_frac": 0.1, "min_lr_frac": 0.1, "weight_decay": 0.0,
                             "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
                             "precision": "fp32", "seed": 1337, "checkpoint_every": 2,
                             "log_every": 1, "keep_last": 0})
    config_file = tmp_path / "cfg.toml"
    config_file.write_text(to_toml(config), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.grpo", "--config", str(config_file)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (Path(config["out_dir"]) / "checkpoints" / "step_000002.pt").exists()
