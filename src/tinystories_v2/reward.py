"""Reward Model stage: distill Judge preferences into a scalar reward with
hand-written Bradley-Terry loss (issue 05).

Invoke standalone:
    ts2-reward --config configs/reward_fixture.toml [--resume]
    (or: python -m tinystories_v2.reward --config ...)

Initializes the backbone from an SFT checkpoint ([init] section), attaches a
fresh scalar head, and trains on order-swap-consistent preference pairs (issue
10 schema). Reuses issue 02's checkpoint-resume contract, optimizer conventions
(build_optimizer), LR schedule (lr_at), precision knob, W&B logging, and Hub
sync verbatim; only the data source (preference pairs) and the loss (Bradley-
Terry over chosen/rejected scores) differ.

Holds out a deterministic slice of pairs and records held-out pair accuracy and
the split recipe in the manifest (schema: docs/schemas/reward-model-artifact-v1.md).
The accuracy gate that protects RLAIF lives in tinystories_v2.gate and reads
that manifest.

Artifacts in <out_dir>:
    checkpoints/step_XXXXXX.pt   full training state (atomic; resume contract)
    metrics.jsonl                one line per log_every steps: loss, lr, accuracy, pairs_seen
    manifest.json                stage, version, final step/loss, heldout_accuracy,
                                 pair_split recipe, pairs_path, config

Determinism contract: backbone init is loaded from a fixed checkpoint, the
held-out split is a pure function of (n_pairs, holdout_frac, split_seed),
batches are a pure function of (seed, step, micro_step), and optimizer state
round-trips, so an interrupted-and-resumed run reproduces the uninterrupted run
exactly (fp32 CPU; asserted by tests/test_reward_resume.py).
"""

import argparse
import json
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinystories_v2 import __version__
from tinystories_v2.checkpoint import (
    latest_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint,
)
from tinystories_v2.config import load_config, load_env
from tinystories_v2.gate import DEFAULT_ACCURACY_GATE
from tinystories_v2.hub import fetch_from, try_sync_to
from tinystories_v2.model import ModelConfig
from tinystories_v2.preferences import PreferencePair, validate_preference_pair
from tinystories_v2.pretrain import build_optimizer, lr_at
from tinystories_v2.reward_model import (
    RewardModel, bradley_terry_loss, pad_sequences, pair_accuracy, score_sequences,
)
from tinystories_v2.slot_prompt import encode_example
from tinystories_v2.tracking import MetricsLogger


def load_pairs(path: str | Path) -> list[PreferencePair]:
    """Read a preference-pair jsonl, validating each line against schema v1."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(validate_preference_pair(json.loads(line)))
    return pairs


def encode_pairs(tokenizer: Tokenizer, pairs: list[PreferencePair]) -> list[dict]:
    """Precompute (chosen_ids, rejected_ids) per pair via the Slot Prompt encoder
    (each is the full <|character|>…<|fable|>{body}<|end|> sequence)."""
    encoded = []
    for pair in pairs:
        chosen = encode_example(tokenizer, pair.scaffold, pair.chosen).input_ids
        rejected = encode_example(tokenizer, pair.scaffold, pair.rejected).input_ids
        encoded.append({"chosen_ids": chosen, "rejected_ids": rejected})
    return encoded


def split_pairs(encoded: list[dict], holdout_frac: float,
                seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic train/holdout split: a seeded permutation, the last
    round(n*holdout_frac) held out. A pure function of (n, holdout_frac, seed) so
    a resumed run reproduces the same split (and thus the same held-out accuracy)."""
    n = len(encoded)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    n_holdout = round(n * holdout_frac)
    train = [encoded[i] for i in perm[:n - n_holdout]]
    holdout = [encoded[i] for i in perm[n - n_holdout:]] if n_holdout else []
    return train, holdout


def get_pair_batch(train: list[dict], micro_batch_size: int, context: int, *,
                   seed: int, step: int, micro_step: int,
                   device: str = "cpu") -> tuple[torch.Tensor, ...]:
    """A (chosen_idx, chosen_len, rejected_idx, rejected_len) micro-batch sampled
    with replacement; a pure function of (seed, step, micro_step) for resume."""
    generator = torch.Generator()
    generator.manual_seed(((seed * 1_000_003 + step) * 1_009 + micro_step) % 2**63)
    idx = torch.randint(0, len(train), (micro_batch_size,), generator=generator)
    picks = idx.tolist()
    chosen_idx, chosen_len = pad_sequences(
        [train[i]["chosen_ids"] for i in picks], context, device)
    rejected_idx, rejected_len = pad_sequences(
        [train[i]["rejected_ids"] for i in picks], context, device)
    return chosen_idx, chosen_len, rejected_idx, rejected_len


@torch.no_grad()
def evaluate_accuracy(model: RewardModel, holdout: list[dict], *,
                      device: str = "cpu") -> float:
    """Held-out pair accuracy: fraction of holdout pairs the model scores
    chosen > rejected. Returns NaN for an empty holdout."""
    if not holdout:
        return float("nan")
    chosen = score_sequences(model, [p["chosen_ids"] for p in holdout], device=device)
    rejected = score_sequences(model, [p["rejected_ids"] for p in holdout], device=device)
    return pair_accuracy(chosen, rejected).item()


def _init_backbone_from_sft(config: dict, device: str) -> RewardModel:
    """Fresh Reward Model start: build the model from [model], load SFT backbone
    weights into it, and validate the architecture matches. Fetches the init
    artifact from Hub first if the local checkpoint is absent (fresh VM)."""
    model = RewardModel(ModelConfig(**config["model"])).to(device)
    init = config["init"]
    init_dir = Path(init["local_dir"])
    init_ckpt_dir = init_dir / "checkpoints"
    if latest_checkpoint(init_ckpt_dir) is None and init.get("hub_source"):
        fetch_from(init["hub_source"], init_dir)  # fresh Colab VM: pull SFT
    init_ckpt = latest_checkpoint(init_ckpt_dir)
    if init_ckpt is None:
        raise ValueError(
            f"no SFT checkpoint under {init_ckpt_dir}; point [init].local_dir "
            f"(and optionally [init].hub_source) at the SFT artifact")
    state = load_checkpoint(init_ckpt)
    if ModelConfig(**state["config"]["model"]) != model.config:
        raise ValueError(
            f"[model] does not match the SFT checkpoint at {init_ckpt}; the "
            f"Reward Model must reuse the SFT architecture")
    model.load_backbone_state_dict(state["model"])
    print(f"initialized Reward Model backbone from {init_ckpt}")
    return model


def run(config: dict, resume: bool = False) -> dict:
    load_env()  # W&B/HF keys before wandb.init or hub sync; never printed
    train = config["train"]
    out_dir = Path(config["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    hub_target = config.get("hub", {}).get("target")

    tokenizer = Tokenizer.from_file(config["data"]["tokenizer_path"])
    pairs = load_pairs(config["data"]["pairs_path"])
    if not pairs:
        raise ValueError(f"no preference pairs in {config['data']['pairs_path']}")
    encoded = encode_pairs(tokenizer, pairs)
    split = config["split"]
    train_pairs, holdout_pairs = split_pairs(encoded, split["holdout_frac"], split["seed"])
    if not train_pairs:
        raise ValueError("no training pairs after the holdout split; lower "
                         "[split].holdout_frac or add more pairs")

    # -- precision knob: fp32 | bf16 (autocast) | fp16 (autocast + GradScaler)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = train["precision"]
    amp_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    autocast = (torch.autocast(device_type=device, dtype=amp_dtype)
                if amp_dtype else nullcontext())
    scaler = torch.amp.GradScaler(device, enabled=(precision == "fp16"))

    torch.manual_seed(train["seed"])
    model = _init_backbone_from_sft(config, device)
    optimizer = build_optimizer(model, train["peak_lr"],
                                (train["beta1"], train["beta2"]),
                                train["weight_decay"])

    start_step, pairs_seen = 0, 0
    if resume:
        if latest_checkpoint(ckpt_dir) is None and hub_target:
            try:
                fetch_from(hub_target, out_dir)  # fresh VM: pull previous session
            except Exception as err:  # noqa: BLE001 — first run: repo may not exist yet
                warnings.warn(
                    f"could not fetch a prior Reward Model run from {hub_target!r}; "
                    f"starting fresh: {err}", stacklevel=2)
        ckpt_path = latest_checkpoint(ckpt_dir)
        if ckpt_path is not None:
            state = load_checkpoint(ckpt_path)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start_step, pairs_seen = state["step"], state["pairs_seen"]
            print(f"resumed from {ckpt_path.name} at step {start_step}")

    def checkpoint(step: int) -> None:
        save_checkpoint(ckpt_dir, step, {
            "step": step, "pairs_seen": pairs_seen,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "config": config,
        })
        prune_checkpoints(ckpt_dir, train.get("keep_last", 0))
        if hub_target:
            try_sync_to(hub_target, out_dir)

    logger = MetricsLogger(out_dir, config.get("wandb"))
    steps, accum = train["steps"], train["grad_accum"]
    micro_bs, context = train["micro_batch_size"], config["model"]["context"]
    loss_value, batch_acc = float("nan"), float("nan")
    model.train()
    for step in range(start_step, steps):
        lr = lr_at(step, steps, train["peak_lr"],
                   train["warmup_frac"], train["min_lr_frac"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(accum):
            c_idx, c_len, r_idx, r_len = get_pair_batch(
                train_pairs, micro_bs, context, seed=train["seed"],
                step=step, micro_step=micro_step, device=device)
            with autocast:
                chosen_scores = model(c_idx, c_len)
                rejected_scores = model(r_idx, r_len)
                loss = bradley_terry_loss(chosen_scores, rejected_scores)
            scaler.scale(loss / accum).backward()
            pairs_seen += micro_bs
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        loss_value = loss.item()          # last micro-batch Bradley-Terry loss
        batch_acc = pair_accuracy(chosen_scores.detach(),
                                  rejected_scores.detach()).item()
        done = step + 1
        if done % train["log_every"] == 0:
            logger.log({"loss": loss_value, "lr": lr, "accuracy": batch_acc,
                        "pairs_seen": pairs_seen}, step=done)
        if done % train["checkpoint_every"] == 0:
            checkpoint(done)
    if steps % train["checkpoint_every"] != 0:
        checkpoint(steps)

    heldout_accuracy = evaluate_accuracy(model, holdout_pairs, device=device)
    logger.finish()

    manifest = {
        "stage": "reward_model", "package_version": __version__,
        "final_step": steps, "final_loss": loss_value,
        "heldout_accuracy": heldout_accuracy,
        "pair_split": {"seed": split["seed"], "holdout_frac": split["holdout_frac"],
                       "n_pairs": len(encoded), "n_train": len(train_pairs),
                       "n_holdout": len(holdout_pairs)},
        "pairs_path": config["data"]["pairs_path"], "n_pairs": len(pairs),
        "config": config,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    if hub_target:
        try_sync_to(hub_target, out_dir)
    print(f"held-out pair accuracy: {heldout_accuracy:.3f} "
          f"(gate {DEFAULT_ACCURACY_GATE:.2f})")
    return {"step": steps, "loss": loss_value, "heldout_accuracy": heldout_accuracy}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest checkpoint in out_dir")
    args = parser.parse_args(argv)
    run(load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
