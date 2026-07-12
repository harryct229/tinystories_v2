"""Pretraining stage: pack the pretrain split, train FableLM, checkpoint-resume.

Invoke standalone:
    ts2-pretrain --config configs/pretrain_fixture.toml [--resume]
    (or: python -m tinystories_v2.pretrain --config ...)

Artifacts in <out_dir>:
    checkpoints/step_XXXXXX.pt   full training state (atomic; resume contract)
    metrics.jsonl                one line per log_every steps: loss, lr, tokens_seen
    manifest.json                stage, version, final step/loss, config
Plus the packed binary at [data].packed_path (skipped if already present).

Determinism contract: model init is seeded, batches are a pure function of
(seed, step, micro_step), and optimizer state round-trips through checkpoints,
so an interrupted-and-resumed run reproduces the uninterrupted run exactly
(fp32 CPU; asserted by tests/test_resume.py).
"""

import argparse
import json
import math
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch

from tinystories_v2 import __version__
from tinystories_v2.checkpoint import (
    latest_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint,
)
from tinystories_v2.config import load_config, load_env
from tinystories_v2.hub import fetch_from, try_sync_to
from tinystories_v2.tracking import MetricsLogger
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.pack import get_batch, load_packed, pack_split


def lr_at(step: int, total_steps: int, peak_lr: float,
          warmup_frac: float, min_lr_frac: float) -> float:
    """Linear warmup to peak_lr, then cosine decay to min_lr_frac * peak_lr."""
    warmup_steps = max(1, round(total_steps * warmup_frac))
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    min_lr = peak_lr * min_lr_frac
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def build_optimizer(model: FableLM, peak_lr: float, betas: tuple[float, float],
                    weight_decay: float) -> torch.optim.AdamW:
    # Decay 2D+ params (matmul weights, embeddings); never norms (1-D).
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=peak_lr, betas=betas,
    )


def run(config: dict, resume: bool = False) -> dict:
    load_env()  # W&B/HF keys from .env before wandb.init or hub sync; never printed
    train = config["train"]
    out_dir = Path(config["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    hub_target = config.get("hub", {}).get("target")

    # -- data: pack once, reuse thereafter (the packed binary is its own artifact)
    packed_path = Path(config["data"]["packed_path"])
    manifest_path = Path(str(packed_path) + ".json")
    if not packed_path.exists() or not manifest_path.exists():
        pack_split(config["data"]["split_path"],
                   config["data"]["tokenizer_path"], packed_path)
    else:
        pack_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_vocab = config["model"]["vocab_size"]
        if pack_manifest["vocab_size"] != expected_vocab:
            raise ValueError(
                f"packed binary at {packed_path} was built with vocab_size="
                f"{pack_manifest['vocab_size']}, but config requests vocab_size="
                f"{expected_vocab}. Delete the stale packed binary (and its "
                f".json manifest) to re-pack, or fix the config."
            )
    data = load_packed(packed_path)

    # -- precision knob: fp32 | bf16 (autocast) | fp16 (autocast + GradScaler)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = train["precision"]
    amp_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    autocast = (torch.autocast(device_type=device, dtype=amp_dtype)
                if amp_dtype else nullcontext())
    scaler = torch.amp.GradScaler(device, enabled=(precision == "fp16"))

    # -- model + optimizer (seeded init so runs are reproducible from config)
    torch.manual_seed(train["seed"])
    model = FableLM(ModelConfig(**config["model"])).to(device)
    optimizer = build_optimizer(model, train["peak_lr"],
                                (train["beta1"], train["beta2"]),
                                train["weight_decay"])

    start_step, tokens_seen = 0, 0
    if resume:
        if latest_checkpoint(ckpt_dir) is None and hub_target:
            # Fresh Colab VM: pull a previous session's checkpoints. A missing
            # target repo (or any hub error) just means "nothing to resume" —
            # start fresh rather than crash on the first-ever run.
            try:
                fetch_from(hub_target, out_dir)
            except Exception as err:  # noqa: BLE001 — resume is best-effort
                warnings.warn(f"resume fetch from {hub_target!r} failed; "
                              f"starting fresh: {err}", stacklevel=2)
        ckpt_path = latest_checkpoint(ckpt_dir)
        if ckpt_path is not None:
            state = load_checkpoint(ckpt_path)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start_step, tokens_seen = state["step"], state["tokens_seen"]
            print(f"resumed from {ckpt_path.name} at step {start_step}")

    def checkpoint(step: int) -> None:
        save_checkpoint(ckpt_dir, step, {
            "step": step, "tokens_seen": tokens_seen,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "config": config,
        })
        prune_checkpoints(ckpt_dir, train.get("keep_last", 0))
        if hub_target:
            try_sync_to(hub_target, out_dir)

    logger = MetricsLogger(out_dir, config.get("wandb"))
    steps, accum = train["steps"], train["grad_accum"]
    micro_bs, context = train["micro_batch_size"], config["model"]["context"]
    loss_value = float("nan")
    model.train()
    for step in range(start_step, steps):
        lr = lr_at(step, steps, train["peak_lr"],
                   train["warmup_frac"], train["min_lr_frac"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(accum):
            x, y = get_batch(data, micro_bs, context, seed=train["seed"],
                             step=step, micro_step=micro_step, device=device)
            with autocast:
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
            scaler.scale(loss / accum).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        tokens_seen += micro_bs * context * accum
        loss_value = loss.item()  # last micro-batch loss (cheap, logged raw)
        done = step + 1
        if done % train["log_every"] == 0:
            logger.log({"loss": loss_value, "lr": lr, "tokens_seen": tokens_seen},
                       step=done)
        if done % train["checkpoint_every"] == 0:
            checkpoint(done)
    if steps % train["checkpoint_every"] != 0:
        checkpoint(steps)
    logger.finish()

    manifest = {
        "stage": "pretrain", "package_version": __version__,
        "final_step": steps, "final_loss": loss_value,
        "tokens_seen": tokens_seen, "config": config,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    if hub_target:
        try_sync_to(hub_target, out_dir)
    return {"step": steps, "loss": loss_value}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest checkpoint in out_dir")
    args = parser.parse_args(argv)
    run(load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
