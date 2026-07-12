"""SFT stage: fine-tune a Pretraining checkpoint on Slot Prompt -> Fable
examples (issue 12's sft_data artifact) with prompt-masked loss.

Invoke standalone:
    ts2-sft --config configs/sft_fixture.toml [--resume]
    (or: python -m tinystories_v2.sft --config ...)

Initializes model weights from a Pretraining checkpoint ([init] section), then
trains with its own optimizer/schedule/checkpoints. Reuses issue 02's
checkpoint-resume contract, optimizer conventions (build_optimizer), LR
schedule (lr_at), precision knob, W&B logging, and Hub sync verbatim; only the
data source (variable-length masked examples, not packed windows) and the loss
(masked to the fable body + <|end|>) differ.

Artifacts in <out_dir>:
    checkpoints/step_XXXXXX.pt   full training state (atomic; resume contract)
    metrics.jsonl                one line per log_every steps: loss, lr, tokens_seen
    manifest.json                stage, version, final step/loss, examples_path, config

Determinism contract (same as pretrain): model init is loaded from a fixed
checkpoint, batches are a pure function of (seed, step, micro_step), optimizer
state round-trips, so an interrupted-and-resumed run reproduces the
uninterrupted run exactly (fp32 CPU; asserted by tests/test_sft_resume.py).
"""

import argparse
import json
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
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.pretrain import build_optimizer, lr_at
from tinystories_v2.tracking import MetricsLogger


def load_sft_examples(path: str | Path) -> list[dict]:
    """Read the sft_data artifact (examples.jsonl) into memory. Each record has
    input_ids and loss_mask (schema: docs/schemas/sft-example-v1.md)."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def get_sft_batch(examples: list[dict], micro_batch_size: int, context: int, *,
                  seed: int, step: int, micro_step: int,
                  device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A padded (x, y, mask) micro-batch sampled with replacement. Batch
    selection is a pure function of (seed, step, micro_step) so an interrupted
    run resumed from a checkpoint replays identical batches (resume contract).

    Each example is truncated to context+1 ids, then shifted for next-token
    prediction: x = ids[:-1], y = ids[1:], mask = loss_mask[1:] (active over the
    fable body + <|end|>). Rows are right-padded to the batch's longest x with
    id 0 and mask 0; padding never contributes to the loss, and causal attention
    makes right-padding safe without an attention mask.
    """
    generator = torch.Generator()
    generator.manual_seed(((seed * 1_000_003 + step) * 1_009 + micro_step) % 2**63)
    idx = torch.randint(0, len(examples), (micro_batch_size,), generator=generator)

    rows = []
    for i in idx.tolist():
        ids = examples[i]["input_ids"][:context + 1]
        mask = examples[i]["loss_mask"][:context + 1]
        rows.append((ids[:-1], ids[1:], mask[1:]))
    width = max(len(x) for x, _, _ in rows)

    xs, ys, ms = [], [], []
    for x, y, m in rows:
        pad = width - len(x)
        xs.append(x + [0] * pad)
        ys.append(y + [0] * pad)
        ms.append([float(v) for v in m] + [0.0] * pad)
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    mask = torch.tensor(ms, dtype=torch.float)
    return x.to(device), y.to(device), mask.to(device)


def masked_lm_loss(logits: torch.Tensor, y: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy over active (mask==1) positions only.
    Clamps the denominator so an all-padding batch cannot divide by zero."""
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), y.view(-1), reduction="none"
    )
    loss = loss.view_as(y) * mask
    return loss.sum() / mask.sum().clamp(min=1.0)


def _init_model_from_pretrain(config: dict, device: str) -> FableLM:
    """Fresh SFT start: build the model from [model] and load Pretraining
    weights. Fetches the init artifact from Hub first if the local checkpoint is
    absent, then validates the pretrained architecture matches [model]."""
    model = FableLM(ModelConfig(**config["model"])).to(device)
    init = config["init"]
    init_dir = Path(init["local_dir"])
    init_ckpt_dir = init_dir / "checkpoints"
    if latest_checkpoint(init_ckpt_dir) is None and init.get("hub_source"):
        fetch_from(init["hub_source"], init_dir)  # fresh Colab VM: pull pretrain
    init_ckpt = latest_checkpoint(init_ckpt_dir)
    if init_ckpt is None:
        raise ValueError(
            f"no Pretraining checkpoint under {init_ckpt_dir}; point "
            f"[init].local_dir (and optionally [init].hub_source) at the "
            f"Pretraining artifact"
        )
    state = load_checkpoint(init_ckpt)
    if ModelConfig(**state["config"]["model"]) != model.config:
        raise ValueError(
            f"[model] does not match the Pretraining checkpoint at {init_ckpt}; "
            f"SFT must use the pretrained architecture"
        )
    model.load_state_dict(state["model"])
    print(f"initialized from Pretraining checkpoint {init_ckpt}")
    return model


def run(config: dict, resume: bool = False) -> dict:
    load_env()  # W&B/HF keys before wandb.init or hub sync; never printed
    train = config["train"]
    out_dir = Path(config["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    hub_target = config.get("hub", {}).get("target")

    examples = load_sft_examples(config["data"]["examples_path"])
    if not examples:
        raise ValueError(f"no examples in {config['data']['examples_path']}")

    # -- precision knob: fp32 | bf16 (autocast) | fp16 (autocast + GradScaler)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = train["precision"]
    amp_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    autocast = (torch.autocast(device_type=device, dtype=amp_dtype)
                if amp_dtype else nullcontext())
    scaler = torch.amp.GradScaler(device, enabled=(precision == "fp16"))

    torch.manual_seed(train["seed"])
    model = _init_model_from_pretrain(config, device)
    optimizer = build_optimizer(model, train["peak_lr"],
                                (train["beta1"], train["beta2"]),
                                train["weight_decay"])

    start_step, tokens_seen = 0, 0
    if resume:
        if latest_checkpoint(ckpt_dir) is None and hub_target:
            try:
                fetch_from(hub_target, out_dir)  # fresh VM: pull previous session
            except Exception as err:  # noqa: BLE001 — first run: the repo may not exist yet
                warnings.warn(
                    f"could not fetch a prior SFT run from {hub_target!r}; "
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
        # Token-weight grad-accum: scale each micro-batch's masked mean by its
        # share of the effective batch's active tokens (n_i / N) so the
        # accumulated gradient is the true token-weighted mean over all accum
        # micro-batches, not a mean of per-micro-batch means (which biases toward
        # micro-batches with fewer active tokens under SFT masking). At accum=1
        # this is loss * 1.0 == the previous loss / 1, so it is bitwise-identical
        # to the old path on the single-micro-batch tests (incl. resume).
        batches = [
            get_sft_batch(examples, micro_bs, context, seed=train["seed"],
                          step=step, micro_step=micro_step, device=device)
            for micro_step in range(accum)
        ]
        counts = [int(mask.sum().item()) for _, _, mask in batches]
        total_active = max(sum(counts), 1)
        loss_sum = 0.0
        for (x, y, mask), n in zip(batches, counts):
            with autocast:
                logits = model(x)
                loss = masked_lm_loss(logits, y, mask)  # mean over this micro-batch
            scaler.scale(loss * (n / total_active)).backward()
            loss_sum += loss.item() * n
            tokens_seen += n  # cumulative active target tokens
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        loss_value = loss_sum / total_active  # token-weighted masked mean over the batch
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
        "stage": "sft", "package_version": __version__,
        "final_step": steps, "final_loss": loss_value,
        "tokens_seen": tokens_seen,
        "examples_path": config["data"]["examples_path"],
        "n_examples": len(examples),
        "config": config,
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
