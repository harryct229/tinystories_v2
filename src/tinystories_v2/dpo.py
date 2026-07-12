"""DPO fallback stage: fine-tune the SFT policy directly on Judge preference
pairs against a frozen SFT reference, with a hand-written DPO loss (issue 08).

Invoke standalone:
    ts2-dpo --config configs/dpo_fixture.toml [--resume]
    (or: python -m tinystories_v2.dpo --config ...)

The pre-committed stage-3 fallback (ADR-0004): if GRPO is unstable or the Reward
Model can't clear its gate by the schedule checkpoint, ship DPO as the aligned
model. It consumes the *identical* preference-pair artifact as the Reward Model
(issue 05) and produces a plain FableLM checkpoint that is a drop-in third model
for the eval suite (issue 07). Reuses issue 02's checkpoint-resume contract,
optimizer conventions (build_optimizer), LR schedule (lr_at), precision knob,
W&B logging, and Hub sync verbatim (ADR-0005: libraries only at the edges).

Both the policy and the frozen reference initialize from the SFT checkpoint in
[init]; the reference is always re-derived from that fixed checkpoint (never a
resumed policy), so it is a pure function of [init] and resume stays bitwise.

Artifacts in <out_dir> (schema: docs/schemas/dpo-artifact-v1.md):
    checkpoints/step_XXXXXX.pt   full training state (atomic; resume contract)
    metrics.jsonl                one line per log_every steps: loss, lr, margin, pairs_seen
    manifest.json                stage, version, final step/loss, heldout_margin,
                                 beta, pair_split recipe, pairs_path, config
"""

import argparse
import json
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from tinystories_v2 import __version__
from tinystories_v2.checkpoint import (
    latest_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint,
)
from tinystories_v2.config import load_config, load_env
from tinystories_v2.hub import fetch_from, try_sync_to
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.pretrain import build_optimizer, lr_at
from tinystories_v2.reward import load_pairs, split_pairs
from tinystories_v2.slot_prompt import encode_example
from tinystories_v2.tracking import MetricsLogger


def sequence_logprobs(logits: torch.Tensor, y: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """Sum of per-token target log-probs over active (mask==1) positions.

    logits [B, T, V] are next-token scores for inputs x = ids[:-1]; y [B, T] are
    the shifted targets ids[1:]; mask [B, T] is 1 over the fable body + <|end|>
    and 0 over the prompt prefix and right-padding. Returns [B]: the completion
    log-probability log p(completion | prompt) the model assigns to each row."""
    logp = F.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)   # [B, T]
    return (token_logp * mask).sum(dim=-1)                       # [B]


def implicit_reward_margins(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
                            ref_chosen: torch.Tensor, ref_rejected: torch.Tensor,
                            beta: float) -> torch.Tensor:
    """Per-pair DPO implicit-reward margin (Rafailov et al. 2023):
    beta * [ (logπ_c - logπ_ref_c) - (logπ_r - logπ_ref_r) ]. Positive means the
    policy prefers chosen over rejected more than the frozen reference does. [B]."""
    return beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor,
             beta: float) -> torch.Tensor:
    """-log σ(beta * [(logπ_c - logπ_r) - (logπ_ref_c - logπ_ref_r)]), averaged
    (ADR-0005, hand-written; no TRL DPOTrainer). Minimized when the policy raises
    the chosen-minus-rejected completion log-ratio above the frozen reference's."""
    logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    return -F.logsigmoid(beta * logits).mean()


def encode_pairs(tokenizer, pairs: list) -> list[dict]:
    """Precompute (ids, loss_mask) for the chosen and rejected completion of each
    pair via the Slot Prompt encoder: each is <|character|>…<|fable|>{body}<|end|>
    with the mask active over the body + <|end|> only (encode_example)."""
    encoded = []
    for pair in pairs:
        chosen = encode_example(tokenizer, pair.scaffold, pair.chosen)
        rejected = encode_example(tokenizer, pair.scaffold, pair.rejected)
        encoded.append({"chosen_ids": chosen.input_ids, "chosen_mask": chosen.loss_mask,
                        "rejected_ids": rejected.input_ids,
                        "rejected_mask": rejected.loss_mask})
    return encoded


def _pad_shifted(ids_list: list[list[int]], mask_list: list[list[int]],
                 context: int, device: str) -> tuple[torch.Tensor, ...]:
    """Right-pad (ids, loss_mask) rows into next-token (x, y, mask) tensors. Each
    row is truncated to context+1 ids, then shifted: x = ids[:-1], y = ids[1:],
    mask = loss_mask[1:] (active over body + <|end|>). Rows are padded to the
    batch's longest x with id 0 / mask 0; causal attention makes right-padding
    safe and padding never contributes to a completion log-prob."""
    rows = []
    for ids, m in zip(ids_list, mask_list):
        ids, m = ids[:context + 1], m[:context + 1]
        rows.append((ids[:-1], ids[1:], m[1:]))
    width = max(len(x) for x, _, _ in rows)
    xs, ys, ms = [], [], []
    for x, y, m in rows:
        pad = width - len(x)
        xs.append(x + [0] * pad)
        ys.append(y + [0] * pad)
        ms.append([float(v) for v in m] + [0.0] * pad)
    return (torch.tensor(xs, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device),
            torch.tensor(ms, dtype=torch.float, device=device))


def get_pair_batch(train: list[dict], micro_batch_size: int, context: int, *,
                   seed: int, step: int, micro_step: int,
                   device: str = "cpu") -> tuple[tuple, tuple]:
    """A (chosen_xyz, rejected_xyz) micro-batch sampled with replacement; a pure
    function of (seed, step, micro_step) so a resumed run replays it (resume
    contract). chosen and rejected are padded independently."""
    generator = torch.Generator()
    generator.manual_seed(((seed * 1_000_003 + step) * 1_009 + micro_step) % 2**63)
    picks = torch.randint(0, len(train), (micro_batch_size,),
                          generator=generator).tolist()
    chosen = _pad_shifted([train[i]["chosen_ids"] for i in picks],
                          [train[i]["chosen_mask"] for i in picks], context, device)
    rejected = _pad_shifted([train[i]["rejected_ids"] for i in picks],
                            [train[i]["rejected_mask"] for i in picks], context, device)
    return chosen, rejected


def _load_sft_state(config: dict, device: str) -> dict:
    """Load the SFT checkpoint declared in [init], fetching the artifact from the
    Hub first if the local checkpoint is absent (fresh VM), and validate its
    architecture matches [model]. Returns the loaded checkpoint state (contains
    'model'); both the policy and the frozen reference are built from it."""
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
    if ModelConfig(**state["config"]["model"]) != ModelConfig(**config["model"]):
        raise ValueError(
            f"[model] does not match the SFT checkpoint at {init_ckpt}; DPO must "
            f"fine-tune the SFT architecture")
    print(f"loaded SFT weights from {init_ckpt}")
    return state


def _build_model(config: dict, state: dict, device: str) -> FableLM:
    """Build a FableLM from [model] and load the SFT weights (strict)."""
    model = FableLM(ModelConfig(**config["model"])).to(device)
    model.load_state_dict(state["model"])
    return model


@torch.no_grad()
def evaluate_margin(policy: FableLM, reference: FableLM, holdout: list[dict],
                    context: int, beta: float, *, device: str = "cpu",
                    batch_size: int = 32) -> float:
    """Mean held-out implicit-reward margin. > 0 means the policy shifted toward
    the chosen completions relative to the frozen SFT reference. NaN for an empty
    holdout. Both models are read in eval mode with no grad."""
    if not holdout:
        return float("nan")
    was_training = policy.training
    policy.eval()
    reference.eval()
    margins = []
    for start in range(0, len(holdout), batch_size):
        chunk = holdout[start:start + batch_size]
        cx, cy, cm = _pad_shifted([p["chosen_ids"] for p in chunk],
                                  [p["chosen_mask"] for p in chunk], context, device)
        rx, ry, rm = _pad_shifted([p["rejected_ids"] for p in chunk],
                                  [p["rejected_mask"] for p in chunk], context, device)
        pc = sequence_logprobs(policy(cx), cy, cm)
        pr = sequence_logprobs(policy(rx), ry, rm)
        rc = sequence_logprobs(reference(cx), cy, cm)
        rr = sequence_logprobs(reference(rx), ry, rm)
        margins.append(implicit_reward_margins(pc, pr, rc, rr, beta))
    if was_training:
        policy.train()
    return torch.cat(margins).mean().item()


def run(config: dict, resume: bool = False) -> dict:
    load_env()  # W&B/HF keys before wandb.init or hub sync; never printed
    train = config["train"]
    out_dir = Path(config["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    hub_target = config.get("hub", {}).get("target")
    beta = config["dpo"]["beta"]

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
    # Policy and frozen reference both start from the fixed SFT checkpoint. The
    # reference is re-derived here on every run (fresh or resume), never stored
    # in the DPO checkpoint, so it stays a pure function of [init].
    sft_state = _load_sft_state(config, device)
    policy = _build_model(config, sft_state, device)
    reference = _build_model(config, sft_state, device).requires_grad_(False)
    reference.eval()
    optimizer = build_optimizer(policy, train["peak_lr"],
                                (train["beta1"], train["beta2"]),
                                train["weight_decay"])

    start_step, pairs_seen = 0, 0
    if resume:
        if latest_checkpoint(ckpt_dir) is None and hub_target:
            try:
                fetch_from(hub_target, out_dir)  # fresh VM: pull previous session
            except Exception as err:  # noqa: BLE001 — first run: repo may not exist yet
                warnings.warn(
                    f"could not fetch a prior DPO run from {hub_target!r}; "
                    f"starting fresh: {err}", stacklevel=2)
        ckpt_path = latest_checkpoint(ckpt_dir)
        if ckpt_path is not None:
            state = load_checkpoint(ckpt_path)
            policy.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start_step, pairs_seen = state["step"], state["pairs_seen"]
            print(f"resumed from {ckpt_path.name} at step {start_step}")

    def checkpoint(step: int) -> None:
        save_checkpoint(ckpt_dir, step, {
            "step": step, "pairs_seen": pairs_seen,
            "model": policy.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "config": config,
        })
        prune_checkpoints(ckpt_dir, train.get("keep_last", 0))
        if hub_target:
            try_sync_to(hub_target, out_dir)

    logger = MetricsLogger(out_dir, config.get("wandb"))
    steps, accum = train["steps"], train["grad_accum"]
    micro_bs, context = train["micro_batch_size"], config["model"]["context"]
    loss_value, batch_margin = float("nan"), float("nan")
    policy.train()
    for step in range(start_step, steps):
        lr = lr_at(step, steps, train["peak_lr"],
                   train["warmup_frac"], train["min_lr_frac"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(accum):
            (cx, cy, cm), (rx, ry, rm) = get_pair_batch(
                train_pairs, micro_bs, context, seed=train["seed"],
                step=step, micro_step=micro_step, device=device)
            with autocast:
                pc = sequence_logprobs(policy(cx), cy, cm)
                pr = sequence_logprobs(policy(rx), ry, rm)
                with torch.no_grad():
                    rc = sequence_logprobs(reference(cx), cy, cm)
                    rr = sequence_logprobs(reference(rx), ry, rm)
                loss = dpo_loss(pc, pr, rc, rr, beta)
            scaler.scale(loss / accum).backward()
            pairs_seen += micro_bs
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), train["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        loss_value = loss.item()          # last micro-batch DPO loss
        batch_margin = implicit_reward_margins(
            pc.detach(), pr.detach(), rc, rr, beta).mean().item()
        done = step + 1
        if done % train["log_every"] == 0:
            logger.log({"loss": loss_value, "lr": lr, "margin": batch_margin,
                        "pairs_seen": pairs_seen}, step=done)
        if done % train["checkpoint_every"] == 0:
            checkpoint(done)
    if steps % train["checkpoint_every"] != 0:
        checkpoint(steps)
    logger.finish()

    heldout_margin = evaluate_margin(policy, reference, holdout_pairs,
                                     context, beta, device=device)

    manifest = {
        "stage": "dpo", "package_version": __version__,
        "final_step": steps, "final_loss": loss_value,
        "heldout_margin": heldout_margin, "beta": beta,
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
    print(f"held-out reward margin: {heldout_margin:.4f} (beta {beta})")
    return {"step": steps, "loss": loss_value, "heldout_margin": heldout_margin}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest checkpoint in out_dir")
    args = parser.parse_args(argv)
    run(load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
