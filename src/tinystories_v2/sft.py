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

import json
from pathlib import Path

import torch


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
