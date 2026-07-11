"""Held-out perplexity for any next-token model (toy or real checkpoints).

Contract: model(input_ids) maps a (batch, seq_len) int64 tensor to
(batch, seq_len, vocab_size) logits. The flat token id sequence is split
into non-overlapping blocks of block_size, so every token after the first
is predicted exactly once. PyTorch is imported lazily: importing this
module (like tinystories_v2.metrics) never requires torch.
"""

import itertools
import math
from typing import Any


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "perplexity requires PyTorch; "
            "install with: uv pip install -e '.[dev]'"
        ) from exc
    return torch


def _equal_length_batches(
    blocks: list[tuple[Any, Any]],
    batch_size: int,
):
    """Batch consecutive equal-length blocks (only the last can be short)."""
    for _, group in itertools.groupby(
        blocks, key=lambda pair: pair[0].numel()
    ):
        members = list(group)
        for start in range(0, len(members), batch_size):
            yield members[start : start + batch_size]


def perplexity(
    model: Any,
    token_ids: Any,
    *,
    block_size: int,
    batch_size: int = 8,
    device: str = "cpu",
) -> float:
    """exp(mean next-token NLL) of a flat held-out token id sequence.

    Works with any checkpoint or toy module satisfying the logits
    contract in the module docstring. Inputs are moved to `device`; the
    model must already live there and be in eval mode. Deterministic:
    no sampling, evaluated under torch.inference_mode().
    """
    torch = _require_torch()
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    ids = torch.as_tensor(token_ids, dtype=torch.long)
    if ids.dim() != 1:
        raise ValueError("token_ids must be a flat 1-D sequence")
    if ids.numel() < 2:
        raise ValueError("token_ids must hold at least two tokens")

    blocks = []
    for start in range(0, ids.numel() - 1, block_size):
        targets = ids[start + 1 : start + block_size + 1]
        inputs = ids[start : start + block_size][: targets.numel()]
        blocks.append((inputs, targets))

    total_nll = 0.0
    total_targets = 0
    with torch.inference_mode():
        for batch in _equal_length_batches(blocks, batch_size):
            inputs = torch.stack([pair[0] for pair in batch]).to(device)
            targets = torch.stack([pair[1] for pair in batch]).to(device)
            logits = model(inputs)
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_nll += float(nll)
            total_targets += targets.numel()
    return math.exp(total_nll / total_targets)
