"""Atomic training-state checkpoints (checkpoint-resume contract).

A checkpoint file only ever appears under its final name via os.replace, so a
process killed mid-write can never leave a corrupt step_*.pt behind — resume
always finds the last complete state. State dicts stay weights_only-safe
(plain containers of tensors/numbers/strings) so loading never unpickles code.
"""

import re
from pathlib import Path

import torch

_STEP_RE = re.compile(r"step_(\d{6})\.pt$")


def save_checkpoint(ckpt_dir: Path, step: int, state: dict) -> Path:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final = ckpt_dir / f"step_{step:06d}.pt"
    tmp = final.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    tmp.replace(final)
    return final


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    steps = {
        int(m.group(1)): p
        for p in ckpt_dir.iterdir()
        if (m := _STEP_RE.fullmatch(p.name))
    }
    return steps[max(steps)] if steps else None


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def prune_checkpoints(ckpt_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    paths = sorted(Path(ckpt_dir).glob("step_*.pt"))
    for path in paths[:-keep_last]:
        path.unlink()
