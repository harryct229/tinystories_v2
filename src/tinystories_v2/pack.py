"""Packed Pretraining data: Fable text -> flat uint16 token-ID binary.

Format (documented contract, see <out>.json manifest):
    - dtype uint16 little-endian (vocab 8192 < 65536), flat 1-D array
    - each Fable's token IDs followed by the <|end|> ID, docs concatenated
    - no header; length in tokens = file size / 2 and is recorded in the manifest

Batches for the training loop are a pure function of (seed, step, micro_step)
so an interrupted run resumed from a checkpoint sees exactly the batches the
uninterrupted run would have seen (checkpoint-resume contract).
"""

import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from tinystories_v2 import __version__
from tinystories_v2.slots import SLOT_SPECIAL_TOKENS

END_TOKEN = SLOT_SPECIAL_TOKENS[-1]  # "<|end|>"


def pack_split(split_path: str | Path, tokenizer_path: str | Path,
               out_path: str | Path) -> dict:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    end_id = tokenizer.token_to_id(END_TOKEN)
    if end_id is None:
        raise ValueError(f"tokenizer at {tokenizer_path} lacks the {END_TOKEN} token")
    if tokenizer.get_vocab_size() > 2**16:
        raise ValueError("vocab does not fit uint16")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_docs = n_tokens = 0
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(split_path, encoding="utf-8") as src, open(tmp, "wb") as dst:
        for line in src:
            if not line.strip():
                continue
            ids = tokenizer.encode(json.loads(line)["fable"]).ids + [end_id]
            np.asarray(ids, dtype=np.uint16).tofile(dst)
            n_docs += 1
            n_tokens += len(ids)
    tmp.replace(out_path)

    manifest = {
        "stage": "pack",
        "package_version": __version__,
        "dtype": "uint16",
        "n_tokens": n_tokens,
        "n_docs": n_docs,
        "vocab_size": tokenizer.get_vocab_size(),
        "end_id": end_id,
    }
    Path(str(out_path) + ".json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def load_packed(path: str | Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data: np.memmap, micro_batch_size: int, context: int, *,
              seed: int, step: int, micro_step: int,
              device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(((seed * 1_000_003 + step) * 1_009 + micro_step) % 2**63)
    offsets = torch.randint(0, len(data) - context - 1, (micro_batch_size,),
                            generator=generator)
    x = torch.stack([
        torch.from_numpy(data[o:o + context].astype(np.int64)) for o in offsets
    ])
    y = torch.stack([
        torch.from_numpy(data[o + 1:o + 1 + context].astype(np.int64)) for o in offsets
    ])
    return x.to(device), y.to(device)
