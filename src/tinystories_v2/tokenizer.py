"""Tokenizer stage: byte-level BPE with Slot Prompt tokens reserved (ADR-0003).

Invoke standalone:
    ts2-tokenizer --config configs/tokenizer_fixture.toml
    (or: python -m tinystories_v2.tokenizer --config ...)

Artifacts in <out_dir>:
    tokenizer.json    load with tokenizers.Tokenizer.from_file
    manifest.json     stage, version, vocab size, special tokens, config

The special tokens come from slots.SLOT_SPECIAL_TOKENS, not from config:
the reserved set is an ADR-0003 invariant, not a tunable.
"""

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from tinystories_v2 import __version__
from tinystories_v2.config import load_config
from tinystories_v2.slots import SLOT_SPECIAL_TOKENS


def iter_corpus(paths: Iterable[str], text_field: str, max_docs: int = 0) -> Iterator[str]:
    seen = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)[text_field]
                seen += 1
                if max_docs and seen >= max_docs:
                    return


def train_tokenizer(texts: Iterator[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SLOT_SPECIAL_TOKENS),
        # Full byte alphabet up front -> any text round-trips losslessly,
        # even bytes absent from the training sample.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def run(config: dict) -> None:
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = iter_corpus(
        config["corpus"], config.get("text_field", "fable"), config.get("max_docs", 0)
    )
    tokenizer = train_tokenizer(texts, config["vocab_size"])
    tokenizer.save(str(out_dir / "tokenizer.json"))
    manifest = {
        "stage": "tokenizer",
        "package_version": __version__,
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": list(SLOT_SPECIAL_TOKENS),
        "config": config,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
