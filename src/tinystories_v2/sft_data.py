"""SFT dataset builder: the data-prep `sft` split -> masked-loss training
examples (issue 12).

Invoke standalone:
    ts2-sft-data --config configs/sft_data_fixture.toml
    (or: python -m tinystories_v2.sft_data --config ...)

Reads the tokenizer artifact and the `sft` split; writes, per the stage
convention:

Artifacts in <out_dir>:
    examples.jsonl   one training example per line: prompt_hash, input_ids,
                     loss_mask, n_prompt_tokens
                     (schema: docs/schemas/sft-example-v1.md)
    manifest.json    stage, version, count, config

Deterministic: examples are emitted in split order with no randomness, so two
runs on the same inputs are byte-identical.
"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from tinystories_v2 import __version__
from tinystories_v2.config import load_config
from tinystories_v2.slot_prompt import SLOT_FIELDS, encode_example
from tinystories_v2.slots import Scaffold

SCHEMA_VERSION = 1


def _read_split(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_example_record(tokenizer: Tokenizer, row: dict) -> dict:
    scaffold = Scaffold(**{field: row[field] for field in SLOT_FIELDS})
    example = encode_example(tokenizer, scaffold, row["fable"])
    return {"prompt_hash": row["prompt_hash"], **example.to_dict()}


def run(config: dict) -> None:
    tokenizer = Tokenizer.from_file(config["tokenizer"])
    split_path = Path(config["sft_split"])
    max_examples = config.get("max_examples", 0)

    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    examples_path = out_dir / "examples.jsonl"
    out = examples_path.open("w", encoding="utf-8")
    try:
        for row in _read_split(split_path):
            record = build_example_record(tokenizer, row)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if max_examples and count >= max_examples:
                break
    except Exception:
        out.close()
        examples_path.unlink(missing_ok=True)
        raise
    else:
        out.close()

    manifest = {
        "stage": "sft_data",
        "package_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "count": count,
        "config": config,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
