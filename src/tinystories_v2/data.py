"""Data-prep stage: TF1-EN-3M -> slot-extracted, disjoint-by-fable splits.

Invoke standalone:
    ts2-data-prep --config configs/data_prep_fixture.toml
    (or: python -m tinystories_v2.data --config ...)

Artifacts in <out_dir>:
    splits/{pretrain,sft,pref,eval}.jsonl   one row per Fable:
        prompt_hash + six Scaffold slots + fable text
    membership.json                         {split: [prompt_hash, ...]}
    manifest.json                           stage, version, counts, config

Split membership is a pure function of (seed, prompt_hash), so runs are
deterministic and Fables sharing a Scaffold never straddle splits.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from tinystories_v2 import __version__
from tinystories_v2.config import load_config, load_env
from tinystories_v2.slots import SlotExtractionError, extract_slots

SPLIT_NAMES = ("pretrain", "sft", "pref", "eval")


def assign_split(prompt_hash: str, fractions: dict[str, float], seed: str) -> str | None:
    digest = hashlib.sha256(f"{seed}:{prompt_hash}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    upper = 0.0
    for name in SPLIT_NAMES:
        upper += fractions[name]
        if position < upper:
            return name
    return None  # remainder of the corpus stays unused


def iter_source(source: dict):
    kind = source["kind"]
    if kind == "jsonl":
        with open(source["path"], encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif kind == "hub":
        from datasets import load_dataset  # network path; tests only use jsonl

        load_env()  # HF token, if any, from environment/.env — never printed
        rows = load_dataset(source["dataset"], split=source["split"], streaming=True)
        limit = source.get("limit", 0)
        for i, row in enumerate(rows):
            if limit and i >= limit:
                break
            yield {k: row[k] for k in ("prompt_hash", "prompt", "fable")}
    else:
        raise ValueError(f"unknown source kind: {kind!r}")


def run(config: dict) -> None:
    fractions = {name: float(config["splits"][name]) for name in SPLIT_NAMES}
    if sum(fractions.values()) > 1.0 + 1e-9:
        raise ValueError(f"split fractions sum to more than 1: {fractions}")
    seed = config["splits"]["seed"]
    max_failures = config.get("max_extraction_failures", 0)

    out_dir = Path(config["out_dir"])
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    membership: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    failures = 0
    writers = {
        name: (splits_dir / f"{name}.jsonl").open("w", encoding="utf-8")
        for name in SPLIT_NAMES
    }
    try:
        for record in iter_source(config["source"]):
            split = assign_split(record["prompt_hash"], fractions, seed)
            if split is None:
                continue
            try:
                scaffold = extract_slots(record["prompt"])
            except SlotExtractionError:
                failures += 1
                if failures > max_failures:
                    raise
                continue
            row = {"prompt_hash": record["prompt_hash"], **asdict(scaffold), "fable": record["fable"]}
            writers[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            membership[split].append(record["prompt_hash"])
    finally:
        for writer in writers.values():
            writer.close()

    (out_dir / "membership.json").write_text(json.dumps(membership, indent=2), encoding="utf-8")
    manifest = {
        "stage": "data_prep",
        "package_version": __version__,
        "counts": {name: len(ids) for name, ids in membership.items()},
        "skipped_extraction_failures": failures,
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
