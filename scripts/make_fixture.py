"""One-time fixture builder: downloads the first 120 real TF1-EN-3M records.

Writes tests/fixtures/tf1_sample.jsonl with only the fields the pipeline uses
(prompt_hash, prompt, fable). Needs network; the dataset is public so the
HF token in .env is optional but loaded for parity with real stage runs.
The committed output is what the offline test suite runs on.

Run from the repo root:  .venv/bin/python scripts/make_fixture.py
"""

import json
from pathlib import Path

from datasets import load_dataset

from tinystories_v2.config import load_env

N_RECORDS = 120
OUT = Path("tests/fixtures/tf1_sample.jsonl")


def main() -> None:
    load_env()
    rows = load_dataset("klusai/ds-tf1-en-3m", split="train", streaming=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for i, record in enumerate(rows):
            if i >= N_RECORDS:
                break
            row = {k: record[k] for k in ("prompt_hash", "prompt", "fable")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {N_RECORDS} records to {OUT}")


if __name__ == "__main__":
    main()
