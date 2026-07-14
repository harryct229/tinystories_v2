"""One-command real evaluation run for Colab (issue 07).

Turns a fresh L4 VM into a running eval job with a single command. Idempotent:
re-running after an L4 preemption skips already-present downloads and re-runs
the (single-pass) eval stage. The stage pulls each stage checkpoint from its
[[stages]].hub_source on a fresh VM.

Steps:
  1. load .env secrets (HF_TOKEN) so Hub download/sync work
  2. download tokenizer.json + the held-out eval split (issue 01's data repo)
     from the Hub (retry-wrapped) if the local copies are absent
  3. run the eval stage (ts2-eval): generate per-stage completions, score
     cross-family win-rates, compute reference-free metrics + perplexity, and
     write results.json + report.md (synced to [hub].target when configured)

Run on the VM (in-kernel, survives disconnects — never nohup-detach):
    python scripts/eval_colab.py            # download + eval
    python scripts/eval_colab.py --skip-eval    # download only
    colab exec -f scripts/eval_colab.py

See docs/colab-notes.md for the CLI gotchas. The eval split lives in issue 01's
data repo (`DATA_REPO`); edit the constants below if it moves.
"""

import argparse
from pathlib import Path

from tinystories_v2 import eval
from tinystories_v2.config import load_config, load_env
from tinystories_v2.hub_download import download_file  # noqa: F401 — monkeypatched in tests

TOKENIZER_REPO = "congthanh991/tinystories-v2-tokenizer"
DATA_REPO = "congthanh991/tinystories-v2-data"
EVAL_FILENAME = "splits/eval.jsonl"
DEFAULT_EVAL_CONFIG = "configs/eval_full.toml"


def prepare(eval_config, *, download=None) -> tuple[Path, Path]:
    """Ensure the tokenizer + eval split are present (download if missing).
    Returns (tokenizer_path, eval_split_path). `download` is injectable for
    tests; it defaults to download_file. Idempotent: each step is guarded by an
    existence check, so re-running on a warm VM is a no-op up to the eval run."""
    if download is None:
        download = download_file
    cfg = load_config(eval_config)
    tokenizer_path = Path(cfg["data"]["tokenizer"])
    eval_path = Path(cfg["data"]["eval_split"])   # .../<data_dir>/splits/eval.jsonl

    if not tokenizer_path.exists():
        print(f"[eval_colab] downloading tokenizer -> {tokenizer_path}")
        download(TOKENIZER_REPO, "tokenizer.json", tokenizer_path.parent)
    if not eval_path.exists():
        print(f"[eval_colab] downloading eval split -> {eval_path}")
        # download_file writes to local_dir/filename, so the data dir (parent of
        # splits/) is the local_dir and the filename keeps its "splits/" prefix.
        download(DATA_REPO, EVAL_FILENAME, eval_path.parent.parent)
    return tokenizer_path, eval_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG, type=Path)
    parser.add_argument("--skip-eval", action="store_true",
                        help="download the tokenizer + eval split only, then stop")
    args = parser.parse_args(argv)

    load_env()  # HF_TOKEN reaches Hub download/sync; never printed
    prepare(args.eval_config)
    if args.skip_eval:
        print("[eval_colab] --skip-eval: inputs ready; skipping eval")
        return
    print("[eval_colab] starting evaluation (ts2-eval --resume)")
    # resume=True: reuse cached completions and logged judgments from a prior
    # preempted session (fetched from [hub].target on a fresh VM).
    eval.run(load_config(args.eval_config), resume=True)
    print("[eval_colab] done: results.json + report.md written")


if __name__ == "__main__":
    main()
