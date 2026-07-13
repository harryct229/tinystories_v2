"""One-command real GRPO run for Colab (issue 06).

Turns a fresh L4 VM into a running GRPO job with a single command. Idempotent:
safe to re-run after an L4 preemption — it skips already-present artifacts and
`ts2-grpo --resume` continues from the last Hub checkpoint.

Steps:
  1. load .env secrets (HF_TOKEN, WANDB_API_KEY) so Hub download/sync + W&B work
  2. download tokenizer.json (tokenizer repo) + splits/pref.jsonl (data repo) —
     the rollout prompts — if the local copies are absent
  3. run the GRPO stage (ts2-grpo, resume=True): enforces issue 05's accuracy
     gate, fetches the SFT checkpoint via [init].hub_source (policy + reference)
     and the Reward Model via [reward].hub_source, resumes any prior GRPO
     checkpoint from the Hub, optimizes the policy, and checkpoints back to the
     Hub every checkpoint_every steps

Run on the VM (in-kernel, survives disconnects — never nohup-detach):
    python scripts/grpo_colab.py            # download + train
    python scripts/grpo_colab.py --skip-train   # download only
    colab exec -f scripts/grpo_colab.py

See docs/colab-notes.md for the CLI gotchas (push main first, .env via upload,
background + poll long commands, retries, always stop the VM). The pref split
lives in issue 01's data repo (`DATA_REPO`); edit `DATA_REPO` /
`PREF_SPLIT_FILENAME` in this file if it moves.
"""

import argparse
from pathlib import Path

from tinystories_v2 import grpo
from tinystories_v2.config import load_config, load_env
from tinystories_v2.hub_download import download_file  # noqa: F401 — monkeypatched in tests

TOKENIZER_REPO = "congthanh991/tinystories-v2-tokenizer"
DATA_REPO = "congthanh991/tinystories-v2-data"
PREF_SPLIT_FILENAME = "splits/pref.jsonl"
DEFAULT_GRPO_CONFIG = "configs/grpo_full.toml"


def prepare(grpo_config, *, download=None) -> Path:
    """Ensure the tokenizer + pref split are present (download if missing).
    Returns the local pref-split path. `download` is injectable for tests; it
    defaults to download_file resolved at call time. Idempotent: each step is
    guarded by an existence check, so re-running on a warm VM is a no-op up to
    training."""
    if download is None:
        download = download_file
    cfg = load_config(grpo_config)
    tokenizer_path = Path(cfg["data"]["tokenizer_path"])   # artifacts/tokenizer_full/tokenizer.json
    pref_split = Path(cfg["data"]["pref_split"])           # artifacts/data_prep_full/splits/pref.jsonl

    if not tokenizer_path.exists():
        print(f"[grpo_colab] downloading tokenizer -> {tokenizer_path}")
        download(TOKENIZER_REPO, "tokenizer.json", tokenizer_path.parent)
    if not pref_split.exists():
        print(f"[grpo_colab] downloading pref split -> {pref_split}")
        # download_file lands the repo-relative "splits/pref.jsonl" under the
        # given dir, so pass the split dir's parent (the data-prep root).
        download(DATA_REPO, PREF_SPLIT_FILENAME, pref_split.parent.parent)
    return pref_split


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grpo-config", default=DEFAULT_GRPO_CONFIG, type=Path)
    parser.add_argument("--skip-train", action="store_true",
                        help="download the tokenizer + pref split only, then stop")
    args = parser.parse_args(argv)

    load_env()  # HF_TOKEN / WANDB_API_KEY reach Hub sync + wandb; never printed
    prepare(args.grpo_config)
    if args.skip_train:
        print("[grpo_colab] --skip-train: inputs ready; skipping training")
        return
    print("[grpo_colab] starting GRPO training (ts2-grpo --resume)")
    summary = grpo.run(load_config(args.grpo_config), resume=True)
    print(f"[grpo_colab] done: step {summary['step']}, loss {summary['loss']:.4f}, "
          f"mean reward {summary['reward_mean']:.4f}, KL {summary['kl']:.4f}")


if __name__ == "__main__":
    main()
