"""One-command real Reward Model run for Colab (issue 05).

Turns a fresh L4 VM into a running Reward Model job with a single command.
Idempotent: safe to re-run after an L4 preemption — it skips already-present
artifacts and `ts2-reward --resume` continues from the last Hub checkpoint.

Steps:
  1. load .env secrets (HF_TOKEN, WANDB_API_KEY) so Hub download/sync + W&B work
  2. download tokenizer.json + the preference pairs (issue 04's artifact) from
     the Hub (retry-wrapped) if the local copies are absent
  3. run the Reward Model stage (ts2-reward, resume=True): fetches the SFT
     checkpoint via [init].hub_source, resumes any prior Reward Model checkpoint
     from the Hub, trains with Bradley-Terry loss, and checkpoints back to the
     Hub every checkpoint_every steps

Run on the VM (in-kernel, survives disconnects — never nohup-detach):
    python scripts/reward_colab.py            # download + train
    python scripts/reward_colab.py --skip-train   # download only
    colab exec -f scripts/reward_colab.py

See docs/colab-notes.md for the CLI gotchas (push main first, .env via upload,
background + poll long commands, retries, always stop the VM). The preference
pairs live in issue 04's Hub repo (`PAIRS_REPO`); edit `PAIRS_REPO` /
`PAIRS_FILENAME` in this file if it moves.
"""

import argparse
from pathlib import Path

from tinystories_v2 import reward
from tinystories_v2.config import load_config, load_env
from tinystories_v2.hub_download import download_file  # noqa: F401 — monkeypatched in tests

TOKENIZER_REPO = "congthanh991/tinystories-v2-tokenizer"
PAIRS_REPO = "congthanh991/tinystories-v2-pref-pairs"   # issue 04's preference artifact
PAIRS_FILENAME = "pairs.jsonl"
DEFAULT_REWARD_CONFIG = "configs/reward_full.toml"


def prepare(reward_config, *, download=None) -> Path:
    """Ensure the tokenizer + preference pairs are present (download if missing).
    Returns the local pairs path. `download` is injectable for tests; it defaults
    to download_file resolved at call time. Idempotent: each step is guarded by
    an existence check, so re-running on a warm VM is a no-op up to training."""
    if download is None:
        download = download_file
    cfg = load_config(reward_config)
    tokenizer_path = Path(cfg["data"]["tokenizer_path"])   # artifacts/tokenizer_full/tokenizer.json
    pairs_path = Path(cfg["data"]["pairs_path"])           # artifacts/pref_full/pairs.jsonl

    if not tokenizer_path.exists():
        print(f"[reward_colab] downloading tokenizer -> {tokenizer_path}")
        download(TOKENIZER_REPO, "tokenizer.json", tokenizer_path.parent)
    if not pairs_path.exists():
        print(f"[reward_colab] downloading preference pairs -> {pairs_path}")
        download(PAIRS_REPO, PAIRS_FILENAME, pairs_path.parent)
    return pairs_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reward-config", default=DEFAULT_REWARD_CONFIG, type=Path)
    parser.add_argument("--skip-train", action="store_true",
                        help="download the tokenizer + pairs only, then stop")
    args = parser.parse_args(argv)

    load_env()  # HF_TOKEN / WANDB_API_KEY reach Hub sync + wandb; never printed
    prepare(args.reward_config)
    if args.skip_train:
        print("[reward_colab] --skip-train: inputs ready; skipping training")
        return
    print("[reward_colab] starting Reward Model training (ts2-reward --resume)")
    summary = reward.run(load_config(args.reward_config), resume=True)
    print(f"[reward_colab] done: step {summary['step']}, loss {summary['loss']:.4f}, "
          f"held-out accuracy {summary['heldout_accuracy']:.3f}")


if __name__ == "__main__":
    main()
