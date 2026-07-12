"""One-command real SFT run for Colab (issue 03).

Turns a fresh L4 VM into a running SFT job with a single command. Idempotent:
safe to re-run after an L4 preemption — it skips already-present artifacts and
`ts2-sft --resume` continues from the last Hub checkpoint.

Steps:
  1. load .env secrets (HF_TOKEN, WANDB_API_KEY) so Hub download/sync + W&B work
  2. download tokenizer.json + splits/sft.jsonl from the Hub (retry-wrapped) if
     the local copies are absent
  3. build the SFT dataset (the ts2-sft-data stage) if examples.jsonl is absent
  4. run SFT (the ts2-sft stage, resume=True): fetches the Pretraining
     checkpoint via [init].hub_source, resumes any prior SFT checkpoint from the
     Hub, trains, and checkpoints back to the Hub every checkpoint_every steps

Run on the VM:
    python scripts/sft_colab.py            # download + build + train
    python scripts/sft_colab.py --skip-train   # download + build only

Or via the Colab CLI (in-kernel, survives disconnects — never nohup-detach):
    colab exec -f scripts/sft_colab.py

The pretraining Hub artifacts (tokenizer, split, checkpoint) live under
`congthanh991`; override the repo ids with --tokenizer-repo / --data-repo if
they move.
"""

import argparse
import time
from pathlib import Path

from tinystories_v2 import sft, sft_data
from tinystories_v2.config import load_config, load_env

TOKENIZER_REPO = "congthanh991/tinystories-v2-tokenizer"
DATA_REPO = "congthanh991/tinystories-v2-data"
DEFAULT_SFT_DATA_CONFIG = "configs/sft_data_full.toml"
DEFAULT_SFT_CONFIG = "configs/sft_full.toml"


def retry(fn, *, attempts: int = 5, base_delay: float = 2.0, what: str = "operation"):
    """Call fn(), retrying on any exception with exponential backoff. Colab's
    Hub/network calls hit intermittent ConnectionResetError; a few retries make
    the bootstrap survive them instead of dying on a fresh VM."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as err:  # noqa: BLE001 — transient network faults are the norm here
            if attempt == attempts - 1:
                raise
            delay = base_delay * 2**attempt
            print(f"[sft_colab] {what} failed ({err!r}); "
                  f"retry {attempt + 1}/{attempts - 1} in {delay:.0f}s")
            time.sleep(delay)


def download_file(repo_id: str, filename: str, local_dir: Path) -> Path:
    """Download one file from the Hub into local_dir (so it lands at
    local_dir/filename), retry-wrapped and tolerant of the repo being a model or
    a dataset repo. Returns the local path."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    def _fetch() -> str:
        last: Exception | None = None
        for repo_type in ("model", "dataset"):
            try:
                return hf_hub_download(repo_id=repo_id, filename=filename,
                                       repo_type=repo_type, local_dir=str(local_dir))
            except RepositoryNotFoundError as err:  # try the other repo type
                last = err
        raise last  # both repo types missing — surface the last error

    retry(_fetch, what=f"download {repo_id}/{filename}")
    return local_dir / filename


def prepare(sft_data_config, sft_config, *, download=None) -> Path:
    """Ensure the tokenizer + sft split are present (download if missing), then
    build the SFT dataset if examples.jsonl is missing. Returns the examples
    path. `download` is injectable for tests; it defaults to download_file
    resolved at call time. Idempotent: every step is guarded by an existence
    check, so re-running on a warm VM is a no-op up to training."""
    if download is None:
        download = download_file
    data_cfg = load_config(sft_data_config)
    tokenizer_path = Path(data_cfg["tokenizer"])          # artifacts/tokenizer_full/tokenizer.json
    split_path = Path(data_cfg["sft_split"])              # artifacts/data_prep_full/splits/sft.jsonl

    if not tokenizer_path.exists():
        print(f"[sft_colab] downloading tokenizer -> {tokenizer_path}")
        download(TOKENIZER_REPO, "tokenizer.json", tokenizer_path.parent)
    if not split_path.exists():
        print(f"[sft_colab] downloading sft split -> {split_path}")
        # local_dir is the artifact root so filename 'splits/sft.jsonl' resolves under it
        download(DATA_REPO, "splits/sft.jsonl", split_path.parent.parent)

    examples_path = Path(load_config(sft_config)["data"]["examples_path"])
    if examples_path.exists():
        print(f"[sft_colab] SFT dataset already built: {examples_path}")
    else:
        print(f"[sft_colab] building SFT dataset -> {examples_path}")
        sft_data.run(data_cfg)
    return examples_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sft-data-config", default=DEFAULT_SFT_DATA_CONFIG, type=Path)
    parser.add_argument("--sft-config", default=DEFAULT_SFT_CONFIG, type=Path)
    parser.add_argument("--skip-train", action="store_true",
                        help="download + build the SFT dataset only, then stop")
    args = parser.parse_args(argv)

    load_env()  # HF_TOKEN / WANDB_API_KEY reach Hub sync + wandb; never printed
    prepare(args.sft_data_config, args.sft_config)
    if args.skip_train:
        print("[sft_colab] --skip-train: SFT dataset ready; skipping training")
        return
    print("[sft_colab] starting SFT (ts2-sft --resume)")
    summary = sft.run(load_config(args.sft_config), resume=True)
    print(f"[sft_colab] done: step {summary['step']}, loss {summary['loss']:.4f}")


if __name__ == "__main__":
    main()
