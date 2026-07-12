"""Retry-wrapped single-file Hub downloads for the Colab bootstrap scripts.

Shared by scripts/*_colab.py so each one-command bootstrap survives Colab's
intermittent network faults (ConnectionResetError) without duplicating the
logic. The hf_hub_download import stays local to download_file so tests can
monkeypatch huggingface_hub.hf_hub_download.
"""

import time
from pathlib import Path


def retry(fn, *, attempts: int = 5, base_delay: float = 2.0, what: str = "operation"):
    """Call fn(), retrying on any exception with exponential backoff."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as err:  # noqa: BLE001 — transient network faults are the norm here
            if attempt == attempts - 1:
                raise
            delay = base_delay * 2**attempt
            print(f"[hub_download] {what} failed ({err!r}); "
                  f"retry {attempt + 1}/{attempts - 1} in {delay:.0f}s")
            time.sleep(delay)


def download_file(repo_id: str, filename: str, local_dir) -> Path:
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
