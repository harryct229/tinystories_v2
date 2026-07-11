"""Thin artifact sync: local checkpoint dir <-> HF Hub or another local path.

Targets:
    hf://<repo_id>   private HF Hub model repo (created on first sync);
                     token comes from env/.env via load_env — never printed
    anything else    a local directory (tests, Drive scratch)

Stages write artifacts locally first and sync as a separate step, so the
training loop never blocks on (or fails because of) the network — a failed
sync is a warning, not a dead run. Uses module-level attribute access
(huggingface_hub.HfApi) so tests can monkeypatch without network.
"""

import shutil
import warnings
from pathlib import Path

import huggingface_hub

from tinystories_v2.config import load_env

_HF_PREFIX = "hf://"


def sync_to(target: str, local_dir: Path) -> None:
    local_dir = Path(local_dir)
    if target.startswith(_HF_PREFIX):
        load_env()
        repo_id = target[len(_HF_PREFIX):]
        api = huggingface_hub.HfApi()
        api.create_repo(repo_id, private=True, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=str(local_dir), repo_id=repo_id,
                          repo_type="model")
    else:
        dst = Path(target)
        for src_file in local_dir.rglob("*"):
            if not src_file.is_file():
                continue
            dst_file = dst / src_file.relative_to(local_dir)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)


def fetch_from(target: str, local_dir: Path) -> None:
    local_dir = Path(local_dir)
    if target.startswith(_HF_PREFIX):
        load_env()
        repo_id = target[len(_HF_PREFIX):]
        huggingface_hub.snapshot_download(
            repo_id=repo_id, local_dir=str(local_dir), repo_type="model"
        )
    else:
        sync_to(str(local_dir), Path(target))


def try_sync_to(target: str, local_dir: Path) -> None:
    """Best-effort sync for use inside the training loop."""
    try:
        sync_to(target, local_dir)
    except Exception as err:  # noqa: BLE001 — network errors must not kill training
        warnings.warn(f"hub sync to {target!r} failed: {err}", stacklevel=2)
