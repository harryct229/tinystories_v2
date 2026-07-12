"""Thin artifact sync: local checkpoint dir <-> HF Hub or another local path.

Targets:
    hf://<repo_id>   private HF Hub model repo (created on first sync);
                     token comes from env/.env via load_env — never printed
    anything else    a local directory (tests, Drive scratch)

Stages write artifacts locally first and sync as a separate step, so the
training loop never blocks on (or fails because of) the network — a failed
sync is a warning, not a dead run. Uses module-level attribute access
(huggingface_hub.HfApi) so tests can monkeypatch without network.

Mirror semantics: Push (sync_to) mirrors checkpoint deletions, so once local
`prune_checkpoints` removes a stale checkpoint, the next sync removes it from
the remote/mirror too — otherwise every checkpoint ever synced would
accumulate remotely even after local pruning, and a fresh-VM resume would
re-download all of them. Fetch (fetch_from) is additive: never deletes
destination checkpoints even if source has none. All other files (metrics.jsonl,
manifest.json, ...) keep add/overwrite semantics only — nothing else is ever
deleted.
"""

import shutil
import warnings
from pathlib import Path

import huggingface_hub

from tinystories_v2.config import load_env

_HF_PREFIX = "hf://"


_CHECKPOINT_GLOB = "checkpoints/step_*.pt"


def sync_to(target: str, local_dir: Path, *, mirror_checkpoints: bool = True) -> None:
    local_dir = Path(local_dir)
    if target.startswith(_HF_PREFIX):
        load_env()
        repo_id = target[len(_HF_PREFIX):]
        api = huggingface_hub.HfApi()
        api.create_repo(repo_id, private=True, exist_ok=True, repo_type="model")
        delete_patterns = [_CHECKPOINT_GLOB] if mirror_checkpoints else None
        api.upload_folder(folder_path=str(local_dir), repo_id=repo_id,
                          repo_type="model", delete_patterns=delete_patterns)
    else:
        dst = Path(target)
        for src_file in local_dir.rglob("*"):
            if not src_file.is_file():
                continue
            dst_file = dst / src_file.relative_to(local_dir)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
        # Mirror deletions for checkpoint files only: anything pruned locally
        # (by prune_checkpoints) is removed from the mirror too.
        if mirror_checkpoints:
            local_ckpt_dir = local_dir / "checkpoints"
            dst_ckpt_dir = dst / "checkpoints"
            if dst_ckpt_dir.is_dir():
                local_names = ({p.name for p in local_ckpt_dir.glob("step_*.pt")}
                              if local_ckpt_dir.is_dir() else set())
                for dst_file in dst_ckpt_dir.glob("step_*.pt"):
                    if dst_file.name not in local_names:
                        dst_file.unlink()


def fetch_from(target: str, local_dir: Path) -> None:
    local_dir = Path(local_dir)
    if target.startswith(_HF_PREFIX):
        load_env()
        repo_id = target[len(_HF_PREFIX):]
        huggingface_hub.snapshot_download(
            repo_id=repo_id, local_dir=str(local_dir), repo_type="model"
        )
    else:
        sync_to(str(local_dir), Path(target), mirror_checkpoints=False)


def fetch_file_from(target: str, relative_path: str, dest: Path) -> None:
    """Fetch one file from a sync target (additive, like fetch_from) — for
    repos where a full snapshot would pull far more than needed (e.g. the
    data repo's ~1 GB pretrain split when only splits/pref.jsonl is wanted).
    Tries the model repo type first, then dataset: the data-splits repo is a
    dataset repo, everything else this project syncs is a model repo (same
    tolerance as scripts/sft_colab.py's downloader)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if target.startswith(_HF_PREFIX):
        load_env()
        repo_id = target[len(_HF_PREFIX):]
        try:
            downloaded = huggingface_hub.hf_hub_download(
                repo_id=repo_id, filename=relative_path, repo_type="model"
            )
        except huggingface_hub.utils.RepositoryNotFoundError:
            downloaded = huggingface_hub.hf_hub_download(
                repo_id=repo_id, filename=relative_path, repo_type="dataset"
            )
        source = Path(downloaded)
    else:
        source = Path(target) / relative_path
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.copy2(source, tmp)
    tmp.replace(dest)


def try_sync_to(target: str, local_dir: Path) -> None:
    """Best-effort sync for use inside the training loop."""
    try:
        sync_to(target, local_dir)
    except Exception as err:  # noqa: BLE001 — network errors must not kill training
        warnings.warn(f"hub sync to {target!r} failed: {err}", stacklevel=2)
