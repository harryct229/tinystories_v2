from pathlib import Path

import huggingface_hub
import pytest

from tinystories_v2.hub import fetch_from, sync_to


def make_tree(root: Path) -> None:
    (root / "checkpoints").mkdir(parents=True)
    (root / "checkpoints" / "step_000002.pt").write_bytes(b"ckpt-bytes")
    (root / "metrics.jsonl").write_text('{"step": 1}\n', encoding="utf-8")


def test_local_roundtrip(tmp_path):
    src, mirror, restored = tmp_path / "src", tmp_path / "mirror", tmp_path / "restored"
    make_tree(src)
    sync_to(str(mirror), src)
    assert (mirror / "checkpoints" / "step_000002.pt").read_bytes() == b"ckpt-bytes"
    assert (mirror / "metrics.jsonl").exists()
    fetch_from(str(mirror), restored)
    assert (restored / "checkpoints" / "step_000002.pt").read_bytes() == b"ckpt-bytes"


def test_sync_overwrites_stale_files(tmp_path):
    src, mirror = tmp_path / "src", tmp_path / "mirror"
    make_tree(src)
    sync_to(str(mirror), src)
    (src / "metrics.jsonl").write_text('{"step": 2}\n', encoding="utf-8")
    sync_to(str(mirror), src)
    assert '"step": 2' in (mirror / "metrics.jsonl").read_text(encoding="utf-8")


def test_hf_target_dispatches_to_hub_api(tmp_path, monkeypatch):
    calls = []

    class FakeApi:
        def create_repo(self, repo_id, private, exist_ok, repo_type):
            calls.append(("create_repo", repo_id, private, exist_ok, repo_type))

        def upload_folder(self, folder_path, repo_id, repo_type):
            calls.append(("upload_folder", folder_path, repo_id, repo_type))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    make_tree(tmp_path / "src")
    sync_to("hf://team/tinystories-v2-pretrain", tmp_path / "src")
    assert calls == [
        ("create_repo", "team/tinystories-v2-pretrain", True, True, "model"),
        ("upload_folder", str(tmp_path / "src"), "team/tinystories-v2-pretrain", "model"),
    ]


def test_hf_fetch_dispatches_to_snapshot_download(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download",
        lambda repo_id, local_dir, repo_type: calls.append(
            (repo_id, local_dir, repo_type)
        ),
    )
    fetch_from("hf://team/tinystories-v2-pretrain", tmp_path / "dst")
    assert calls == [
        ("team/tinystories-v2-pretrain", str(tmp_path / "dst"), "model")
    ]


def test_try_sync_to_warns_instead_of_raising(tmp_path, monkeypatch):
    from tinystories_v2 import hub

    def boom(target, local_dir):
        raise ConnectionError("network down")

    monkeypatch.setattr(hub, "sync_to", boom)
    with pytest.warns(UserWarning, match="hub sync .* failed"):
        hub.try_sync_to("hf://team/repo", tmp_path)
