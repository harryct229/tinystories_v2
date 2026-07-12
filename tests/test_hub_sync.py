from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from tinystories_v2.hub import fetch_from, sync_to, fetch_file_from


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

        def upload_folder(self, folder_path, repo_id, repo_type, delete_patterns):
            calls.append(("upload_folder", folder_path, repo_id, repo_type,
                          delete_patterns))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    make_tree(tmp_path / "src")
    sync_to("hf://team/tinystories-v2-pretrain", tmp_path / "src")
    assert calls == [
        ("create_repo", "team/tinystories-v2-pretrain", True, True, "model"),
        ("upload_folder", str(tmp_path / "src"), "team/tinystories-v2-pretrain", "model",
         ["checkpoints/step_*.pt"]),
    ]


def test_local_sync_removes_pruned_checkpoints(tmp_path):
    src, mirror = tmp_path / "src", tmp_path / "mirror"
    make_tree(src)
    sync_to(str(mirror), src)
    # keep_last pruning removed step 2 locally, step 4 replaced it
    (src / "checkpoints" / "step_000002.pt").unlink()
    (src / "checkpoints" / "step_000004.pt").write_bytes(b"newer")
    sync_to(str(mirror), src)
    assert not (mirror / "checkpoints" / "step_000002.pt").exists()
    assert (mirror / "checkpoints" / "step_000004.pt").read_bytes() == b"newer"
    assert (mirror / "metrics.jsonl").exists()  # non-checkpoint files untouched


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


def test_fetch_from_never_deletes_destination_checkpoints(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "notes").mkdir(parents=True)
    (src / "notes" / "readme.txt").write_text("hi", encoding="utf-8")
    make_tree(dst)  # destination already has checkpoints/step_000002.pt
    fetch_from(str(src), dst)
    assert (dst / "checkpoints" / "step_000002.pt").exists()  # untouched
    assert (dst / "notes" / "readme.txt").exists()  # fetched


def test_fetch_file_from_local_target(tmp_path):
    src = tmp_path / "artifact"
    (src / "splits").mkdir(parents=True)
    (src / "splits" / "pref.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    dest = tmp_path / "elsewhere" / "pref.jsonl"
    fetch_file_from(str(src), "splits/pref.jsonl", dest)
    assert dest.read_text(encoding="utf-8") == '{"x": 1}\n'


def test_fetch_file_from_hf_dispatches_to_hf_hub_download(tmp_path, monkeypatch):
    calls = {}

    def fake_download(*, repo_id, filename, repo_type):
        calls.update(repo_id=repo_id, filename=filename, repo_type=repo_type)
        src = tmp_path / "downloaded.jsonl"
        src.write_text('{"y": 2}\n', encoding="utf-8")
        return str(src)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    dest = tmp_path / "dest" / "pref.jsonl"
    fetch_file_from("hf://someone/some-repo", "splits/pref.jsonl", dest)
    assert calls == {"repo_id": "someone/some-repo",
                     "filename": "splits/pref.jsonl", "repo_type": "model"}
    assert dest.read_text(encoding="utf-8") == '{"y": 2}\n'


def test_fetch_file_from_falls_back_to_dataset_repo(tmp_path, monkeypatch):
    # The data-splits repo is a dataset repo (the real labeling run 404'd on
    # the hardcoded model type); fetch_file_from must try dataset next, like
    # scripts/sft_colab.py's downloader.
    calls = []
    fake_response = SimpleNamespace(headers={}, request=None)

    def fake_download(*, repo_id, filename, repo_type):
        calls.append(repo_type)
        if repo_type == "model":
            raise huggingface_hub.utils.RepositoryNotFoundError(
                "not a model", response=fake_response)
        src = tmp_path / "downloaded.jsonl"
        src.write_text('{"z": 3}\n', encoding="utf-8")
        return str(src)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    dest = tmp_path / "dest" / "pref.jsonl"
    fetch_file_from("hf://someone/data-repo", "splits/pref.jsonl", dest)
    assert calls == ["model", "dataset"]
    assert dest.read_text(encoding="utf-8") == '{"z": 3}\n'
