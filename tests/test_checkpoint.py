import torch

from tinystories_v2.checkpoint import (
    latest_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint,
)


def _state(step):
    return {"step": step, "model": {"w": torch.full((2, 2), float(step))}}


def test_save_load_roundtrip(tmp_path):
    path = save_checkpoint(tmp_path, 7, _state(7))
    assert path.name == "step_000007.pt"
    loaded = load_checkpoint(path)
    assert loaded["step"] == 7
    assert torch.equal(loaded["model"]["w"], torch.full((2, 2), 7.0))


def test_latest_picks_highest_step(tmp_path):
    for step in (2, 10, 6):
        save_checkpoint(tmp_path, step, _state(step))
    assert latest_checkpoint(tmp_path).name == "step_000010.pt"


def test_latest_ignores_partial_tmp_files(tmp_path):
    save_checkpoint(tmp_path, 3, _state(3))
    (tmp_path / "step_000099.pt.tmp").write_bytes(b"partial garbage")
    assert latest_checkpoint(tmp_path).name == "step_000003.pt"


def test_latest_none_when_empty(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    assert latest_checkpoint(tmp_path / "missing") is None


def test_no_tmp_left_behind_after_save(tmp_path):
    save_checkpoint(tmp_path, 1, _state(1))
    assert list(tmp_path.glob("*.tmp")) == []


def test_prune_keeps_newest(tmp_path):
    for step in (1, 2, 3, 4):
        save_checkpoint(tmp_path, step, _state(step))
    prune_checkpoints(tmp_path, keep_last=2)
    assert sorted(p.name for p in tmp_path.glob("step_*.pt")) == [
        "step_000003.pt", "step_000004.pt",
    ]
    prune_checkpoints(tmp_path, keep_last=0)  # 0 = keep all
    assert len(list(tmp_path.glob("step_*.pt"))) == 2
