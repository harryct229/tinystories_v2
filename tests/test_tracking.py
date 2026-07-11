import json
import sys
import types

import pytest

from tinystories_v2.tracking import MetricsLogger


def read_lines(out_dir):
    text = (out_dir / "metrics.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_jsonl_written_and_flushed_per_line(tmp_path):
    logger = MetricsLogger(tmp_path)
    logger.log({"loss": 2.5, "lr": 1e-4, "tokens_seen": 4096}, step=1)
    # Readable BEFORE finish(): a SIGKILLed run must not lose logged lines.
    assert read_lines(tmp_path) == [
        {"step": 1, "loss": 2.5, "lr": 1e-4, "tokens_seen": 4096}
    ]
    logger.log({"loss": 2.0}, step=2)
    logger.finish()
    assert [line["step"] for line in read_lines(tmp_path)] == [1, 2]


def test_append_mode_survives_reopen(tmp_path):
    a = MetricsLogger(tmp_path)
    a.log({"loss": 3.0}, step=1)
    a.finish()
    b = MetricsLogger(tmp_path)  # a resumed run re-opens the same file
    b.log({"loss": 2.0}, step=2)
    b.finish()
    assert [line["step"] for line in read_lines(tmp_path)] == [1, 2]


def test_wandb_streams_when_enabled(tmp_path, monkeypatch):
    calls = []
    run = types.SimpleNamespace(
        log=lambda data, step: calls.append(("log", data, step)),
        finish=lambda: calls.append(("finish",)),
    )
    fake = types.ModuleType("wandb")
    fake.init = lambda **kw: calls.append(("init", kw)) or run
    monkeypatch.setitem(sys.modules, "wandb", fake)

    logger = MetricsLogger(
        tmp_path, {"enabled": True, "project": "p", "run_name": "r"}
    )
    logger.log({"loss": 1.5}, step=3)
    logger.finish()

    assert calls[0] == ("init", {"project": "p", "name": "r", "resume": "allow"})
    assert calls[1] == ("log", {"loss": 1.5}, 3)
    assert calls[2] == ("finish",)
    assert read_lines(tmp_path)[0]["loss"] == 1.5  # JSONL still written


def test_degrades_to_jsonl_when_wandb_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # import wandb -> ImportError
    with pytest.warns(UserWarning, match="wandb"):
        logger = MetricsLogger(tmp_path, {"enabled": True, "project": "p"})
    logger.log({"loss": 1.0}, step=1)
    logger.finish()
    assert read_lines(tmp_path) == [{"step": 1, "loss": 1.0}]


def test_disabled_wandb_never_imported(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    logger = MetricsLogger(tmp_path, {"enabled": False})  # must not raise or warn
    logger.log({"loss": 1.0}, step=1)
    logger.finish()
