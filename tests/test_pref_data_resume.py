"""Kill-and-resume: the preference-labeling commit protocol, end to end.

Sized so the run is slow enough to SIGKILL mid-flight: a d_model-128 sampler
generating 128 new tokens x 4 completions takes a noticeable fraction of a
second per Scaffold, giving several commits inside the kill window.
Per-Scaffold sampling seeds make each Scaffold's work independent of history,
so the resumed artifact must be byte-identical to an uninterrupted reference
run — a duplicated pair or a lost pair would break equality.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tinystories_v2.data import run as data_run
from tinystories_v2.pref_data import run as pref_run
from tinystories_v2.tokenizer import run as tokenizer_run

MODEL = {"vocab_size": 512, "d_model": 128, "n_layers": 4,
         "n_heads": 4, "context": 256, "ffn_hidden": 384}
MAX_SCAFFOLDS = 12
KILL_AFTER_DONE = 3


def pref_config(out_dir, pref_split, init_dir) -> dict:
    return {
        "out_dir": str(out_dir),
        "max_scaffolds": MAX_SCAFFOLDS,
        "data": {"pref_split": str(pref_split)},
        "checkpoint": {"local_dir": str(init_dir)},
        "sampling": {"num_completions": 4, "pairs_per_scaffold": 3,
                     "temperature": 1.0, "top_p": 0.95,
                     "max_new_tokens": 128, "seed": 1337},
        "judge": {"kind": "fake_slot_coverage"},
    }


def to_toml(config: dict) -> str:
    lines = [f'out_dir = "{config["out_dir"]}"',
             f"max_scaffolds = {config['max_scaffolds']}"]
    for section in ("data", "checkpoint", "sampling", "judge"):
        lines.append(f"[{section}]")
        for key, value in config[section].items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"


def done_count(progress_path: Path) -> int:
    try:
        return len(
            json.loads(progress_path.read_text(encoding="utf-8"))["done"])
    except (FileNotFoundError, json.JSONDecodeError):
        return 0   # not yet written


def test_killed_labeling_resumes_to_identical_artifact(
        tmp_path, fixture_path, make_init_checkpoint):
    # Shared inputs: one pref split, one tokenizer, one sampling checkpoint.
    data_dir, tok_dir = tmp_path / "data", tmp_path / "tok"
    data_run({
        "out_dir": str(data_dir), "max_extraction_failures": 0,
        "source": {"kind": "jsonl", "path": str(fixture_path)},
        "splits": {"seed": "fixture-v1", "pretrain": 0.3, "sft": 0.2,
                   "pref": 0.4, "eval": 0.1},
    })
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    pref_split = data_dir / "splits" / "pref.jsonl"
    init_dir = make_init_checkpoint(tmp_path / "init", MODEL,
                                    tok_dir / "tokenizer.json")

    # Reference: identical config, never interrupted.
    reference = pref_config(tmp_path / "reference", pref_split, init_dir)
    pref_run(reference)

    # Interrupted: run as a subprocess, SIGKILL after a few commits.
    interrupted = pref_config(tmp_path / "interrupted", pref_split, init_dir)
    config_file = tmp_path / "interrupted.toml"
    config_file.write_text(to_toml(interrupted), encoding="utf-8")
    progress_path = Path(interrupted["out_dir"]) / "progress.json"

    proc = subprocess.Popen(
        [sys.executable, "-m", "tinystories_v2.pref_data",
         "--config", str(config_file)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 120
        while done_count(progress_path) < KILL_AFTER_DONE:
            if proc.poll() is not None:
                pytest.fail(
                    f"stage finished (rc={proc.returncode}) before the kill "
                    f"window; enlarge MODEL or lower KILL_AFTER_DONE")
            if time.monotonic() > deadline:
                pytest.fail("timed out waiting for committed Scaffolds")
            time.sleep(0.005)
        os.kill(proc.pid, signal.SIGKILL)
    finally:
        proc.kill()
        proc.wait(timeout=30)

    killed_at = done_count(progress_path)
    assert KILL_AFTER_DONE <= killed_at < MAX_SCAFFOLDS

    pref_run(interrupted, resume=True)

    ref_out = Path(reference["out_dir"])
    res_out = Path(interrupted["out_dir"])
    assert (res_out / "pairs.jsonl").read_bytes() == \
        (ref_out / "pairs.jsonl").read_bytes()
    ref_manifest = json.loads(
        (ref_out / "manifest.json").read_text(encoding="utf-8"))
    res_manifest = json.loads(
        (res_out / "manifest.json").read_text(encoding="utf-8"))
    assert res_manifest["counters"] == ref_manifest["counters"]
    assert res_manifest["scaffolds_done"] == \
        ref_manifest["scaffolds_done"] == MAX_SCAFFOLDS
