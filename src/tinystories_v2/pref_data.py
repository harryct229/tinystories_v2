"""Preference-labeling stage: pref-split Scaffolds -> Judge-labeled preference
pairs, the Reward Model training data (issue 04).

Invoke standalone:
    ts2-pref-data --config configs/pref_data_fixture.toml [--resume]
    (or: python -m tinystories_v2.pref_data --config ...)

For each Scaffold in the pref split: sample [sampling].num_completions from
the SFT checkpoint (issue 02's generation utility), form pairs_per_scaffold
pairs on a round-robin schedule, and label each pair through issue 10's
order-swap consistency filter with the config-selected Judge. Only consistent
verdicts become records (docs/schemas/preference-pair-v1.md).

Artifacts in <out_dir>:
    pairs.jsonl      one schema-valid preference-pair v1 record per line
    progress.json    the commit point: committed line count, done prompt
                     hashes, counters
    manifest.json    stage, version, counters, discard_rate, judge_id, config

Commit protocol (kill-safe, one Scaffold at a time): append the Scaffold's
retained pairs to pairs.jsonl and fsync, then atomically replace
progress.json. A crash between the two leaves uncommitted trailing lines that
the next resume truncates before continuing — never a duplicate, never a lost
pair. Per-Scaffold sampling seeds derive from (seed, prompt_hash), so a
resumed run reproduces exactly what the uninterrupted run would have written
(on the same device type and torch build; resuming on different hardware
stays valid and duplicate-free, just not byte-identical), and labeling
accumulates across Colab sessions into one growing artifact synced to
[hub].target.
"""

import argparse
import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinystories_v2 import __version__
from tinystories_v2.checkpoint import latest_checkpoint, load_checkpoint
from tinystories_v2.config import load_config, load_env
from tinystories_v2.generate import sample
from tinystories_v2.hub import fetch_file_from, fetch_from, try_sync_to
from tinystories_v2.judge import (
    Judge, JudgeOutputError, build_judge, judge_with_order_swap,
    normalize_text,
)
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.preferences import SCHEMA_VERSION, PreferencePair
from tinystories_v2.slot_prompt import END_TOKEN, SLOT_FIELDS, render_prompt
from tinystories_v2.slots import Scaffold


def pair_indices(n_completions: int, n_pairs: int) -> list[tuple[int, int]]:
    """The first n_pairs of a round-robin (circle method) schedule over all
    C(n,2) index pairs. Scheduling rounds use every completion at most once,
    so small n_pairs spreads comparisons across completions instead of
    reusing completion 0 (what plain itertools.combinations order would do)."""
    if n_completions < 2:
        raise ValueError("need at least two completions to form pairs")
    max_pairs = n_completions * (n_completions - 1) // 2
    if not 1 <= n_pairs <= max_pairs:
        raise ValueError(
            f"pairs_per_scaffold must be in [1, {max_pairs}] for "
            f"{n_completions} completions, got {n_pairs}"
        )
    ids = list(range(n_completions))
    if len(ids) % 2:
        ids.append(-1)  # bye slot for odd counts
    half = len(ids) // 2
    schedule = []
    for _ in range(len(ids) - 1):
        for a, b in zip(ids[:half], reversed(ids[half:])):
            if -1 not in (a, b):
                schedule.append((min(a, b), max(a, b)))
        ids = [ids[0]] + [ids[-1]] + ids[1:-1]
    return schedule[:n_pairs]


def scaffold_seed(seed: int, prompt_hash: str) -> int:
    """Per-Scaffold sampling seed: a pure function of (seed, prompt_hash), so
    completions are independent of processing order and a resumed run
    regenerates exactly what the uninterrupted run would have (mirrors
    data.assign_split's hashing)."""
    digest = hashlib.sha256(f"{seed}:{prompt_hash}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2**63


def sample_completions(model: FableLM, tokenizer: Tokenizer,
                       scaffold: Scaffold, *, num_completions: int,
                       max_new_tokens: int, temperature: float, top_p: float,
                       seed: int, device: str = "cpu") -> list[str]:
    """N seeded completions of a Scaffold's Slot Prompt: decoded fable bodies
    (the prompt prefix and special tokens are excluded; decode drops <|end|>)."""
    prompt_ids = tokenizer.encode(render_prompt(scaffold)).ids
    sequences = sample(
        model, prompt_ids, num_samples=num_completions,
        max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
        seed=seed, end_id=tokenizer.token_to_id(END_TOKEN), device=device,
    )
    return [tokenizer.decode(seq[len(prompt_ids):]).strip()
            for seq in sequences]


def _degenerate(fable_a: str, fable_b: str) -> bool:
    """True when the Judge could not accept this pair: empty or effectively
    identical candidates (the Judge seam's own candidate normalization)."""
    a, b = normalize_text(fable_a), normalize_text(fable_b)
    return not a or not b or a == b


def label_scaffold(judge: Judge, scaffold: Scaffold, completions: list[str],
                   n_pairs: int) -> tuple[list[PreferencePair], dict[str, int]]:
    """Form the round-robin pairs and label each through order-swap
    consistency filtering. Degenerate pairs are skipped before the Judge sees
    them; inconsistent verdicts are discarded (position-bias filter)."""
    pairs: list[PreferencePair] = []
    counters = {"kept": 0, "discarded_inconsistent": 0,
               "skipped_degenerate": 0, "judge_error": 0}
    for i, j in pair_indices(len(completions), n_pairs):
        fable_a, fable_b = completions[i], completions[j]
        if _degenerate(fable_a, fable_b):
            counters["skipped_degenerate"] += 1
            continue
        try:
            pair = judge_with_order_swap(judge, scaffold, fable_a, fable_b)
        except JudgeOutputError:
            # One malformed verdict must not wedge the offline batch: resume
            # replays the same seed -> same completions -> same greedy Judge
            # output -> the same crash forever. Count it and move on.
            counters["judge_error"] += 1
            continue
        if pair is None:
            counters["discarded_inconsistent"] += 1
        else:
            counters["kept"] += 1
            pairs.append(pair)
    return pairs, counters


@dataclass
class Progress:
    """The stage's resume state. pairs_written is the commit point: the number
    of pairs.jsonl lines that are durably part of the artifact."""

    pairs_written: int = 0
    done: list[str] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)


def _truncate_pairs(pairs_path: Path, committed_lines: int) -> None:
    """Drop uncommitted trailing lines (a crash between append and commit)."""
    if not pairs_path.exists():
        if committed_lines:
            raise ValueError(
                f"{pairs_path} is missing but progress.json committed "
                f"{committed_lines} pairs; the artifact is corrupt"
            )
        return
    raw = pairs_path.read_text(encoding="utf-8")
    # json.dumps escapes "\n" inside records, so "\n" count == record count.
    # str.splitlines would also split on U+2028/U+2029/U+0085 — which
    # ensure_ascii=False emits raw — and a miscount here would slice
    # committed records apart. Split on "\n" only.
    parts = raw.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])  # uncommitted partial trailing line
    if len(lines) < committed_lines:
        raise ValueError(
            f"{pairs_path} has {len(lines)} lines but progress.json committed "
            f"{committed_lines}; the artifact is corrupt"
        )
    if len(lines) > committed_lines:
        tmp = pairs_path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(lines[:committed_lines]), encoding="utf-8")
        tmp.replace(pairs_path)


def load_progress(out_dir: Path) -> Progress:
    out_dir = Path(out_dir)
    progress_path = out_dir / "progress.json"
    if progress_path.exists():
        raw = json.loads(progress_path.read_text(encoding="utf-8"))
        progress = Progress(pairs_written=raw["pairs_written"],
                            done=list(raw["done"]),
                            counters=dict(raw["counters"]))
    else:
        progress = Progress()
    _truncate_pairs(out_dir / "pairs.jsonl", progress.pairs_written)
    return progress


def commit_scaffold(out_dir: Path, progress: Progress, prompt_hash: str,
                    pairs: list[PreferencePair],
                    counters: dict[str, int]) -> None:
    """Append this Scaffold's retained pairs, then advance the commit point.

    Order matters for kill-safety: pair lines are appended and fsynced first;
    the atomic progress.json replace is the commit. A crash in between leaves
    trailing lines that the next load_progress truncates away."""
    out_dir = Path(out_dir)
    with (out_dir / "pairs.jsonl").open("a", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    progress.pairs_written += len(pairs)
    progress.done.append(prompt_hash)
    for key, value in counters.items():
        progress.counters[key] = progress.counters.get(key, 0) + value
    payload = json.dumps({"pairs_written": progress.pairs_written,
                          "done": progress.done,
                          "counters": progress.counters}, indent=2)
    tmp = out_dir / "progress.json.tmp"
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(out_dir / "progress.json")


def _read_split(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_sampler(config: dict, device: str) -> tuple[FableLM, Tokenizer]:
    """Load the SFT (or any FableLM) checkpoint to sample completions from,
    plus its tokenizer. Fetches missing pieces from the Hub (fresh VM):
    the checkpoint artifact via [checkpoint].hub_source, the tokenizer via
    [data].tokenizer_hub_source (the checkpoint's recorded tokenizer_path is
    a local path that does not exist on a fresh VM)."""
    ckpt_cfg = config["checkpoint"]
    local_dir = Path(ckpt_cfg["local_dir"])
    if (latest_checkpoint(local_dir / "checkpoints") is None
            and ckpt_cfg.get("hub_source")):
        fetch_from(ckpt_cfg["hub_source"], local_dir)
    ckpt_path = latest_checkpoint(local_dir / "checkpoints")
    if ckpt_path is None:
        raise ValueError(
            f"no step_*.pt checkpoint under {local_dir / 'checkpoints'}; "
            f"point [checkpoint].local_dir (and optionally hub_source) at "
            f"the SFT artifact"
        )
    state = load_checkpoint(ckpt_path)
    model = FableLM(ModelConfig(**state["config"]["model"]))
    model.load_state_dict(state["model"])

    data = config["data"]
    tokenizer_path = Path(
        data.get("tokenizer") or state["config"]["data"]["tokenizer_path"])
    if not tokenizer_path.exists() and data.get("tokenizer_hub_source"):
        fetch_file_from(data["tokenizer_hub_source"], "tokenizer.json",
                        tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"sampling from checkpoint {ckpt_path}")
    return model.to(device).eval(), tokenizer


def run(config: dict, resume: bool = False) -> dict:
    load_env()  # HF token for hub sync/fetch — never printed
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    hub_target = config.get("hub", {}).get("target")

    # A fresh run over an existing labeling dir would append duplicates into
    # pairs.jsonl — hours of Judge time silently corrupted. Refuse instead.
    if not resume and progress_path.exists():
        raise ValueError(
            f"{progress_path} already exists; pass --resume to continue that "
            f"labeling run, or remove {out_dir} to start over"
        )
    if resume and not progress_path.exists() and hub_target:
        try:
            fetch_from(hub_target, out_dir)  # fresh VM: pull prior sessions
        except Exception as err:  # noqa: BLE001 — first run: the repo may not exist yet
            warnings.warn(
                f"could not fetch a prior labeling run from {hub_target!r}; "
                f"starting fresh: {err}", stacklevel=2)
    progress = load_progress(out_dir)

    data = config["data"]
    split_path = Path(data["pref_split"])
    if not split_path.exists() and data.get("hub_source"):
        fetch_file_from(data["hub_source"], "splits/pref.jsonl", split_path)
    rows = _read_split(split_path)
    if not rows:
        raise ValueError(f"no Scaffolds in {split_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = _load_sampler(config, device)
    judge = build_judge(config["judge"])

    sampling = config["sampling"]
    max_scaffolds = config.get("max_scaffolds", 0)
    sync_every = config.get("sync_every", 25)

    def write_manifest() -> None:
        kept = progress.counters.get("kept", 0)
        discarded = progress.counters.get("discarded_inconsistent", 0)
        manifest = {
            "stage": "pref_data",
            "package_version": __version__,
            "schema_version": SCHEMA_VERSION,  # preference-pair schema
            "scaffolds_total": len(rows),
            "scaffolds_done": len(progress.done),
            "counters": dict(progress.counters),
            "discard_rate": discarded / max(kept + discarded, 1),
            "judge_id": judge.judge_id,
            "config": config,
        }
        tmp = out_dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp.replace(out_dir / "manifest.json")

    done = set(progress.done)
    for row in rows:
        if max_scaffolds and len(progress.done) >= max_scaffolds:
            break
        prompt_hash = row["prompt_hash"]
        if prompt_hash in done:
            continue
        scaffold = Scaffold(**{f: row[f] for f in SLOT_FIELDS})
        prompt_ids = tokenizer.encode(render_prompt(scaffold)).ids
        if len(prompt_ids) > model.config.context:
            pairs, counters = [], {"skipped_long_prompt": 1}
        else:
            completions = sample_completions(
                model, tokenizer, scaffold,
                num_completions=sampling["num_completions"],
                max_new_tokens=sampling["max_new_tokens"],
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                seed=scaffold_seed(sampling["seed"], prompt_hash),
                device=device,
            )
            pairs, counters = label_scaffold(
                judge, scaffold, completions, sampling["pairs_per_scaffold"])
        commit_scaffold(out_dir, progress, prompt_hash, pairs, counters)
        done.add(prompt_hash)
        write_manifest()
        print(f"[{len(progress.done)}/{len(rows)}] "
              f"kept={progress.counters.get('kept', 0)} "
              f"discarded={progress.counters.get('discarded_inconsistent', 0)}")
        if hub_target and len(progress.done) % sync_every == 0:
            try_sync_to(hub_target, out_dir)

    write_manifest()
    if hub_target:
        try_sync_to(hub_target, out_dir)
    return {"scaffolds": len(progress.done), "pairs": progress.pairs_written}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted labeling run in out_dir")
    args = parser.parse_args(argv)
    run(load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
