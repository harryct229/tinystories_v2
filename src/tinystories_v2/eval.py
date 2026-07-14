"""Evaluation stage: cross-family win-rates, reference-free metrics, and a
qualitative sample sheet over stage checkpoints (issue 07).

Invoke standalone:
    ts2-eval --config configs/eval_fixture.toml
    (or: python -m tinystories_v2.eval --config ...)

For each Scaffold in the held-out eval split, generate one seeded completion
per configured stage checkpoint (base / SFT / optional RLAIF) using identical
Scaffolds and sampling settings, then:
  - win-rates: score every stage pair with the config-selected cross-family
    eval Judge (issue 10) under order-swapped double judging;
  - reference-free metrics: issue 11's Self-BLEU, Distinct-n, Flesch Reading
    Ease per stage, plus held-out perplexity of each checkpoint on the eval
    fables;
  - sample sheet: the first sample_sheet_k eval Scaffolds rendered by every
    stage side by side.

Artifacts in <out_dir>:
    results.json   eval_judge_id, sampling, per-pair win-rate tables (counts),
                   per-stage metric tables, and the config (schema:
                   docs/schemas/eval-results-v1.md)
    report.md      report-pastable win-rate tables, metric tables, and the
                   embedded sample sheet
"""

import argparse
import itertools
import json
import os
import shutil
import warnings
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinystories_v2 import __version__
from tinystories_v2.checkpoint import latest_checkpoint, load_checkpoint
from tinystories_v2.config import load_config, load_env
from tinystories_v2.generate import sample
from tinystories_v2.hub import fetch_file_from, fetch_from, try_sync_to
from tinystories_v2.judge import JudgeOutputError, Verdict, build_judge, normalize_text
from tinystories_v2.metrics import (
    distinct_n, mean_distinct_n, mean_flesch_reading_ease, self_bleu,
    tokenize_words,
)
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.perplexity import perplexity
from tinystories_v2.pref_data import scaffold_seed
from tinystories_v2.slot_prompt import END_TOKEN, SLOT_FIELDS, render_prompt
from tinystories_v2.slots import Scaffold


def _degenerate(fable_a: str, fable_b: str) -> bool:
    """True when the Judge could not accept this pair: empty or effectively
    identical candidates (the Judge seam's own candidate normalization)."""
    a, b = normalize_text(fable_a), normalize_text(fable_b)
    return not a or not b or a == b


def _read_jsonl_lines(path: Path) -> list[dict]:
    """Complete JSON lines only: content after the last newline (a torn append
    from a killed run) is ignored, so resume never trips on a partial write."""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    complete = raw[: raw.rfind("\n") + 1] if "\n" in raw else ""
    return [json.loads(line) for line in complete.split("\n") if line]


def _completions_path(out_dir: Path, stage: str) -> Path:
    return Path(out_dir) / "completions" / f"{stage}.jsonl"


def _load_cached_fables(out_dir: Path, stage: str,
                        hashes: list[str]) -> list[str] | None:
    """The stage's cached completions aligned to hashes, or None when the
    cache is absent or does not cover every requested Scaffold."""
    by_hash = {r["prompt_hash"]: r["fable"]
               for r in _read_jsonl_lines(_completions_path(out_dir, stage))}
    if all(h in by_hash for h in hashes):
        return [by_hash[h] for h in hashes]
    return None


def _store_fables(out_dir: Path, stage: str, hashes: list[str],
                  fables: list[str]) -> None:
    path = _completions_path(out_dir, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for prompt_hash, fable in zip(hashes, fables):
            f.write(json.dumps({"prompt_hash": prompt_hash, "fable": fable},
                               ensure_ascii=False) + "\n")
    tmp.replace(path)


def _judgments_path(out_dir: Path, stage_a: str, stage_b: str) -> Path:
    return Path(out_dir) / "judgments" / f"{stage_a}--{stage_b}.jsonl"


def stage_win(judge, scaffold: Scaffold, fable_a: str, fable_b: str) -> str:
    """Order-swapped double judging of two stages' fables for one Scaffold.

    Returns "a"/"b" only when the same candidate is preferred under both
    presentation orders (position bias cancels); otherwise "tie". Assumes
    non-degenerate candidates — callers skip degenerate pairs first."""
    first = judge.compare(scaffold, fable_a, fable_b)
    swapped = judge.compare(scaffold, fable_b, fable_a)
    if first is Verdict.A and swapped is Verdict.B:
        return "a"
    if first is Verdict.B and swapped is Verdict.A:
        return "b"
    return "tie"


def win_rate_table(judge, scaffolds: list[Scaffold], stage_a: str,
                   fables_a: list[str], stage_b: str,
                   fables_b: list[str], *, hashes: list[str] | None = None,
                   log_path: Path | None = None,
                   on_progress=None) -> dict:
    """Tally wins/ties/skips of stage_a vs stage_b over aligned per-Scaffold
    completions. Degenerate pairs (empty or identical) the Judge cannot compare
    are skipped, not counted as ties.

    With log_path (and hashes), judgments stream to an append-only JSONL: one
    line per Scaffold as it is judged, fsynced, so a killed run resumes from
    the next unjudged Scaffold instead of re-paying the judged ones. Lines are
    validated against hashes positionally — a mismatch means the Scaffold set
    changed and the log cannot be trusted."""
    if not (len(scaffolds) == len(fables_a) == len(fables_b)):
        raise ValueError("scaffolds and both fable lists must align")

    outcomes: list[str] = []
    if log_path is not None:
        if hashes is None or len(hashes) != len(scaffolds):
            raise ValueError("streaming judgments require aligned hashes")
        for i, record in enumerate(_read_jsonl_lines(log_path)):
            if i >= len(hashes) or record["prompt_hash"] != hashes[i]:
                raise ValueError(
                    f"{log_path} does not match the eval Scaffold set; "
                    f"remove the stale judgments to re-judge"
                )
            outcomes.append(record["outcome"])
        log_path.parent.mkdir(parents=True, exist_ok=True)

    log_file = log_path.open("a", encoding="utf-8") if log_path else None
    try:
        for i in range(len(outcomes), len(scaffolds)):
            fa, fb = fables_a[i], fables_b[i]
            if _degenerate(fa, fb):
                outcome = "skipped"
            else:
                try:
                    outcome = stage_win(judge, scaffolds[i], fa, fb)
                except JudgeOutputError:
                    # One malformed real-Judge verdict must not abort the whole
                    # eval after all generation (mirrors
                    # pref_data.label_scaffold). Count it and move on; it is
                    # neither a win nor a genuine tie.
                    outcome = "judge_error"
            outcomes.append(outcome)
            if log_file is not None:
                log_file.write(json.dumps(
                    {"prompt_hash": hashes[i], "outcome": outcome}) + "\n")
                log_file.flush()
                os.fsync(log_file.fileno())
            if on_progress is not None:
                on_progress()
    finally:
        if log_file is not None:
            log_file.close()

    return {"stage_a": stage_a, "stage_b": stage_b,
            "wins_a": outcomes.count("a"), "wins_b": outcomes.count("b"),
            "ties": outcomes.count("tie"), "skipped": outcomes.count("skipped"),
            "judge_error": outcomes.count("judge_error"),
            "n": len(scaffolds)}


def all_pairwise_win_rates(judge, scaffolds: list[Scaffold],
                           stage_fables: dict[str, list[str]], *,
                           hashes: list[str] | None = None,
                           out_dir: Path | None = None,
                           on_progress=None) -> list[dict]:
    """A win_rate_table for every unordered stage pair, in stage_fables order.
    With out_dir, each pairing streams its judgments to a resumable log."""
    names = list(stage_fables)
    return [win_rate_table(
                judge, scaffolds, a, stage_fables[a], b, stage_fables[b],
                hashes=hashes,
                log_path=_judgments_path(out_dir, a, b) if out_dir else None,
                on_progress=on_progress)
            for a, b in itertools.combinations(names, 2)]


def reference_free_metrics(fables: list[str], *,
                           self_bleu_sample_size: int | None = None,
                           self_bleu_seed: int = 0) -> dict:
    """Aggregate issue 11's reference-free metrics over one stage's fables.

    Wordless generations (an empty body from an early/toy checkpoint) carry no
    lexical signal and are dropped first. A metric undefined for the usable set
    is None: distinct_2 when fewer than two usable fables remain or the pooled
    set has no bigram, self_bleu with fewer than two usable fables, and every
    metric when nothing is usable. Distinct-1 is the paper's per-Fable mean
    (mean_distinct_n); distinct_2 is pooled (distinct_n) and reported only when
    at least two usable fables are present."""
    usable = [f for f in fables if tokenize_words(f)]
    metrics = {
        "n_usable": len(usable),
        "mean_distinct_1": None,
        "distinct_2": None,
        "self_bleu": None,
        "mean_flesch_reading_ease": None,
    }
    if not usable:
        return metrics
    metrics["mean_distinct_1"] = mean_distinct_n(usable, 1)
    metrics["mean_flesch_reading_ease"] = mean_flesch_reading_ease(usable)
    if len(usable) >= 2:
        try:
            metrics["distinct_2"] = distinct_n(usable, 2)
        except ValueError:
            metrics["distinct_2"] = None  # no fable has two tokens
        metrics["self_bleu"] = self_bleu(
            usable, sample_size=self_bleu_sample_size, seed=self_bleu_seed)
    return metrics


def sample_sheet_md(scaffolds: list[Scaffold],
                    stage_fables: dict[str, list[str]], k: int) -> str:
    """The first k eval Scaffolds rendered by every stage side by side."""
    names = list(stage_fables)
    lines = ["# Qualitative sample sheet", ""]
    for i, scaffold in enumerate(scaffolds[:k]):
        lines.append(f"## Scaffold {i + 1}")
        for field in SLOT_FIELDS:
            lines.append(f"- **{field}**: {getattr(scaffold, field)}")
        lines.append("")
        for name in names:
            lines.append(f"### {name}")
            lines.append(stage_fables[name][i].strip() or "_(empty)_")
            lines.append("")
    return "\n".join(lines)


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_report(results: dict, sample_sheet: str) -> str:
    """Report-pastable Markdown: win-rate tables, metric tables, sample sheet."""
    lines = [
        "# Evaluation report",
        "",
        f"Eval Judge: `{results['eval_judge_id']}`",
        f"Held-out Scaffolds: {results['n_scaffolds']}",
        "",
        "## Win-rates (order-swapped double judging)",
        "",
        "| A | B | A wins | B wins | ties | skipped | judge err | n |",
        "| - | - | ------ | ------ | ---- | ------- | --------- | - |",
    ]
    for w in results["win_rates"]:
        lines.append(
            f"| {w['stage_a']} | {w['stage_b']} | {w['wins_a']} | "
            f"{w['wins_b']} | {w['ties']} | {w['skipped']} | "
            f"{w['judge_error']} | {w['n']} |")
    lines += [
        "",
        "## Reference-free metrics",
        "",
        "| stage | Distinct-1 | Distinct-2 | Self-BLEU | Flesch | Perplexity |",
        "| ----- | ---------- | ---------- | --------- | ------ | ---------- |",
    ]
    for name, m in results["metrics"].items():
        lines.append(
            f"| {name} | {_fmt(m['mean_distinct_1'])} | {_fmt(m['distinct_2'])} "
            f"| {_fmt(m['self_bleu'])} | {_fmt(m['mean_flesch_reading_ease'])} "
            f"| {_fmt(m['perplexity'])} |")
    return "\n".join(lines) + "\n\n" + sample_sheet


def load_stage_model(stage_cfg: dict, device: str) -> FableLM:
    """Load one stage's FableLM checkpoint, fetching the artifact from the Hub
    first if the local checkpoint is absent (fresh VM). Every stage checkpoint
    is a plain FableLM (base/SFT/RLAIF share the architecture)."""
    local_dir = Path(stage_cfg["local_dir"])
    ckpt_dir = local_dir / "checkpoints"
    if latest_checkpoint(ckpt_dir) is None and stage_cfg.get("hub_source"):
        fetch_from(stage_cfg["hub_source"], local_dir)
    ckpt = latest_checkpoint(ckpt_dir)
    if ckpt is None:
        raise ValueError(
            f"no checkpoint for stage {stage_cfg['name']!r} under {ckpt_dir}; "
            f"point [[stages]].local_dir (and optionally hub_source) at the "
            f"stage artifact")
    state = load_checkpoint(ckpt)
    model = FableLM(ModelConfig(**state["config"]["model"]))
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def generate_stage_fables(model, tokenizer, scaffolds: list[Scaffold],
                          seeds: list[int], sampling: dict, *,
                          device: str = "cpu") -> list[str]:
    """One seeded completion per Scaffold, decoded to a fable body (prompt
    prefix and <|end|> excluded). A Slot Prompt longer than the model context
    yields "" so the caller can skip it rather than crash the whole eval."""
    end_id = tokenizer.token_to_id(END_TOKEN)
    fables = []
    for scaffold, seed in zip(scaffolds, seeds):
        prompt_ids = tokenizer.encode(render_prompt(scaffold)).ids
        if len(prompt_ids) > model.config.context:
            fables.append("")
            continue
        seq = sample(
            model, prompt_ids, num_samples=1,
            max_new_tokens=sampling["max_new_tokens"],
            temperature=sampling["temperature"], top_p=sampling["top_p"],
            seed=seed, end_id=end_id, device=device)[0]
        fables.append(tokenizer.decode(seq[len(prompt_ids):]).strip())
    return fables


def generate_all_stages(stage_models: dict[str, FableLM], tokenizer,
                        scaffolds: list[Scaffold], seeds: list[int],
                        sampling: dict, *, device: str = "cpu",
                        generate_fn=None) -> dict[str, list[str]]:
    """Generate per-stage completions with identical Scaffolds, seeds, and
    sampling across every checkpoint (the apples-to-apples eval contract).
    generate_fn is injectable for tests; it defaults to generate_stage_fables."""
    gen = generate_fn or generate_stage_fables
    return {name: gen(model, tokenizer, scaffolds, seeds, sampling, device=device)
            for name, model in stage_models.items()}


def _read_split(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _encode_eval_tokens(tokenizer, rows: list[dict]) -> list[int]:
    """Flatten the eval fables into one held-out token stream for perplexity."""
    ids: list[int] = []
    for row in rows:
        ids.extend(tokenizer.encode(row["fable"]).ids)
    return ids


def run(config: dict, *, resume: bool = False, generate_fn=None,
        judge_factory=None) -> dict:
    load_env()  # HF token for hub fetch/sync — never printed
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    hub_target = config.get("hub", {}).get("target")
    if resume:
        # Fresh VM: pull a prior session's partial completions/judgments.
        if hub_target and not (out_dir / "judgments").exists():
            try:
                fetch_from(hub_target, out_dir)
            except Exception as err:  # noqa: BLE001 — first run: repo may not exist yet
                warnings.warn(
                    f"could not fetch a prior eval run from {hub_target!r}; "
                    f"starting fresh: {err}", stacklevel=2)
    else:
        # A fresh run must never silently reuse stale resume state.
        shutil.rmtree(out_dir / "judgments", ignore_errors=True)
        shutil.rmtree(out_dir / "completions", ignore_errors=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = config["data"]
    split_path = Path(data["eval_split"])
    if not split_path.exists() and data.get("hub_source"):
        fetch_file_from(data["hub_source"], "splits/eval.jsonl", split_path)
    rows = _read_split(split_path)
    max_scaffolds = config.get("max_eval_scaffolds", 0)
    if max_scaffolds:
        rows = rows[:max_scaffolds]
    if not rows:
        raise ValueError(f"no eval Scaffolds in {split_path}")

    tokenizer_path = Path(data["tokenizer"])
    if not tokenizer_path.exists() and data.get("tokenizer_hub_source"):
        fetch_file_from(data["tokenizer_hub_source"], "tokenizer.json", tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    scaffolds = [Scaffold(**{f: row[f] for f in SLOT_FIELDS}) for row in rows]
    sampling = config["sampling"]
    seeds = [scaffold_seed(sampling["seed"], row["prompt_hash"]) for row in rows]

    stage_models = {s["name"]: load_stage_model(s, device) for s in config["stages"]}
    hashes = [row["prompt_hash"] for row in rows]
    gen = generate_fn or generate_stage_fables
    stage_fables: dict[str, list[str]] = {}
    for name, model in stage_models.items():
        cached = _load_cached_fables(out_dir, name, hashes)
        if cached is not None:
            stage_fables[name] = cached
            continue
        stage_fables[name] = gen(model, tokenizer, scaffolds, seeds, sampling,
                                 device=device)
        _store_fables(out_dir, name, hashes, stage_fables[name])
        if hub_target:
            try_sync_to(hub_target, out_dir)

    judge = (judge_factory or build_judge)(config["judge"])
    meta_path = out_dir / "judgments" / "meta.json"
    if meta_path.exists():
        recorded = json.loads(meta_path.read_text(encoding="utf-8"))
        if recorded["eval_judge_id"] != judge.judge_id:
            raise ValueError(
                f"resuming with a different judge: the logged judgments were "
                f"made by {recorded['eval_judge_id']!r} but the config builds "
                f"{judge.judge_id!r}; remove {meta_path.parent} to re-judge"
            )
    else:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"eval_judge_id": judge.judge_id}),
                       encoding="utf-8")
        tmp.replace(meta_path)

    judged = {"count": 0}
    sync_every = config.get("sync_every", 25)

    def on_progress() -> None:
        judged["count"] += 1
        if hub_target and judged["count"] % sync_every == 0:
            try_sync_to(hub_target, out_dir)

    win_rates = all_pairwise_win_rates(judge, scaffolds, stage_fables,
                                       hashes=hashes, out_dir=out_dir,
                                       on_progress=on_progress)

    metrics_cfg = config.get("metrics", {})
    sample_size = metrics_cfg.get("self_bleu_sample_size") or None
    eval_ids = _encode_eval_tokens(tokenizer, rows)
    metrics = {}
    for name, model in stage_models.items():
        m = reference_free_metrics(
            stage_fables[name], self_bleu_sample_size=sample_size,
            self_bleu_seed=metrics_cfg.get("self_bleu_seed", 0))
        m["perplexity"] = perplexity(
            model, eval_ids, block_size=model.config.context, device=device)
        metrics[name] = m

    sheet = sample_sheet_md(scaffolds, stage_fables, config.get("sample_sheet_k", 8))
    results = {
        "stage": "eval",
        "package_version": __version__,
        "eval_judge_id": judge.judge_id,
        "sampling": {key: sampling[key]
                     for key in ("max_new_tokens", "temperature", "top_p", "seed")},
        "eval_scaffolds": [row["prompt_hash"] for row in rows],
        "n_scaffolds": len(rows),
        "stages": list(stage_models),
        "win_rates": win_rates,
        "metrics": metrics,
        "config": config,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2),
                                          encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(results, sheet),
                                       encoding="utf-8")
    if hub_target:
        try_sync_to(hub_target, out_dir)
    print(f"eval done: {len(rows)} Scaffolds, {len(stage_models)} stages, "
          f"judge {judge.judge_id}")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="reuse cached completions and logged judgments "
                             "from an interrupted eval in out_dir (fetched "
                             "from [hub].target on a fresh VM)")
    args = parser.parse_args(argv)
    run(load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
