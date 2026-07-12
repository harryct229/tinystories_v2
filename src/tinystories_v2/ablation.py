"""Issue 09: guarded report for the matched ~5M architecture ablation.

The existing Pretraining stage creates each run. This helper is evaluation
only: it loads the latest checkpoints, rejects unequal token budgets or model
sizes outside the declared tolerance, computes held-out perplexity on one
shared packed eval slice, and generates each variant from identical fixed
Scaffolds. It writes machine-readable JSON plus a report-ready Markdown table.
"""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from tokenizers import Tokenizer

from tinystories_v2.checkpoint import latest_checkpoint, load_checkpoint
from tinystories_v2.config import load_config
from tinystories_v2.generate import sample
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.pack import load_packed, pack_split
from tinystories_v2.perplexity import perplexity
from tinystories_v2.slot_prompt import SLOT_FIELDS
from tinystories_v2.slots import Scaffold, extract_slots


def validate_comparability(
    rows: list[dict], *, baseline: str, param_tolerance: float
) -> int:
    """Return shared tokens_seen after enforcing the fairness contract."""
    if not rows:
        raise ValueError("at least one ablation variant is required")
    if param_tolerance < 0:
        raise ValueError("param_tolerance must be non-negative")
    by_name = {row["variant"]: row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError("ablation variant names must be unique")
    if baseline not in by_name:
        raise ValueError(f"baseline variant {baseline!r} is missing")

    token_counts = {int(row["tokens_seen"]) for row in rows}
    if len(token_counts) != 1:
        observed = sorted(token_counts)
        raise ValueError(
            "matched training tokens required; observed " f"{observed}"
        )

    baseline_params = int(by_name[baseline]["params"])
    for row in rows:
        drift = abs(int(row["params"]) - baseline_params) / baseline_params
        if drift > param_tolerance:
            raise ValueError(
                f"variant {row['variant']!r} exceeds parameter tolerance: "
                f"{drift:.6f} > {param_tolerance:.6f}"
            )
    return token_counts.pop()


def _resolve_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path)
    if checkpoint.is_dir():
        latest = latest_checkpoint(checkpoint)
        if latest is None:
            raise ValueError(f"no step_*.pt checkpoints in {checkpoint}")
        return latest
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _load_variants(config: dict) -> list[tuple[dict, FableLM]]:
    loaded = []
    for variant in config["variants"]:
        checkpoint = _resolve_checkpoint(variant["checkpoint"])
        state = load_checkpoint(checkpoint)
        model_config = ModelConfig(**state["config"]["model"])
        model = FableLM(model_config)
        model.load_state_dict(state["model"])
        row = {
            "variant": variant["name"],
            "position_encoding": model_config.position_encoding,
            "mlp_type": model_config.mlp_type,
            "ffn_hidden": model_config.ffn_hidden,
            "params": model.num_params(),
            "step": int(state["step"]),
            "tokens_seen": int(state["tokens_seen"]),
            "checkpoint": str(checkpoint),
        }
        loaded.append((row, model))
    return loaded


def _load_eval_tokens(eval_config: dict):
    packed_path = Path(eval_config["packed_path"])
    pack_split(
        eval_config["split_path"],
        eval_config["tokenizer_path"],
        packed_path,
    )
    packed = load_packed(packed_path)
    max_tokens = int(eval_config["max_tokens"])
    if max_tokens < 2:
        raise ValueError("eval max_tokens must be at least 2")
    if len(packed) < max_tokens:
        raise ValueError(
            f"rebuilt packed eval split has {len(packed)} tokens; "
            f"configured max_tokens requires {max_tokens}"
        )
    return packed[:max_tokens].copy()


def _load_scaffolds(path: str | Path, count: int) -> list[Scaffold]:
    if count < 1:
        raise ValueError("sample_count must be at least 1")
    scaffolds = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if all(field in record for field in SLOT_FIELDS):
                scaffold = Scaffold(
                    **{field: record[field] for field in SLOT_FIELDS}
                )
            else:
                scaffold = extract_slots(record["prompt"])
            scaffolds.append(scaffold)
            if len(scaffolds) == count:
                return scaffolds
    raise ValueError(
        f"eval split contains {len(scaffolds)} Scaffolds; "
        f"sample_count requests {count}"
    )


def render_scaffold_seed(scaffold: Scaffold) -> str:
    """Build a Fable-like prefix from a Scaffold for Pretraining models."""
    return f"In {scaffold.setting}, a {scaffold.trait} {scaffold.character}"


def render_markdown(report: dict) -> str:
    lines = [
        "# 5M Architecture Ablation",
        "",
        f"Baseline: `{report['baseline']}`  ",
        f"Required training tokens: **{report['expected_tokens']:,}**  ",
        f"Matched training tokens: **{report['matched_tokens']:,}**  ",
        f"Held-out evaluation tokens: **{report['eval_tokens']:,}**  ",
        f"Parameter tolerance: **{report['param_tolerance']:.1%}**",
        "",
        "| Variant | Position | MLP | FFN hidden | Params | Training tokens | Val loss | Perplexity |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| `{row['variant']}` | {row['position_encoding']} | "
            f"{row['mlp_type']} | {row['ffn_hidden']:,} | "
            f"{row['params']:,} | {row['tokens_seen']:,} | "
            f"{row['val_loss']:.6f} | {row['perplexity']:.6f} |"
        )

    for index, sample_row in enumerate(report["samples"], start=1):
        scaffold = sample_row["scaffold"]
        lines.extend(
            [
                "",
                f"## Scaffold {index}",
                "",
                ", ".join(
                    f"{field}={scaffold[field]}" for field in SLOT_FIELDS
                ),
                "",
                f"Seed: {sample_row['seed']}",
            ]
        )
        for variant, fable in sample_row["generations"].items():
            lines.extend(
                [
                    "",
                    f"### {variant}",
                    "",
                    fable.strip() or "_Empty generation_",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def run(config: dict) -> dict:
    loaded = _load_variants(config)
    rows = [row for row, _ in loaded]
    matched_tokens = validate_comparability(
        rows,
        baseline=config["baseline"],
        param_tolerance=float(config["param_tolerance"]),
    )
    expected_tokens = int(config["expected_tokens"])
    if matched_tokens != expected_tokens:
        raise ValueError(
            f"completed token budget required: expected {expected_tokens}, "
            f"observed {matched_tokens}"
        )

    eval_config = config["eval"]
    eval_tokens = _load_eval_tokens(eval_config)
    tokenizer = Tokenizer.from_file(str(eval_config["tokenizer_path"]))
    scaffolds = _load_scaffolds(
        eval_config["split_path"], int(eval_config["sample_count"])
    )
    sample_rows = [
        {
            "scaffold": asdict(scaffold),
            "seed": render_scaffold_seed(scaffold),
            "generations": {},
        }
        for scaffold in scaffolds
    ]
    end_id = tokenizer.token_to_id("<|end|>")
    device = eval_config["device"]

    for row, model in loaded:
        model = model.to(device).eval()
        value = perplexity(
            model,
            eval_tokens,
            block_size=model.config.context,
            batch_size=int(eval_config["batch_size"]),
            device=device,
        )
        row["perplexity"] = value
        row["val_loss"] = math.log(value)

        for index, scaffold in enumerate(scaffolds):
            seed_ids = tokenizer.encode(render_scaffold_seed(scaffold)).ids
            sequence = sample(
                model,
                seed_ids,
                num_samples=1,
                max_new_tokens=int(eval_config["max_new_tokens"]),
                temperature=float(eval_config["temperature"]),
                top_p=float(eval_config["top_p"]),
                seed=int(eval_config["seed"]) + index,
                end_id=end_id,
                device=device,
            )[0]
            sample_rows[index]["generations"][row["variant"]] = (
                tokenizer.decode(sequence)
            )
        model.to("cpu")

    report = {
        "baseline": config["baseline"],
        "param_tolerance": float(config["param_tolerance"]),
        "expected_tokens": expected_tokens,
        "matched_tokens": matched_tokens,
        "eval_tokens": len(eval_tokens),
        "rows": rows,
        "samples": sample_rows,
    }
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
