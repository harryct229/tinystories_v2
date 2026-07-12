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
resumed run reproduces exactly what the uninterrupted run would have written,
and labeling accumulates across Colab sessions into one growing artifact
synced to [hub].target.
"""

import hashlib


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
