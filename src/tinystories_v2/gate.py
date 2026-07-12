"""Shared Reward Model accuracy gate (issue 05).

RLAIF refuses to start against a Reward Model whose held-out pair accuracy is
below the gate: below it, the fix is better Judge labels, not RL — a policy
optimized against a near-chance reward learns noise. GRPO (issue 06) calls
check_reward_gate at startup; the reward stage records the accuracy this reads.
"""

import json
import math
from pathlib import Path

DEFAULT_ACCURACY_GATE = 0.68


class RewardGateError(RuntimeError):
    """Raised when a Reward Model artifact is missing, malformed, or below the
    accuracy gate."""


def load_reward_manifest(reward_dir: str | Path) -> dict:
    """Read and sanity-check a Reward Model artifact's manifest.json."""
    path = Path(reward_dir) / "manifest.json"
    if not path.exists():
        raise RewardGateError(f"no Reward Model manifest at {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "reward_model":
        raise RewardGateError(
            f"{path} is not a Reward Model artifact (stage="
            f"{manifest.get('stage')!r})")
    if "heldout_accuracy" not in manifest:
        raise RewardGateError(
            f"{path} records no heldout_accuracy; re-run the reward stage")
    return manifest


def check_reward_gate(reward_dir: str | Path,
                      gate: float = DEFAULT_ACCURACY_GATE) -> float:
    """Return the Reward Model's held-out accuracy, or raise RewardGateError if it
    is undefined or below `gate`. Downstream RLAIF calls this before training."""
    accuracy = load_reward_manifest(reward_dir)["heldout_accuracy"]
    if accuracy is None or (isinstance(accuracy, float) and math.isnan(accuracy)):
        raise RewardGateError(
            "Reward Model held-out accuracy is undefined (NaN); the holdout "
            "split was empty — lower [split].holdout_frac or add more pairs")
    if accuracy < gate:
        raise RewardGateError(
            f"Reward Model held-out accuracy {accuracy:.3f} is below the gate "
            f"{gate:.2f}: improve Judge labels before RL, do not optimize a "
            f"policy against a near-chance reward")
    return accuracy
