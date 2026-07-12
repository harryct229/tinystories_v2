# Reward Model Artifact v1

Issue 05 pins the Reward Model stage's `out_dir` artifact: the checkpoints,
metrics, and the `manifest.json` metadata the accuracy gate reads.

## Artifact layout

```
<out_dir>/
  checkpoints/step_XXXXXX.pt   full training state (atomic; resume contract)
  metrics.jsonl                one line per log_every steps
  manifest.json                stage metadata (below)
```

Each `metrics.jsonl` line: `{"step", "loss", "lr", "accuracy", "pairs_seen"}`,
where `loss` is the batch Bradley-Terry loss and `accuracy` is the batch pair
accuracy (chance = 0.5).

## manifest.json

```json
{
  "stage": "reward_model",
  "package_version": "0.1.0",
  "final_step": 400,
  "final_loss": 0.31,
  "heldout_accuracy": 0.74,
  "pair_split": {
    "seed": 20260712,
    "holdout_frac": 0.1,
    "n_pairs": 10000,
    "n_train": 9000,
    "n_holdout": 1000
  },
  "pairs_path": "artifacts/pref_full/pairs.jsonl",
  "n_pairs": 10000,
  "config": { "...": "the full stage config" }
}
```

`heldout_accuracy` is the fraction of held-out pairs the trained Reward Model
scores `chosen > rejected`. The `pair_split` recipe makes the held-out slice
reproducible: the split is a seeded permutation of the encoded pairs with the
last `round(n_pairs * holdout_frac)` held out.

## Gate contract

Downstream RLAIF (issue 06 GRPO) calls
`tinystories_v2.gate.check_reward_gate(reward_dir, gate=0.68)` before training.
It returns `heldout_accuracy` or raises `RewardGateError` when the manifest is
missing/not a Reward Model artifact, `heldout_accuracy` is undefined (NaN, i.e.
an empty holdout), or `heldout_accuracy < gate`. Below the gate the fix is
better Judge labels, not RL.
