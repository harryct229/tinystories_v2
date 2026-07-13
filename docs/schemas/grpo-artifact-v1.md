# GRPO Artifact v1

Issue 06 pins the GRPO stage's `out_dir` artifact: the checkpoints, metrics, and
`manifest.json` metadata. The GRPO checkpoint is a plain FableLM policy — the
**RLAIF** model, a drop-in third model for the eval suite (issue 07), loaded by
`eval.load_stage_model` exactly like the base and SFT checkpoints.

## Artifact layout

```
<out_dir>/
  checkpoints/step_XXXXXX.pt   full training state (atomic; resume contract)
  metrics.jsonl                one line per log_every steps
  manifest.json                stage metadata (below)
```

Each `checkpoints/step_XXXXXX.pt` holds `{step, rollouts_seen, model, optimizer,
scaler, config}`, where `model` is the **policy** `state_dict` (a FableLM). The
frozen reference and the Reward Model are **not** stored — the reference is
re-derived from the `[init]` SFT checkpoint and the Reward Model re-loaded from
the `[reward]` artifact on every run, so both are pure functions of their inputs
and resume is bitwise-identical.

Each `metrics.jsonl` line: `{"step", "loss", "lr", "reward_mean", "kl",
"self_bleu", "policy_loss", "rollouts_seen"}`. `reward_mean` is the batch-mean
rollout reward (should rise), `kl` is the masked-mean KL(policy‖reference)
(should stay bounded), `self_bleu` is the batch's Self-BLEU (rising = diversity
collapse; NaN when fewer than two rollouts have words).

## manifest.json

```json
{
  "stage": "grpo",
  "package_version": "0.1.0",
  "final_step": 300,
  "final_loss": -0.12,
  "final_reward_mean": 1.84,
  "final_kl": 0.07,
  "reward_gate": {
    "accuracy": 0.739,
    "gate": 0.68,
    "reward_dir": "artifacts/reward_full"
  },
  "grpo": {
    "group_size": 8,
    "clip_eps": 0.2,
    "kl_beta": 0.03,
    "ppo_epochs": 2
  },
  "pref_split": "artifacts/data_prep_full/splits/pref.jsonl",
  "n_scaffolds": 4053,
  "config": { "...": "the full stage config" }
}
```

`reward_gate.accuracy` is the held-out accuracy `gate.check_reward_gate` read at
startup; the stage refuses to run when it is below `reward_gate.gate` (default
0.68). `final_reward_mean` / `final_kl` are the last logged step's values.

## Gate contract

GRPO calls `tinystories_v2.gate.check_reward_gate(reward_dir, gate)` before
loading any model or making `out_dir`. Below the gate it raises `RewardGateError`
and does not train — the fix is better Judge labels, not RL.

## Consumer contract

The GRPO checkpoint is consumed by the eval suite (issue 07) as any other stage:
add an `[[stages]]` block named `rlaif` pointing `local_dir` (and optionally
`hub_source`) at the GRPO `out_dir`. It is scored, metriced, and win-rated
identically to base and SFT — no GRPO-specific eval path. The pre-committed DPO
fallback (issue 08) is the alternative stage-3 model if GRPO is unstable at the
schedule checkpoint (DESIGN.md).
