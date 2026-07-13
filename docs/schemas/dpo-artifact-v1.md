# DPO Artifact v1

Issue 08 pins the DPO fallback stage's `out_dir` artifact: the checkpoints,
metrics, and `manifest.json` metadata. The DPO checkpoint is a plain FableLM
policy — a drop-in third model for the eval suite (issue 07), loaded by
`eval.load_stage_model` exactly like the base and SFT checkpoints.

## Artifact layout

```
<out_dir>/
  checkpoints/step_XXXXXX.pt   full training state (atomic; resume contract)
  metrics.jsonl                one line per log_every steps
  manifest.json                stage metadata (below)
```

Each `checkpoints/step_XXXXXX.pt` holds `{step, pairs_seen, model, optimizer,
scaler, config}`, where `model` is the **policy** `state_dict` (a FableLM). The
frozen reference is **not** stored — it is re-derived from the `[init]` SFT
checkpoint on every run, so it is a pure function of `[init]` and resume is
bitwise-identical.

Each `metrics.jsonl` line: `{"step", "loss", "lr", "margin", "pairs_seen"}`,
where `loss` is the batch DPO loss and `margin` is the batch-mean implicit
reward margin `beta * [(logπ_c - logπ_r) - (logπ_ref_c - logπ_ref_r)]`.

## manifest.json

```json
{
  "stage": "dpo",
  "package_version": "0.1.0",
  "final_step": 400,
  "final_loss": 0.52,
  "heldout_margin": 0.31,
  "beta": 0.1,
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

`heldout_margin` is the mean implicit reward margin over the held-out pairs: a
value `> 0` means the trained policy prefers the chosen completions over the
rejected ones more strongly than the frozen SFT reference does (the policy
shifted the intended way). The `pair_split` recipe makes the held-out slice
reproducible: a seeded permutation of the encoded pairs with the last
`round(n_pairs * holdout_frac)` held out (shared with the Reward Model via
`reward.split_pairs`).

## Consumer contract

The DPO checkpoint is consumed by the eval suite (issue 07) as any other stage:
add an `[[stages]]` block pointing `local_dir` (and optionally `hub_source`) at
the DPO `out_dir`. It is scored, metriced, and win-rated identically to base and
SFT — no DPO-specific eval path. DPO consumes the identical preference-pair
artifact as the Reward Model (issue 05); there is no separate labeling path.
