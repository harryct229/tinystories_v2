# eval-results-v1

The evaluation stage (`ts2-eval`, issue 07) writes two artifacts to its
`out_dir`: `results.json` (machine-readable, this schema) and `report.md`
(report-pastable win-rate tables, metric tables, and the qualitative sample
sheet). Stages share nothing in memory; downstream readers consume `results.json`.

## `results.json`

| Field | Type | Meaning |
|-------|------|---------|
| `stage` | string | Always `"eval"`. |
| `package_version` | string | `tinystories_v2.__version__` at run time. |
| `eval_judge_id` | string | Identity of the cross-family eval Judge (`judge.judge_id`). Records "who judged" so the report is never ambiguous. Never a Qwen Judge id. |
| `sampling` | object | Shared decoding settings applied to **every** stage: `max_new_tokens`, `temperature`, `top_p`, `seed`. |
| `eval_scaffolds` | string[] | The held-out eval Scaffold `prompt_hash`es scored, in order — the same set for every stage. |
| `n_scaffolds` | int | `len(eval_scaffolds)`. |
| `stages` | string[] | Stage names in config order (e.g. `["base","sft"]`, `["base","sft","rlaif"]`). |
| `win_rates` | object[] | One entry per unordered stage pair. Each: `stage_a`, `stage_b`, `wins_a`, `wins_b`, `ties`, `skipped`, `judge_error`, `n`. A win requires consistency under order-swapped double judging; `ties` are inconsistent verdicts; `skipped` are degenerate (empty/identical) comparisons; `judge_error` are comparisons the real Judge answered with an unparseable verdict (`JudgeOutputError`), skipped rather than aborting the run; `wins_a + wins_b + ties + skipped + judge_error == n`. |
| `metrics` | object | Keyed by stage name. Each value: `mean_distinct_1`, `distinct_2`, `self_bleu`, `mean_flesch_reading_ease` (issue 11's reference-free metrics; `null` when undefined for the usable set), `n_usable` (fables with word tokens), and `perplexity` (held-out perplexity of that checkpoint on the eval fables). Per-stage `perplexity` uses each checkpoint's own `context` as the block size, so it is strictly comparable only across checkpoints that share the same context length (all current FableLM stages do). |
| `config` | object | The exact TOML config, echoed for provenance. |

## Guarantees

- The eval Judge is selected via issue 10's Judge interface (`kind` in `[judge]`)
  and is cross-family (Llama-3.1-8B-Instruct for real runs), never the Qwen
  Judge that produced the reward signal.
- All stages are scored on identical Scaffolds and identical sampling settings.
- The RLAIF column is present only when a third `[[stages]]` block is configured.
