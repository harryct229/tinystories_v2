# Final Review Fix Report

## Status

All three minor final-review findings are resolved with regression tests. The
new tests exercised the existing production behavior without exposing a
production defect, so no production code changed.

## Findings and resolutions

### Minor 1: Retained B/A winner branch lacked direct regression coverage

Added `test_consistent_fake_preserves_original_b_winner`. It presents the
weaker Fable as original A and the stronger Fable as original B, drives the
real `judge_with_order_swap` path to first-pass B / swapped-pass A, and asserts
that original B is chosen, original A is rejected, and both metadata labels
are preserved.

### Minor 2: Production Judge identities were checked only by substrings

Expanded the production-config parameter table with the two canonical
identities and replaced substring assertions with exact `judge_id` equality:

- `transformers:Qwen/Qwen3-8B;precision=fp16;thinking=false;rubric=fable-pairwise-v1`
- `transformers:Qwen/Qwen3-4B-Instruct-2507;precision=fp16;thinking=default;rubric=fable-pairwise-v1`

### Minor 3: Lazy backend and adapter behavior was not exercised offline

Added complete lightweight doubles at the imported Torch/Transformers
boundary. The coverage now verifies:

- construction performs no tokenizer or model load;
- absence of either `torch` or `transformers` produces the documented optional
  dependency error and preserves the `ImportError` cause;
- each production config loads its tokenizer and model once across two real
  `TransformersJudge.compare` calls;
- exact Judge identity and parsed `Verdict.B` output;
- fp16 selection, CUDA model/input movement, evaluation mode, and inference
  context use;
- exact chat-template arguments, including `enable_thinking=false` for L4 and
  omission of the option for T4;
- exact deterministic generation arguments;
- slicing away all prompt tokens before decode;
- decode output handoff through the real `parse_verdict` path.

The doubles implement every API shape consumed by the adapter: Torch dtype
attributes and inference context management; tokenizer/model factories;
tokenizer template application, batch mapping and `.to`; tensor `.shape`;
model `.to`, `.eval`, and `.generate`; generated row indexing/slicing; and
tokenizer decode.

## Files changed

- `tests/test_judge.py`
- `tests/test_judge_config.py`
- `.superpowers/sdd/final-fix-report.md`

## Covering tests

Command:

```text
rtk .venv/bin/pytest -q tests/test_judge.py tests/test_judge_config.py
```

Output:

```text
.....................                                                    [100%]
21 passed in 0.02s
```

## Full suite

Command:

```text
rtk .venv/bin/pytest -q
```

Output:

```text
...............................................                          [100%]
47 passed in 0.19s
```

## Production-code changes

None. The existing implementation passed all new behavior-level regression
coverage.

## Self-review

- Assertions target real `judge_with_order_swap` / `TransformersJudge`
  outputs, exact identities, caching effects, and outgoing external-boundary
  calls; they do not assert that a double merely exists or expose test-only
  production methods.
- The boundary doubles include all shapes and operations the adapter consumes,
  including attention masks and prompt/completion token structure.
- No Judge extras were installed, no network was accessed, and no model was
  loaded.
- No plans, issue status, schema documentation, configs, or progress ledgers
  were modified.
- `git diff --check` reported no whitespace errors before the final suite.

## Concerns

- Offline adapter coverage cannot replace the recommended Colab GPU smoke run
  for both production configs in the release checklist.
- `torch_dtype` remains intentionally unchanged; its Transformers v5
  deprecation should be handled with the future version-floor migration noted
  by the reviewer.
