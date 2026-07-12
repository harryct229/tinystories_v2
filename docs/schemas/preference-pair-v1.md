# Preference-Pair Schema v1

Issue 10 pins the record shared by preference labeling, Reward Model
training, DPO, and any fixture-driven RLAIF tests. Each JSONL line is one
retained, order-swap-consistent Judge preference.

## Record

```json
{
  "schema_version": 1,
  "scaffold": {
    "character": "fox",
    "trait": "greedy",
    "setting": "a dense forest",
    "conflict": "loses food to a trick",
    "resolution": "the trickster is exposed",
    "moral": "honesty is the best policy"
  },
  "chosen": "A greedy fox admitted the trick, returned the food, and learned that honesty is the best policy.",
  "rejected": "A fox walked through a forest and went home.",
  "verdict": {
    "judge_id": "fake:slot-coverage-v1",
    "first_pass": "A",
    "swapped_pass": "B",
    "consistent": true
  }
}
```

All five top-level fields are required and additional fields are rejected.
All six Scaffold fields, both completion fields, and `judge_id` are non-empty
strings. `chosen` and `rejected` must differ.

## Order-swap semantics

`first_pass` is relative to the original presentation `(A, B)`.
`swapped_pass` is relative to the second presentation `(B, A)`. Therefore,
opposite labels identify the same underlying Fable:

- `first_pass = "A"` and `swapped_pass = "B"` selects original A.
- `first_pass = "B"` and `swapped_pass = "A"` selects original B.

Equal labels reveal position bias or another inconsistency. Such a comparison
is discarded and is not a preference-pair record. Consequently every stored
v1 record has `consistent = true` and opposite pass labels.

Margin judges (`judge_id` prefix `transformers-margin:`) decide from the
debiased A/B logit margin measured across both presentation orders, so their
verdict is order-symmetric by construction: the pass labels record the winner
under each presentation and are always opposite. Pairs whose |margin| does
not clear the judge's threshold (`tau` in `judge_id`) are discarded, playing
the same filtering role the order-swap inconsistency discard plays for
verdict judges.

## Consumer contract

Decode each JSONL line and pass it through
`tinystories_v2.preferences.validate_preference_pair` before training. The
helper returns the typed `PreferencePair` or raises
`PreferencePairValidationError`. Consumers must not silently accept another
schema version, infer a missing Judge identity, or reconstruct discarded
pairs.
