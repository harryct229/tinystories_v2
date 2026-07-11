# SFT-Example Schema v1

Issue 12 pins the record the SFT dataset builder (`tinystories_v2.sft_data`)
writes and the SFT trainer (issue 03) reads. Each JSONL line in
`examples.jsonl` is one masked-loss training example.

## Record

```json
{
  "prompt_hash": "71df0b5f…",
  "input_ids": [4, 812, 7, 233, 5, 91, 6, 44, 2, 118, 3, 77, 8, 501, 320, 9],
  "loss_mask": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
  "n_prompt_tokens": 13
}
```

- `prompt_hash` — the data-prep identity of the source Fable (join key).
- `input_ids` — the tokenized Slot Prompt example, in the fixed order
  `<|character|>…<|moral|><|fable|>{body}<|end|>`. Special tokens each encode
  to a single reserved ID.
- `loss_mask` — same length as `input_ids`; `0` over the conditioning prefix
  (through and including `<|fable|>`), `1` over the fable body and `<|end|>`.
- `n_prompt_tokens` — count of leading masked tokens; equals the index of
  `<|fable|>` plus one, and the number of leading zeros in `loss_mask`.

## Consumer contract

`loss_mask` and `input_ids` are always equal length. Train next-token
prediction only where `loss_mask` is `1` (the model learns to write the Fable,
not to parrot the Slot Prompt). Recover the Scaffold and fable body with
`tinystories_v2.slot_prompt.parse_example(tokenizer, input_ids)`. The builder
is deterministic: two runs over the same `sft_split` and tokenizer produce a
byte-identical `examples.jsonl`.
