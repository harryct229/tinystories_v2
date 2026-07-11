# 03 — SFT stage: Slot Prompt format and masked-loss fine-tuning

Status: ready-for-agent

## Parent

`.scratch/tinystories-v2-pipeline/PRD.md`

## What to build

The stage that turns the pretrained base into a Scaffold-conditioned fable
generator, demoable as: give six slot values, get a Fable.

**Slot Prompt formatting**: render extracted slots (from issue 01's data-prep)
into the compact special-token format decided in the design (schema below,
from the design session — the exact token order is the contract every later
stage relies on):

```
<|character|>…<|trait|>…<|setting|>…<|conflict|>…<|resolution|>…<|moral|>…<|fable|>{fable text}<|end|>
```

**SFT stage**: fine-tunes a Pretraining checkpoint on the sft split with loss
masked on Slot Prompt tokens (loss only on the Fable body and `<|end|>`),
reusing the checkpoint-resume contract, optimizer conventions, and W&B
logging from issue 02.

**Demo script**: takes six slot values (or samples a Scaffold from the eval
split) and prints the model's Fable — the live artifact for the final
presentation, pointed at any checkpoint.

## Acceptance criteria

- [ ] A test asserts the rendered Slot Prompt token sequence matches the schema exactly (special tokens as single IDs, correct order, loss mask covers exactly the prompt segment)
- [ ] Toy SFT run through the stage entrypoint decreases loss on Slot-Prompt-formatted fixture data and resumes after a kill
- [ ] After toy SFT, generation conditioned on a fixture Scaffold terminates with `<|end|>` (format learned at toy scale)
- [ ] Demo script generates from a checkpoint given six slot values on CPU
- [ ] Thin Colab notebook exists for the real SFT run
- [ ] Stage reads the sft split artifact only — a test guards against pretrain/eval split leakage into SFT training data

## Blocked by

- `02-model-pretraining-stage.md`
