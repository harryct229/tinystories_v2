# 03 — SFT stage: Slot Prompt format and masked-loss fine-tuning

Status: ready-for-agent

## Parent

`.scratch/tinystories-v2-pipeline/PRD.md`

## What to build

The stage that turns the pretrained base into a Scaffold-conditioned fable
generator, demoable as: give six slot values, get a Fable.

**SFT stage**: fine-tunes a Pretraining checkpoint on the SFT dataset
artifact built by issue 12 (Slot Prompt rendering, parsing, and loss masking
live there), reusing the checkpoint-resume contract, optimizer conventions,
and W&B logging from issue 02.

**Demo script**: takes six slot values (or samples a Scaffold from the eval
split) and prints the model's Fable — the live artifact for the final
presentation, pointed at any checkpoint.

## Acceptance criteria

- [ ] Toy SFT run through the stage entrypoint decreases loss on issue 12's dataset artifact built from fixture data, and resumes after a kill
- [ ] After toy SFT, generation conditioned on a fixture Scaffold terminates with `<|end|>` (format learned at toy scale)
- [ ] Demo script generates from a checkpoint given six slot values on CPU
- [ ] Thin Colab notebook exists for the real SFT run
- [ ] Stage reads the sft split artifact only — a test guards against pretrain/eval split leakage into SFT training data

## Blocked by

- `02-model-pretraining-stage.md`
- `12-slot-prompt-renderer.md`
