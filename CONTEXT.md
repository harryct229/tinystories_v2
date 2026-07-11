# tinystories_v2

A university Generative AI course project: train a small language model from scratch to generate moral fables, using the TF1-EN-3M dataset, with an AI-feedback alignment stage. The academic focus is justifying every architecture choice.

## Language

**Fable**:
A short moral story (~325 words, ages 4–7 reading level) from the TF1-EN-3M dataset, built on the six-slot Scaffold and ending with an explicit moral.
_Avoid_: story, tale (too generic — a Fable always has the Scaffold)

**Scaffold**:
The six structural slots every Fable is generated from: character, trait, setting, conflict, resolution, moral.
_Avoid_: template, prompt schema

**Pretraining**:
Stage 1 — training the model from randomly initialized weights on Fable text via next-token prediction.
_Avoid_: training (ambiguous across stages)

**SFT**:
Stage 2 — supervised fine-tuning of the pretrained model on prompt→Fable pairs so it generates fables from a Scaffold instruction.
_Avoid_: instruction tuning (use SFT consistently)

**RLAIF**:
Stage 3 — alignment via reinforcement learning against a Reward Model whose training labels come from the Judge instead of human annotators.
_Avoid_: RLHF (we have no human feedback; the feedback source is a model)

**Judge**:
The local Qwen model that labels generated Fables (offline) to produce the Reward Model's training data.
_Avoid_: critic, evaluator, feedback system

**Reward Model**:
A small scalar-output model, initialized from the SFT model, trained on Judge preference pairs to score Fables during RLAIF.
_Avoid_: RM without first introducing it, scorer

**Slot Prompt**:
The compact special-token encoding of a Scaffold (e.g. `<|character|>fox<|trait|>greedy…<|fable|>`) used as the model's input format from SFT onward.
_Avoid_: prompt (ambiguous — the dataset's `prompt` field is the verbose natural-language version)
