"""After a toy SFT overfit, generation from a training Scaffold terminates with
<|end|> — the Slot Prompt format learned at toy scale (issue 03 criterion 2).

Four short examples encoded through the real slot_prompt format fit the toy
context (<|end|> reachable); the small model overfits and greedy decoding of a
training Slot Prompt reproduces the body and emits <|end|>.
"""

import json

from tokenizers import Tokenizer

from tinystories_v2.checkpoint import latest_checkpoint, load_checkpoint
from tinystories_v2.generate import sample
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.slot_prompt import encode_example, render_prompt
from tinystories_v2.slots import Scaffold
from tinystories_v2.sft import run as sft_run
from tinystories_v2.tokenizer import run as tokenizer_run

MODEL = {"vocab_size": 512, "d_model": 64, "n_layers": 2,
         "n_heads": 2, "context": 128, "ffn_hidden": 192}

PAIRS = [
    (Scaffold("fox", "sly", "a green wood", "a locked henhouse",
              "the fox waited", "patience wins"),
     "The sly fox waited by the henhouse and at last it opened."),
    (Scaffold("mouse", "brave", "a tall barn", "a hungry cat",
              "the mouse hid", "courage helps"),
     "The brave mouse hid from the cat and stayed safe all night."),
    (Scaffold("crow", "proud", "a dry field", "a shiny stone",
              "the crow shared", "pride can pass"),
     "The proud crow found a stone and learned to share it."),
    (Scaffold("bee", "busy", "a bright garden", "a coming storm",
              "the bee worked", "work pays off"),
     "The busy bee worked before the storm and saved the honey."),
]


def test_toy_sft_generation_terminates_with_end_token(
        tmp_path, fixture_path, make_init_checkpoint):
    tok_dir = tmp_path / "tok"
    tokenizer_run({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer_path = tok_dir / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    end_id = tokenizer.token_to_id("<|end|>")

    # Build a tiny examples.jsonl via the real encoder (short bodies fit ctx 128).
    examples_path = tmp_path / "examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as f:
        for i, (scaffold, fable) in enumerate(PAIRS):
            example = encode_example(tokenizer, scaffold, fable)
            f.write(json.dumps({"prompt_hash": str(i), **example.to_dict()}) + "\n")

    init_dir = make_init_checkpoint(tmp_path / "init", MODEL, tokenizer_path)
    config = {
        "out_dir": str(tmp_path / "out"),
        "model": dict(MODEL),
        "data": {"examples_path": str(examples_path),
                 "tokenizer_path": str(tokenizer_path)},
        "init": {"local_dir": str(init_dir)},
        "train": {"steps": 250, "micro_batch_size": 4, "grad_accum": 1,
                  "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                  "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
                  "grad_clip": 1.0, "precision": "fp32", "seed": 0,
                  "checkpoint_every": 250, "log_every": 50, "keep_last": 0},
        "wandb": {"enabled": False},
    }
    summary = sft_run(config)
    assert summary["loss"] < 0.5  # overfit drove masked loss near zero

    ckpts = tmp_path / "out" / "checkpoints"
    state = load_checkpoint(latest_checkpoint(ckpts))
    model = FableLM(ModelConfig(**state["config"]["model"]))
    model.load_state_dict(state["model"])

    scaffold = PAIRS[0][0]
    prompt_ids = tokenizer.encode(render_prompt(scaffold)).ids
    sequence = sample(model, prompt_ids, max_new_tokens=60,
                      temperature=0.0, end_id=end_id)[0]
    generated = sequence[len(prompt_ids):]
    assert end_id in generated  # generation terminated with <|end|>
