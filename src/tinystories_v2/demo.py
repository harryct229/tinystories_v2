"""Demo: render six slot values (or a sampled eval Scaffold) into a Slot Prompt
and print the model's Fable. The live artifact for the final presentation; point
--checkpoint at any Pretraining/SFT/RLAIF checkpoint. Runs on CPU.

Invoke standalone:
    ts2-demo --checkpoint artifacts/sft_full/checkpoints \
        --character fox --trait greedy --setting "a sunny orchard" \
        --conflict "a locked gate" --resolution "the fox shared" \
        --moral "sharing brings friends"
    ts2-demo --checkpoint ... --sample-eval artifacts/data_prep_full/splits/eval.jsonl

--tokenizer defaults to the tokenizer_path recorded in the checkpoint's config.
"""

import argparse
import json
import random
from pathlib import Path

from tokenizers import Tokenizer

from tinystories_v2.checkpoint import latest_checkpoint, load_checkpoint
from tinystories_v2.generate import sample
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.slot_prompt import END_TOKEN, SLOT_FIELDS, render_prompt
from tinystories_v2.slots import Scaffold


def _scaffold_from_args(args) -> Scaffold:
    if args.sample_eval:
        rows = [json.loads(line) for line in
                open(args.sample_eval, encoding="utf-8") if line.strip()]
        if not rows:
            raise SystemExit(f"no rows in {args.sample_eval}")
        row = random.Random(args.seed).choice(rows)
        return Scaffold(**{field: row[field] for field in SLOT_FIELDS})
    values = {field: getattr(args, field) for field in SLOT_FIELDS}
    missing = [field for field, value in values.items() if not value]
    if missing:
        raise SystemExit(
            f"provide all six slots or --sample-eval; missing: {', '.join(missing)}"
        )
    return Scaffold(**values)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="a step_*.pt file or a directory of them")
    for field in SLOT_FIELDS:
        parser.add_argument(f"--{field}", help=f"the {field} slot value")
    parser.add_argument("--sample-eval", type=Path, default=None,
                        help="jsonl split to sample a Scaffold from instead of slots")
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    scaffold = _scaffold_from_args(args)

    path = args.checkpoint
    if path.is_dir():
        path = latest_checkpoint(path)
        if path is None:
            raise SystemExit(f"no step_*.pt checkpoints in {args.checkpoint}")
    state = load_checkpoint(path)
    model = FableLM(ModelConfig(**state["config"]["model"]))
    model.load_state_dict(state["model"])

    tokenizer_path = args.tokenizer or state["config"]["data"]["tokenizer_path"]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    end_id = tokenizer.token_to_id(END_TOKEN)

    prompt_ids = tokenizer.encode(render_prompt(scaffold)).ids
    sequence = sample(
        model, prompt_ids, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
        seed=args.seed, end_id=end_id,
    )[0]
    body = tokenizer.decode(sequence[len(prompt_ids):])  # skips <|end|> by default

    print("Scaffold")
    for field in SLOT_FIELDS:
        print(f"  {field}: {getattr(scaffold, field)}")
    print("\nFable\n" + body.strip())


if __name__ == "__main__":
    main()
