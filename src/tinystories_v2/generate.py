"""Seedable temperature/top-p sampling from any checkpoint.

The shared sampling path for later stages: preference-data rollouts (N
completions per Slot Prompt), evaluation, and the demo. Batched over
num_samples for one prompt; loop over prompts for more (no KV cache yet —
at ~30M params and 512 context a full forward per token is fast enough;
revisit if preference-data generation becomes the bottleneck).

Invoke standalone:
    ts2-generate --checkpoint artifacts/pretrain_fixture/checkpoints \
        --prompt "Once upon a time" --num-samples 2 --seed 7
    (or: python -m tinystories_v2.generate ...)

--tokenizer defaults to the tokenizer_path recorded in the checkpoint's config.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from tinystories_v2.checkpoint import latest_checkpoint, load_checkpoint
from tinystories_v2.model import FableLM, ModelConfig


@torch.no_grad()
def sample(model: FableLM, prompt_ids: list[int], *, num_samples: int = 1,
           max_new_tokens: int, temperature: float = 1.0, top_p: float = 1.0,
           seed: int | None = None, end_id: int | None = None,
           device: str = "cpu") -> list[list[int]]:
    context = model.config.context
    if len(prompt_ids) > context:
        raise ValueError(f"prompt length {len(prompt_ids)} exceeds context {context}")
    model = model.to(device).eval()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    idx = idx.unsqueeze(0).expand(num_samples, -1).contiguous()
    finished = torch.zeros(num_samples, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -context:])[:, -1]  # [num_samples, vocab]
        if temperature == 0.0:
            next_ids = logits.argmax(dim=-1)
        else:
            logits = logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_ix = logits.sort(dim=-1, descending=True)
                cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                # Drop tokens once the cumulative mass before them exceeds top_p
                # (the first token always survives).
                drop = cumulative - sorted_logits.softmax(dim=-1) > top_p
                sorted_logits[drop] = -float("inf")
                logits = torch.full_like(logits, -float("inf")).scatter(
                    -1, sorted_ix, sorted_logits)
            probs = logits.softmax(dim=-1)
            next_ids = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        idx = torch.cat([idx, next_ids.unsqueeze(-1)], dim=-1)
        if end_id is not None:
            finished |= next_ids == end_id
            if bool(finished.all()):
                break

    sequences = []
    for row in idx.tolist():
        if end_id is not None and end_id in row[len(prompt_ids):]:
            cut = row.index(end_id, len(prompt_ids))
            row = row[:cut + 1]
        sequences.append(row)
    return sequences


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="a step_*.pt file or a directory of them")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    from tokenizers import Tokenizer

    from tinystories_v2.slots import SLOT_SPECIAL_TOKENS

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
    end_id = tokenizer.token_to_id(SLOT_SPECIAL_TOKENS[-1])

    sequences = sample(
        model, tokenizer.encode(args.prompt).ids,
        num_samples=args.num_samples, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
        seed=args.seed, end_id=end_id,
    )
    print("\n---\n".join(tokenizer.decode(seq) for seq in sequences))


if __name__ == "__main__":
    main()
