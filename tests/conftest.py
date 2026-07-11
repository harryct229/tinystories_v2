import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "tf1_sample.jsonl"


@pytest.fixture(scope="session")
def fixture_records(fixture_path) -> list[dict]:
    with fixture_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def make_init_checkpoint():
    """Write a minimal Pretraining-style checkpoint the SFT stage can init from
    without running the pretrain stage. The SFT init path reads only
    state['model'] and state['config']['model']; the rest satisfies the schema
    and lets generate/demo find the tokenizer."""
    import torch
    from tinystories_v2.checkpoint import save_checkpoint
    from tinystories_v2.model import FableLM, ModelConfig

    def _make(init_dir, model_cfg: dict, tokenizer_path) -> Path:
        torch.manual_seed(0)
        model = FableLM(ModelConfig(**model_cfg))
        save_checkpoint(Path(init_dir) / "checkpoints", 0, {
            "step": 0, "tokens_seen": 0,
            "model": model.state_dict(), "optimizer": {}, "scaler": {},
            "config": {"model": dict(model_cfg),
                       "data": {"tokenizer_path": str(tokenizer_path)}},
        })
        return Path(init_dir)

    return _make
