import builtins
import math
import subprocess
import sys

import pytest
import torch

from tinystories_v2.perplexity import perplexity


class UniformLogitsModel(torch.nn.Module):
    """Zero logits everywhere, so perplexity must equal vocab_size."""

    def __init__(
        self,
        vocab_size: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dtype = dtype

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        return torch.zeros(
            batch,
            seq_len,
            self.vocab_size,
            dtype=self.dtype,
            device=input_ids.device,
        )


class ToyByteModel(torch.nn.Module):
    """Tiny seeded byte-vocab model standing in for a real checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(256, 8)
        self.head = torch.nn.Linear(8, 256)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids))


@pytest.fixture()
def fixture_byte_ids(fixture_records) -> list[int]:
    # Real fixture Fable text, tokenized with the trivial byte vocab so
    # the test needs no trained tokenizer artifact.
    return list(fixture_records[0]["fable"].encode("utf-8"))


def test_uniform_model_perplexity_is_vocab_size():
    token_ids = list(range(11)) * 3
    result = perplexity(UniformLogitsModel(11), token_ids, block_size=7)
    assert result == pytest.approx(11.0)


def test_half_precision_logits_do_not_overflow_summed_loss():
    vocab_size = 4
    token_ids = [0] * 50_001
    model = UniformLogitsModel(vocab_size, dtype=torch.float16)

    result = perplexity(model, token_ids, block_size=50_000, batch_size=1)

    assert math.isfinite(result)
    assert result == pytest.approx(vocab_size)


def test_matches_hand_rolled_nll_on_fixture(fixture_byte_ids):
    model = ToyByteModel()
    result = perplexity(model, fixture_byte_ids, block_size=16, batch_size=4)

    ids = torch.tensor(fixture_byte_ids, dtype=torch.long)
    total_nll = 0.0
    total_targets = 0
    with torch.inference_mode():
        for start in range(0, ids.numel() - 1, 16):
            targets = ids[start + 1 : start + 17]
            inputs = ids[start : start + 16][: targets.numel()]
            log_probs = torch.log_softmax(
                model(inputs.unsqueeze(0))[0], dim=-1
            )
            picked = log_probs[torch.arange(targets.numel()), targets]
            total_nll -= float(picked.sum())
            total_targets += targets.numel()

    assert result == pytest.approx(math.exp(total_nll / total_targets))


def test_batch_size_never_changes_the_result(fixture_byte_ids):
    model = ToyByteModel()
    single = perplexity(model, fixture_byte_ids, block_size=16, batch_size=1)
    batched = perplexity(model, fixture_byte_ids, block_size=16, batch_size=5)
    assert batched == pytest.approx(single)


def test_deterministic_across_calls(fixture_byte_ids):
    model = ToyByteModel()
    first = perplexity(model, fixture_byte_ids, block_size=16)
    second = perplexity(model, fixture_byte_ids, block_size=16)
    assert first == second


def test_rejects_fewer_than_two_tokens():
    with pytest.raises(ValueError, match="at least two"):
        perplexity(UniformLogitsModel(11), [5], block_size=4)


def test_rejects_non_1d_token_ids():
    with pytest.raises(ValueError, match="flat 1-D"):
        perplexity(UniformLogitsModel(11), [[1, 2], [3, 4]], block_size=2)


@pytest.mark.parametrize(
    "kwargs",
    [{"block_size": 0}, {"block_size": 4, "batch_size": 0}],
)
def test_rejects_non_positive_sizes(kwargs):
    with pytest.raises(ValueError, match="at least 1"):
        perplexity(UniformLogitsModel(11), [1, 2, 3], **kwargs)


def test_import_never_eagerly_pulls_torch():
    code = (
        "import sys\n"
        "import tinystories_v2.metrics\n"
        "import tinystories_v2.perplexity\n"
        "assert 'torch' not in sys.modules, 'torch imported eagerly'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_torch_raises_install_guidance(monkeypatch):
    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("PyTorch deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)

    with pytest.raises(RuntimeError, match=r"requires PyTorch.*install"):
        perplexity(None, [1, 2], block_size=1)
