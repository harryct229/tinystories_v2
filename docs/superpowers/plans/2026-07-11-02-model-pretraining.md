# Issue 02 — Model + Pretraining Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The hand-written Llama-style model and the Pretraining stage, demoable as: pack the fixture, train a toy config on CPU, kill the process, resume from the latest checkpoint, and generate text from the result.

**Architecture:** `model.py` is a plain PyTorch `nn.Module` (pre-norm RMSNorm, RoPE, SwiGLU, no biases, tied embeddings), fully driven by a `ModelConfig` dataclass. The Pretraining stage (`ts2-pretrain`) follows the established config→artifacts convention: it packs the pretrain split into a uint16 binary (once, skipped if present), then runs a hand-written training loop with a precision knob, grad accumulation, AdamW, warmup+cosine, grad clipping, JSONL-always/W&B-optional metrics, and atomic checkpoints. **Batches are a pure function of `(seed, step, micro_step)`** — no global RNG state in the data path — which is what makes kill-and-resume bitwise-reproducible and testable. A thin `hub.py` sync layer treats a plain local path and `hf://<repo_id>` uniformly so tests never touch the network.

**Tech Stack:** PyTorch ≥2.6 (verified: torch 2.13.0 ships cp314 macOS arm64 wheels — the existing Python 3.14 venv works), numpy (memmap packed data), `tokenizers` (already a dep), `huggingface_hub` (sync layer), optional `wandb` extra, `pytest`.

## Global Constraints

Copied from the spec (`.scratch/tinystories-v2-pipeline/issues/02-model-pretraining-stage.md`, PRD, `docs/DESIGN.md`, ADR-0002/0005). Every task's requirements implicitly include these.

- **Model (ADR-0002, ADR-0005):** hand-written plain PyTorch `nn.Module`, Llama-style decoder-only — pre-norm RMSNorm (eps 1e-5), RoPE (θ 10 000), SwiGLU FFN, **no biases anywhere**, tied embeddings, dropout 0.0. No HF Trainer/TRL — libraries only at the edges.
- **Real config:** vocab 8192, d_model 512, 8 layers, 8 heads (head_dim 64), context 512, SwiGLU hidden 1408. Exact unique-parameter count for these numbers is **29,893,120**; budget assertion is `< 32_000_000`. (DESIGN.md's "≈27M" was a rough early estimate — the exact count follows from the formula in Task 1; a docs correction is Task 9.)
- **Packed data:** Fable text only (no prompts), each fable's token IDs followed by the `<|end|>` ID, concatenated flat, dtype **uint16** (vocab 8192 < 65 536), loaded with `np.memmap`.
- **Optimizer/schedule defaults (DESIGN.md):** AdamW β=(0.9, 0.95), weight decay 0.1 (2D+ params only), grad clip 1.0, peak LR 6e-4, 1.5% linear warmup, cosine to 10% of peak. Real batch: micro 32 × ctx 512 × accum 8; ~3 800 steps.
- **Precision is a config knob:** `"fp32"` (CPU/tests), `"bf16"` (L4 autocast, no scaler), `"fp16"` (T4 autocast + GradScaler). One uniform code path — a disabled GradScaler passes through.
- **Checkpoint-resume contract:** full training state (model, optimizer, scaler, step, tokens_seen, config) persisted periodically with **atomic writes** (tmp + `os.replace`); resume via one `--resume` flag from the latest checkpoint.
- **Hub sync:** thin layer; target `hf://<repo_id>` → private HF Hub repo, anything else → local directory path. Tests use only local paths — **no network, no GPU in the test suite**.
- Metrics **always** append to `<out_dir>/metrics.jsonl` (flushed per line — must survive SIGKILL); W&B streams additionally when enabled and degrades gracefully (warn, continue) when unavailable.
- Stage convention: declarative TOML in → versioned artifacts + `manifest.json` in `out_dir`; stages share nothing in memory. Entrypoints runnable as console script and `python -m`.
- Secrets (HF token, W&B key) from environment/`.env` via `config.load_env()`; never printed or committed.
- Use CONTEXT.md vocabulary in code comments/docstrings: Fable, Scaffold, Slot Prompt, Pretraining (never plain "training" across stages).
- New dependencies: `torch>=2.6`, `numpy>=1.26` in main deps; `wandb>=0.17` as optional extra `track` (never imported unless enabled). Install into the existing `.venv` with `uv pip install -e '.[dev]'`.
- The Colab notebook contains **no logic** beyond clone → install → secrets → stage invocation.
- Every commit message ends with the trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

## File Structure

```
pyproject.toml                      # + torch, numpy deps; + ts2-pretrain / ts2-generate scripts; + track extra
configs/
    pretrain_fixture.toml           # toy CPU run against the committed fixture
    pretrain_full.toml              # real ~30M / 500M-token run (design-doc defaults)
notebooks/
    pretrain_colab.ipynb            # thin wrapper: clone -> install -> secrets -> ts2-pretrain
src/tinystories_v2/
    model.py                        # ModelConfig, RMSNorm, RoPE, Attention, SwiGLU MLP, Block, FableLM
    pack.py                         # pack_split, load_packed, get_batch (pure fn of seed/step/micro_step)
    checkpoint.py                   # atomic save_checkpoint, latest_checkpoint, load_checkpoint, prune
    tracking.py                     # MetricsLogger: JSONL always, W&B optional
    hub.py                          # sync_to / fetch_from: local path or hf://<repo_id>
    pretrain.py                     # Pretraining stage: run, main (--config, --resume), lr_at, build_optimizer
    generate.py                     # sample() + ts2-generate CLI
tests/
    test_model.py
    test_pack.py
    test_checkpoint.py
    test_tracking.py
    test_hub_sync.py
    test_pretrain_stage.py          # loss decreases, schedule, full-config budget
    test_resume.py                  # kill-and-resume via subprocess SIGKILL
    test_generate.py
    test_notebook.py
```

Existing modules consumed: `config.load_config`/`load_env`, `slots.SLOT_SPECIAL_TOKENS` (`"<|end|>"` is its last element), tokenizer artifacts (`tokenizer.json` loadable via `tokenizers.Tokenizer.from_file`), `tests/conftest.py` fixtures `fixture_path`/`fixture_records`, committed fixture `tests/fixtures/tf1_sample.jsonl` (~120 real records with `fable` field).

---

### Task 1: Dependencies + the hand-written model

**Files:**
- Modify: `pyproject.toml`
- Create: `src/tinystories_v2/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `ModelConfig(vocab_size: int, d_model: int, n_layers: int, n_heads: int, context: int, ffn_hidden: int, rope_theta: float = 10000.0, norm_eps: float = 1e-5)` — frozen dataclass; later tasks build it as `ModelConfig(**config["model"])` from TOML.
  - `FableLM(config: ModelConfig)` — `forward(idx: LongTensor[B,T]) -> FloatTensor[B,T,vocab_size]` (logits, no loss); attribute `model.config`; `model.num_params() -> int` (tied weights counted once).
  - Tied embeddings: `model.lm_head.weight is model.tok_emb.weight`.

- [ ] **Step 1: Add torch + numpy deps and the wandb extra**

In `pyproject.toml`, change the `dependencies` and `optional-dependencies` sections to:

```toml
dependencies = [
    "datasets>=2.19",
    "tokenizers>=0.19",
    "huggingface_hub>=0.23",
    "torch>=2.6",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = ["pytest>=8"]
track = ["wandb>=0.17"]
```

Then install: `uv pip install -e '.[dev]'` (from repo root, venv `.venv`). Verify: `.venv/bin/python -c "import torch; print(torch.__version__)"` prints ≥2.6.

- [ ] **Step 2: Write the failing model tests**

Create `tests/test_model.py`:

```python
import dataclasses

import pytest
import torch

from tinystories_v2.model import FableLM, ModelConfig

TOY = ModelConfig(
    vocab_size=512, d_model=64, n_layers=2, n_heads=2, context=64, ffn_hidden=192
)
# The real run's architecture numbers (DESIGN.md). Duplicated in
# configs/pretrain_full.toml; test_pretrain_stage.py ties the two together.
REAL = ModelConfig(
    vocab_size=8192, d_model=512, n_layers=8, n_heads=8, context=512, ffn_hidden=1408
)
PARAM_BUDGET = 32_000_000


def expected_params(c: ModelConfig) -> int:
    per_layer = (
        4 * c.d_model * c.d_model        # q, k, v, o projections (no biases)
        + 3 * c.d_model * c.ffn_hidden   # SwiGLU gate, up, down
        + 2 * c.d_model                  # two RMSNorm weights
    )
    return c.vocab_size * c.d_model + c.n_layers * per_layer + c.d_model  # + final norm


@pytest.fixture(scope="module")
def toy_model():
    torch.manual_seed(0)
    return FableLM(TOY).eval()


def test_forward_shape_fp32(toy_model):
    idx = torch.randint(TOY.vocab_size, (2, 16))
    logits = toy_model(idx)
    assert logits.shape == (2, 16, TOY.vocab_size)
    assert logits.dtype == torch.float32


def test_causality_future_tokens_do_not_affect_past_logits(toy_model):
    torch.manual_seed(1)
    a = torch.randint(TOY.vocab_size, (2, 32))
    b = a.clone()
    b[:, 20:] = (b[:, 20:] + 7) % TOY.vocab_size  # perturb only the future
    with torch.no_grad():
        la, lb = toy_model(a), toy_model(b)
    assert torch.allclose(la[:, :20], lb[:, :20], atol=1e-5)
    assert not torch.allclose(la[:, 20:], lb[:, 20:], atol=1e-5)


def test_tied_embeddings_share_storage(toy_model):
    assert toy_model.lm_head.weight is toy_model.tok_emb.weight


def test_no_biases_anywhere(toy_model):
    for name, _ in toy_model.named_parameters():
        assert not name.endswith("bias"), name


def test_param_count_matches_analytic_formula(toy_model):
    assert toy_model.num_params() == expected_params(TOY)


def test_real_config_param_count_within_budget():
    assert expected_params(REAL) == 29_893_120
    torch.manual_seed(0)
    model = FableLM(REAL)
    assert model.num_params() == 29_893_120 < PARAM_BUDGET


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_under_autocast(toy_model, dtype):
    idx = torch.randint(TOY.vocab_size, (2, 16))
    with torch.autocast(device_type="cpu", dtype=dtype), torch.no_grad():
        logits = toy_model(idx)
    assert logits.shape == (2, 16, TOY.vocab_size)
    assert torch.isfinite(logits.float()).all()


def test_config_is_frozen_and_buildable_from_dict():
    cfg = ModelConfig(**{
        "vocab_size": 512, "d_model": 64, "n_layers": 2, "n_heads": 2,
        "context": 64, "ffn_hidden": 192,
    })
    assert cfg.rope_theta == 10000.0 and cfg.norm_eps == 1e-5
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.d_model = 1


def test_d_model_must_divide_by_heads():
    with pytest.raises(ValueError, match="n_heads"):
        FableLM(ModelConfig(vocab_size=64, d_model=65, n_layers=1, n_heads=2,
                            context=8, ffn_hidden=32))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_model.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'tinystories_v2.model'`.

- [ ] **Step 4: Implement the model**

Create `src/tinystories_v2/model.py`:

```python
"""Hand-written Llama-style decoder-only LM (ADR-0002, ADR-0005).

Pre-norm RMSNorm, RoPE, SwiGLU FFN, no biases, tied embeddings, dropout 0.0
(single-pass data regime). Fully config-driven: the toy test config and the
real ~30M config differ only in numbers. Report citations per component:
RMSNorm (Zhang & Sennrich 2019), RoPE (Su et al. 2021), SwiGLU (Shazeer 2020).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    context: int
    ffn_hidden: int
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in fp32 for stability under bf16/fp16 autocast, then cast back.
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm.type_as(x)


def _rope_cache(head_dim: int, context: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    positions = torch.arange(context).float()
    freqs = torch.outer(positions, inv_freq)  # [context, head_dim//2]
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]; rotate interleaved pairs (x0,x1), (x2,x3), ...
    t = x.size(-2)
    cos, sin = cos[:t].to(x.dtype), sin[:t].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        shape = (b, t, self.n_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).reshape(b, t, d))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class FableLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError(
                f"d_model={config.d_model} not divisible by n_heads={config.n_heads}"
            )
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied embeddings (param budget)
        cos, sin = _rope_cache(config.d_model // config.n_heads, config.context,
                               config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # GPT-2-style scaled init on residual-output projections.
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        # parameters() deduplicates shared tensors, so tied weights count once.
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.size(1) > self.config.context:
            raise ValueError(
                f"sequence length {idx.size(1)} exceeds context {self.config.context}"
            )
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        return self.lm_head(self.final_norm(x))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_model.py -v`
Expected: all PASS. Then run the whole suite to check nothing regressed: `.venv/bin/python -m pytest`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/tinystories_v2/model.py tests/test_model.py
git commit -m "feat: hand-written Llama-style FableLM (RMSNorm, RoPE, SwiGLU, tied embeddings)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Packed pretrain data (uint16 binary + deterministic batches)

**Files:**
- Create: `src/tinystories_v2/pack.py`
- Test: `tests/test_pack.py`

**Interfaces:**
- Consumes: a split JSONL (rows with a `fable` field, as written by the data-prep stage — the committed fixture has the same shape) and a `tokenizer.json` artifact; `slots.SLOT_SPECIAL_TOKENS[-1]` == `"<|end|>"`.
- Produces (used by Tasks 6–7):
  - `pack_split(split_path: str | Path, tokenizer_path: str | Path, out_path: str | Path) -> dict` — writes `<out_path>` (flat uint16 token IDs) and `<out_path>.json` (manifest documenting dtype/shape) and returns the manifest dict.
  - `load_packed(path: str | Path) -> np.memmap` — read-only uint16 memmap.
  - `get_batch(data, micro_batch_size: int, context: int, *, seed: int, step: int, micro_step: int, device: str = "cpu") -> tuple[Tensor, Tensor]` — `(x, y)` LongTensors `[B, context]`, y shifted by one. **Pure function of its arguments** — no global RNG.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pack.py`:

```python
import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from tinystories_v2.pack import get_batch, load_packed, pack_split
from tinystories_v2.slots import SLOT_SPECIAL_TOKENS
from tinystories_v2.tokenizer import run as run_tokenizer

END_TOKEN = SLOT_SPECIAL_TOKENS[-1]


@pytest.fixture(scope="module")
def tokenizer_path(tmp_path_factory, fixture_path):
    out = tmp_path_factory.mktemp("tok")
    run_tokenizer({
        "out_dir": str(out), "corpus": [str(fixture_path)],
        "text_field": "fable", "vocab_size": 512,
    })
    return out / "tokenizer.json"


@pytest.fixture(scope="module")
def packed(tmp_path_factory, fixture_path, tokenizer_path):
    out = tmp_path_factory.mktemp("packed") / "pretrain.bin"
    manifest = pack_split(fixture_path, tokenizer_path, out)
    return out, manifest


def test_manifest_documents_dtype_and_shape(packed, fixture_records):
    out, manifest = packed
    assert manifest["dtype"] == "uint16"
    assert manifest["n_docs"] == len(fixture_records)
    assert manifest["n_tokens"] == out.stat().st_size // 2  # uint16 = 2 bytes
    on_disk = json.loads((out.parent / "pretrain.bin.json").read_text(encoding="utf-8"))
    assert on_disk == manifest


def test_roundtrip_first_fable_against_tokenizer(packed, fixture_records, tokenizer_path):
    out, manifest = packed
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    data = load_packed(out)
    end_id = manifest["end_id"]
    assert end_id == tokenizer.token_to_id(END_TOKEN)
    first_end = int(np.argmax(data == end_id))
    decoded = tokenizer.decode(data[:first_end].tolist())
    assert decoded == fixture_records[0]["fable"]


def test_every_doc_is_end_separated(packed, fixture_records):
    out, manifest = packed
    data = load_packed(out)
    assert int((data == manifest["end_id"]).sum()) == len(fixture_records)
    assert int(data[-1]) == manifest["end_id"]


def test_get_batch_shapes_and_shift(packed):
    out, _ = packed
    data = load_packed(out)
    x, y = get_batch(data, 4, 32, seed=1, step=0, micro_step=0)
    assert x.shape == y.shape == (4, 32)
    assert x.dtype == y.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])  # y is x shifted left by one


def test_get_batch_is_pure_function_of_seed_step_microstep(packed):
    out, _ = packed
    data = load_packed(out)
    a = get_batch(data, 4, 32, seed=1, step=5, micro_step=2)
    b = get_batch(data, 4, 32, seed=1, step=5, micro_step=2)
    c = get_batch(data, 4, 32, seed=1, step=5, micro_step=3)
    d = get_batch(data, 4, 32, seed=2, step=5, micro_step=2)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0])
    assert not torch.equal(a[0], d[0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pack.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'tinystories_v2.pack'`.

- [ ] **Step 3: Implement pack.py**

Create `src/tinystories_v2/pack.py`:

```python
"""Packed Pretraining data: Fable text -> flat uint16 token-ID binary.

Format (documented contract, see <out>.json manifest):
    - dtype uint16 little-endian (vocab 8192 < 65536), flat 1-D array
    - each Fable's token IDs followed by the <|end|> ID, docs concatenated
    - no header; length in tokens = file size / 2 and is recorded in the manifest

Batches for the training loop are a pure function of (seed, step, micro_step)
so an interrupted run resumed from a checkpoint sees exactly the batches the
uninterrupted run would have seen (checkpoint-resume contract).
"""

import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from tinystories_v2 import __version__
from tinystories_v2.slots import SLOT_SPECIAL_TOKENS

END_TOKEN = SLOT_SPECIAL_TOKENS[-1]  # "<|end|>"


def pack_split(split_path: str | Path, tokenizer_path: str | Path,
               out_path: str | Path) -> dict:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    end_id = tokenizer.token_to_id(END_TOKEN)
    if end_id is None:
        raise ValueError(f"tokenizer at {tokenizer_path} lacks the {END_TOKEN} token")
    if tokenizer.get_vocab_size() > 2**16:
        raise ValueError("vocab does not fit uint16")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_docs = n_tokens = 0
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(split_path, encoding="utf-8") as src, open(tmp, "wb") as dst:
        for line in src:
            if not line.strip():
                continue
            ids = tokenizer.encode(json.loads(line)["fable"]).ids + [end_id]
            np.asarray(ids, dtype=np.uint16).tofile(dst)
            n_docs += 1
            n_tokens += len(ids)
    tmp.replace(out_path)

    manifest = {
        "stage": "pack",
        "package_version": __version__,
        "dtype": "uint16",
        "n_tokens": n_tokens,
        "n_docs": n_docs,
        "vocab_size": tokenizer.get_vocab_size(),
        "end_id": end_id,
    }
    Path(str(out_path) + ".json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def load_packed(path: str | Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data: np.memmap, micro_batch_size: int, context: int, *,
              seed: int, step: int, micro_step: int,
              device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(((seed * 1_000_003 + step) * 1_009 + micro_step) % 2**63)
    offsets = torch.randint(0, len(data) - context - 1, (micro_batch_size,),
                            generator=generator)
    x = torch.stack([
        torch.from_numpy(data[o:o + context].astype(np.int64)) for o in offsets
    ])
    y = torch.stack([
        torch.from_numpy(data[o + 1:o + 1 + context].astype(np.int64)) for o in offsets
    ])
    return x.to(device), y.to(device)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pack.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/pack.py tests/test_pack.py
git commit -m "feat: uint16 packed Pretraining data with deterministic batch sampling

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Atomic checkpoints

**Files:**
- Create: `src/tinystories_v2/checkpoint.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: nothing from other tasks (state dicts are plain dicts of tensors/ints/strs).
- Produces (used by Tasks 6–8):
  - `save_checkpoint(ckpt_dir: Path, step: int, state: dict) -> Path` — writes `step_{step:06d}.pt` atomically (tmp + `os.replace`), returns the final path. `state` must contain only `torch.load(weights_only=True)`-safe values (tensors, dicts, lists, numbers, strings, None).
  - `latest_checkpoint(ckpt_dir: Path) -> Path | None` — highest-step `step_*.pt`; ignores tmp/partial files; `None` if none exist.
  - `load_checkpoint(path: Path) -> dict` — `torch.load(path, map_location="cpu", weights_only=True)`.
  - `prune_checkpoints(ckpt_dir: Path, keep_last: int) -> None` — deletes all but the newest `keep_last` (no-op when `keep_last <= 0`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_checkpoint.py`:

```python
import torch

from tinystories_v2.checkpoint import (
    latest_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint,
)


def _state(step):
    return {"step": step, "model": {"w": torch.full((2, 2), float(step))}}


def test_save_load_roundtrip(tmp_path):
    path = save_checkpoint(tmp_path, 7, _state(7))
    assert path.name == "step_000007.pt"
    loaded = load_checkpoint(path)
    assert loaded["step"] == 7
    assert torch.equal(loaded["model"]["w"], torch.full((2, 2), 7.0))


def test_latest_picks_highest_step(tmp_path):
    for step in (2, 10, 6):
        save_checkpoint(tmp_path, step, _state(step))
    assert latest_checkpoint(tmp_path).name == "step_000010.pt"


def test_latest_ignores_partial_tmp_files(tmp_path):
    save_checkpoint(tmp_path, 3, _state(3))
    (tmp_path / "step_000099.pt.tmp").write_bytes(b"partial garbage")
    assert latest_checkpoint(tmp_path).name == "step_000003.pt"


def test_latest_none_when_empty(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    assert latest_checkpoint(tmp_path / "missing") is None


def test_no_tmp_left_behind_after_save(tmp_path):
    save_checkpoint(tmp_path, 1, _state(1))
    assert list(tmp_path.glob("*.tmp")) == []


def test_prune_keeps_newest(tmp_path):
    for step in (1, 2, 3, 4):
        save_checkpoint(tmp_path, step, _state(step))
    prune_checkpoints(tmp_path, keep_last=2)
    assert sorted(p.name for p in tmp_path.glob("step_*.pt")) == [
        "step_000003.pt", "step_000004.pt",
    ]
    prune_checkpoints(tmp_path, keep_last=0)  # 0 = keep all
    assert len(list(tmp_path.glob("step_*.pt"))) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_checkpoint.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'tinystories_v2.checkpoint'`.

- [ ] **Step 3: Implement checkpoint.py**

Create `src/tinystories_v2/checkpoint.py`:

```python
"""Atomic training-state checkpoints (checkpoint-resume contract).

A checkpoint file only ever appears under its final name via os.replace, so a
process killed mid-write can never leave a corrupt step_*.pt behind — resume
always finds the last complete state. State dicts stay weights_only-safe
(plain containers of tensors/numbers/strings) so loading never unpickles code.
"""

import re
from pathlib import Path

import torch

_STEP_RE = re.compile(r"step_(\d{6})\.pt$")


def save_checkpoint(ckpt_dir: Path, step: int, state: dict) -> Path:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final = ckpt_dir / f"step_{step:06d}.pt"
    tmp = final.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    tmp.replace(final)
    return final


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    steps = {
        int(m.group(1)): p
        for p in ckpt_dir.iterdir()
        if (m := _STEP_RE.fullmatch(p.name))
    }
    return steps[max(steps)] if steps else None


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def prune_checkpoints(ckpt_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    paths = sorted(Path(ckpt_dir).glob("step_*.pt"))
    for path in paths[:-keep_last]:
        path.unlink()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_checkpoint.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: atomic checkpoint save/load/latest/prune

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Training-metrics tracking (JSONL always, W&B optional)

(Named `tracking.py`, not `metrics.py` — issue 11's plan reserves `tinystories_v2.metrics` for the reference-free evaluation metrics library.)

**Files:**
- Create: `src/tinystories_v2/tracking.py`
- Test: `tests/test_tracking.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (used by Task 6):
  - `MetricsLogger(out_dir: Path, wandb_config: dict | None = None)` — `wandb_config` is the TOML `[wandb]` table, e.g. `{"enabled": true, "project": "tinystories-v2", "run_name": "pretrain-500M"}`. `None` or `enabled: false` → JSONL only.
  - `logger.log(metrics: dict, step: int) -> None` — appends one JSON line `{"step": ..., **metrics}` to `<out_dir>/metrics.jsonl`, flushed immediately; forwards to W&B when active.
  - `logger.finish() -> None` — closes the file and the W&B run if any.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tracking.py`:

```python
import json
import sys
import types

import pytest

from tinystories_v2.tracking import MetricsLogger


def read_lines(out_dir):
    text = (out_dir / "metrics.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_jsonl_written_and_flushed_per_line(tmp_path):
    logger = MetricsLogger(tmp_path)
    logger.log({"loss": 2.5, "lr": 1e-4, "tokens_seen": 4096}, step=1)
    # Readable BEFORE finish(): a SIGKILLed run must not lose logged lines.
    assert read_lines(tmp_path) == [
        {"step": 1, "loss": 2.5, "lr": 1e-4, "tokens_seen": 4096}
    ]
    logger.log({"loss": 2.0}, step=2)
    logger.finish()
    assert [line["step"] for line in read_lines(tmp_path)] == [1, 2]


def test_append_mode_survives_reopen(tmp_path):
    a = MetricsLogger(tmp_path)
    a.log({"loss": 3.0}, step=1)
    a.finish()
    b = MetricsLogger(tmp_path)  # a resumed run re-opens the same file
    b.log({"loss": 2.0}, step=2)
    b.finish()
    assert [line["step"] for line in read_lines(tmp_path)] == [1, 2]


def test_wandb_streams_when_enabled(tmp_path, monkeypatch):
    calls = []
    run = types.SimpleNamespace(
        log=lambda data, step: calls.append(("log", data, step)),
        finish=lambda: calls.append(("finish",)),
    )
    fake = types.ModuleType("wandb")
    fake.init = lambda **kw: calls.append(("init", kw)) or run
    monkeypatch.setitem(sys.modules, "wandb", fake)

    logger = MetricsLogger(
        tmp_path, {"enabled": True, "project": "p", "run_name": "r"}
    )
    logger.log({"loss": 1.5}, step=3)
    logger.finish()

    assert calls[0] == ("init", {"project": "p", "name": "r", "resume": "allow"})
    assert calls[1] == ("log", {"loss": 1.5}, 3)
    assert calls[2] == ("finish",)
    assert read_lines(tmp_path)[0]["loss"] == 1.5  # JSONL still written


def test_degrades_to_jsonl_when_wandb_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # import wandb -> ImportError
    with pytest.warns(UserWarning, match="wandb"):
        logger = MetricsLogger(tmp_path, {"enabled": True, "project": "p"})
    logger.log({"loss": 1.0}, step=1)
    logger.finish()
    assert read_lines(tmp_path) == [{"step": 1, "loss": 1.0}]


def test_disabled_wandb_never_imported(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    logger = MetricsLogger(tmp_path, {"enabled": False})  # must not raise or warn
    logger.log({"loss": 1.0}, step=1)
    logger.finish()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tracking.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'tinystories_v2.tracking'`.

- [ ] **Step 3: Implement tracking.py**

Create `src/tinystories_v2/tracking.py`:

```python
"""Training metrics: local JSONL always, W&B stream when enabled.

The JSONL file is the source of truth (metrics must survive session death and
SIGKILL, and the resume test replays it); W&B is an additive stream. wandb is
imported only when enabled so the package works without the `track` extra,
degrading with a warning if enabled but unavailable.
"""

import json
import warnings
from pathlib import Path


class MetricsLogger:
    def __init__(self, out_dir: Path, wandb_config: dict | None = None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._file = (out_dir / "metrics.jsonl").open("a", encoding="utf-8")
        self._run = None
        if wandb_config and wandb_config.get("enabled"):
            try:
                import wandb
            except ImportError:
                warnings.warn(
                    "wandb enabled in config but not importable; "
                    "logging to metrics.jsonl only",
                    stacklevel=2,
                )
            else:
                self._run = wandb.init(
                    project=wandb_config.get("project", "tinystories-v2"),
                    name=wandb_config.get("run_name"),
                    resume="allow",
                )

    def log(self, metrics: dict, step: int) -> None:
        self._file.write(json.dumps({"step": step, **metrics}) + "\n")
        self._file.flush()
        if self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        self._file.close()
        if self._run is not None:
            self._run.finish()
```

Note: the fake-wandb test asserts `init` is called with exactly `project`, `name`, `resume` — keep that call signature. `monkeypatch.setitem(sys.modules, "wandb", None)` makes `import wandb` raise `ImportError` (stdlib behavior for `None` in `sys.modules`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tracking.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/tracking.py tests/test_tracking.py
git commit -m "feat: training-metrics tracking — JSONL always, W&B optional with graceful degrade

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Hub sync layer

**Files:**
- Create: `src/tinystories_v2/hub.py`
- Test: `tests/test_hub_sync.py`

**Interfaces:**
- Consumes: `config.load_env` (HF token into env, never printed).
- Produces (used by Task 6 and the Colab workflow):
  - `sync_to(target: str, local_dir: Path) -> None` — mirror `local_dir`'s files into the target. `target` starting with `hf://` → `huggingface_hub` upload to a private model repo (created if missing); any other string → local directory path (tests, Drive scratch).
  - `fetch_from(target: str, local_dir: Path) -> None` — inverse: populate `local_dir` from the target.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hub_sync.py`:

```python
from pathlib import Path

import huggingface_hub

from tinystories_v2.hub import fetch_from, sync_to


def make_tree(root: Path) -> None:
    (root / "checkpoints").mkdir(parents=True)
    (root / "checkpoints" / "step_000002.pt").write_bytes(b"ckpt-bytes")
    (root / "metrics.jsonl").write_text('{"step": 1}\n', encoding="utf-8")


def test_local_roundtrip(tmp_path):
    src, mirror, restored = tmp_path / "src", tmp_path / "mirror", tmp_path / "restored"
    make_tree(src)
    sync_to(str(mirror), src)
    assert (mirror / "checkpoints" / "step_000002.pt").read_bytes() == b"ckpt-bytes"
    assert (mirror / "metrics.jsonl").exists()
    fetch_from(str(mirror), restored)
    assert (restored / "checkpoints" / "step_000002.pt").read_bytes() == b"ckpt-bytes"


def test_sync_overwrites_stale_files(tmp_path):
    src, mirror = tmp_path / "src", tmp_path / "mirror"
    make_tree(src)
    sync_to(str(mirror), src)
    (src / "metrics.jsonl").write_text('{"step": 2}\n', encoding="utf-8")
    sync_to(str(mirror), src)
    assert '"step": 2' in (mirror / "metrics.jsonl").read_text(encoding="utf-8")


def test_hf_target_dispatches_to_hub_api(tmp_path, monkeypatch):
    calls = []

    class FakeApi:
        def create_repo(self, repo_id, private, exist_ok, repo_type):
            calls.append(("create_repo", repo_id, private, exist_ok, repo_type))

        def upload_folder(self, folder_path, repo_id, repo_type):
            calls.append(("upload_folder", folder_path, repo_id, repo_type))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    make_tree(tmp_path / "src")
    sync_to("hf://team/tinystories-v2-pretrain", tmp_path / "src")
    assert calls == [
        ("create_repo", "team/tinystories-v2-pretrain", True, True, "model"),
        ("upload_folder", str(tmp_path / "src"), "team/tinystories-v2-pretrain", "model"),
    ]


def test_hf_fetch_dispatches_to_snapshot_download(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download",
        lambda repo_id, local_dir, repo_type: calls.append(
            (repo_id, local_dir, repo_type)
        ),
    )
    fetch_from("hf://team/tinystories-v2-pretrain", tmp_path / "dst")
    assert calls == [
        ("team/tinystories-v2-pretrain", str(tmp_path / "dst"), "model")
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hub_sync.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'tinystories_v2.hub'`.

- [ ] **Step 3: Implement hub.py**

Create `src/tinystories_v2/hub.py`:

```python
"""Thin artifact sync: local checkpoint dir <-> HF Hub or another local path.

Targets:
    hf://<repo_id>   private HF Hub model repo (created on first sync);
                     token comes from env/.env via load_env — never printed
    anything else    a local directory (tests, Drive scratch)

Stages write artifacts locally first and sync as a separate step, so the
training loop never blocks on (or fails because of) the network — a failed
sync is a warning, not a dead run. Uses module-level attribute access
(huggingface_hub.HfApi) so tests can monkeypatch without network.
"""

import shutil
import warnings
from pathlib import Path

import huggingface_hub

from tinystories_v2.config import load_env

_HF_PREFIX = "hf://"


def sync_to(target: str, local_dir: Path) -> None:
    local_dir = Path(local_dir)
    if target.startswith(_HF_PREFIX):
        load_env()
        repo_id = target[len(_HF_PREFIX):]
        api = huggingface_hub.HfApi()
        api.create_repo(repo_id, private=True, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=str(local_dir), repo_id=repo_id,
                          repo_type="model")
    else:
        dst = Path(target)
        for src_file in local_dir.rglob("*"):
            if not src_file.is_file():
                continue
            dst_file = dst / src_file.relative_to(local_dir)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)


def fetch_from(target: str, local_dir: Path) -> None:
    local_dir = Path(local_dir)
    if target.startswith(_HF_PREFIX):
        load_env()
        repo_id = target[len(_HF_PREFIX):]
        huggingface_hub.snapshot_download(
            repo_id=repo_id, local_dir=str(local_dir), repo_type="model"
        )
    else:
        sync_to(str(local_dir), Path(target))


def try_sync_to(target: str, local_dir: Path) -> None:
    """Best-effort sync for use inside the training loop."""
    try:
        sync_to(target, local_dir)
    except Exception as err:  # noqa: BLE001 — network errors must not kill training
        warnings.warn(f"hub sync to {target!r} failed: {err}", stacklevel=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hub_sync.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tinystories_v2/hub.py tests/test_hub_sync.py
git commit -m "feat: thin hub sync layer (local path or hf:// target)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Pretraining stage (pack → train loop → checkpoints)

**Files:**
- Create: `src/tinystories_v2/pretrain.py`
- Create: `configs/pretrain_fixture.toml`
- Create: `configs/pretrain_full.toml`
- Modify: `pyproject.toml` (add `ts2-pretrain` console script)
- Test: `tests/test_pretrain_stage.py`

**Interfaces:**
- Consumes: `FableLM`/`ModelConfig` (Task 1), `pack_split`/`load_packed`/`get_batch` (Task 2), `save_checkpoint`/`latest_checkpoint`/`load_checkpoint`/`prune_checkpoints` (Task 3), `MetricsLogger` (Task 4), `sync_to`/`fetch_from`/`try_sync_to` (Task 5), `config.load_config`.
- Produces (used by Tasks 7–9):
  - `run(config: dict, resume: bool = False) -> dict` — executes the stage, returns `{"step": final_step, "loss": last_loss}`; artifacts in `out_dir`: `pretrain.bin`(+`.json`) at `config["data"]["packed_path"]`, `checkpoints/step_*.pt`, `metrics.jsonl`, `manifest.json`.
  - `main(argv: list[str] | None = None)` — `--config PATH [--resume]`; console script `ts2-pretrain` and `python -m tinystories_v2.pretrain`.
  - `lr_at(step: int, total_steps: int, peak_lr: float, warmup_frac: float, min_lr_frac: float) -> float`.
  - Checkpoint state schema (Tasks 7–8 read it): `{"step": int, "tokens_seen": int, "model": state_dict, "optimizer": state_dict, "scaler": state_dict, "config": dict}` where `config` is the full TOML dict (so `generate.py` can rebuild `ModelConfig(**ckpt["config"]["model"])`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pretrain_stage.py`:

```python
import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from tinystories_v2.model import ModelConfig
from tinystories_v2.pretrain import lr_at, run
from tinystories_v2.tokenizer import run as run_tokenizer

REPO_ROOT = Path(__file__).parent.parent


def toy_config(tmp_path: Path, fixture_path: Path, tokenizer_path: Path, **train_overrides) -> dict:
    train = {
        "steps": 30, "micro_batch_size": 8, "grad_accum": 1,
        "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
        "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
        "precision": "fp32", "seed": 1337,
        "checkpoint_every": 10, "log_every": 1, "keep_last": 0,
    }
    train.update(train_overrides)
    return {
        "out_dir": str(tmp_path / "out"),
        "model": {"vocab_size": 512, "d_model": 64, "n_layers": 2,
                  "n_heads": 2, "context": 64, "ffn_hidden": 192},
        "data": {"split_path": str(fixture_path),
                 "tokenizer_path": str(tokenizer_path),
                 "packed_path": str(tmp_path / "packed" / "pretrain.bin")},
        "train": train,
        "wandb": {"enabled": False},
    }


@pytest.fixture(scope="module")
def tokenizer_path(tmp_path_factory, fixture_path):
    out = tmp_path_factory.mktemp("tok")
    run_tokenizer({"out_dir": str(out), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    return out / "tokenizer.json"


def read_metrics(out_dir: Path) -> list[dict]:
    lines = (out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_toy_run_decreases_loss_through_stage_entrypoint(tmp_path, fixture_path, tokenizer_path):
    config = toy_config(tmp_path, fixture_path, tokenizer_path)
    summary = run(config)
    metrics = read_metrics(Path(config["out_dir"]))
    assert len(metrics) == 30
    first, last = metrics[0], metrics[-1]
    assert last["loss"] < first["loss"] - 0.5  # random init starts near ln(512) ~ 6.2
    assert summary["step"] == 30
    # loss, LR, tokens seen all present per line (W&B-off degrade path)
    assert {"step", "loss", "lr", "tokens_seen"} <= first.keys()
    assert last["tokens_seen"] == 30 * 8 * 64  # steps * micro_batch * context


def test_stage_artifacts_and_manifest(tmp_path, fixture_path, tokenizer_path):
    config = toy_config(tmp_path, fixture_path, tokenizer_path, steps=4,
                        checkpoint_every=2)
    run(config)
    out = Path(config["out_dir"])
    assert Path(config["data"]["packed_path"]).exists()
    ckpts = sorted(p.name for p in (out / "checkpoints").glob("step_*.pt"))
    assert ckpts == ["step_000002.pt", "step_000004.pt"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "pretrain"
    assert manifest["final_step"] == 4
    assert manifest["config"]["train"]["steps"] == 4


def test_packing_skipped_when_binary_exists(tmp_path, fixture_path, tokenizer_path):
    config = toy_config(tmp_path, fixture_path, tokenizer_path, steps=2,
                        checkpoint_every=2)
    run(config)
    before = Path(config["data"]["packed_path"]).stat().st_mtime_ns
    config["out_dir"] = str(tmp_path / "out2")
    run(config)
    assert Path(config["data"]["packed_path"]).stat().st_mtime_ns == before


def test_lr_schedule_warmup_peak_cosine_floor():
    peak = 6e-4
    kwargs = {"total_steps": 1000, "peak_lr": peak,
              "warmup_frac": 0.1, "min_lr_frac": 0.1}
    assert lr_at(0, **kwargs) == pytest.approx(peak / 100)   # first step, warming up
    assert lr_at(100, **kwargs) == pytest.approx(peak)       # end of warmup
    assert lr_at(1000, **kwargs) == pytest.approx(peak * 0.1)  # cosine floor
    mid = lr_at(550, **kwargs)
    assert peak * 0.1 < mid < peak
    assert mid == pytest.approx(
        peak * 0.1 + (peak - peak * 0.1) * 0.5 * (1 + math.cos(math.pi * 0.5))
    )


def test_hub_sync_after_checkpoint(tmp_path, fixture_path, tokenizer_path):
    config = toy_config(tmp_path, fixture_path, tokenizer_path, steps=2,
                        checkpoint_every=2)
    config["hub"] = {"target": str(tmp_path / "mirror")}
    run(config)
    assert (tmp_path / "mirror" / "checkpoints" / "step_000002.pt").exists()
    assert (tmp_path / "mirror" / "metrics.jsonl").exists()


def test_full_config_parses_and_matches_budgeted_model():
    config = tomllib.loads(
        (REPO_ROOT / "configs" / "pretrain_full.toml").read_text(encoding="utf-8")
    )
    assert ModelConfig(**config["model"]) == ModelConfig(
        vocab_size=8192, d_model=512, n_layers=8, n_heads=8,
        context=512, ffn_hidden=1408,
    )
    train = config["train"]
    assert train["peak_lr"] == 6e-4 and train["precision"] == "bf16"
    assert train["micro_batch_size"] == 32 and train["grad_accum"] == 8


def test_cli_entrypoint_runs_standalone(tmp_path, fixture_path, tokenizer_path):
    config = toy_config(tmp_path, fixture_path, tokenizer_path, steps=2,
                        checkpoint_every=2, log_every=1)
    config_file = tmp_path / "cfg.toml"
    config_file.write_text(to_toml(config), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.pretrain", "--config", str(config_file)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (Path(config["out_dir"]) / "checkpoints" / "step_000002.pt").exists()


def to_toml(config: dict) -> str:
    """Serialize the nested toy config as TOML (stdlib has no writer)."""
    lines = [f'out_dir = "{config["out_dir"]}"']
    for section in ("model", "data", "train", "wandb", "hub"):
        if section not in config:
            continue
        lines.append(f"[{section}]")
        for key, value in config[section].items():
            if isinstance(value, bool):
                lines.append(f"{key} = {str(value).lower()}")
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pretrain_stage.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'tinystories_v2.pretrain'`.

- [ ] **Step 3: Implement the Pretraining stage**

Create `src/tinystories_v2/pretrain.py`:

```python
"""Pretraining stage: pack the pretrain split, train FableLM, checkpoint-resume.

Invoke standalone:
    ts2-pretrain --config configs/pretrain_fixture.toml [--resume]
    (or: python -m tinystories_v2.pretrain --config ...)

Artifacts in <out_dir>:
    checkpoints/step_XXXXXX.pt   full training state (atomic; resume contract)
    metrics.jsonl                one line per log_every steps: loss, lr, tokens_seen
    manifest.json                stage, version, final step/loss, config
Plus the packed binary at [data].packed_path (skipped if already present).

Determinism contract: model init is seeded, batches are a pure function of
(seed, step, micro_step), and optimizer state round-trips through checkpoints,
so an interrupted-and-resumed run reproduces the uninterrupted run exactly
(fp32 CPU; asserted by tests/test_resume.py).
"""

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch

from tinystories_v2 import __version__
from tinystories_v2.checkpoint import (
    latest_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint,
)
from tinystories_v2.config import load_config
from tinystories_v2.hub import fetch_from, try_sync_to
from tinystories_v2.tracking import MetricsLogger
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.pack import get_batch, load_packed, pack_split


def lr_at(step: int, total_steps: int, peak_lr: float,
          warmup_frac: float, min_lr_frac: float) -> float:
    """Linear warmup to peak_lr, then cosine decay to min_lr_frac * peak_lr."""
    warmup_steps = max(1, round(total_steps * warmup_frac))
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    min_lr = peak_lr * min_lr_frac
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def build_optimizer(model: FableLM, peak_lr: float, betas: tuple[float, float],
                    weight_decay: float) -> torch.optim.AdamW:
    # Decay 2D+ params (matmul weights, embeddings); never norms (1-D).
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=peak_lr, betas=betas,
    )


def run(config: dict, resume: bool = False) -> dict:
    train = config["train"]
    out_dir = Path(config["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    hub_target = config.get("hub", {}).get("target")

    # -- data: pack once, reuse thereafter (the packed binary is its own artifact)
    packed_path = Path(config["data"]["packed_path"])
    if not packed_path.exists():
        pack_split(config["data"]["split_path"],
                   config["data"]["tokenizer_path"], packed_path)
    data = load_packed(packed_path)

    # -- precision knob: fp32 | bf16 (autocast) | fp16 (autocast + GradScaler)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = train["precision"]
    amp_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    autocast = (torch.autocast(device_type=device, dtype=amp_dtype)
                if amp_dtype else nullcontext())
    scaler = torch.amp.GradScaler(device, enabled=(precision == "fp16"))

    # -- model + optimizer (seeded init so runs are reproducible from config)
    torch.manual_seed(train["seed"])
    model = FableLM(ModelConfig(**config["model"])).to(device)
    optimizer = build_optimizer(model, train["peak_lr"],
                                (train["beta1"], train["beta2"]),
                                train["weight_decay"])

    start_step, tokens_seen = 0, 0
    if resume:
        if latest_checkpoint(ckpt_dir) is None and hub_target:
            fetch_from(hub_target, out_dir)  # fresh Colab VM: pull previous session
        ckpt_path = latest_checkpoint(ckpt_dir)
        if ckpt_path is not None:
            state = load_checkpoint(ckpt_path)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start_step, tokens_seen = state["step"], state["tokens_seen"]
            print(f"resumed from {ckpt_path.name} at step {start_step}")

    def checkpoint(step: int) -> None:
        save_checkpoint(ckpt_dir, step, {
            "step": step, "tokens_seen": tokens_seen,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "config": config,
        })
        prune_checkpoints(ckpt_dir, train.get("keep_last", 0))
        if hub_target:
            try_sync_to(hub_target, out_dir)

    logger = MetricsLogger(out_dir, config.get("wandb"))
    steps, accum = train["steps"], train["grad_accum"]
    micro_bs, context = train["micro_batch_size"], config["model"]["context"]
    loss_value = float("nan")
    model.train()
    for step in range(start_step, steps):
        lr = lr_at(step, steps, train["peak_lr"],
                   train["warmup_frac"], train["min_lr_frac"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(accum):
            x, y = get_batch(data, micro_bs, context, seed=train["seed"],
                             step=step, micro_step=micro_step, device=device)
            with autocast:
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
            scaler.scale(loss / accum).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        tokens_seen += micro_bs * context * accum
        loss_value = loss.item()  # last micro-batch loss (cheap, logged raw)
        done = step + 1
        if done % train["log_every"] == 0:
            logger.log({"loss": loss_value, "lr": lr, "tokens_seen": tokens_seen},
                       step=done)
        if done % train["checkpoint_every"] == 0:
            checkpoint(done)
    if steps % train["checkpoint_every"] != 0:
        checkpoint(steps)
    logger.finish()

    manifest = {
        "stage": "pretrain", "package_version": __version__,
        "final_step": steps, "final_loss": loss_value,
        "tokens_seen": tokens_seen, "config": config,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    if hub_target:
        try_sync_to(hub_target, out_dir)
    return {"step": steps, "loss": loss_value}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest checkpoint in out_dir")
    args = parser.parse_args(argv)
    run(load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create the two stage configs**

Create `configs/pretrain_fixture.toml`:

```toml
# Toy CPU run against the committed fixture — local sanity runs and docs.
out_dir = "artifacts/pretrain_fixture"

[model]
vocab_size = 512
d_model = 64
n_layers = 2
n_heads = 2
context = 64
ffn_hidden = 192

[data]
split_path = "tests/fixtures/tf1_sample.jsonl"
tokenizer_path = "artifacts/tokenizer_fixture/tokenizer.json"
packed_path = "artifacts/pretrain_fixture/pretrain.bin"

[train]
steps = 30
micro_batch_size = 8
grad_accum = 1
peak_lr = 1e-3
warmup_frac = 0.1
min_lr_frac = 0.1
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
precision = "fp32"
seed = 1337
checkpoint_every = 10
log_every = 1
keep_last = 0

[wandb]
enabled = false
```

Create `configs/pretrain_full.toml`:

```toml
# Real 500M-token Pretraining run on Colab Pro (design-doc defaults).
# bf16 on L4 (preferred). On a T4 fallback set precision = "fp16" (Turing has
# no bf16; the stage then engages the GradScaler automatically).
# Prerequisites on Hub/disk: data-prep full split + tokenizer_full artifacts.
out_dir = "artifacts/pretrain_full"

[model]
vocab_size = 8192
d_model = 512
n_layers = 8
n_heads = 8
context = 512
ffn_hidden = 1408

[data]
split_path = "artifacts/data_prep_full/splits/pretrain.jsonl"
tokenizer_path = "artifacts/tokenizer_full/tokenizer.json"
packed_path = "artifacts/packed_full/pretrain.bin"

[train]
steps = 3800                # ~500M tokens at 131k tokens/step
micro_batch_size = 32
grad_accum = 8
peak_lr = 6e-4
warmup_frac = 0.015
min_lr_frac = 0.1
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
precision = "bf16"
seed = 1337
checkpoint_every = 400      # ~10-15 min on L4
log_every = 10
keep_last = 2

[wandb]
enabled = true
project = "tinystories-v2"
run_name = "pretrain-500M"

[hub]
# Edit to the team's private repo before the real run, e.g. "hf://<org>/tinystories-v2-pretrain"
target = "hf://CHANGE-ME/tinystories-v2-pretrain"
```

- [ ] **Step 5: Register the console script**

In `pyproject.toml`, extend `[project.scripts]`:

```toml
[project.scripts]
ts2-data-prep = "tinystories_v2.data:main"
ts2-tokenizer = "tinystories_v2.tokenizer:main"
ts2-pretrain = "tinystories_v2.pretrain:main"
```

Re-install so the script lands: `uv pip install -e '.[dev]'`

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pretrain_stage.py -v`
Expected: all PASS (the loss-decrease test takes a few seconds of CPU).
Then the full suite: `.venv/bin/python -m pytest`

- [ ] **Step 7: Demo the fixture config end-to-end**

```bash
.venv/bin/ts2-tokenizer --config configs/tokenizer_fixture.toml   # ensure tokenizer artifact exists
.venv/bin/ts2-pretrain --config configs/pretrain_fixture.toml
tail -3 artifacts/pretrain_fixture/metrics.jsonl
```
Expected: three JSON lines with `loss` well below the first line's (~6.2 at init).

- [ ] **Step 8: Commit**

```bash
git add src/tinystories_v2/pretrain.py configs/pretrain_fixture.toml configs/pretrain_full.toml pyproject.toml tests/test_pretrain_stage.py
git commit -m "feat: Pretraining stage — precision knob, accumulation, warmup+cosine, checkpoints

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Kill-and-resume test

No production code should be needed — this task proves the Task 3+6 contract by SIGKILLing a real subprocess mid-run and asserting the resumed run matches an uninterrupted reference run **exactly** (same floats). If an assertion fails, the bug is in the stage (state not fully round-tripped, batch selection not pure, non-atomic write) — fix it there; do not weaken the test to `allclose`.

**Files:**
- Test: `tests/test_resume.py`

**Interfaces:**
- Consumes: `pretrain.run` / `python -m tinystories_v2.pretrain` (Task 6), checkpoint state schema (Task 6), `to_toml` helper shape from `tests/test_pretrain_stage.py` (re-declared locally — tests don't import from each other).
- Produces: nothing (leaf task).

Why exact equality is achievable: fp32 on CPU, dropout 0.0, model init from `torch.manual_seed(seed)` consumed identically in both runs, batches keyed by `(seed, step, micro_step)` rather than RNG stream position, and AdamW moments restored from the checkpoint. Both runs execute the identical op sequence for every step ≥ resume point.

- [ ] **Step 1: Write the test**

Create `tests/test_resume.py`:

```python
"""Kill-and-resume: the checkpoint-resume contract, end to end.

Sized so the run is slow enough to SIGKILL mid-flight: d_model 128 / 4 layers
/ ctx 128 / micro-batch 8 is ~0.9M params and ~50-150 ms per CPU step, so 50
steps gives a multi-second window. checkpoint_every=5 guarantees several
checkpoints exist before the kill.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from tinystories_v2.checkpoint import latest_checkpoint, load_checkpoint
from tinystories_v2.pretrain import run
from tinystories_v2.tokenizer import run as run_tokenizer

STEPS = 50
CHECKPOINT_EVERY = 5
KILL_AFTER_STEP = 10  # SIGKILL once this checkpoint appears


def resume_config(base: Path, fixture_path: Path, tokenizer_path: Path,
                  out_name: str) -> dict:
    return {
        "out_dir": str(base / out_name),
        "model": {"vocab_size": 512, "d_model": 128, "n_layers": 4,
                  "n_heads": 4, "context": 128, "ffn_hidden": 384},
        "data": {"split_path": str(fixture_path),
                 "tokenizer_path": str(tokenizer_path),
                 "packed_path": str(base / "packed" / "pretrain.bin")},
        "train": {"steps": STEPS, "micro_batch_size": 8, "grad_accum": 1,
                  "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                  "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
                  "grad_clip": 1.0, "precision": "fp32", "seed": 1337,
                  "checkpoint_every": CHECKPOINT_EVERY, "log_every": 1,
                  "keep_last": 0},
        "wandb": {"enabled": False},
    }


def to_toml(config: dict) -> str:
    lines = [f'out_dir = "{config["out_dir"]}"']
    for section in ("model", "data", "train", "wandb"):
        lines.append(f"[{section}]")
        for key, value in config[section].items():
            if isinstance(value, bool):
                lines.append(f"{key} = {str(value).lower()}")
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"


def read_metrics(out_dir: str) -> dict[int, float]:
    lines = (Path(out_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return {row["step"]: row["loss"] for row in map(json.loads, lines)}


def test_killed_run_resumes_to_identical_final_state(tmp_path, fixture_path):
    tok_dir = tmp_path / "tok"
    run_tokenizer({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    tokenizer_path = tok_dir / "tokenizer.json"

    # Reference: the same config, never interrupted (shares the packed binary).
    reference = resume_config(tmp_path, fixture_path, tokenizer_path, "reference")
    run(reference)

    # Interrupted: identical config except out_dir, run as a subprocess.
    interrupted = resume_config(tmp_path, fixture_path, tokenizer_path, "interrupted")
    config_file = tmp_path / "interrupted.toml"
    config_file.write_text(to_toml(interrupted), encoding="utf-8")
    ckpt_dir = Path(interrupted["out_dir"]) / "checkpoints"
    kill_marker = ckpt_dir / f"step_{KILL_AFTER_STEP:06d}.pt"

    proc = subprocess.Popen(
        [sys.executable, "-m", "tinystories_v2.pretrain", "--config", str(config_file)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 120
        while not kill_marker.exists():
            if proc.poll() is not None:
                pytest.fail(
                    f"stage finished (rc={proc.returncode}) before the kill window; "
                    f"enlarge the toy model or lower KILL_AFTER_STEP"
                )
            if time.monotonic() > deadline:
                pytest.fail("timed out waiting for the kill-marker checkpoint")
            time.sleep(0.01)
        os.kill(proc.pid, signal.SIGKILL)
    finally:
        proc.wait(timeout=30)

    killed_at = load_checkpoint(latest_checkpoint(ckpt_dir))["step"]
    assert KILL_AFTER_STEP <= killed_at < STEPS  # it really died mid-run

    # Resume with the one flag; must continue from the recorded step to the end.
    run(interrupted, resume=True)

    # Training state matches the uninterrupted run bitwise.
    final_ref = load_checkpoint(
        latest_checkpoint(Path(reference["out_dir"]) / "checkpoints"))
    final_res = load_checkpoint(latest_checkpoint(ckpt_dir))
    assert final_res["step"] == final_ref["step"] == STEPS
    assert final_res["tokens_seen"] == final_ref["tokens_seen"]
    for key, tensor in final_ref["model"].items():
        assert torch.equal(final_res["model"][key], tensor), key

    # Post-resume losses replay the reference exactly, from the resumed step on.
    ref_losses, res_losses = read_metrics(reference["out_dir"]), read_metrics(
        interrupted["out_dir"])
    for step in range(killed_at + 1, STEPS + 1):
        assert res_losses[step] == ref_losses[step], step
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/python -m pytest tests/test_resume.py -v`
Expected: PASS in roughly 15–30 s of CPU. If it fails on `torch.equal`, debug the stage — the usual culprits are (a) optimizer/scaler state not restored, (b) anything in the loop consuming the global RNG, (c) `lr_at` depending on wall-clock rather than step. If the subprocess finishes before the kill (fast machine), raise `n_layers` to 6 in `resume_config` rather than adding sleeps. If final states differ only between the subprocess and in-process runs (thread-count float drift), run the reference via the same subprocess route instead of in-process.

- [ ] **Step 3: Run the full suite**

Run: `.venv/bin/python -m pytest`
Expected: everything green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_resume.py
git commit -m "test: kill-and-resume reproduces the uninterrupted run bitwise

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Generation utility (`sample` + `ts2-generate`)

**Files:**
- Create: `src/tinystories_v2/generate.py`
- Modify: `pyproject.toml` (add `ts2-generate` console script)
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `FableLM`/`ModelConfig` (Task 1), `latest_checkpoint`/`load_checkpoint` (Task 3), checkpoint state schema (Task 6: `ckpt["config"]["model"]` rebuilds the `ModelConfig`, `ckpt["config"]["data"]["tokenizer_path"]` locates the tokenizer).
- Produces (the shared sampling path for SFT/preference-data/eval/demo in later issues):
  - `sample(model: FableLM, prompt_ids: list[int], *, num_samples: int = 1, max_new_tokens: int, temperature: float = 1.0, top_p: float = 1.0, seed: int | None = None, end_id: int | None = None, device: str = "cpu") -> list[list[int]]` — batched over `num_samples` (one prompt replicated; the later preference-data use case is exactly N samples per Slot Prompt). Returns full sequences (prompt + continuation), each truncated after the first `end_id`. `temperature == 0.0` → greedy argmax. `seed=None` → nondeterministic.
  - CLI `ts2-generate --checkpoint <file-or-dir> --prompt TEXT [--tokenizer PATH] [--num-samples N] [--max-new-tokens N] [--temperature T] [--top-p P] [--seed S]` — prints one decoded sample per line-block.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generate.py`:

```python
import subprocess
import sys

import pytest
import torch

from tinystories_v2.generate import sample
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.pretrain import run as run_pretrain
from tinystories_v2.tokenizer import run as run_tokenizer

TOY = ModelConfig(vocab_size=512, d_model=64, n_layers=2, n_heads=2,
                  context=64, ffn_hidden=192)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return FableLM(TOY).eval()


def test_seeded_sampling_is_reproducible(model):
    a = sample(model, [1, 2, 3], num_samples=2, max_new_tokens=20,
               temperature=1.0, top_p=0.9, seed=7)
    b = sample(model, [1, 2, 3], num_samples=2, max_new_tokens=20,
               temperature=1.0, top_p=0.9, seed=7)
    assert a == b
    c = sample(model, [1, 2, 3], num_samples=2, max_new_tokens=20,
               temperature=1.0, top_p=0.9, seed=8)
    assert a != c


def test_batched_sampling_shapes_and_prompt_prefix(model):
    out = sample(model, [5, 6], num_samples=3, max_new_tokens=10, seed=0)
    assert len(out) == 3
    for seq in out:
        assert seq[:2] == [5, 6]
        assert len(seq) <= 2 + 10


def test_stops_at_end_token(model):
    # With end_id ranging over the whole vocab the argmax token at step 1 is
    # end for greedy decoding of *some* id; instead force it: temperature 0
    # gives a deterministic continuation, then rerun with that continuation's
    # first token as end_id and expect immediate stop.
    greedy = sample(model, [1, 2, 3], max_new_tokens=5, temperature=0.0)[0]
    first_generated = greedy[3]
    out = sample(model, [1, 2, 3], max_new_tokens=5, temperature=0.0,
                 end_id=first_generated)[0]
    assert out == [1, 2, 3, first_generated]  # truncated right after end_id


def test_greedy_needs_no_seed(model):
    a = sample(model, [4], max_new_tokens=8, temperature=0.0)
    b = sample(model, [4], max_new_tokens=8, temperature=0.0)
    assert a == b


def test_prompt_longer_than_context_rejected(model):
    with pytest.raises(ValueError, match="context"):
        sample(model, list(range(TOY.context + 1)), max_new_tokens=1)


def test_cli_generates_from_toy_checkpoint(tmp_path, fixture_path):
    tok_dir = tmp_path / "tok"
    run_tokenizer({"out_dir": str(tok_dir), "corpus": [str(fixture_path)],
                   "text_field": "fable", "vocab_size": 512})
    config = {
        "out_dir": str(tmp_path / "out"),
        "model": {"vocab_size": 512, "d_model": 64, "n_layers": 2,
                  "n_heads": 2, "context": 64, "ffn_hidden": 192},
        "data": {"split_path": str(fixture_path),
                 "tokenizer_path": str(tok_dir / "tokenizer.json"),
                 "packed_path": str(tmp_path / "packed" / "pretrain.bin")},
        "train": {"steps": 2, "micro_batch_size": 4, "grad_accum": 1,
                  "peak_lr": 1e-3, "warmup_frac": 0.1, "min_lr_frac": 0.1,
                  "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
                  "grad_clip": 1.0, "precision": "fp32", "seed": 1,
                  "checkpoint_every": 2, "log_every": 1, "keep_last": 0},
        "wandb": {"enabled": False},
    }
    run_pretrain(config)
    result = subprocess.run(
        [sys.executable, "-m", "tinystories_v2.generate",
         "--checkpoint", str(tmp_path / "out" / "checkpoints"),
         "--prompt", "Once upon a time", "--max-new-tokens", "16",
         "--num-samples", "2", "--seed", "3"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    samples = [s.strip() for s in result.stdout.split("\n---\n") if s.strip()]
    assert len(samples) == 2
    assert all(s.startswith("Once upon a time") for s in samples)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_generate.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'tinystories_v2.generate'`.

- [ ] **Step 3: Implement generate.py**

Create `src/tinystories_v2/generate.py`:

```python
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
```

- [ ] **Step 4: Register the console script**

In `pyproject.toml`, extend `[project.scripts]` with:

```toml
ts2-generate = "tinystories_v2.generate:main"
```

Re-install: `uv pip install -e '.[dev]'`

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_generate.py -v`
Expected: all PASS.

- [ ] **Step 6: Demo — generate from the fixture checkpoint trained in Task 6**

```bash
.venv/bin/ts2-generate --checkpoint artifacts/pretrain_fixture/checkpoints \
    --prompt "Once upon a time" --num-samples 2 --max-new-tokens 60 --seed 7
```
Expected: two blocks of (mostly incoherent — 30 steps of Pretraining) text, both starting with "Once upon a time". Re-running the exact command prints identical text.

- [ ] **Step 7: Commit**

```bash
git add src/tinystories_v2/generate.py pyproject.toml tests/test_generate.py
git commit -m "feat: batched seedable temperature/top-p generation + ts2-generate CLI

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Thin Colab notebook + docs

**Files:**
- Create: `notebooks/pretrain_colab.ipynb`
- Modify: `docs/DESIGN.md` (one-line param-count correction)
- Modify: `.scratch/tinystories-v2-pipeline/issues/02-model-pretraining-stage.md` (check off criteria)
- Test: `tests/test_notebook.py`

**Interfaces:**
- Consumes: `ts2-pretrain` console script (Task 6), `configs/pretrain_full.toml` (Task 6).
- Produces: the runbook for the real Colab Pro run (human follow-up; GPU throughput validation is explicitly out of this issue's acceptance).

- [ ] **Step 1: Write the failing notebook-thinness test**

Create `tests/test_notebook.py`:

```python
"""The Colab notebook must stay a thin wrapper: setup + stage invocation only.

Any Python logic (function/class definitions, torch imports, loops) belongs in
the package where it is reviewed and tested — never in notebook JSON.
"""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "pretrain_colab.ipynb"


def test_notebook_is_thin():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert 1 <= len(code_cells) <= 4
    source = "\n".join("".join(c["source"]) for c in code_cells)
    for forbidden in ("def ", "class ", "import torch", "for ", "while "):
        assert forbidden not in source, forbidden
    assert "ts2-pretrain" in source
    assert "--resume" in source


def test_notebook_has_no_secrets_or_outputs():
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "hf_" not in text  # no literal HF token prefixes
    cells = json.loads(text)["cells"]
    assert all(not c.get("outputs") for c in cells)  # committed clean
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_notebook.py -v`
Expected: FAIL with `FileNotFoundError` (notebook doesn't exist yet).

- [ ] **Step 3: Create the notebook**

Create `notebooks/pretrain_colab.ipynb` with exactly this JSON:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Pretraining on Colab Pro (L4 preferred, T4 fallback)\n",
    "\n",
    "Thin wrapper per docs/DESIGN.md: clone → install → secrets → run stage.\n",
    "All logic lives in the package; edit `configs/pretrain_full.toml` in the repo, not here.\n",
    "\n",
    "Before running: set `HF_TOKEN` and `WANDB_API_KEY` in Colab **Secrets** (key icon, left sidebar),\n",
    "set the repo URL below, and on a T4 change `precision = \"fp16\"` in the config (Turing has no bf16)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "REPO_URL = \"https://github.com/CHANGE-ME/tinystories_v2.git\"\n",
    "!git clone {REPO_URL}\n",
    "%cd tinystories_v2\n",
    "!pip install -q -e '.[track]'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from google.colab import userdata\n",
    "os.environ[\"HF_TOKEN\"] = userdata.get(\"HF_TOKEN\")\n",
    "os.environ[\"WANDB_API_KEY\"] = userdata.get(\"WANDB_API_KEY\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!ts2-pretrain --config configs/pretrain_full.toml --resume"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {"provenance": []},
  "kernelspec": {"display_name": "Python 3", "name": "python3"},
  "language_info": {"name": "python"}
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
```

(`--resume` is safe on a fresh run — with no checkpoints and no fetched state the stage starts from step 0 — so the same cell works for both first run and every rerun after a session death.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_notebook.py -v`
Expected: both PASS.

- [ ] **Step 5: Correct the param estimate in DESIGN.md**

In `docs/DESIGN.md`, replace the line:

```
- Total ≈ 27M params (≈4.2M embeddings).
```

with:

```
- Total 29,893,120 params exactly (≈4.2M embeddings; the design-phase "≈27M"
  was a rough estimate — exact formula in tests/test_model.py).
```

- [ ] **Step 6: Check off the issue's acceptance criteria**

In `.scratch/tinystories-v2-pipeline/issues/02-model-pretraining-stage.md`, flip every satisfied `- [ ]` to `- [x]` (all eight, provided the full suite is green and the Task 6/8 demos worked).

- [ ] **Step 7: Full-suite verification**

Run: `.venv/bin/python -m pytest -v`
Expected: every test green, no network access, wall time a few minutes on laptop CPU. Also re-run the two demo commands (Task 6 Step 7, Task 8 Step 6) if anything changed since.

- [ ] **Step 8: Commit**

```bash
git add notebooks/pretrain_colab.ipynb tests/test_notebook.py docs/DESIGN.md .scratch/tinystories-v2-pipeline/issues/02-model-pretraining-stage.md
git commit -m "docs: thin Colab notebook, exact param count, check off issue 02

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Execution Notes

- Tasks 1–5 are independent of each other except that all need Task 1's `pyproject.toml` dependency bump (torch/numpy). Task 6 needs 1–5; Task 7 needs 6; Task 8 needs 6; Task 9 needs 6.
- Total new test count ≈ 40; the slow ones are the loss-decrease test (~5–10 s) and the kill-and-resume test (~15–30 s).
- Out of scope, per the issue: real-GPU throughput validation on Colab (human follow-up), val-perplexity tracking (arrives with the evaluation suite, issue 07), KV-cached generation (revisit if preference-data rollouts are slow).
