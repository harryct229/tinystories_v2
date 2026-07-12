# Issue 09 — 5M Architecture Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add config-selectable RoPE/learned-position and SwiGLU/GELU model variants, train a fair three-run ablation at ~5M parameters, and produce a reproducible held-out comparison table plus fixed-Scaffold samples.

**Architecture:** Extend `ModelConfig` with backward-compatible component selectors while keeping the current Llama-style path as the default, so the existing Pretraining stage remains unchanged and old checkpoints still load. Three declarative configs use the same data, optimizer, seed, effective batch, and 498,073,600-token budget; a separate report helper loads their latest checkpoints, rejects unfair comparisons, computes held-out perplexity with the existing metric, and generates from the same fixed Scaffolds.

**Tech Stack:** Python >=3.11, PyTorch >=2.6 (`nn.Embedding`, scaled-dot-product attention, GELU), `tokenizers`, TOML configs, pytest, existing checkpoint/packing/perplexity/generation helpers, W&B through the existing `[track]` extra.

## Global Constraints

Copied from `.scratch/tinystories-v2-pipeline/issues/09-architecture-ablation.md`, its parent PRD user story 29, `docs/DESIGN.md`, `CONTEXT.md`, and the existing stage contracts. Every task implicitly includes these requirements.

- The only planned variants are the Llama-style baseline (`RoPE + SwiGLU`), learned absolute positional embeddings instead of RoPE, and a GELU MLP instead of SwiGLU. Ablations beyond this single planned 5M-scale comparison are out of scope.
- Each non-baseline variant differs from the baseline in exactly one component. The GELU variant widens its hidden dimension because a two-projection GELU MLP otherwise has fewer parameters than a three-projection SwiGLU MLP.
- Real model dimensions are fixed at vocabulary 8192, `d_model=256`, 4 layers, 4 heads, context 512. SwiGLU hidden size is 704; GELU hidden size is 1056.
- Exact unique-parameter counts are 5,310,720 for RoPE + SwiGLU, 5,441,792 for learned positions + SwiGLU, and 5,310,720 for RoPE + GELU.
- Parameter comparability means `abs(variant_params - baseline_params) / baseline_params <= 0.03`. The GELU variant is exact; the learned-position variant is +2.468% because its `context * d_model = 131,072` learned table is the component being measured. Do not narrow another layer to hide that component's intrinsic cost.
- All three real runs consume the same packed Pretraining binary, use seed 1337, and process exactly `3800 * 128 * 2 * 512 = 498,073,600` tokens. Their `[data]` and `[train]` tables must otherwise be byte-for-byte equivalent.
- `src/tinystories_v2/pretrain.py` must not change. All variants run through `ts2-pretrain` solely by changing `[model]` config values.
- `ModelConfig` defaults remain `position_encoding="rope"` and `mlp_type="swiglu"`. Existing configs and checkpoints omit these keys and must continue to construct the exact current model/state-dict layout.
- Held-out comparison uses the disjoint `artifacts/data_prep_full/splits/eval.jsonl` split, the same first 1,000,000 packed evaluation tokens, and the latest checkpoint from each run only when every `tokens_seen` value equals the configured 498,073,600-token budget.
- Report output includes variant, component choices, FFN hidden size, parameter count, matched training tokens, held-out loss, and held-out perplexity. It also includes generations for the same first three held-out Scaffolds with identical decoding settings and seeds.
- Qualitative generation starts each Pretraining model with the same in-distribution natural-language seed, `In {setting}, a {trait} {character}`, derived from each fixed Scaffold. Do not feed a Slot Prompt to these models: Slot Prompt conditioning is learned only during SFT and would confound this Pretraining architecture comparison.
- Training loss curves continue to come from the existing `metrics.jsonl`/W&B logger; each real config uses a distinct W&B run name in the same `tinystories-v2` project.
- All automated tests run on laptop CPU with no GPU, network, Hub access, or W&B dependency. Real bf16 runs happen on a Colab Pro L4; a T4 operator may change all three configs to fp16, but must make the same change in every config.
- Use project vocabulary: Fable, Scaffold, Pretraining, Slot Prompt. Do not call a Fable a "story" or "tale" in new code/docs.
- Secrets remain in `.env` or Colab Secrets and are never committed or printed.
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`, matching repository history.

## File Structure

```text
src/tinystories_v2/
    model.py
        # backward-compatible component selectors, learned positions, GELU MLP
    ablation.py
        # checkpoint comparability gate, held-out eval, fixed-Scaffold samples,
        # JSON + Markdown report, CLI
tests/
    test_model.py
        # every variant's construction, causality, analytic parameters, defaults
    test_pretrain_stage.py
        # parameterized CPU loss-decrease run through the unchanged stage
    test_ablation_configs.py
        # exact real dimensions, parameter tolerance, identical token/data budget
    test_ablation.py
        # report validation, toy-checkpoint evaluation, samples, CLI
configs/
    ablation_5m_rope_swiglu.toml
    ablation_5m_learned_swiglu.toml
    ablation_5m_rope_gelu.toml
        # three matched real Pretraining runs
    ablation_5m_report.toml
        # held-out evaluation and fixed-Scaffold report inputs
pyproject.toml
    # register ts2-ablation-report
docs/DESIGN.md
    # record exact architecture sizes, fairness protocol, and launch commands
docs/experiments/5m-architecture-ablation/
    report.json
    report.md
        # generated, versioned real-run evidence for the course report
.scratch/tinystories-v2-pipeline/issues/09-architecture-ablation.md
PROGRESS.md
    # acceptance evidence and completion tracking after verification
```

`pretrain.py`, `checkpoint.py`, `pack.py`, `perplexity.py`, `generate.py`, `slot_prompt.py`, and `tracking.py` are reused unchanged.

---

### Task 1: Config-selectable positional encoding and MLP

**Files:**
- Modify: `src/tinystories_v2/model.py`
- Modify: `tests/test_model.py`

**Interfaces:**
- Consumes: the current `FableLM(idx: LongTensor[B,T]) -> FloatTensor[B,T,vocab_size]` contract and checkpoint construction via `ModelConfig(**state["config"]["model"])`.
- Produces:
  - `ModelConfig.position_encoding: Literal["rope", "learned"] = "rope"`.
  - `ModelConfig.mlp_type: Literal["swiglu", "gelu"] = "swiglu"`.
  - `GELUMLP(config: ModelConfig)`, with bias-free `up_proj` and `down_proj`.
  - `FableLM` whose default state dict is unchanged, whose learned-position variant owns `pos_emb.weight`, and whose blocks instantiate the selected MLP.
  - `ValueError` containing `position_encoding` or `mlp_type` for unsupported selectors.

- [ ] **Step 1: Replace the model tests with variant-aware failing tests**

Replace `tests/test_model.py` with:

```python
import dataclasses

import pytest
import torch

from tinystories_v2.model import GELUMLP, FableLM, ModelConfig, SwiGLU

TOY = ModelConfig(
    vocab_size=512,
    d_model=64,
    n_layers=2,
    n_heads=2,
    context=64,
    ffn_hidden=192,
)
TOY_VARIANTS = {
    "rope_swiglu": TOY,
    "learned_swiglu": dataclasses.replace(
        TOY, position_encoding="learned"
    ),
    "rope_gelu": dataclasses.replace(
        TOY, mlp_type="gelu", ffn_hidden=288
    ),
}

# The production model from DESIGN.md/configs/pretrain_full.toml. Defaults must
# preserve the architecture and state-dict contract of existing checkpoints.
REAL = ModelConfig(
    vocab_size=8192,
    d_model=512,
    n_layers=8,
    n_heads=8,
    context=512,
    ffn_hidden=1408,
)
PARAM_BUDGET = 32_000_000

ABLATION_VARIANTS = {
    "rope_swiglu": ModelConfig(
        vocab_size=8192,
        d_model=256,
        n_layers=4,
        n_heads=4,
        context=512,
        ffn_hidden=704,
        position_encoding="rope",
        mlp_type="swiglu",
    ),
    "learned_swiglu": ModelConfig(
        vocab_size=8192,
        d_model=256,
        n_layers=4,
        n_heads=4,
        context=512,
        ffn_hidden=704,
        position_encoding="learned",
        mlp_type="swiglu",
    ),
    "rope_gelu": ModelConfig(
        vocab_size=8192,
        d_model=256,
        n_layers=4,
        n_heads=4,
        context=512,
        ffn_hidden=1056,
        position_encoding="rope",
        mlp_type="gelu",
    ),
}


def expected_params(config: ModelConfig) -> int:
    mlp_projections = {"swiglu": 3, "gelu": 2}[config.mlp_type]
    per_layer = (
        4 * config.d_model * config.d_model
        + mlp_projections * config.d_model * config.ffn_hidden
        + 2 * config.d_model
    )
    learned_positions = (
        config.context * config.d_model
        if config.position_encoding == "learned"
        else 0
    )
    return (
        config.vocab_size * config.d_model
        + learned_positions
        + config.n_layers * per_layer
        + config.d_model
    )


@pytest.fixture(scope="module")
def toy_model():
    torch.manual_seed(0)
    return FableLM(TOY).eval()


def test_forward_shape_fp32(toy_model):
    idx = torch.randint(TOY.vocab_size, (2, 16))
    logits = toy_model(idx)
    assert logits.shape == (2, 16, TOY.vocab_size)
    assert logits.dtype == torch.float32


@pytest.mark.parametrize(
    "config", TOY_VARIANTS.values(), ids=TOY_VARIANTS.keys()
)
def test_causality_future_tokens_do_not_affect_past_logits(config):
    torch.manual_seed(1)
    model = FableLM(config).eval()
    a = torch.randint(config.vocab_size, (2, 32))
    b = a.clone()
    b[:, 20:] = (b[:, 20:] + 7) % config.vocab_size
    with torch.no_grad():
        logits_a, logits_b = model(a), model(b)
    assert torch.allclose(logits_a[:, :20], logits_b[:, :20], atol=1e-5)
    assert not torch.allclose(logits_a[:, 20:], logits_b[:, 20:], atol=1e-5)


@pytest.mark.parametrize(
    "name,config", TOY_VARIANTS.items(), ids=TOY_VARIANTS.keys()
)
def test_variant_selects_only_requested_components(name, config):
    model = FableLM(config)
    if name == "learned_swiglu":
        assert isinstance(model.pos_emb, torch.nn.Embedding)
        assert model.rope_cos is None and model.rope_sin is None
    else:
        assert model.pos_emb is None
        assert model.rope_cos is not None and model.rope_sin is not None
    expected_mlp = GELUMLP if name == "rope_gelu" else SwiGLU
    assert all(isinstance(block.mlp, expected_mlp) for block in model.blocks)


@pytest.mark.parametrize(
    "config", TOY_VARIANTS.values(), ids=TOY_VARIANTS.keys()
)
def test_tied_embeddings_and_no_biases_for_every_variant(config):
    model = FableLM(config)
    assert model.lm_head.weight is model.tok_emb.weight
    for name, _ in model.named_parameters():
        assert not name.endswith("bias"), name


@pytest.mark.parametrize(
    "config", TOY_VARIANTS.values(), ids=TOY_VARIANTS.keys()
)
def test_param_count_matches_variant_analytic_formula(config):
    assert FableLM(config).num_params() == expected_params(config)


def test_real_config_param_count_within_budget():
    assert expected_params(REAL) == 29_893_120
    torch.manual_seed(0)
    model = FableLM(REAL)
    assert model.num_params() == 29_893_120 < PARAM_BUDGET


def test_ablation_variants_are_within_three_percent_of_baseline():
    counts = {
        name: FableLM(config).num_params()
        for name, config in ABLATION_VARIANTS.items()
    }
    assert counts == {
        "rope_swiglu": 5_310_720,
        "learned_swiglu": 5_441_792,
        "rope_gelu": 5_310_720,
    }
    baseline = counts["rope_swiglu"]
    assert all(abs(count - baseline) / baseline <= 0.03 for count in counts.values())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_under_autocast(toy_model, dtype):
    idx = torch.randint(TOY.vocab_size, (2, 16))
    with torch.autocast(device_type="cpu", dtype=dtype), torch.no_grad():
        logits = toy_model(idx)
    assert logits.shape == (2, 16, TOY.vocab_size)
    assert torch.isfinite(logits.float()).all()


def test_config_is_frozen_backward_compatible_and_buildable_from_dict():
    config = ModelConfig(
        **{
            "vocab_size": 512,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 2,
            "context": 64,
            "ffn_hidden": 192,
        }
    )
    assert config.rope_theta == 10000.0 and config.norm_eps == 1e-5
    assert config.position_encoding == "rope"
    assert config.mlp_type == "swiglu"
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.d_model = 1


def test_d_model_must_divide_by_heads():
    with pytest.raises(ValueError, match="n_heads"):
        FableLM(
            ModelConfig(
                vocab_size=64,
                d_model=65,
                n_layers=1,
                n_heads=2,
                context=8,
                ffn_hidden=32,
            )
        )


@pytest.mark.parametrize(
    "field,value",
    [("position_encoding", "sinusoidal"), ("mlp_type", "relu")],
)
def test_unknown_component_selector_is_rejected(field, value):
    values = dataclasses.asdict(TOY)
    values[field] = value
    with pytest.raises(ValueError, match=field):
        FableLM(ModelConfig(**values))
```

- [ ] **Step 2: Run the model tests to verify the selectors do not exist yet**

Run: `rtk .venv/bin/pytest tests/test_model.py -v`

Expected: collection fails because `GELUMLP` is not importable and `ModelConfig` does not accept `position_encoding`/`mlp_type`.

- [ ] **Step 3: Replace the model module with the minimal component factories**

Replace `src/tinystories_v2/model.py` with:

```python
"""Hand-written decoder-only Fable LM (ADR-0002, ADR-0005).

The default remains the production Llama-style stack: pre-norm RMSNorm,
RoPE, SwiGLU, no biases, tied embeddings, and dropout 0.0. Issue 09 adds two
config-selected, one-component ablations at ~5M scale: learned absolute
positions instead of RoPE, and a parameter-matched GELU MLP instead of
SwiGLU. Existing configs omit both selectors and therefore retain the exact
original architecture and checkpoint key layout.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

PositionEncoding = Literal["rope", "learned"]
MLPType = Literal["swiglu", "gelu"]


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
    position_encoding: PositionEncoding = "rope"
    mlp_type: MLPType = "swiglu"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return self.weight * norm.type_as(x)


def _rope_cache(
    head_dim: int, context: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / theta ** (
        torch.arange(0, head_dim, 2).float() / head_dim
    )
    positions = torch.arange(context).float()
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    t = x.size(-2)
    cos, sin = cos[:t].to(x.dtype), sin[:t].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        (x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1
    ).flatten(-2)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, time, width = x.shape
        shape = (batch, time, self.n_heads, self.head_dim)
        query = self.q_proj(x).view(shape).transpose(1, 2)
        key = self.k_proj(x).view(shape).transpose(1, 2)
        value = self.v_proj(x).view(shape).transpose(1, 2)
        if cos is not None and sin is not None:
            query = _apply_rope(query, cos, sin)
            key = _apply_rope(key, cos, sin)
        output = F.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )
        return self.o_proj(output.transpose(1, 2).reshape(batch, time, width))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.d_model, config.ffn_hidden, bias=False
        )
        self.up_proj = nn.Linear(
            config.d_model, config.ffn_hidden, bias=False
        )
        self.down_proj = nn.Linear(
            config.ffn_hidden, config.d_model, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class GELUMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.up_proj = nn.Linear(
            config.d_model, config.ffn_hidden, bias=False
        )
        self.down_proj = nn.Linear(
            config.ffn_hidden, config.d_model, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = (
            SwiGLU(config)
            if config.mlp_type == "swiglu"
            else GELUMLP(config)
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class FableLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError(
                f"d_model={config.d_model} not divisible by "
                f"n_heads={config.n_heads}"
            )
        if config.position_encoding not in ("rope", "learned"):
            raise ValueError(
                "position_encoding must be 'rope' or 'learned', got "
                f"{config.position_encoding!r}"
            )
        if config.mlp_type not in ("swiglu", "gelu"):
            raise ValueError(
                f"mlp_type must be 'swiglu' or 'gelu', got {config.mlp_type!r}"
            )

        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = (
            nn.Embedding(config.context, config.d_model)
            if config.position_encoding == "learned"
            else None
        )
        self.blocks = nn.ModuleList(
            Block(config) for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )
        self.lm_head.weight = self.tok_emb.weight

        if config.position_encoding == "rope":
            cos, sin = _rope_cache(
                config.d_model // config.n_heads,
                config.context,
                config.rope_theta,
            )
        else:
            cos, sin = None, None
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            nn.init.normal_(
                block.attn.o_proj.weight, mean=0.0, std=residual_std
            )
            nn.init.normal_(
                block.mlp.down_proj.weight, mean=0.0, std=residual_std
            )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.size(1) > self.config.context:
            raise ValueError(
                f"sequence length {idx.size(1)} exceeds "
                f"context {self.config.context}"
            )
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            positions = torch.arange(idx.size(1), device=idx.device)
            x = x + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        return self.lm_head(self.final_norm(x))
```

- [ ] **Step 4: Run focused model tests and checkpoint consumers**

```bash
rtk .venv/bin/pytest tests/test_model.py tests/test_checkpoint.py tests/test_generate.py tests/test_sft_stage.py -q
rtk git diff --check
```

Expected: all selected tests pass with zero failures; `git diff --check` prints nothing. In particular, existing configs without selector keys still construct RoPE + SwiGLU models and existing checkpoint consumers still load.

- [ ] **Step 5: Commit the component-selection model change**

```bash
rtk git add src/tinystories_v2/model.py tests/test_model.py
rtk git commit -m "feat: add config-selected architecture variants

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Prove stage compatibility and add the matched real-run config set

**Files:**
- Modify: `tests/test_pretrain_stage.py:45-61`
- Create: `tests/test_ablation_configs.py`
- Create: `configs/ablation_5m_rope_swiglu.toml`
- Create: `configs/ablation_5m_learned_swiglu.toml`
- Create: `configs/ablation_5m_rope_gelu.toml`
- Modify: `docs/DESIGN.md` immediately after the existing `## Evaluation` section

**Interfaces:**
- Consumes: `pretrain.run(config: dict, resume: bool = False) -> dict`, unchanged; `FableLM(ModelConfig(**config["model"]))` from Task 1; existing full data/tokenizer artifacts.
- Produces:
  - Three complete TOML configs accepted by the existing `ts2-pretrain --config ... --resume` CLI.
  - Identical parsed `[data]` and `[train]` mappings across all variants.
  - Exactly 498,073,600 tokens per completed run and three unique W&B/Hub destinations.
  - A parameterized CPU test demonstrating decreasing loss for every component selection through the real stage path.

- [ ] **Step 1: Parameterize the existing toy loss-decrease test over all variants**

In `tests/test_pretrain_stage.py`, replace `test_toy_run_decreases_loss_through_stage_entrypoint` with:

```python
@pytest.mark.parametrize(
    "position_encoding,mlp_type,ffn_hidden",
    [
        pytest.param("rope", "swiglu", 192, id="rope-swiglu"),
        pytest.param("learned", "swiglu", 192, id="learned-swiglu"),
        pytest.param("rope", "gelu", 288, id="rope-gelu"),
    ],
)
def test_toy_ablation_variant_decreases_loss_through_stage_entrypoint(
    tmp_path,
    fixture_path,
    tokenizer_path,
    position_encoding,
    mlp_type,
    ffn_hidden,
):
    config = toy_config(tmp_path, fixture_path, tokenizer_path)
    config["model"].update(
        {
            "position_encoding": position_encoding,
            "mlp_type": mlp_type,
            "ffn_hidden": ffn_hidden,
        }
    )

    summary = run(config)

    metrics = read_metrics(Path(config["out_dir"]))
    assert len(metrics) == 30
    first_mean = sum(row["loss"] for row in metrics[:5]) / 5
    last_mean = sum(row["loss"] for row in metrics[-5:]) / 5
    assert last_mean < first_mean - 0.25
    assert summary["step"] == 30
    assert {"step", "loss", "lr", "tokens_seen"} <= metrics[0].keys()
    assert metrics[-1]["tokens_seen"] == 30 * 8 * 64
```

- [ ] **Step 2: Add exact config-contract tests before the configs exist**

Create `tests/test_ablation_configs.py`:

```python
import tomllib
from pathlib import Path

from tinystories_v2.model import FableLM, ModelConfig

REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATHS = {
    "rope_swiglu": REPO_ROOT / "configs" / "ablation_5m_rope_swiglu.toml",
    "learned_swiglu": (
        REPO_ROOT / "configs" / "ablation_5m_learned_swiglu.toml"
    ),
    "rope_gelu": REPO_ROOT / "configs" / "ablation_5m_rope_gelu.toml",
}
EXPECTED_MODELS = {
    "rope_swiglu": {
        "vocab_size": 8192,
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 4,
        "context": 512,
        "ffn_hidden": 704,
        "position_encoding": "rope",
        "mlp_type": "swiglu",
    },
    "learned_swiglu": {
        "vocab_size": 8192,
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 4,
        "context": 512,
        "ffn_hidden": 704,
        "position_encoding": "learned",
        "mlp_type": "swiglu",
    },
    "rope_gelu": {
        "vocab_size": 8192,
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 4,
        "context": 512,
        "ffn_hidden": 1056,
        "position_encoding": "rope",
        "mlp_type": "gelu",
    },
}


def load_configs() -> dict[str, dict]:
    return {
        name: tomllib.loads(path.read_text(encoding="utf-8"))
        for name, path in CONFIG_PATHS.items()
    }


def test_real_ablation_model_configs_and_exact_parameter_counts():
    configs = load_configs()
    assert {name: config["model"] for name, config in configs.items()} == (
        EXPECTED_MODELS
    )
    counts = {
        name: FableLM(ModelConfig(**config["model"])).num_params()
        for name, config in configs.items()
    }
    assert counts == {
        "rope_swiglu": 5_310_720,
        "learned_swiglu": 5_441_792,
        "rope_gelu": 5_310_720,
    }
    baseline = counts["rope_swiglu"]
    assert all(abs(count - baseline) / baseline <= 0.03 for count in counts.values())


def test_each_variant_changes_only_its_named_component():
    baseline = EXPECTED_MODELS["rope_swiglu"]
    learned = EXPECTED_MODELS["learned_swiglu"]
    gelu = EXPECTED_MODELS["rope_gelu"]
    assert {key for key in baseline if baseline[key] != learned[key]} == {
        "position_encoding"
    }
    # ffn_hidden is widened only to compensate for GELU's two projections.
    assert {key for key in baseline if baseline[key] != gelu[key]} == {
        "ffn_hidden",
        "mlp_type",
    }


def test_variants_share_data_schedule_seed_and_exact_token_budget():
    configs = load_configs()
    baseline = configs["rope_swiglu"]
    assert all(
        config["data"] == baseline["data"] for config in configs.values()
    )
    assert all(
        config["train"] == baseline["train"] for config in configs.values()
    )
    train = baseline["train"]
    context = baseline["model"]["context"]
    tokens = (
        train["steps"]
        * train["micro_batch_size"]
        * train["grad_accum"]
        * context
    )
    assert train["seed"] == 1337
    assert tokens == 498_073_600


def test_tracking_and_artifact_destinations_are_unique():
    configs = load_configs()
    assert all(config["wandb"]["enabled"] for config in configs.values())
    assert len(
        {config["wandb"]["run_name"] for config in configs.values()}
    ) == 3
    assert len({config["hub"]["target"] for config in configs.values()}) == 3
    assert len({config["out_dir"] for config in configs.values()}) == 3
```

- [ ] **Step 3: Run the new tests to verify the real configs are missing**

Run: `rtk .venv/bin/pytest tests/test_pretrain_stage.py::test_toy_ablation_variant_decreases_loss_through_stage_entrypoint tests/test_ablation_configs.py -v`

Expected: the three parameterized toy stage cases pass, while the config tests fail with `FileNotFoundError` for `configs/ablation_5m_rope_swiglu.toml` and its peers.

- [ ] **Step 4: Create the RoPE + SwiGLU baseline config**

Create `configs/ablation_5m_rope_swiglu.toml`:

```toml
# Issue 09 baseline: ~5M RoPE + SwiGLU model, matched 498,073,600-token run.
# L4 default is bf16. If using a T4, change precision to fp16 in all three
# ablation configs so hardware/precision remains controlled.
out_dir = "artifacts/ablation_5m/rope_swiglu"

[model]
vocab_size = 8192
d_model = 256
n_layers = 4
n_heads = 4
context = 512
ffn_hidden = 704
position_encoding = "rope"
mlp_type = "swiglu"

[data]
split_path = "artifacts/data_prep_full/splits/pretrain.jsonl"
tokenizer_path = "artifacts/tokenizer_full/tokenizer.json"
packed_path = "artifacts/packed_full/pretrain.bin"

[train]
steps = 3800
micro_batch_size = 128
grad_accum = 2
peak_lr = 6e-4
warmup_frac = 0.015
min_lr_frac = 0.1
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
precision = "bf16"
seed = 1337
checkpoint_every = 400
log_every = 10
keep_last = 2

[wandb]
enabled = true
project = "tinystories-v2"
run_name = "ablation-5m-rope-swiglu"

[hub]
target = "hf://congthanh991/tinystories-v2-ablation-rope-swiglu"
```

- [ ] **Step 5: Create the learned-position + SwiGLU config**

Create `configs/ablation_5m_learned_swiglu.toml`:

```toml
# Issue 09 position ablation: learned absolute positions replace RoPE.
# The learned context*d_model table adds 131,072 params (+2.468%, under the
# documented 3% tolerance); every other model/training value matches baseline.
out_dir = "artifacts/ablation_5m/learned_swiglu"

[model]
vocab_size = 8192
d_model = 256
n_layers = 4
n_heads = 4
context = 512
ffn_hidden = 704
position_encoding = "learned"
mlp_type = "swiglu"

[data]
split_path = "artifacts/data_prep_full/splits/pretrain.jsonl"
tokenizer_path = "artifacts/tokenizer_full/tokenizer.json"
packed_path = "artifacts/packed_full/pretrain.bin"

[train]
steps = 3800
micro_batch_size = 128
grad_accum = 2
peak_lr = 6e-4
warmup_frac = 0.015
min_lr_frac = 0.1
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
precision = "bf16"
seed = 1337
checkpoint_every = 400
log_every = 10
keep_last = 2

[wandb]
enabled = true
project = "tinystories-v2"
run_name = "ablation-5m-learned-swiglu"

[hub]
target = "hf://congthanh991/tinystories-v2-ablation-learned-swiglu"
```

- [ ] **Step 6: Create the RoPE + parameter-matched GELU config**

Create `configs/ablation_5m_rope_gelu.toml`:

```toml
# Issue 09 MLP ablation: GELU replaces SwiGLU. GELU uses two projections, so
# hidden 1056 (= 704 * 3/2) exactly matches SwiGLU's 540,672 MLP params/layer.
out_dir = "artifacts/ablation_5m/rope_gelu"

[model]
vocab_size = 8192
d_model = 256
n_layers = 4
n_heads = 4
context = 512
ffn_hidden = 1056
position_encoding = "rope"
mlp_type = "gelu"

[data]
split_path = "artifacts/data_prep_full/splits/pretrain.jsonl"
tokenizer_path = "artifacts/tokenizer_full/tokenizer.json"
packed_path = "artifacts/packed_full/pretrain.bin"

[train]
steps = 3800
micro_batch_size = 128
grad_accum = 2
peak_lr = 6e-4
warmup_frac = 0.015
min_lr_frac = 0.1
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
precision = "bf16"
seed = 1337
checkpoint_every = 400
log_every = 10
keep_last = 2

[wandb]
enabled = true
project = "tinystories-v2"
run_name = "ablation-5m-rope-gelu"

[hub]
target = "hf://congthanh991/tinystories-v2-ablation-rope-gelu"
```

- [ ] **Step 7: Document the protocol and config-set launch path**

In `docs/DESIGN.md`, insert this section after the current Evaluation section and before `## Infrastructure & workflow`:

````markdown
## 5M architecture ablation (issue 09)

The report's layer justification uses one controlled three-run ablation:

| Variant | Position | MLP | FFN hidden | Exact params |
|---|---|---|---:|---:|
| `rope_swiglu` | RoPE | SwiGLU | 704 | 5,310,720 |
| `learned_swiglu` | learned absolute | SwiGLU | 704 | 5,441,792 |
| `rope_gelu` | RoPE | GELU | 1056 | 5,310,720 |

The GELU width is `704 * 3/2 = 1056`, so its two matrices have exactly the
same parameters as SwiGLU's three. Learned positions retain every baseline
dimension and add the measured `512 * 256 = 131,072` table (+2.468%). Fairness
tolerance is 3% relative to `rope_swiglu`.

All runs read the same packed Pretraining binary, use seed 1337 and identical
optimizer/schedule settings, and process exactly 498,073,600 tokens. Their
separate W&B runs provide matched-token loss curves. The config set is the
thin Colab launch surface; no training logic belongs in a notebook:

```bash
ts2-pretrain --config configs/ablation_5m_rope_swiglu.toml --resume
ts2-pretrain --config configs/ablation_5m_learned_swiglu.toml --resume
ts2-pretrain --config configs/ablation_5m_rope_gelu.toml --resume
ts2-ablation-report --config configs/ablation_5m_report.toml
```

The report command refuses unequal `tokens_seen` values or a parameter drift
above 3%, then writes held-out loss/perplexity and fixed-Scaffold generations
under `docs/experiments/5m-architecture-ablation/` for versioned report evidence.
````

- [ ] **Step 8: Verify all variants train through the unchanged stage**

```bash
rtk .venv/bin/pytest tests/test_model.py tests/test_ablation_configs.py tests/test_pretrain_stage.py -q
rtk git diff -- src/tinystories_v2/pretrain.py
rtk git diff --check
```

Expected: all selected tests pass with zero failures; the `pretrain.py` diff is empty; the whitespace check prints nothing.

- [ ] **Step 9: Commit the matched ablation run surface**

```bash
rtk git add tests/test_pretrain_stage.py tests/test_ablation_configs.py configs/ablation_5m_rope_swiglu.toml configs/ablation_5m_learned_swiglu.toml configs/ablation_5m_rope_gelu.toml docs/DESIGN.md
rtk git commit -m "feat: add matched 5M ablation run configs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Build the guarded held-out report and fixed-Scaffold sampler

**Files:**
- Create: `src/tinystories_v2/ablation.py`
- Create: `tests/test_ablation.py`
- Modify: `tests/test_ablation_configs.py`
- Create: `configs/ablation_5m_report.toml`
- Modify: `pyproject.toml:31-37`

**Interfaces:**
- Consumes:
  - A report config with top-level `out_dir`, `baseline`, `param_tolerance`, `expected_tokens`; an `[eval]` table; and ordered `[[variants]]` entries containing `name` and `checkpoint`.
  - Existing `latest_checkpoint`, `load_checkpoint`, `FableLM`, `pack_split`, `load_packed`, `perplexity`, `sample`, and `Scaffold` interfaces without modifying them.
- Produces:
  - `validate_comparability(rows: list[dict], *, baseline: str, param_tolerance: float) -> int`, returning the shared `tokens_seen` or raising `ValueError` on duplicate/missing baseline names, unequal token counts, or excessive parameter drift.
  - `render_scaffold_seed(scaffold: Scaffold) -> str`, returning the exact in-distribution prefix `In {setting}, a {trait} {character}`.
  - `run(config: dict) -> dict`, writing `<out_dir>/report.json` and `<out_dir>/report.md` and returning the JSON-compatible report.
  - `render_markdown(report: dict) -> str` with the required comparison table and sample sections.
  - CLI: `ts2-ablation-report --config configs/ablation_5m_report.toml` and `python -m tinystories_v2.ablation --config ...`.
  - Report schema: `baseline`, `param_tolerance`, `expected_tokens`, `matched_tokens`, `eval_tokens`, `rows`, `samples`; each row includes `variant`, `position_encoding`, `mlp_type`, `ffn_hidden`, `params`, `step`, `tokens_seen`, `val_loss`, `perplexity`, and `checkpoint`.

- [ ] **Step 1: Write report validation, artifact, sample, and CLI tests**

Create `tests/test_ablation.py`:

```python
import dataclasses
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tinystories_v2.ablation import run, validate_comparability
from tinystories_v2.checkpoint import save_checkpoint
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.tokenizer import run as run_tokenizer


@pytest.fixture()
def toy_artifacts(tmp_path, fixture_path):
    tokenizer_dir = tmp_path / "tokenizer"
    run_tokenizer(
        {
            "out_dir": str(tokenizer_dir),
            "corpus": [str(fixture_path)],
            "text_field": "fable",
            "vocab_size": 512,
        }
    )
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    base = ModelConfig(
        vocab_size=512,
        d_model=64,
        n_layers=2,
        n_heads=2,
        context=64,
        ffn_hidden=192,
    )
    configs = {
        "rope_swiglu": base,
        "learned_swiglu": dataclasses.replace(
            base, position_encoding="learned"
        ),
        "rope_gelu": dataclasses.replace(
            base, mlp_type="gelu", ffn_hidden=288
        ),
    }
    variants = []
    for index, (name, model_config) in enumerate(configs.items()):
        torch.manual_seed(index)
        model = FableLM(model_config)
        checkpoint_dir = tmp_path / name / "checkpoints"
        save_checkpoint(
            checkpoint_dir,
            2,
            {
                "step": 2,
                "tokens_seen": 256,
                "model": model.state_dict(),
                "optimizer": {},
                "scaler": {},
                "config": {
                    "model": dataclasses.asdict(model_config),
                    "data": {"tokenizer_path": str(tokenizer_path)},
                },
            },
        )
        variants.append(
            {"name": name, "checkpoint": str(checkpoint_dir)}
        )
    return {
        "tokenizer_path": tokenizer_path,
        "variants": variants,
    }


def report_config(tmp_path, fixture_path, toy_artifacts):
    return {
        "out_dir": str(tmp_path / "report"),
        "baseline": "rope_swiglu",
        "param_tolerance": 0.03,
        "expected_tokens": 256,
        "eval": {
            "split_path": str(fixture_path),
            "tokenizer_path": str(toy_artifacts["tokenizer_path"]),
            "packed_path": str(tmp_path / "eval" / "eval.bin"),
            "max_tokens": 128,
            "batch_size": 2,
            "device": "cpu",
            "sample_count": 1,
            "max_new_tokens": 2,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 7,
        },
        "variants": toy_artifacts["variants"],
    }


def write_report_config(path: Path, config: dict) -> None:
    lines = [
        f'out_dir = "{config["out_dir"]}"',
        f'baseline = "{config["baseline"]}"',
        f'param_tolerance = {config["param_tolerance"]}',
        f'expected_tokens = {config["expected_tokens"]}',
        "",
        "[eval]",
    ]
    for key, value in config["eval"].items():
        if isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        else:
            lines.append(f"{key} = {str(value).lower()}")
    for variant in config["variants"]:
        lines.extend(
            [
                "",
                "[[variants]]",
                f'name = "{variant["name"]}"',
                f'checkpoint = "{variant["checkpoint"]}"',
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_validate_comparability_accepts_matched_rows():
    rows = [
        {"variant": "base", "params": 100, "tokens_seen": 1_000},
        {"variant": "position", "params": 102, "tokens_seen": 1_000},
        {"variant": "mlp", "params": 100, "tokens_seen": 1_000},
    ]
    assert validate_comparability(
        rows, baseline="base", param_tolerance=0.03
    ) == 1_000


def test_validate_comparability_rejects_mismatched_training_tokens():
    rows = [
        {"variant": "base", "params": 100, "tokens_seen": 1_000},
        {"variant": "position", "params": 102, "tokens_seen": 900},
    ]
    with pytest.raises(ValueError, match="matched training tokens"):
        validate_comparability(
            rows, baseline="base", param_tolerance=0.03
        )


def test_validate_comparability_rejects_parameter_drift():
    rows = [
        {"variant": "base", "params": 100, "tokens_seen": 1_000},
        {"variant": "oversized", "params": 104, "tokens_seen": 1_000},
    ]
    with pytest.raises(ValueError, match="parameter tolerance"):
        validate_comparability(
            rows, baseline="base", param_tolerance=0.03
        )


def test_run_rejects_equally_incomplete_checkpoints(
    tmp_path, fixture_path, toy_artifacts
):
    config = report_config(tmp_path, fixture_path, toy_artifacts)
    config["expected_tokens"] = 512
    with pytest.raises(ValueError, match="completed token budget"):
        run(config)


def test_run_writes_comparison_table_and_fixed_scaffold_samples(
    tmp_path, fixture_path, toy_artifacts
):
    config = report_config(tmp_path, fixture_path, toy_artifacts)

    report = run(config)

    assert report["baseline"] == "rope_swiglu"
    assert report["expected_tokens"] == 256
    assert report["matched_tokens"] == 256
    assert report["eval_tokens"] == 128
    assert [row["variant"] for row in report["rows"]] == [
        "rope_swiglu",
        "learned_swiglu",
        "rope_gelu",
    ]
    assert {row["variant"]: row["params"] for row in report["rows"]} == {
        "rope_swiglu": 139_584,
        "learned_swiglu": 143_680,
        "rope_gelu": 139_584,
    }
    for row in report["rows"]:
        assert row["tokens_seen"] == 256
        assert math.isfinite(row["val_loss"])
        assert math.isfinite(row["perplexity"])
        assert row["val_loss"] == pytest.approx(math.log(row["perplexity"]))

    assert len(report["samples"]) == 1
    assert report["samples"][0]["seed"] == (
        "In a canyon, a persuasive firefly"
    )
    generations = report["samples"][0]["generations"]
    assert set(generations) == {
        "rope_swiglu",
        "learned_swiglu",
        "rope_gelu",
    }
    assert all(isinstance(text, str) for text in generations.values())

    out_dir = Path(config["out_dir"])
    assert json.loads(
        (out_dir / "report.json").read_text(encoding="utf-8")
    ) == report
    markdown = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "| Variant | Position | MLP | FFN hidden | Params |" in markdown
    assert "Matched training tokens: **256**" in markdown
    assert "## Scaffold 1" in markdown
    assert all(name in markdown for name in generations)


def test_module_cli_writes_report(tmp_path, fixture_path, toy_artifacts):
    config = report_config(tmp_path, fixture_path, toy_artifacts)
    config["out_dir"] = str(tmp_path / "cli-report")
    config_path = tmp_path / "report.toml"
    write_report_config(config_path, config)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tinystories_v2.ablation",
            "--config",
            str(config_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (Path(config["out_dir"]) / "report.json").exists()
    assert (Path(config["out_dir"]) / "report.md").exists()
```

- [ ] **Step 2: Run the report tests to verify the helper is absent**

Run: `rtk .venv/bin/pytest tests/test_ablation.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'tinystories_v2.ablation'`.

- [ ] **Step 3: Implement checkpoint validation, held-out evaluation, and sampling**

Create `src/tinystories_v2/ablation.py`:

```python
"""Issue 09: guarded report for the matched ~5M architecture ablation.

The existing Pretraining stage creates each run. This helper is evaluation
only: it loads the latest checkpoints, rejects unequal token budgets or model
sizes outside the declared tolerance, computes held-out perplexity on one
shared packed eval slice, and generates each variant from identical fixed
Scaffolds. It writes machine-readable JSON plus a report-ready Markdown table.
"""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from tokenizers import Tokenizer

from tinystories_v2.checkpoint import latest_checkpoint, load_checkpoint
from tinystories_v2.config import load_config
from tinystories_v2.generate import sample
from tinystories_v2.model import FableLM, ModelConfig
from tinystories_v2.pack import load_packed, pack_split
from tinystories_v2.perplexity import perplexity
from tinystories_v2.slot_prompt import SLOT_FIELDS
from tinystories_v2.slots import Scaffold, extract_slots


def validate_comparability(
    rows: list[dict], *, baseline: str, param_tolerance: float
) -> int:
    """Return shared tokens_seen after enforcing the fairness contract."""
    if not rows:
        raise ValueError("at least one ablation variant is required")
    if param_tolerance < 0:
        raise ValueError("param_tolerance must be non-negative")
    by_name = {row["variant"]: row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError("ablation variant names must be unique")
    if baseline not in by_name:
        raise ValueError(f"baseline variant {baseline!r} is missing")

    token_counts = {int(row["tokens_seen"]) for row in rows}
    if len(token_counts) != 1:
        observed = sorted(token_counts)
        raise ValueError(
            "matched training tokens required; observed " f"{observed}"
        )

    baseline_params = int(by_name[baseline]["params"])
    for row in rows:
        drift = abs(int(row["params"]) - baseline_params) / baseline_params
        if drift > param_tolerance:
            raise ValueError(
                f"variant {row['variant']!r} exceeds parameter tolerance: "
                f"{drift:.6f} > {param_tolerance:.6f}"
            )
    return token_counts.pop()


def _resolve_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path)
    if checkpoint.is_dir():
        latest = latest_checkpoint(checkpoint)
        if latest is None:
            raise ValueError(f"no step_*.pt checkpoints in {checkpoint}")
        return latest
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _load_variants(config: dict) -> list[tuple[dict, FableLM]]:
    loaded = []
    for variant in config["variants"]:
        checkpoint = _resolve_checkpoint(variant["checkpoint"])
        state = load_checkpoint(checkpoint)
        model_config = ModelConfig(**state["config"]["model"])
        model = FableLM(model_config)
        model.load_state_dict(state["model"])
        row = {
            "variant": variant["name"],
            "position_encoding": model_config.position_encoding,
            "mlp_type": model_config.mlp_type,
            "ffn_hidden": model_config.ffn_hidden,
            "params": model.num_params(),
            "step": int(state["step"]),
            "tokens_seen": int(state["tokens_seen"]),
            "checkpoint": str(checkpoint),
        }
        loaded.append((row, model))
    return loaded


def _load_eval_tokens(eval_config: dict):
    packed_path = Path(eval_config["packed_path"])
    manifest_path = Path(str(packed_path) + ".json")
    if not packed_path.exists() or not manifest_path.exists():
        pack_split(
            eval_config["split_path"],
            eval_config["tokenizer_path"],
            packed_path,
        )
    packed = load_packed(packed_path)
    max_tokens = int(eval_config["max_tokens"])
    if max_tokens < 2:
        raise ValueError("eval max_tokens must be at least 2")
    count = min(len(packed), max_tokens)
    if count < 2:
        raise ValueError("packed eval split must contain at least two tokens")
    return packed[:count].copy()


def _load_scaffolds(path: str | Path, count: int) -> list[Scaffold]:
    if count < 1:
        raise ValueError("sample_count must be at least 1")
    scaffolds = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if all(field in record for field in SLOT_FIELDS):
                scaffold = Scaffold(
                    **{field: record[field] for field in SLOT_FIELDS}
                )
            else:
                scaffold = extract_slots(record["prompt"])
            scaffolds.append(scaffold)
            if len(scaffolds) == count:
                return scaffolds
    raise ValueError(
        f"eval split contains {len(scaffolds)} Scaffolds; "
        f"sample_count requests {count}"
    )


def render_scaffold_seed(scaffold: Scaffold) -> str:
    """Build a Fable-like prefix from a Scaffold for Pretraining models."""
    return f"In {scaffold.setting}, a {scaffold.trait} {scaffold.character}"


def render_markdown(report: dict) -> str:
    lines = [
        "# 5M Architecture Ablation",
        "",
        f"Baseline: `{report['baseline']}`  ",
        f"Required training tokens: **{report['expected_tokens']:,}**  ",
        f"Matched training tokens: **{report['matched_tokens']:,}**  ",
        f"Held-out evaluation tokens: **{report['eval_tokens']:,}**  ",
        f"Parameter tolerance: **{report['param_tolerance']:.1%}**",
        "",
        "| Variant | Position | MLP | FFN hidden | Params | Training tokens | Val loss | Perplexity |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| `{row['variant']}` | {row['position_encoding']} | "
            f"{row['mlp_type']} | {row['ffn_hidden']:,} | "
            f"{row['params']:,} | {row['tokens_seen']:,} | "
            f"{row['val_loss']:.6f} | {row['perplexity']:.6f} |"
        )

    for index, sample_row in enumerate(report["samples"], start=1):
        scaffold = sample_row["scaffold"]
        lines.extend(
            [
                "",
                f"## Scaffold {index}",
                "",
                ", ".join(
                    f"{field}={scaffold[field]}" for field in SLOT_FIELDS
                ),
                "",
                f"Seed: {sample_row['seed']}",
            ]
        )
        for variant, fable in sample_row["generations"].items():
            lines.extend(
                [
                    "",
                    f"### {variant}",
                    "",
                    fable.strip() or "_Empty generation_",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def run(config: dict) -> dict:
    loaded = _load_variants(config)
    rows = [row for row, _ in loaded]
    matched_tokens = validate_comparability(
        rows,
        baseline=config["baseline"],
        param_tolerance=float(config["param_tolerance"]),
    )
    expected_tokens = int(config["expected_tokens"])
    if matched_tokens != expected_tokens:
        raise ValueError(
            f"completed token budget required: expected {expected_tokens}, "
            f"observed {matched_tokens}"
        )

    eval_config = config["eval"]
    eval_tokens = _load_eval_tokens(eval_config)
    tokenizer = Tokenizer.from_file(str(eval_config["tokenizer_path"]))
    scaffolds = _load_scaffolds(
        eval_config["split_path"], int(eval_config["sample_count"])
    )
    sample_rows = [
        {
            "scaffold": asdict(scaffold),
            "seed": render_scaffold_seed(scaffold),
            "generations": {},
        }
        for scaffold in scaffolds
    ]
    end_id = tokenizer.token_to_id("<|end|>")
    device = eval_config["device"]

    for row, model in loaded:
        model = model.to(device).eval()
        value = perplexity(
            model,
            eval_tokens,
            block_size=model.config.context,
            batch_size=int(eval_config["batch_size"]),
            device=device,
        )
        row["perplexity"] = value
        row["val_loss"] = math.log(value)

        for index, scaffold in enumerate(scaffolds):
            seed_ids = tokenizer.encode(render_scaffold_seed(scaffold)).ids
            sequence = sample(
                model,
                seed_ids,
                num_samples=1,
                max_new_tokens=int(eval_config["max_new_tokens"]),
                temperature=float(eval_config["temperature"]),
                top_p=float(eval_config["top_p"]),
                seed=int(eval_config["seed"]) + index,
                end_id=end_id,
                device=device,
            )[0]
            sample_rows[index]["generations"][row["variant"]] = (
                tokenizer.decode(sequence)
            )
        model.to("cpu")

    report = {
        "baseline": config["baseline"],
        "param_tolerance": float(config["param_tolerance"]),
        "expected_tokens": expected_tokens,
        "matched_tokens": matched_tokens,
        "eval_tokens": len(eval_tokens),
        "rows": rows,
        "samples": sample_rows,
    }
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Register the CLI and add the real report config**

In `pyproject.toml`, add this line to `[project.scripts]` after `ts2-generate`:

```toml
ts2-ablation-report = "tinystories_v2.ablation:main"
```

Create `configs/ablation_5m_report.toml`:

```toml
# Issue 09 report: compare only completed, matched-token 5M checkpoints.
out_dir = "docs/experiments/5m-architecture-ablation"
baseline = "rope_swiglu"
param_tolerance = 0.03
expected_tokens = 498073600

[eval]
split_path = "artifacts/data_prep_full/splits/eval.jsonl"
tokenizer_path = "artifacts/tokenizer_full/tokenizer.json"
packed_path = "artifacts/ablation_5m/eval.bin"
max_tokens = 1000000
batch_size = 32
device = "cuda"
sample_count = 3
max_new_tokens = 400
temperature = 0.8
top_p = 0.95
seed = 2026

[[variants]]
name = "rope_swiglu"
checkpoint = "artifacts/ablation_5m/rope_swiglu/checkpoints"

[[variants]]
name = "learned_swiglu"
checkpoint = "artifacts/ablation_5m/learned_swiglu/checkpoints"

[[variants]]
name = "rope_gelu"
checkpoint = "artifacts/ablation_5m/rope_gelu/checkpoints"
```

- [ ] **Step 5: Pin the committed report config in the config tests**

Append to `tests/test_ablation_configs.py`:

```python
def test_real_report_config_uses_all_variants_and_fixed_eval_slice():
    path = REPO_ROOT / "configs" / "ablation_5m_report.toml"
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    assert config["out_dir"] == "docs/experiments/5m-architecture-ablation"
    assert config["baseline"] == "rope_swiglu"
    assert config["param_tolerance"] == 0.03
    assert config["expected_tokens"] == 498_073_600
    assert [variant["name"] for variant in config["variants"]] == [
        "rope_swiglu",
        "learned_swiglu",
        "rope_gelu",
    ]
    assert [variant["checkpoint"] for variant in config["variants"]] == [
        "artifacts/ablation_5m/rope_swiglu/checkpoints",
        "artifacts/ablation_5m/learned_swiglu/checkpoints",
        "artifacts/ablation_5m/rope_gelu/checkpoints",
    ]
    assert config["eval"] == {
        "split_path": "artifacts/data_prep_full/splits/eval.jsonl",
        "tokenizer_path": "artifacts/tokenizer_full/tokenizer.json",
        "packed_path": "artifacts/ablation_5m/eval.bin",
        "max_tokens": 1_000_000,
        "batch_size": 32,
        "device": "cuda",
        "sample_count": 3,
        "max_new_tokens": 400,
        "temperature": 0.8,
        "top_p": 0.95,
        "seed": 2026,
    }
```

- [ ] **Step 6: Run the helper, CLI, and config tests**

```bash
rtk .venv/bin/pytest tests/test_ablation.py tests/test_ablation_configs.py -v
rtk .venv/bin/python -m tinystories_v2.ablation --help
rtk git diff --check
```

Expected: all ablation/report tests pass; the module help exits 0 and includes `--config`; the whitespace check prints nothing. The tests build only tiny local checkpoints and never contact W&B or the Hub.

- [ ] **Step 7: Commit the report helper**

```bash
rtk git add src/tinystories_v2/ablation.py tests/test_ablation.py tests/test_ablation_configs.py configs/ablation_5m_report.toml pyproject.toml
rtk git commit -m "feat: report matched architecture ablation results

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Run the real matched experiment and version its evidence

**Files:**
- Create (generated): `docs/experiments/5m-architecture-ablation/report.json`
- Create (generated): `docs/experiments/5m-architecture-ablation/report.md`

**Interfaces:**
- Consumes: the four configs and report CLI from Tasks 2-3; private Hub artifacts `congthanh991/tinystories-v2-tokenizer`, `congthanh991/tinystories-v2-packed`, and `congthanh991/tinystories-v2-data`; a Colab Pro L4 with `HF_TOKEN` and `WANDB_API_KEY` in the environment.
- Produces: three completed Hub-backed Pretraining runs at exactly 498,073,600 tokens, three W&B loss curves, and a committed JSON/Markdown report containing real held-out metrics and fixed-Scaffold generations.

- [ ] **Step 1: On the L4 runtime, install the branch and restore immutable input artifacts**

From the repository root in the Colab runtime, with `HF_TOKEN` and `WANDB_API_KEY` already loaded from Colab Secrets, run:

```bash
rtk pip install -q -e '.[track]'
rtk hf download congthanh991/tinystories-v2-tokenizer tokenizer.json --local-dir artifacts/tokenizer_full
rtk hf download congthanh991/tinystories-v2-packed pretrain.bin pretrain.bin.json --local-dir artifacts/packed_full
rtk hf download congthanh991/tinystories-v2-data splits/eval.jsonl --local-dir artifacts/data_prep_full
rtk python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

Expected: each `hf download` exits 0; the final command prints an L4 device and `True`. The packed binary and its `.json` manifest prevent `ts2-pretrain` from needing the 1.1M-row raw Pretraining split.

- [ ] **Step 2: Run or resume the baseline to its configured token budget**

Run: `rtk ts2-pretrain --config configs/ablation_5m_rope_swiglu.toml --resume`

Expected: the command exits 0, syncs the final checkpoint to `congthanh991/tinystories-v2-ablation-rope-swiglu`, and writes `artifacts/ablation_5m/rope_swiglu/manifest.json` with step 3800 and 498,073,600 tokens. If Colab preempts the process, reconnect and run this same command until it exits 0.

- [ ] **Step 3: Run or resume the learned-position variant to the same budget**

Run: `rtk ts2-pretrain --config configs/ablation_5m_learned_swiglu.toml --resume`

Expected: the command exits 0, syncs the final checkpoint to `congthanh991/tinystories-v2-ablation-learned-swiglu`, and writes `artifacts/ablation_5m/learned_swiglu/manifest.json` with step 3800 and 498,073,600 tokens. On preemption, rerun the same command.

- [ ] **Step 4: Run or resume the parameter-matched GELU variant**

Run: `rtk ts2-pretrain --config configs/ablation_5m_rope_gelu.toml --resume`

Expected: the command exits 0, syncs the final checkpoint to `congthanh991/tinystories-v2-ablation-rope-gelu`, and writes `artifacts/ablation_5m/rope_gelu/manifest.json` with step 3800 and 498,073,600 tokens. On preemption, rerun the same command.

- [ ] **Step 5: Verify completion and architecture identity before evaluation**

```bash
rtk jq -e '.final_step == 3800 and .tokens_seen == 498073600 and .config.model.position_encoding == "rope" and .config.model.mlp_type == "swiglu" and .config.model.ffn_hidden == 704' artifacts/ablation_5m/rope_swiglu/manifest.json
rtk jq -e '.final_step == 3800 and .tokens_seen == 498073600 and .config.model.position_encoding == "learned" and .config.model.mlp_type == "swiglu" and .config.model.ffn_hidden == 704' artifacts/ablation_5m/learned_swiglu/manifest.json
rtk jq -e '.final_step == 3800 and .tokens_seen == 498073600 and .config.model.position_encoding == "rope" and .config.model.mlp_type == "gelu" and .config.model.ffn_hidden == 1056' artifacts/ablation_5m/rope_gelu/manifest.json
```

Expected: all three `jq -e` commands exit 0 and print `true`. Do not generate a report if any command fails; resume that variant first.

- [ ] **Step 6: Generate the real held-out table and fixed-Scaffold samples**

Run: `rtk ts2-ablation-report --config configs/ablation_5m_report.toml`

Expected: the command exits 0 and creates `docs/experiments/5m-architecture-ablation/report.json` plus `report.md`. It evaluates exactly 1,000,000 held-out tokens per variant on CUDA and generates three seeded Fables per variant from the same three held-out Scaffolds.

- [ ] **Step 7: Validate the generated experiment artifact mechanically**

```bash
rtk jq -e '.expected_tokens == 498073600 and .matched_tokens == 498073600 and .eval_tokens == 1000000 and (.rows | length) == 3 and ([.rows[].tokens_seen] | all(. == 498073600)) and ([.rows[].params] == [5310720, 5441792, 5310720]) and (.samples | length) == 3 and ([.samples[].generations | keys | length] | all(. == 3))' docs/experiments/5m-architecture-ablation/report.json
rtk rg -n '^\| `(?:rope_swiglu|learned_swiglu|rope_gelu)`|^## Scaffold [123]$' docs/experiments/5m-architecture-ablation/report.md
rtk git diff --check
```

Expected: `jq` exits 0 and prints `true`; `rg` prints exactly three variant-table rows and three Scaffold headings; the whitespace check prints nothing. The W&B project `tinystories-v2` contains the three run names from the configs for matched-token loss-curve plotting.

- [ ] **Step 8: Commit the real empirical evidence**

```bash
rtk git add docs/experiments/5m-architecture-ablation/report.json docs/experiments/5m-architecture-ablation/report.md
rtk git commit -m "docs: record 5M architecture ablation results

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Audit acceptance criteria and close issue 09

**Files:**
- Modify: `.scratch/tinystories-v2-pipeline/issues/09-architecture-ablation.md:3,30-35`
- Modify: `PROGRESS.md:10-14,40,59,63`

**Interfaces:**
- Consumes: every test, config, real checkpoint manifest, and versioned report delivered by Tasks 1-4.
- Produces: a checked source-of-truth issue and team progress board whose completion claim points to executable tests and real empirical evidence.

- [ ] **Step 1: Run the issue-specific acceptance suite**

Run: `rtk .venv/bin/pytest tests/test_model.py tests/test_pretrain_stage.py tests/test_ablation_configs.py tests/test_ablation.py -q`

Expected: all selected tests pass with zero failures, including three causality cases, three toy loss-decrease runs through `pretrain.run`, exact real-config checks, comparability rejection tests, report generation, fixed-Scaffold sampling, and the module CLI.

- [ ] **Step 2: Run repository-wide verification and inspect the versioned result**

```bash
rtk .venv/bin/pytest -q
rtk jq -e '.matched_tokens == 498073600 and (.rows | length) == 3 and (.samples | length) == 3' docs/experiments/5m-architecture-ablation/report.json
rtk git ls-files --error-unmatch docs/experiments/5m-architecture-ablation/report.json docs/experiments/5m-architecture-ablation/report.md
rtk git diff -- src/tinystories_v2/pretrain.py
rtk git diff --check
```

Expected: the full suite passes with zero failures; `jq` prints `true`; both report paths are tracked; the `pretrain.py` diff and whitespace check are empty.

- [ ] **Step 3: Mark every local issue acceptance item complete with named evidence**

In `.scratch/tinystories-v2-pipeline/issues/09-architecture-ablation.md`, change:

```markdown
Status: ready-for-agent
```

to:

```markdown
Status: complete
```

Replace its acceptance checklist with:

```markdown
- [x] Model config selects positional encoding (`rope` / `learned`) and MLP (`swiglu` / `gelu`); parameterized causality and analytic-count invariants pass for every variant (`tests/test_model.py`)
- [x] All variants train through the unchanged Pretraining stage using config only (`test_toy_ablation_variant_decreases_loss_through_stage_entrypoint`; `src/tinystories_v2/pretrain.py` has no issue-09 diff)
- [x] CPU toy runs for RoPE + SwiGLU, learned + SwiGLU, and RoPE + GELU decrease loss (`test_toy_ablation_variant_decreases_loss_through_stage_entrypoint`)
- [x] `ts2-ablation-report` produces a guarded table with variant, params, matched tokens, held-out loss, and perplexity; toy artifact/CLI coverage is in `tests/test_ablation.py`, and real evidence is versioned at `docs/experiments/5m-architecture-ablation/`
- [x] The three `configs/ablation_5m_*.toml` Pretraining configs launch the real runs; exact config contracts are pinned by `tests/test_ablation_configs.py`
- [x] Variants are within the documented 3% parameter tolerance: 5,310,720 / 5,441,792 / 5,310,720; GELU hidden 1056 exactly matches SwiGLU hidden 704 projection parameters
```

- [ ] **Step 4: Update the team-facing progress snapshot**

Make these exact edits in `PROGRESS.md`:

1. In the `## Now` highest-leverage bullet, change `**05, 07, 08, 09**` to `**05, 07, 08**`.
2. In the issue board, replace the issue 09 row with:

```markdown
| 09 | Architecture ablation at 5M scale | 02 ✅ | ✅ complete — real matched-token report versioned |
```

3. In the W7-8 milestone row, replace the Actual cell with:

```markdown
reference-free metrics (11) ✅; 5M ablation (09) ✅; eval suite (07) ready
```

4. Insert this entry at the top of `## Log`:

```markdown
- **2026-07-12** — Issue 09 (5M architecture ablation) complete: the model now
  config-selects RoPE/learned positions and SwiGLU/parameter-matched GELU while
  preserving old checkpoint defaults. Three 498,073,600-token runs compare
  5,310,720 / 5,441,792 / 5,310,720 parameters on identical data, schedule,
  and seed; held-out loss/perplexity plus fixed-Scaffold generations are
  versioned in `docs/experiments/5m-architecture-ablation/`.
```

- [ ] **Step 5: Commit the acceptance audit and tracker update**

```bash
rtk git add .scratch/tinystories-v2-pipeline/issues/09-architecture-ablation.md PROGRESS.md
rtk git commit -m "docs: complete issue 09 architecture ablation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Acceptance Mapping

- Config-selected RoPE/learned positions and SwiGLU/GELU, with causality and parameter invariants for every variant: Task 1.
- Existing Pretraining entrypoint unchanged and toy loss decreases for every variant: Task 2.
- Three real configs, identical data/schedule/seed/token budget, W&B loss curves, documented 3% tolerance: Task 2.
- Guarded held-out loss/perplexity table and fixed-Scaffold generations, tested on toy checkpoints: Task 3.
- Real 498,073,600-token runs and versioned empirical report artifact: Task 4.
- Full-suite verification, source-of-truth checklist, and progress closure: Task 5.
