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
