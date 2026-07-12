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

    def hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """Token embeddings through the final RMSNorm: the [B, T, d_model] states
        the LM head reads. Exposed as a seam so the Reward Model (issue 05) can
        attach a scalar head to the same backbone (ADR-0005)."""
        if idx.size(1) > self.config.context:
            raise ValueError(
                f"sequence length {idx.size(1)} exceeds context {self.config.context}"
            )
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        return self.final_norm(x)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.hidden_states(idx))
