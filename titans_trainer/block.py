"""
TITANS Transformer Block (MAC Variant)
=======================================
Memory-as-Context: neural memory output is prepended to the attention
input, giving the core attention access to long-term context.

Architecture per block:
    input → [neural_memory_ctx; input; persistent_memory] → attention → FFN → output

Usage:
    from titans_trainer import TitansBlock

    block = TitansBlock(d_model=256, n_heads=4)
    x = torch.randn(4, 2048, 256)
    out = block(x)  # same shape: (4, 2048, 256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .memory import NeuralMemory, PersistentMemory


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head self-attention.
    Uses PyTorch's native scaled_dot_product_attention for automatic
    kernel selection (Flash Attention when available).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn_out)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020)."""

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TitansBlock(nn.Module):
    """
    Single TITANS block (MAC — Memory-as-Context variant).

    Three memory systems working together:
    1. Neural memory (long-term): MLP with test-time weight updates
    2. Core attention (short-term): standard multi-head attention
    3. Persistent memory (fixed): learned task-independent priors

    The neural memory output is prepended to the attention input,
    giving attention direct access to long-term context. A learned
    gate blends the attention output with the direct memory signal.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: FFN hidden dimension (default: d_model * 4).
        memory_depth: Layers in the memory MLP (default: 2).
        n_persistent: Number of persistent memory tokens (default: 64).
        chunk_size: Tokens per memory gradient step (default: 128).
        dropout: Dropout rate (default: 0.1).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        memory_depth: int = 2,
        n_persistent: int = 64,
        chunk_size: int = 128,
        dropout: float = 0.1,
        # Accepted for config compatibility
        window_size: int = 512,
    ):
        super().__init__()

        # Three memory components
        self.neural_memory = NeuralMemory(
            d_model=d_model,
            memory_depth=memory_depth,
            chunk_size=chunk_size,
        )
        self.persistent_memory = PersistentMemory(
            d_model=d_model,
            n_persistent=n_persistent,
        )

        # Core attention (short-term memory)
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Feed-forward
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_memory = nn.LayerNorm(d_model)

        # Gate to blend memory context with attention
        self.memory_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: optional attention mask
        Returns:
            (batch, seq_len, d_model)
        """
        B, L, D = x.shape

        # 1. Get long-term memory context
        memory_ctx, surprise = self.neural_memory(
            self.norm_memory(x), return_surprise=True
        )

        # 2. Get persistent memory
        persistent = self.persistent_memory(B)

        # 3. Construct augmented sequence: [memory_ctx; x; persistent]
        augmented = torch.cat([memory_ctx, x, persistent], dim=1)

        # 4. Core attention over augmented sequence
        normed = self.norm1(augmented)
        attn_out = self.attention(normed, mask=None)

        # Extract only the positions corresponding to input x
        attn_x = attn_out[:, L:2*L, :]

        # 5. Gate: blend attention output with direct memory
        gate = self.memory_gate(torch.cat([attn_x, memory_ctx], dim=-1))
        gated = gate * attn_x + (1 - gate) * memory_ctx

        # 6. Residual + FFN
        x = x + gated
        x = x + self.ffn(self.norm2(x))

        return x
