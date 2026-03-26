"""
TITANS Model
=============
High-level model wrapper that stacks TITANS blocks with an embedding
layer and optional task heads. Domain-agnostic — works for NLP,
time series, or any sequence modeling task.

Usage:
    from titans_trainer import TitansModel

    # Language model
    model = TitansModel(vocab_size=32000, d_model=512, n_layers=6)
    logits = model(token_ids)  # (batch, seq_len, vocab_size)

    # With custom embedding (e.g., continuous features)
    model = TitansModel(vocab_size=None, d_model=256, n_layers=4)
    x = torch.randn(4, 1000, 256)  # pre-embedded input
    out = model(x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List
from .block import TitansBlock


class TitansModel(nn.Module):
    """
    Full TITANS model with stacked MAC blocks.

    Supports two modes:
    1. Discrete tokens (vocab_size > 0): adds embedding + LM head
    2. Continuous input (vocab_size = None): pass pre-embedded features

    Args:
        vocab_size: Vocabulary size. None for continuous input.
        d_model: Model dimension (default: 512).
        n_layers: Number of TITANS blocks (default: 6).
        n_heads: Number of attention heads (default: 8).
        d_ff: FFN hidden dimension (default: d_model * 4).
        max_seq_len: Maximum sequence length for positional encoding (default: 2048).
        memory_depth: Layers in each memory MLP (default: 2).
        n_persistent: Persistent memory tokens per block (default: 64).
        chunk_size: Tokens per memory gradient step (default: 128).
        dropout: Dropout rate (default: 0.1).
        padding_idx: Padding token ID (default: 0).
    """

    @classmethod
    def from_config(cls, config) -> 'TitansModel':
        """
        Build model from a TitansConfig object.

        Usage:
            from titans_trainer import TitansConfig, TitansModel
            config = TitansConfig(vocab_size=32000, d_model=512, n_layers=6)
            model = TitansModel.from_config(config)
        """
        return cls(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            memory_depth=config.memory_depth,
            n_persistent=config.n_persistent,
            chunk_size=config.chunk_size,
            dropout=config.dropout,
            padding_idx=config.padding_idx,
            causal=getattr(config, 'causal', False),
        )

    @classmethod
    def from_pretrained(cls, path: str, device: str = 'cpu') -> 'TitansModel':
        """
        Load model from a saved checkpoint.

        Usage:
            model = TitansModel.from_pretrained("outputs/best_model.pt")
        """
        from .config import TitansConfig
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = TitansConfig.from_dict(ckpt.get('config', {}))
        model = cls.from_config(config)
        model.load_state_dict(ckpt['model_state_dict'])
        return model

    def save_pretrained(self, path: str):
        """
        Save model weights and config.

        Usage:
            model.save_pretrained("outputs/my_model.pt")
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
            },
        }, path)

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = None,
        max_seq_len: int = 2048,
        memory_depth: int = 2,
        n_persistent: int = 64,
        chunk_size: int = 128,
        dropout: float = 0.1,
        padding_idx: int = 0,
        causal: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embedding (only for discrete tokens)
        if vocab_size is not None:
            self.embedding = nn.Embedding(
                vocab_size, d_model, padding_idx=padding_idx
            )
            self.pos_encoding = nn.Parameter(
                self._sinusoidal_encoding(max_seq_len, d_model),
                requires_grad=False,
            )
            self.embed_norm = nn.LayerNorm(d_model)
            self.embed_dropout = nn.Dropout(dropout)
        else:
            self.embedding = None
            self.pos_encoding = None

        # Stacked TITANS blocks
        self.blocks = nn.ModuleList([
            TitansBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff or d_model * 4,
                memory_depth=memory_depth,
                n_persistent=n_persistent,
                chunk_size=chunk_size,
                dropout=dropout,
                causal=causal,
            )
            for _ in range(n_layers)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(d_model)

        # LM head (only for discrete tokens)
        if vocab_size is not None:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            # Weight tying
            self.lm_head.weight = self.embedding.weight
        else:
            self.lm_head = None

        # Initialize
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"TitansModel: {n_params / 1e6:.1f}M params "
              f"(d={d_model}, L={n_layers}, H={n_heads}, "
              f"mem_depth={memory_depth}, persistent={n_persistent})")

    @staticmethod
    def _sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Token IDs (batch, seq_len) if vocab_size set,
               or pre-embedded (batch, seq_len, d_model) if not.
            labels: Target token IDs for language modeling loss.
                    -100 for positions to ignore (standard convention).
            output_hidden: If True, return all hidden states.

        Returns:
            dict with:
                'logits': (batch, seq_len, vocab_size) or (batch, seq_len, d_model)
                'loss': scalar loss if labels provided, else None
                'hidden_states': list of (batch, seq_len, d_model) if requested
        """
        # Embed if discrete tokens
        if self.embedding is not None:
            B, L = x.shape
            h = self.embedding(x)
            h = h + self.pos_encoding[:, :L, :]
            h = self.embed_dropout(self.embed_norm(h))
        else:
            h = x

        # Pass through TITANS blocks
        hidden_states = []
        for block in self.blocks:
            h = block(h)
            if output_hidden:
                hidden_states.append(h)

        h = self.final_norm(h)

        # LM head or raw output
        if self.lm_head is not None:
            logits = self.lm_head(h)
        else:
            logits = h

        # Compute loss
        loss = None
        if labels is not None and self.vocab_size is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        result = {'logits': logits, 'loss': loss}
        if output_hidden:
            result['hidden_states'] = hidden_states
        return result

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract sequence-level embedding via mean pooling.
        Useful for classification, retrieval, etc.
        """
        if self.embedding is not None:
            B, L = x.shape
            h = self.embedding(x)
            h = h + self.pos_encoding[:, :L, :]
            h = self.embed_dropout(self.embed_norm(h))
            pad_mask = (x != 0).unsqueeze(-1).float()
        else:
            h = x
            pad_mask = torch.ones(h.shape[0], h.shape[1], 1, device=h.device)

        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)

        # Mean pool over non-padding positions
        pooled = (h * pad_mask).sum(dim=1) / pad_mask.sum(dim=1).clamp(min=1)
        return pooled

    def get_surprise_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get per-token surprise scores from all TITANS blocks.
        Tokens with high surprise deviate from learned patterns.

        Useful for anomaly detection, novelty scoring, out-of-distribution detection.

        Returns:
            (batch, seq_len, n_layers) surprise scores
        """
        if self.embedding is not None:
            B, L = x.shape
            h = self.embedding(x)
            h = h + self.pos_encoding[:, :L, :]
            h = self.embed_dropout(self.embed_norm(h))
        else:
            h = x

        surprises = []
        for block in self.blocks:
            _, surprise = block.neural_memory(
                block.norm_memory(h), return_surprise=True
            )
            surprises.append(surprise)
            h = block(h)

        return torch.cat(surprises, dim=-1)  # (B, L, n_layers)
