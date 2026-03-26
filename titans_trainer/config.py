"""
TITANS Configuration
=====================
HuggingFace-style config for TITANS models.

Usage:
    from titans_trainer import TitansConfig

    config = TitansConfig(d_model=256, n_layers=6, n_heads=4)
    config.save("my_config.json")
    config = TitansConfig.from_file("my_config.json")
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TitansConfig:
    """Configuration for TITANS model architecture and training."""

    # Architecture
    vocab_size: Optional[int] = None
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: Optional[int] = None  # Default: d_model * 4
    max_seq_len: int = 2048
    memory_depth: int = 2
    n_persistent: int = 64
    chunk_size: int = 128
    dropout: float = 0.1
    padding_idx: int = 0
    architecture: str = "titans"  # "titans" or "bert" (for ablation)
    causal: bool = False  # Use causal (autoregressive) attention masking

    # Training
    lr: float = 5e-4
    weight_decay: float = 0.001
    warmup_steps: int = 300
    epochs: float = 3.0
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    mask_prob: float = 0.15
    use_amp: bool = True

    # Logging
    log_interval: int = 20
    val_every_steps: int = 500
    save_every_steps: int = 0
    use_wandb: bool = False
    wandb_project: str = "titans"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Data
    num_workers: int = 0
    val_fraction: float = 0.05

    # Output
    output_dir: str = "./outputs"

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = self.d_model * 4

    def save(self, path: str):
        """Save config to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_file(cls, path: str) -> 'TitansConfig':
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, d: dict) -> 'TitansConfig':
        """Create config from dict (e.g., YAML)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def small(cls, vocab_size: int = 32000) -> 'TitansConfig':
        """Small config for quick experiments (~10M params)."""
        return cls(
            vocab_size=vocab_size,
            d_model=256, n_layers=4, n_heads=4,
            d_ff=512, memory_depth=2, n_persistent=32,
            chunk_size=64,
        )

    @classmethod
    def base(cls, vocab_size: int = 32000) -> 'TitansConfig':
        """Base config (~50M params)."""
        return cls(
            vocab_size=vocab_size,
            d_model=512, n_layers=6, n_heads=8,
            memory_depth=2, n_persistent=64,
            chunk_size=128,
        )

    @classmethod
    def large(cls, vocab_size: int = 32000) -> 'TitansConfig':
        """Large config (~200M params)."""
        return cls(
            vocab_size=vocab_size,
            d_model=1024, n_layers=12, n_heads=16,
            memory_depth=3, n_persistent=128,
            chunk_size=128,
        )
