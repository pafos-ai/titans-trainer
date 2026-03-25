"""
titans-trainer: HuggingFace-style Training for TITANS
======================================================
Train TITANS models (Behrouz et al., Google Research, NeurIPS 2025)
with a simple, familiar API.

Quick start:

    from titans_trainer import TitansConfig, TitansModel, TitansTrainer

    config = TitansConfig.base(vocab_size=32000)
    model = TitansModel.from_config(config)
    trainer = TitansTrainer(model, train_dataset, val_dataset, config)
    trainer.train()

    model.save_pretrained("my_model.pt")
    model = TitansModel.from_pretrained("my_model.pt")

    surprise = model.get_surprise_scores(data)  # test-time learning
"""

__version__ = "0.1.0"

# Core components
from .memory import NeuralMemory, PersistentMemory
from .block import TitansBlock, MultiHeadAttention, SwiGLUFFN

# High-level API
from .config import TitansConfig
from .model import TitansModel
from .trainer import TitansTrainer

__all__ = [
    # Core
    "NeuralMemory",
    "PersistentMemory",
    "TitansBlock",
    "MultiHeadAttention",
    "SwiGLUFFN",
    # High-level
    "TitansConfig",
    "TitansModel",
    "TitansTrainer",
]
