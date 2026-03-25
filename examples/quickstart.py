"""
Quickstart: Train a TITANS Model
==================================
Complete HuggingFace-style workflow in one file.

    pip install titans-trainer
    python examples/quickstart.py
"""

import torch
from torch.utils.data import Dataset
from titans_trainer import TitansConfig, TitansModel, TitansTrainer


# --- 1. Dataset (same format as HuggingFace) ---
class SimpleMLMDataset(Dataset):
    """Minimal MLM dataset. Replace with your real data."""

    def __init__(self, vocab_size, seq_len, n_samples, mask_prob=0.15):
        self.data = torch.randint(1, vocab_size, (n_samples, seq_len))
        self.mask_token = vocab_size - 1
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx].clone()
        labels = tokens.clone()
        mask = torch.rand(tokens.shape) < self.mask_prob
        labels[~mask] = -100
        tokens[mask] = self.mask_token
        return {'input_ids': tokens, 'labels': labels}


# --- 2. Configure ---
config = TitansConfig(
    vocab_size=10000,
    d_model=128,
    n_layers=4,
    n_heads=4,
    memory_depth=2,
    n_persistent=16,
    chunk_size=64,
    max_seq_len=512,

    lr=3e-4,
    epochs=2,
    batch_size=8,
    warmup_steps=50,
    use_amp=True,
    log_interval=10,
    val_every_steps=50,
    use_wandb=False,
    output_dir="./outputs/quickstart",
)

# --- 3. Build model ---
model = TitansModel.from_config(config)

# --- 4. Create datasets ---
train_ds = SimpleMLMDataset(config.vocab_size, config.max_seq_len, n_samples=1000)
val_ds = SimpleMLMDataset(config.vocab_size, config.max_seq_len, n_samples=100)

# --- 5. Train ---
trainer = TitansTrainer(model, train_ds, val_ds, config)
best_loss = trainer.train()

# --- 6. Save & reload ---
model.save_pretrained("outputs/quickstart/model.pt")
reloaded = TitansModel.from_pretrained("outputs/quickstart/model.pt")
print(f"Reloaded: {sum(p.numel() for p in reloaded.parameters()) / 1e6:.1f}M params")

# --- 7. Surprise scores (test-time learning in action) ---
reloaded.eval()
with torch.no_grad():
    test_input = torch.randint(1, 10000, (1, 512))
    surprise = reloaded.get_surprise_scores(test_input)
    print(f"Mean surprise: {surprise.mean():.4f}")
