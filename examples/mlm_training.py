"""
Masked Language Modeling with TITANS
=====================================
BERT-style pre-training: mask 15% of tokens, predict the original.

This is how foundation models like BERT and RoBERTa are trained.
TITANS adds test-time memory — the model adapts to new data at inference.

    python examples/mlm_training.py
"""

import torch
from torch.utils.data import Dataset
from titans_trainer import TitansConfig, TitansModel, TitansTrainer


class MLMDataset(Dataset):
    """
    Masked Language Modeling dataset.

    Each sample:
    - input_ids: token sequence with 15% replaced by [MASK]
    - labels: original tokens at masked positions, -100 elsewhere

    Following BERT convention:
    - 80% of selected tokens → [MASK]
    - 10% → random token
    - 10% → unchanged
    """

    def __init__(self, token_data: torch.Tensor, vocab_size: int, mask_prob: float = 0.15):
        """
        Args:
            token_data: (n_samples, seq_len) pre-tokenized sequences
            vocab_size: vocabulary size (mask token = vocab_size - 1)
            mask_prob: fraction of tokens to mask
        """
        self.data = token_data
        self.vocab_size = vocab_size
        self.mask_token = vocab_size - 1
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx].clone()
        labels = tokens.clone()

        # Select tokens to mask (skip padding = 0)
        pad_mask = tokens == 0
        prob = torch.full(tokens.shape, self.mask_prob)
        prob[pad_mask] = 0.0
        selected = torch.bernoulli(prob).bool()

        labels[~selected] = -100  # Only compute loss on selected tokens

        # 80% → [MASK]
        replace = torch.bernoulli(torch.full(tokens.shape, 0.8)).bool() & selected
        tokens[replace] = self.mask_token

        # 10% → random token
        random = torch.bernoulli(torch.full(tokens.shape, 0.5)).bool() & selected & ~replace
        tokens[random] = torch.randint(1, self.vocab_size - 1, tokens.shape)[random]

        # 10% → keep original (already handled)
        return {'input_ids': tokens, 'labels': labels}


def main():
    # --- Config ---
    vocab_size = 20000
    seq_len = 512
    config = TitansConfig(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        memory_depth=2,
        n_persistent=32,
        chunk_size=64,
        max_seq_len=seq_len,
        dropout=0.1,

        lr=5e-4,
        epochs=3,
        batch_size=16,
        warmup_steps=100,
        use_amp=True,

        log_interval=20,
        val_every_steps=200,
        use_wandb=False,  # Set True to enable W&B
        output_dir="./outputs/mlm",
    )

    # --- Synthetic data (replace with your tokenized corpus) ---
    print("Generating synthetic training data...")
    train_tokens = torch.randint(1, vocab_size - 1, (5000, seq_len))
    val_tokens = torch.randint(1, vocab_size - 1, (500, seq_len))

    train_ds = MLMDataset(train_tokens, vocab_size)
    val_ds = MLMDataset(val_tokens, vocab_size)

    # --- Model ---
    model = TitansModel.from_config(config)

    # --- Train ---
    trainer = TitansTrainer(model, train_ds, val_ds, config)
    trainer.train()

    # --- Extract embeddings ---
    model.eval()
    with torch.no_grad():
        sample = torch.randint(1, vocab_size, (1, seq_len))
        embedding = model.get_embeddings(sample)
        print(f"\nSequence embedding: {embedding.shape}")  # (1, 256)

        # Surprise scores: which tokens are "unexpected"?
        surprise = model.get_surprise_scores(sample)
        print(f"Surprise scores: {surprise.shape}")  # (1, 512, 4)
        print(f"Mean surprise: {surprise.mean():.4f}")
        print(f"Max surprise position: {surprise.mean(dim=-1).argmax(dim=-1).item()}")


if __name__ == '__main__':
    main()
