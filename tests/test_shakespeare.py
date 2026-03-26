"""
End-to-end test: Train a causal TITANS LM on tiny_shakespeare, then generate.
pip install titans-trainer tiktoken
"""

import torch
import pytest
from torch.utils.data import Dataset


class ShakespeareDataset(Dataset):
    def __init__(self, tokens, seq_len=256):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) // self.seq_len - 1

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


@pytest.fixture(scope="module")
def shakespeare_data():
    """Download tiny_shakespeare and tokenize with tiktoken."""
    import urllib.request
    import tiktoken

    text = urllib.request.urlopen(SHAKESPEARE_URL).read().decode("utf-8")

    enc = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor(enc.encode(text), dtype=torch.long)

    split = int(len(tokens) * 0.9)
    train_ds = ShakespeareDataset(tokens[:split])
    val_ds = ShakespeareDataset(tokens[split:])

    return train_ds, val_ds, enc, tokens


def test_shakespeare_train_and_generate(shakespeare_data):
    """End-to-end: train a small causal TITANS LM on Shakespeare, then generate text."""
    from titans_trainer import TitansConfig, TitansModel, TitansTrainer

    train_ds, val_ds, enc, tokens = shakespeare_data
    print(f"\nDataset: {len(tokens):,} tokens")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # --- Config (small for CI speed) ---
    config = TitansConfig(
        vocab_size=enc.n_vocab,  # 50257
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=256,
        memory_depth=2,
        n_persistent=16,
        chunk_size=64,
        max_seq_len=256,
        causal=True,
        lr=3e-4,
        epochs=3,
        batch_size=8,
        warmup_steps=50,
        use_amp=True,
        log_interval=10,
        val_every_steps=100,
        use_wandb=False,
        output_dir="./outputs/shakespeare_test",
    )

    # --- Train ---
    model = TitansModel.from_config(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    trainer = TitansTrainer(model, train_ds, val_ds, config)
    best_val_loss = trainer.train()

    # Loss should be finite and reasonable (well below random ~10.8 for vocab 50k)
    assert best_val_loss < 10.0, f"Val loss too high: {best_val_loss:.2f}"

    # --- Generate ---
    model.eval()
    prompt = enc.encode("ROMEO:")
    input_ids = torch.tensor([prompt]).to(next(model.parameters()).device)

    with torch.no_grad():
        for _ in range(200):
            ctx = input_ids[:, -config.max_seq_len :]
            out = model(ctx)
            logits = out["logits"][:, -1, :] / 0.8
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    generated = enc.decode(input_ids[0].tolist())
    print(f"\n--- Generated ({len(input_ids[0])} tokens) ---")
    print(generated[:500])

    # Sanity checks
    assert len(generated) > len("ROMEO:"), "Generation produced no new text"
    assert input_ids.shape[1] == len(prompt) + 200, "Wrong number of tokens generated"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
