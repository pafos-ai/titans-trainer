"""
Quick demo: Train a TITANS language model on Shakespeare
=========================================================
pip install titans-trainer tiktoken

Trains a 23.5M-param causal TITANS model on tiny_shakespeare (~338K BPE tokens).
Generates sample text showing the model learned Shakespearean English.
"""

import torch
from torch.utils.data import Dataset
import urllib.request
import tiktoken

from titans_trainer import TitansConfig, TitansModel, TitansTrainer

# --- 1. Data ---
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

print("Downloading tiny_shakespeare ...")
text = urllib.request.urlopen(SHAKESPEARE_URL).read().decode("utf-8")

enc = tiktoken.get_encoding("gpt2")
tokens = torch.tensor(enc.encode(text), dtype=torch.long)
print(f"Dataset: {len(tokens):,} tokens")


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


split = int(len(tokens) * 0.9)
train_ds = ShakespeareDataset(tokens[:split])
val_ds = ShakespeareDataset(tokens[split:])
print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")


# --- 2. Config ---
config = TitansConfig(
    vocab_size=enc.n_vocab,  # 50257
    d_model=256,
    n_layers=8,
    n_heads=8,
    d_ff=1024,
    memory_depth=2,
    n_persistent=32,
    chunk_size=64,
    max_seq_len=256,
    causal=True,

    lr=3e-4,
    epochs=25,
    batch_size=8,
    warmup_steps=100,
    use_amp=True,
    log_interval=50,
    val_every_steps=200,
    use_wandb=False,
    output_dir="./outputs/shakespeare",
)


# --- 3. Train ---
model = TitansModel.from_config(config)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

trainer = TitansTrainer(model, train_ds, val_ds, config)
best_val_loss = trainer.train()


# --- 4. Generate samples ---
def generate(model, enc, prompt_text, n_tokens=300, temperature=0.8, top_k=40):
    model.eval()
    device = next(model.parameters()).device
    prompt = enc.encode(prompt_text)
    input_ids = torch.tensor([prompt]).to(device)

    with torch.no_grad():
        for _ in range(n_tokens):
            ctx = input_ids[:, -256:]
            out = model(ctx)
            logits = out["logits"][:, -1, :] / temperature

            if top_k > 0:
                top_vals, _ = torch.topk(logits, top_k)
                logits[logits < top_vals[:, -1:]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return enc.decode(input_ids[0].tolist())


print("\n" + "=" * 60)
print("GENERATED SAMPLES")
print("=" * 60)

for prompt in ["ROMEO:\n", "JULIET:\n", "First Citizen:\n", "KING HENRY IV:\n"]:
    print(f"\n--- Prompt: {prompt.strip()!r} ---")
    print(generate(model, enc, prompt))
    print()

print(f"\nBest val loss: {best_val_loss:.4f}")
