"""
TITANS vs nanoGPT: TinyShakespeare Benchmark
=============================================
1-to-1 comparison against Karpathy's nanoGPT character-level Shakespeare.

nanoGPT reference:
  vocab=65, d_model=384, n_layers=6, n_heads=6, d_ff=1536
  batch=64, lr=1e-3, 5000 iters (~80 epochs), ~10M params
  Best val loss: ~1.47

This script matches: same vocab, same param budget, same training compute.
Only difference: TITANS architecture (neural memory + persistent memory).
"""

import torch
from torch.utils.data import Dataset
import urllib.request
import time

from titans_trainer import TitansConfig, TitansModel, TitansTrainer

# --- 1. Data (character-level, same as nanoGPT) ---
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

print("Downloading tiny_shakespeare ...")
text = urllib.request.urlopen(SHAKESPEARE_URL).read().decode("utf-8")

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda t: "".join(itos[i] for i in t)

tokens = torch.tensor(encode(text), dtype=torch.long)
print(f"Dataset: {len(tokens):,} chars, vocab: {vocab_size}")


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


# --- 2. Config (matched to nanoGPT ~10M params) ---
config = TitansConfig(
    vocab_size=vocab_size,  # 65 chars
    d_model=296,            # tuned to match nanoGPT's ~10.65M params
    n_layers=6,
    n_heads=4,              # 296/4 = 74 head_dim
    d_ff=1184,              # 4x d_model
    memory_depth=2,
    n_persistent=16,
    chunk_size=64,
    max_seq_len=256,
    causal=True,
    dropout=0.35,           # higher than nanoGPT's 0.2 — TITANS memory needs more regularization

    lr=1e-3,                # same as nanoGPT
    batch_size=64,          # same as nanoGPT
    epochs=80,              # matches nanoGPT's 5000 iters
    warmup_steps=100,
    use_amp=True,
    log_interval=10,
    val_every_steps=50,
    use_wandb=False,
    output_dir="./outputs/shakespeare_bench",
)


# --- 3. Train ---
model = TitansModel.from_config(config)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n>>> Total params: {n_params:,} (target: ~10M)")
print(f">>> nanoGPT reference: ~10.65M params, val loss ~1.47")
print()

trainer = TitansTrainer(model, train_ds, val_ds, config)
t0 = time.time()
best_val_loss = trainer.train()
elapsed = time.time() - t0


# --- 4. Generate samples ---
def generate(prompt_text, n_tokens=500, temperature=0.8, top_k=20):
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor([encode(prompt_text)]).to(device)

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

    return decode(input_ids[0].tolist())


# --- 5. Results ---
print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)
print(f"TITANS val loss:  {best_val_loss:.4f}")
print(f"nanoGPT val loss: ~1.47")
print(f"Params:           {n_params:,}")
print(f"Training time:    {elapsed/60:.1f} min")
print("=" * 60)

print("\n--- ROMEO: ---")
print(generate("ROMEO:\n"))

print("\n--- First Citizen: ---")
print(generate("First Citizen:\nBefore we proceed any further, hear me speak.\n"))

print(f"\nBest val loss: {best_val_loss:.4f}")
