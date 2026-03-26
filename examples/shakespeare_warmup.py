"""
TITANS Test-Time Memory Warmup
================================
Load the best checkpoint, warm up neural memory on training data,
then evaluate on val set. The TITANS advantage: memory adapts at
inference without retraining.

Compares "cold" val loss (standard) vs "warm" val loss (after memory adaptation).
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import urllib.request
import copy

from titans_trainer import TitansConfig, TitansModel


# --- 1. Data (same as benchmark) ---
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


class ShakespeareDataset(Dataset):
    def __init__(self, tokens, seq_len=256):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) // self.seq_len - 1

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


split = int(len(tokens) * 0.9)
train_ds = ShakespeareDataset(tokens[:split])
val_ds = ShakespeareDataset(tokens[split:])


# --- 2. Load best model ---
print("Loading best model from outputs/shakespeare_bench/best_model.pt ...")
ckpt = torch.load("outputs/shakespeare_bench/best_model.pt", map_location="cpu", weights_only=False)
model = TitansModel(
    vocab_size=vocab_size, d_model=296, n_layers=6, n_heads=4,
    d_ff=1184, memory_depth=2, n_persistent=16, chunk_size=64,
    max_seq_len=256, causal=True, dropout=0.35,
)
model.load_state_dict(ckpt["model_state_dict"])
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
print(f"Loaded (epoch {ckpt['epoch']}, val_loss={ckpt['metrics']['val_loss']:.4f})")


# --- 3. Evaluate COLD (standard — memory resets each forward call) ---
@torch.no_grad()
def evaluate(model, dataset, desc="eval"):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(ids, labels=labels)
        n = (labels != -100).sum().item()
        total_loss += out["loss"].item() * n
        total_tokens += n
    avg_loss = total_loss / total_tokens
    print(f"  {desc}: loss={avg_loss:.4f}")
    return avg_loss


print("\n--- Cold evaluation (standard, no warmup) ---")
cold_loss = evaluate(model, val_ds, "cold val")


# --- 4. Warm up memory on training data, then evaluate ---
# Strategy: process training sequences through the model. After each sequence,
# persist the adapted memory weights back into the model's base weights.
# This lets the memory accumulate knowledge across the training set.

def warmup_memory(model, dataset, n_passes=1):
    """
    Feed data through the model and persist adapted memory weights.

    For each sequence, NeuralMemory:
    1. Clones base weights
    2. Updates them chunk-by-chunk via surprise-driven gradient descent
    3. Normally discards the updated weights

    We intercept step 3: after processing each sequence, we write the
    adapted weights back into mem_weights/mem_biases so they persist.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for pass_idx in range(n_passes):
        n_seqs = 0
        for batch in loader:
            ids = batch["input_ids"].to(device)

            # For each block's neural memory, we need to:
            # 1. Run the memory update (which adapts weights internally)
            # 2. Capture the final adapted weights
            # 3. Write them back as the new base weights

            for block in model.blocks:
                mem = block.neural_memory
                x = block.norm_memory(model._embed(ids))

                # Run the chunk-wise update and capture final weights
                with torch.enable_grad():
                    B, L, D = x.shape
                    chunk_size = min(mem.chunk_size, L)

                    weights = [w.clone().requires_grad_(True) for w in mem.mem_weights]
                    biases = [b.clone().requires_grad_(True) for b in mem.mem_biases]
                    all_params = weights + biases

                    for start in range(0, L, chunk_size):
                        chunk = x[:, start:start + chunk_size, :]
                        from titans_trainer.memory import _mlp_forward
                        pred = _mlp_forward(weights, biases, chunk)
                        loss = F.mse_loss(pred, chunk.detach())
                        grads = torch.autograd.grad(loss, all_params, create_graph=False)
                        n_layers = len(weights)
                        w_grads = grads[:n_layers]
                        b_grads = grads[n_layers:]

                        effective_lr = mem.lr_memory * mem.lr_gate(chunk).mean()
                        forget = mem.forget_gate(chunk.mean(dim=1)).mean()

                        weights = [(forget * w - effective_lr * g).detach().requires_grad_(True)
                                   for w, g in zip(weights, w_grads)]
                        biases = [(forget * b - effective_lr * g).detach().requires_grad_(True)
                                  for b, g in zip(biases, b_grads)]
                        all_params = weights + biases

                    # Persist adapted weights back
                    with torch.no_grad():
                        for i, (w, b) in enumerate(zip(weights, biases)):
                            mem.mem_weights[i].copy_(w.detach())
                            mem.mem_biases[i].copy_(b.detach())

            n_seqs += 1
            if n_seqs % 500 == 0:
                print(f"    pass {pass_idx+1}: warmed up on {n_seqs}/{len(loader)} sequences")

        print(f"  Pass {pass_idx+1} complete: {n_seqs} sequences processed")


# We need a helper to get embeddings without running through blocks
def _embed_fn(model, x):
    B, L = x.shape
    h = model.embedding(x)
    h = h + model.pos_encoding[:, :L, :]
    h = model.embed_dropout(model.embed_norm(h))
    return h

# Monkey-patch for warmup
model._embed = lambda x: _embed_fn(model, x)

# Save original weights so we can compare
original_weights = {}
for i, block in enumerate(model.blocks):
    mem = block.neural_memory
    original_weights[i] = {
        'weights': [w.clone() for w in mem.mem_weights],
        'biases': [b.clone() for b in mem.mem_biases],
    }

print("\n--- Warming up memory on training data (1 pass) ---")
warmup_memory(model, train_ds, n_passes=1)

# Check how much weights changed
print("\n--- Memory weight changes ---")
for i, block in enumerate(model.blocks):
    mem = block.neural_memory
    total_delta = 0.0
    total_norm = 0.0
    for j, w in enumerate(mem.mem_weights):
        delta = (w - original_weights[i]['weights'][j]).norm().item()
        orig = original_weights[i]['weights'][j].norm().item()
        total_delta += delta
        total_norm += orig
    print(f"  Block {i}: weight delta = {total_delta:.4f} (original norm = {total_norm:.4f}, "
          f"change = {total_delta/total_norm*100:.1f}%)")

print("\n--- Warm evaluation (after memory adaptation) ---")
warm_loss = evaluate(model, val_ds, "warm val")

print("\n" + "=" * 60)
print("TEST-TIME MEMORY ADAPTATION RESULTS")
print("=" * 60)
print(f"Cold val loss:    {cold_loss:.4f}")
print(f"Warm val loss:    {warm_loss:.4f}")
print(f"Improvement:      {(cold_loss - warm_loss) / cold_loss * 100:.1f}%")
print(f"nanoGPT val loss: ~1.47")
print("=" * 60)
