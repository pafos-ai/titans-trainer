"""
Example: Time Series Anomaly Detection with TITANS
====================================================
Use TITANS surprise scores to detect anomalies in time series.
The memory learns "normal" patterns, then flags deviations.
"""

import torch
import numpy as np
from titans_trainer import TitansModel

# --- Generate synthetic time series ---
np.random.seed(42)
seq_len = 1024
d_features = 32

# Normal: smooth sinusoidal patterns
t = np.linspace(0, 10 * np.pi, seq_len)
normal_data = np.stack([
    np.sin(t * (i + 1) / 5) + np.random.randn(seq_len) * 0.1
    for i in range(d_features)
], axis=-1)

# Anomalous: inject spikes at random positions
anomaly_data = normal_data.copy()
spike_positions = np.random.choice(seq_len, size=20, replace=False)
anomaly_data[spike_positions] += np.random.randn(20, d_features) * 5

# Convert to tensors
normal_tensor = torch.FloatTensor(normal_data).unsqueeze(0)   # (1, 1024, 32)
anomaly_tensor = torch.FloatTensor(anomaly_data).unsqueeze(0)

# --- Model (continuous input, no vocab) ---
model = TitansModel(
    vocab_size=None,  # continuous features, no embedding layer
    d_model=d_features,
    n_layers=3,
    n_heads=4,
    memory_depth=2,
    n_persistent=16,
    chunk_size=64,
)

# --- Quick training on normal data ---
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("Training on normal patterns...")
for step in range(50):
    out = model(normal_tensor)
    # Self-supervised: predict input from context
    loss = torch.nn.functional.mse_loss(out['logits'], normal_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step + 1) % 10 == 0:
        print(f"  Step {step+1}: loss={loss.item():.4f}")

# --- Detect anomalies via surprise ---
model.eval()
with torch.no_grad():
    normal_surprise = model.get_surprise_scores(normal_tensor)    # (1, 1024, 3)
    anomaly_surprise = model.get_surprise_scores(anomaly_tensor)

# Average surprise across layers
normal_scores = normal_surprise.mean(dim=-1).squeeze()   # (1024,)
anomaly_scores = anomaly_surprise.mean(dim=-1).squeeze()

print(f"\nNormal data — mean surprise: {normal_scores.mean():.4f}")
print(f"Anomaly data — mean surprise: {anomaly_scores.mean():.4f}")

# Check if spike positions have high surprise
spike_surprise = anomaly_scores[spike_positions].mean()
non_spike_surprise = anomaly_scores[
    np.setdiff1d(np.arange(seq_len), spike_positions)
].mean()
print(f"\nAt spike positions:     {spike_surprise:.4f}")
print(f"At non-spike positions: {non_spike_surprise:.4f}")
print(f"Detection ratio:        {spike_surprise / non_spike_surprise:.1f}x")
