
# titans-trainer

**Train TITANS models in 5 lines.** HuggingFace-style trainer for the [TITANS architecture](https://arxiv.org/abs/2501.00663) (Behrouz et al., Google Research, NeurIPS 2025).

```python
from titans_trainer import TitansConfig, TitansModel, TitansTrainer

config = TitansConfig.base(vocab_size=32000)
model = TitansModel.from_config(config)
trainer = TitansTrainer(model, train_dataset, val_dataset, config)
trainer.train()
```

TITANS introduces neural long-term memory that updates its own weights via gradient descent *during the forward pass*. This enables test-time learning — the model adapts to new patterns at inference time without fine-tuning.

> **Note:** Independent community implementation based on the [original paper](https://arxiv.org/abs/2501.00663). Not affiliated with or endorsed by Google Research.

## Why This Exists

[lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch) provides an excellent implementation of the TITANS architecture with all three variants (MAC, MAG, MAL). **This package provides everything else you need to actually train one:** config management, checkpointing, mixed precision, multi-GPU, W&B logging, LR scheduling, and model save/load.

| | lucidrains/titans-pytorch | **titans-trainer** |
|---|---|---|
| Architecture modules | ✅ All variants (MAC, MAG, MAL) | ✅ MAC variant |
| Training framework | ❌ | ✅ HuggingFace-style Trainer |
| Config presets | ❌ | ✅ `.small()`, `.base()`, `.large()` |
| `from_pretrained` / `save_pretrained` | ❌ | ✅ |
| AMP + Multi-GPU | ❌ DIY | ✅ Automatic |
| W&B logging | ❌ | ✅ Built-in |
| Checkpointing + resume | ❌ | ✅ Best model tracking |
| Callbacks | ❌ | ✅ `on_step`, `on_epoch`, `on_val` |

**Use lucidrains for research exploration. Use titans-trainer to ship.**

## Installation

```bash
pip install titans-trainer
```

Or from source:

```bash
git clone https://github.com/pafos-ai/titans-trainer.git
cd titans-trainer
pip install -e .
```

Requires PyTorch ≥ 2.1.

## Quick Start

### 5-Line Training

```python
from titans_trainer import TitansConfig, TitansModel, TitansTrainer

config = TitansConfig(vocab_size=32000, d_model=512, n_layers=6, n_heads=8)
model = TitansModel.from_config(config)
trainer = TitansTrainer(model, train_dataset, val_dataset, config)
trainer.train()
```

### Preset Configs

```python
config = TitansConfig.small(vocab_size=32000)   # ~10M params — quick experiments
config = TitansConfig.base(vocab_size=32000)    # ~50M params — standard training
config = TitansConfig.large(vocab_size=32000)   # ~200M params — serious runs
```

### Full Config Control

```python
config = TitansConfig(
    # Architecture
    vocab_size=32000,
    d_model=512,
    n_layers=6,
    n_heads=8,
    memory_depth=2,       # TITANS memory MLP depth
    n_persistent=64,      # Persistent memory tokens
    chunk_size=128,       # Tokens per memory gradient step

    # Training
    lr=3e-4,
    epochs=3,
    batch_size=32,
    warmup_steps=300,
    use_amp=True,

    # Logging
    use_wandb=True,
    wandb_project="my-titans",
    val_every_steps=500,

    # Output
    output_dir="./outputs",
)
```

### Save & Load

```python
# Save
model.save_pretrained("my_model.pt")

# Load
model = TitansModel.from_pretrained("my_model.pt")

# Resume training from checkpoint
trainer = TitansTrainer.from_checkpoint(
    "outputs/checkpoint_epoch1.pt",
    model, train_dataset, val_dataset,
)
trainer.train()
```

### Callbacks

```python
def my_logger(trainer, step, loss):
    if step % 100 == 0:
        print(f"Step {step}: loss={loss:.4f}")

trainer = TitansTrainer(
    model, train_data, val_data, config,
    callbacks={
        'on_step': my_logger,
        'on_epoch_end': lambda t, e, m: print(f"Epoch {e} done!"),
    },
)
```

### W&B Integration

```python
config = TitansConfig(
    # ...
    use_wandb=True,
    wandb_entity="my-team",
    wandb_project="titans-experiments",
    wandb_run_name="run-001",
)
# That's it. Trainer logs loss, LR, grad norm, val metrics automatically.
```

## Shakespeare Demo

Train a causal TITANS language model on Shakespeare in under 25 minutes:

```python
pip install titans-trainer tiktoken
```

```python
import torch
from torch.utils.data import Dataset
import urllib.request, tiktoken
from titans_trainer import TitansConfig, TitansModel, TitansTrainer

# Download & tokenize
text = urllib.request.urlopen(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
).read().decode("utf-8")
enc = tiktoken.get_encoding("gpt2")
tokens = torch.tensor(enc.encode(text), dtype=torch.long)

# Dataset: autoregressive (GPT-style) — predict next token
class ShakespeareDataset(Dataset):
    def __init__(self, tokens, seq_len=256):
        self.tokens, self.seq_len = tokens, seq_len
    def __len__(self):
        return len(self.tokens) // self.seq_len - 1
    def __getitem__(self, idx):
        chunk = self.tokens[idx * self.seq_len : idx * self.seq_len + self.seq_len + 1]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}

split = int(len(tokens) * 0.9)
train_ds = ShakespeareDataset(tokens[:split])
val_ds   = ShakespeareDataset(tokens[split:])

# Config — 23.5M params, causal attention
config = TitansConfig(
    vocab_size=enc.n_vocab, d_model=256, n_layers=8, n_heads=8, d_ff=1024,
    memory_depth=2, n_persistent=32, chunk_size=64, max_seq_len=256,
    causal=True,  # autoregressive masking
    lr=3e-4, epochs=25, batch_size=8, warmup_steps=100,
    use_amp=True, use_wandb=False, output_dir="./outputs/shakespeare",
)

# Train (~20 min on single GPU)
model = TitansModel.from_config(config)
trainer = TitansTrainer(model, train_ds, val_ds, config)
trainer.train()  # best val loss: ~2.71
```

After 25 epochs the model generates Shakespearean vocabulary and dialogue structure. See [`examples/shakespeare.py`](examples/shakespeare.py) for the full script with text generation.

## Training Paradigms

### Autoregressive (GPT-style)

```python
# Dataset returns:
#   input_ids: tokens[:-1]
#   labels: tokens[1:]  (shifted by one)
# Set causal=True in config for proper autoregressive masking.
# See examples/shakespeare.py and examples/autoregressive_training.py.
```

### Masked Language Modeling (BERT-style)

```python
# Dataset returns:
#   input_ids: tokens with 15% replaced by [MASK]
#   labels: original tokens at masked positions, -100 elsewhere
# Leave causal=False (default) for bidirectional attention.
# See examples/mlm_training.py for full implementation.
```

### Continuous Features (Time Series, Sensors, etc.)

```python
# No vocabulary — pass pre-embedded features directly
config = TitansConfig(vocab_size=None, d_model=64, n_layers=4, n_heads=4)
model = TitansModel.from_config(config)

x = torch.randn(batch, seq_len, 64)
embeddings = model.get_embeddings(x)  # (batch, 64)
```

## Anomaly Detection via Surprise

The TITANS memory learns "normal" patterns. Inputs that deviate produce high surprise scores — useful for anomaly detection, novelty scoring, and out-of-distribution detection.

```python
model.eval()
surprise = model.get_surprise_scores(data)  # (batch, seq_len, n_layers)

# High surprise = input deviates from learned patterns
anomalous_tokens = surprise.mean(dim=-1) > threshold
```

## Architecture

TITANS uses the **MAC (Memory-as-Context)** variant:

```
Input sequence
    ↓
[Neural Memory Context ; Input ; Persistent Memory]
    ↓
Multi-Head Attention (short-term memory)
    ↓
Memory Gate (blend attention + direct memory)
    ↓
SwiGLU FFN
    ↓
Output
```

Three memory systems:

| Component | Updates When | Function |
|-----------|-------------|----------|
| **Neural Memory** | Every forward pass (inner gradient descent) | Long-term: learns from surprise |
| **Attention** | Per-sequence (context window) | Short-term: precise local context |
| **Persistent Memory** | Training only (frozen at inference) | Priors: task-independent knowledge |

### The Key Innovation

The neural memory MLP's weights update via `torch.autograd.grad` with `create_graph=True` during the forward pass. This is real test-time learning — not a gated residual or adapter. Sequences are processed in chunks (default: 128 tokens) for efficiency.

```python
# What happens inside NeuralMemory (simplified)
for chunk in sequence.chunks(128):
    pred = memory_mlp(chunk)                      # predict
    loss = MSE(pred, chunk)                        # surprise
    grads = autograd.grad(loss, memory_weights)    # inner gradient
    memory_weights = forget * weights - lr * grads # update
    output = memory_mlp(chunk)                     # re-read updated memory
```

## Dataset Format

Same as HuggingFace — your `__getitem__` returns:

```python
{'input_ids': LongTensor, 'labels': LongTensor}
```

Where `labels = -100` at positions to ignore. Works for MLM, causal LM, and classification. See [`examples/datasets.py`](examples/datasets.py) for ready-to-use dataset classes.

## Low-Level API

If you just need the building blocks:

```python
from titans_trainer import NeuralMemory, PersistentMemory, TitansBlock

# Neural memory alone
memory = NeuralMemory(d_model=256, memory_depth=2, chunk_size=128)
x = torch.randn(4, 2048, 256)
memory_out, surprise = memory(x, return_surprise=True)

# Single TITANS block
block = TitansBlock(d_model=256, n_heads=4, n_persistent=64)
out = block(x)  # (4, 2048, 256)
```

## Examples

| Example | Description |
|---------|-------------|
| [`examples/shakespeare.py`](examples/shakespeare.py) | **Start here** — train on Shakespeare, generate text |
| [`examples/quickstart.py`](examples/quickstart.py) | Full HF-style workflow: config → model → train → save |
| [`examples/mlm_training.py`](examples/mlm_training.py) | Masked Language Modeling (BERT-style) |
| [`examples/autoregressive_training.py`](examples/autoregressive_training.py) | Next-token prediction (GPT-style) with text generation |
| [`examples/time_series_anomaly.py`](examples/time_series_anomaly.py) | Continuous features + anomaly detection |
| [`examples/datasets.py`](examples/datasets.py) | Ready-to-use dataset classes for all paradigms |

## Citation

```bibtex
@article{behrouz2025titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal={NeurIPS},
  year={2025}
}

@software{yermekov2026titans_trainer,
  title={titans-trainer: HuggingFace-style Training for TITANS},
  author={Yermekov, Akbar},
  url={https://github.com/pafos-ai/titans-trainer},
  year={2026}
}
```

## License

MIT
