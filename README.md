
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

## Shakespeare Benchmark

Apples-to-apples comparison against Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) on TinyShakespeare — same vocab (character-level, 65 tokens), same param budget (~10.65M), same training compute (80 epochs), same batch size (64), same LR (1e-3).

| | nanoGPT | TITANS |
|---|---|---|
| **Val loss** | ~1.47 | **1.55** |
| Architecture | Vanilla transformer | MAC (memory + attention + persistent) |
| Params | ~10.65M | 10.67M |
| Training | ~80 epochs | 80 epochs |

On this toy benchmark TITANS trades a small gap in val loss for architectural capabilities that vanilla transformers lack. The advantage grows with sequence length — on 256-char snippets there is little long-range structure to exploit, but on longer sequences (documents, codebases, biological data) the memory enables test-time adaptation. See [Real-World Results](#real-world-results) for evidence at scale.

**Sample generation** (10.67M params, 80 epochs, raw model output, unfiltered):

```
ROMEO:
Gerby, coward!
PAULINA:
Believe thee, friend,
Whose dost there, some tapster, which thou think'st,
A virtue of my love against my mercy,
And we have pulic, that he did not say with no
Lonest fire an offer consent, to send them,
I have their afflicts of thy want;
```

See [`examples/shakespeare_benchmark.py`](examples/shakespeare_benchmark.py) for the full benchmark script, and [`examples/shakespeare.py`](examples/shakespeare.py) for a quick BPE demo with text generation.

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
| [`examples/shakespeare_benchmark.py`](examples/shakespeare_benchmark.py) | **Start here** — TITANS vs nanoGPT benchmark |
| [`examples/shakespeare.py`](examples/shakespeare.py) | Quick BPE Shakespeare demo with text generation |
| [`examples/quickstart.py`](examples/quickstart.py) | Full HF-style workflow: config → model → train → save |
| [`examples/mlm_training.py`](examples/mlm_training.py) | Masked Language Modeling (BERT-style) |
| [`examples/autoregressive_training.py`](examples/autoregressive_training.py) | Next-token prediction (GPT-style) with text generation |
| [`examples/time_series_anomaly.py`](examples/time_series_anomaly.py) | Continuous features + anomaly detection |
| [`examples/datasets.py`](examples/datasets.py) | Ready-to-use dataset classes for all paradigms |

## Real-World Results

titans-trainer powers [BioTitan](https://huggingface.co/pafos-ai/biotitan),
the first genomic foundation model built on the TITANS architecture.

Trained on just 255K cells (vs Geneformer's 30M cells and 6 epochs), BioTitan
achieves **92% of Geneformer's performance** across 53 gene property tasks on the
[IBM gene-benchmark](https://arxiv.org/abs/2412.04075) — with 120× less training data.

What makes this possible is a capability unique to TITANS: **test-time memory adaptation.**
BioTitan's static embeddings score 0.636 avg AUC. After processing 60K cells at
inference — no retraining, no optimizer, no labels — they rise to 0.716. That +12.6%
improvement is architecturally impossible in Geneformer, scGPT, or any other existing
single-cell foundation model.

On expression tasks (23 tasks), BioTitan places 5th among 13 models evaluated —
above all text, protein, and DNA models — despite training on the smallest dataset
of any model in the benchmark.

Pre-computed gene embeddings and model weights are available on
[HuggingFace](https://huggingface.co/pafos-ai/biotitan).

## Citation

```bibtex
@software{yermekov2026titans_trainer,
  title={titans-trainer: HuggingFace-style Training for TITANS},
  author={Yermekov, Akbar},
  url={https://github.com/pafos-ai/titans-trainer},
  year={2026}
}

@article{behrouz2025titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal={NeurIPS},
  year={2025}
}
```

## License

MIT
