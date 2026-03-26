"""
Autoregressive Language Modeling with TITANS
=============================================
GPT-style next-token prediction with TITANS memory.

The model predicts the next token given all previous tokens.
TITANS memory enables the model to "remember" patterns from
earlier in the sequence and adapt its predictions — something
vanilla transformers can only do within their context window.

    python examples/autoregressive_training.py
"""

import torch
from torch.utils.data import Dataset
from titans_trainer import TitansConfig, TitansModel, TitansTrainer


class CausalLMDataset(Dataset):
    """
    Autoregressive (GPT-style) dataset.

    Each sample:
    - input_ids: tokens[0:N-1]
    - labels: tokens[1:N] (shifted by one position)

    The model learns: given tokens so far, predict the next one.
    """

    def __init__(self, token_data: torch.Tensor, pad_id: int = 0):
        """
        Args:
            token_data: (n_samples, seq_len) pre-tokenized sequences
            pad_id: padding token ID (labels set to -100 at padding)
        """
        self.data = token_data
        self.pad_id = pad_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]

        # Input: all tokens except last
        input_ids = tokens[:-1].clone()

        # Labels: all tokens except first (shifted by 1)
        labels = tokens[1:].clone()

        # Don't compute loss on padding
        labels[labels == self.pad_id] = -100

        return {'input_ids': input_ids, 'labels': labels}


class TextGenerator:
    """Simple autoregressive text generator for TITANS models."""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            prompt: (1, prompt_len) starting token IDs
            max_new_tokens: how many tokens to generate
            temperature: sampling temperature (lower = more deterministic)
            top_k: only sample from top-k most likely tokens

        Returns:
            (1, prompt_len + max_new_tokens) generated sequence
        """
        self.model.eval()
        tokens = prompt.to(self.device)

        for _ in range(max_new_tokens):
            # Forward pass — TITANS memory adapts with each token
            out = self.model(tokens)
            logits = out['logits'][:, -1, :]  # last position

            # Temperature scaling
            logits = logits / max(temperature, 1e-6)

            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k)
                logits[logits < topk_vals[:, -1:]] = float('-inf')

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens


def main():
    # --- Config ---
    vocab_size = 10000
    seq_len = 256
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
        causal=True,
        dropout=0.1,

        lr=3e-4,
        epochs=3,
        batch_size=16,
        warmup_steps=100,
        use_amp=True,

        log_interval=20,
        val_every_steps=200,
        use_wandb=False,
        output_dir="./outputs/causal_lm",
    )

    # --- Synthetic data ---
    # In practice: tokenize your text corpus, split into fixed-length chunks
    print("Generating synthetic sequences...")

    # Simulate structured sequences (not pure random — has learnable patterns)
    train_data = []
    for _ in range(5000):
        # Each sequence has a "pattern signal": a base value that repeats
        base = torch.randint(1, 100, (1,))
        noise = torch.randint(0, vocab_size - 1, (seq_len + 1,))
        # Every 10th position, insert the base pattern
        noise[::10] = base + torch.arange(0, seq_len + 1, 10) % 100
        train_data.append(noise)

    train_tokens = torch.stack(train_data)
    val_tokens = train_tokens[:500]  # Quick val split

    train_ds = CausalLMDataset(train_tokens)
    val_ds = CausalLMDataset(val_tokens)

    # Note: seq_len in dataset is seq_len+1 (we split into input/label of seq_len each)

    # --- Model ---
    model = TitansModel.from_config(config)

    # --- Train ---
    trainer = TitansTrainer(model, train_ds, val_ds, config)
    trainer.train()

    # --- Generate ---
    print("\n--- Autoregressive Generation ---")
    device = next(model.parameters()).device
    generator = TextGenerator(model, device)

    prompt = torch.randint(1, 100, (1, 10))  # short prompt
    generated = generator.generate(prompt, max_new_tokens=50, temperature=0.8)
    print(f"Prompt:    {prompt[0].tolist()}")
    print(f"Generated: {generated[0, 10:].tolist()}")

    # --- Compare surprise on different inputs ---
    print("\n--- Surprise Analysis ---")
    model.eval()
    with torch.no_grad():
        # Pattern-consistent input
        consistent = torch.randint(1, 100, (1, seq_len)).to(device)
        # Random input (should be more surprising)
        random_input = torch.randint(1, vocab_size, (1, seq_len)).to(device)

        surprise_consistent = model.get_surprise_scores(consistent).mean().item()
        surprise_random = model.get_surprise_scores(random_input).mean().item()

        print(f"Consistent pattern surprise: {surprise_consistent:.4f}")
        print(f"Random input surprise:       {surprise_random:.4f}")
        print(f"Ratio: {surprise_random / max(surprise_consistent, 1e-6):.1f}x")


if __name__ == '__main__':
    main()
