"""
Example Datasets for TitansTrainer
====================================
The trainer expects datasets that return dicts with:
    {'input_ids': LongTensor, 'labels': LongTensor}

This is identical to HuggingFace's format:
- input_ids: token IDs (with masks applied for MLM)
- labels: target token IDs (-100 for positions to ignore)

Below are examples for common use cases.
"""

import torch
from torch.utils.data import Dataset


# ============================================================
# Example 1: Masked Language Model (like BERT)
# ============================================================
class MLMDataset(Dataset):
    """
    Standard masked language modeling dataset.
    Randomly masks 15% of tokens; model predicts the original.

    Works with any tokenized text corpus.

    Usage:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Tokenize
        tokens = [tokenizer(t["text"], truncation=True, max_length=512,
                            return_tensors="pt")["input_ids"].squeeze()
                  for t in raw if len(t["text"]) > 50]

        dataset = MLMDataset(tokens, mask_token_id=tokenizer.mask_token_id,
                             vocab_size=tokenizer.vocab_size)
        trainer = TitansTrainer(model, dataset, config=config)
    """

    def __init__(
        self,
        token_sequences: list,  # List of LongTensors, each (seq_len,)
        mask_token_id: int = 103,
        vocab_size: int = 30522,
        mask_prob: float = 0.15,
        max_len: int = 512,
        pad_id: int = 0,
    ):
        self.data = token_sequences
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
        self.max_len = max_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx][:self.max_len].clone()

        # Pad if needed
        if len(tokens) < self.max_len:
            pad = torch.full((self.max_len - len(tokens),), self.pad_id, dtype=torch.long)
            tokens = torch.cat([tokens, pad])

        labels = tokens.clone()

        # Select 15% of non-padding positions
        pad_mask = tokens == self.pad_id
        prob = torch.full(tokens.shape, self.mask_prob)
        prob[pad_mask] = 0.0
        masked = torch.bernoulli(prob).bool()

        labels[~masked] = -100  # Only predict masked tokens

        # 80% → [MASK], 10% → random, 10% → keep
        replace = torch.bernoulli(torch.full(tokens.shape, 0.8)).bool() & masked
        tokens[replace] = self.mask_token_id

        random = torch.bernoulli(torch.full(tokens.shape, 0.5)).bool() & masked & ~replace
        tokens[random] = torch.randint(1, self.vocab_size, tokens.shape)[random]

        return {'input_ids': tokens, 'labels': labels}


# ============================================================
# Example 2: Causal Language Model (like GPT)
# ============================================================
class CausalLMDataset(Dataset):
    """
    Autoregressive language modeling: predict next token.
    Labels are input shifted by one position.

    Usage:
        texts = ["Hello world this is a test", "Another example sentence"]
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokens = [tokenizer.encode(t, return_tensors="pt").squeeze() for t in texts]

        dataset = CausalLMDataset(tokens, max_len=256)
    """

    def __init__(self, token_sequences: list, max_len: int = 512, pad_id: int = 0):
        self.data = token_sequences
        self.max_len = max_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx][:self.max_len].clone()

        if len(tokens) < self.max_len:
            pad = torch.full((self.max_len - len(tokens),), self.pad_id, dtype=torch.long)
            tokens = torch.cat([tokens, pad])

        # Input: all tokens except last
        # Labels: all tokens except first (shifted by 1)
        input_ids = tokens[:-1]
        labels = tokens[1:]

        # Ignore padding in loss
        labels[labels == self.pad_id] = -100

        return {'input_ids': input_ids, 'labels': labels}


# ============================================================
# Example 3: Text Classification (sentiment, topic, etc.)
# ============================================================
class ClassificationDataset(Dataset):
    """
    Sequence classification. Labels are class indices.

    Note: TitansTrainer expects 'labels' as token-level targets.
    For classification, set labels to the class ID at position 0
    and -100 everywhere else. Or use a custom collate_fn.

    Usage:
        texts = ["I love this movie", "Terrible film"]
        labels = [1, 0]  # positive, negative
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        tokens = [tokenizer.encode(t, return_tensors="pt").squeeze() for t in texts]
        dataset = ClassificationDataset(tokens, labels, max_len=128)
    """

    def __init__(
        self,
        token_sequences: list,
        labels: list,
        max_len: int = 512,
        pad_id: int = 0,
    ):
        self.data = token_sequences
        self.labels = labels
        self.max_len = max_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx][:self.max_len].clone()

        if len(tokens) < self.max_len:
            pad = torch.full((self.max_len - len(tokens),), self.pad_id, dtype=torch.long)
            tokens = torch.cat([tokens, pad])

        # For classification: put class label at position 0, ignore rest
        labels = torch.full((self.max_len,), -100, dtype=torch.long)
        labels[0] = self.labels[idx]

        return {'input_ids': tokens, 'labels': labels}


# ============================================================
# Example 4: Time Series (continuous features, no vocab)
# ============================================================
class TimeSeriesDataset(Dataset):
    """
    Continuous time series data. No vocabulary — pass pre-embedded features.

    For this, use TitansModel with vocab_size=None and a custom training loop
    or modify the trainer's forward to handle continuous inputs.

    Usage:
        # Stock prices: 1000 time steps, 32 features
        data = torch.randn(500, 1000, 32)  # 500 sequences
        dataset = TimeSeriesDataset(data, prediction_horizon=10)
    """

    def __init__(
        self,
        sequences: torch.Tensor,  # (n_samples, seq_len, d_features)
        prediction_horizon: int = 1,
    ):
        self.data = sequences
        self.horizon = prediction_horizon

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        # Self-supervised: predict shifted version of input
        input_ids = seq[:-self.horizon]
        labels = seq[self.horizon:]
        return {'input_ids': input_ids, 'labels': labels}


# ============================================================
# Example 5: From HuggingFace datasets (direct integration)
# ============================================================
class HFDatasetWrapper(Dataset):
    """
    Wraps a HuggingFace dataset for TitansTrainer.

    Usage:
        from datasets import load_dataset

        hf_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        def tokenize(example):
            return tokenizer(example["text"], truncation=True,
                             max_length=512, padding="max_length")

        hf_ds = hf_ds.map(tokenize, remove_columns=["text"])
        hf_ds.set_format("torch")

        dataset = HFDatasetWrapper(hf_ds, mask_token_id=103, vocab_size=30522)
    """

    def __init__(
        self,
        hf_dataset,
        mask_token_id: int = 103,
        vocab_size: int = 30522,
        mask_prob: float = 0.15,
    ):
        self.dataset = hf_dataset
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        tokens = item['input_ids'].clone()

        labels = tokens.clone()

        # Mask 15% for MLM
        pad_mask = tokens == 0
        prob = torch.full(tokens.shape, self.mask_prob)
        prob[pad_mask] = 0.0
        masked = torch.bernoulli(prob).bool()
        labels[~masked] = -100

        replace = torch.bernoulli(torch.full(tokens.shape, 0.8)).bool() & masked
        tokens[replace] = self.mask_token_id

        random = torch.bernoulli(torch.full(tokens.shape, 0.5)).bool() & masked & ~replace
        tokens[random] = torch.randint(1, self.vocab_size, tokens.shape)[random]

        return {'input_ids': tokens, 'labels': labels}
