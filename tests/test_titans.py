"""Tests for titans-trainer."""

import torch
import pytest


def test_neural_memory():
    from titans_trainer import NeuralMemory

    mem = NeuralMemory(d_model=64, memory_depth=2, chunk_size=32)
    x = torch.randn(2, 128, 64)
    out, surprise = mem(x, return_surprise=True)

    assert out.shape == x.shape
    assert surprise.shape == (2, 128, 1)
    assert surprise.min() >= 0  # MSE is non-negative


def test_neural_memory_gradient_flow():
    """Verify gradients flow through the memory update."""
    from titans_trainer import NeuralMemory

    mem = NeuralMemory(d_model=32, memory_depth=2, chunk_size=16)
    x = torch.randn(1, 64, 32, requires_grad=True)
    out, _ = mem(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.abs().sum() > 0
    # Check meta-parameters received gradients
    for p in mem.lr_gate.parameters():
        assert p.grad is not None


def test_neural_memory_inference():
    """Memory should work under torch.no_grad (inference mode)."""
    from titans_trainer import NeuralMemory

    mem = NeuralMemory(d_model=32, memory_depth=2, chunk_size=16)
    mem.eval()
    x = torch.randn(1, 64, 32)

    with torch.no_grad():
        out, surprise = mem(x, return_surprise=True)

    assert out.shape == x.shape
    assert surprise is not None


def test_persistent_memory():
    from titans_trainer import PersistentMemory

    pm = PersistentMemory(d_model=64, n_persistent=16)
    out = pm(batch_size=4)

    assert out.shape == (4, 16, 64)


def test_titans_block():
    from titans_trainer import TitansBlock

    block = TitansBlock(d_model=64, n_heads=4, n_persistent=8, chunk_size=32)
    x = torch.randn(2, 128, 64)
    out = block(x)

    assert out.shape == x.shape


def test_titans_block_gradient():
    from titans_trainer import TitansBlock

    block = TitansBlock(d_model=32, n_heads=2, n_persistent=4, chunk_size=16)
    x = torch.randn(1, 64, 32, requires_grad=True)
    out = block(x)
    out.sum().backward()

    assert x.grad is not None


def test_titans_model_discrete():
    from titans_trainer import TitansModel

    model = TitansModel(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_heads=4,
        memory_depth=2,
        n_persistent=8,
        chunk_size=32,
        max_seq_len=128,
    )
    ids = torch.randint(1, 1000, (2, 128))
    labels = ids.clone()
    labels[torch.rand(2, 128) > 0.15] = -100

    out = model(ids, labels=labels)

    assert out['logits'].shape == (2, 128, 1000)
    assert out['loss'] is not None
    assert out['loss'].item() > 0


def test_titans_model_continuous():
    from titans_trainer import TitansModel

    model = TitansModel(
        vocab_size=None,
        d_model=32,
        n_layers=2,
        n_heads=4,
        chunk_size=16,
    )
    x = torch.randn(2, 64, 32)
    out = model(x)

    assert out['logits'].shape == (2, 64, 32)
    assert out['loss'] is None


def test_get_embeddings():
    from titans_trainer import TitansModel

    model = TitansModel(vocab_size=500, d_model=32, n_layers=2, n_heads=2,
                        chunk_size=16, max_seq_len=64)
    ids = torch.randint(1, 500, (3, 64))
    emb = model.get_embeddings(ids)

    assert emb.shape == (3, 32)


def test_get_surprise_scores():
    from titans_trainer import TitansModel

    model = TitansModel(vocab_size=500, d_model=32, n_layers=2, n_heads=2,
                        chunk_size=16, max_seq_len=64)
    ids = torch.randint(1, 500, (2, 64))
    surprise = model.get_surprise_scores(ids)

    assert surprise.shape == (2, 64, 2)  # n_layers=2
    assert surprise.min() >= 0


def test_training_step():
    """Full training step: forward + backward + optimizer step."""
    from titans_trainer import TitansModel

    model = TitansModel(vocab_size=200, d_model=32, n_layers=2, n_heads=2,
                        chunk_size=16, max_seq_len=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ids = torch.randint(1, 200, (2, 64))
    labels = ids.clone()
    labels[torch.rand(2, 64) > 0.15] = -100

    # Step 1
    out1 = model(ids, labels=labels)
    loss1 = out1['loss'].item()
    out1['loss'].backward()
    optimizer.step()
    optimizer.zero_grad()

    # Step 2 — loss should decrease (or at least not crash)
    out2 = model(ids, labels=labels)
    loss2 = out2['loss'].item()

    assert loss2 < loss1 * 1.5  # shouldn't explode


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
