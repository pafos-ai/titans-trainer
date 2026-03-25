"""
TITANS Neural Long-Term Memory
================================
Pure PyTorch implementation of "Titans: Learning to Memorize at Test Time"
(Behrouz et al., Google Research, 2025).

The memory is a small MLP that updates its own weights via gradient descent
during the forward pass. "Surprise" (prediction error) determines what to
memorize. Learned forget and learning-rate gates control the update dynamics.

Implementation uses functional weight management (no in-place mutation) so
the entire update is differentiable. Sequences are processed in chunks for
efficiency — one gradient step per chunk, not per token.

Usage:
    from titans_trainer import NeuralMemory, PersistentMemory

    memory = NeuralMemory(d_model=256, memory_depth=2, chunk_size=128)
    x = torch.randn(4, 2048, 256)  # (batch, seq_len, d_model)
    memory_out, surprise = memory(x, return_surprise=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def _mlp_forward(
    weights: List[torch.Tensor],
    biases: List[torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Manual MLP forward using explicit weight/bias tensors.
    Layers: linear → SiLU → linear → SiLU → ... → linear (no activation on last).
    """
    n_layers = len(weights)
    for i in range(n_layers):
        x = F.linear(x, weights[i], biases[i])
        if i < n_layers - 1:
            x = F.silu(x)
    return x


class NeuralMemory(nn.Module):
    """
    Neural Long-Term Memory with real test-time weight updates.

    The memory MLP's weights are updated during the forward pass via
    gradient descent on a surprise signal (prediction error). This enables
    test-time learning: the model adapts to new patterns at inference time
    without any external optimizer or fine-tuning step.

    During training: meta-parameters (lr_gate, forget_gate) are learned via
    backprop through the weight update steps (create_graph=True).
    During inference: same weight update mechanism, outer model is frozen.

    Sequences are chunked for efficiency: one gradient step per chunk.
    chunk_size=128 with seq_len=2048 → 16 update steps per layer.

    Args:
        d_model: Model dimension (input and output size).
        memory_depth: Number of layers in the memory MLP (default: 2).
        memory_dim: Hidden dimension of memory MLP (default: same as d_model).
        lr_memory: Base learning rate for memory weight updates (default: 0.01).
        chunk_size: Tokens per gradient step (default: 128).
    """

    def __init__(
        self,
        d_model: int,
        memory_depth: int = 2,
        memory_dim: int = None,
        lr_memory: float = 0.01,
        chunk_size: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.memory_dim = memory_dim or d_model
        self.lr_memory = lr_memory
        self.chunk_size = chunk_size

        # Memory MLP weights stored as ParameterLists for explicit control.
        # This avoids nn.Sequential so we can do manual forward with
        # arbitrary weight tensors (needed for functional weight updates).
        dims = [d_model] + [self.memory_dim] * (memory_depth - 1) + [d_model]
        self.mem_weights = nn.ParameterList()
        self.mem_biases = nn.ParameterList()
        for i in range(len(dims) - 1):
            w = nn.Parameter(torch.empty(dims[i + 1], dims[i]))
            b = nn.Parameter(torch.zeros(dims[i + 1]))
            nn.init.kaiming_uniform_(w, nonlinearity='linear')
            self.mem_weights.append(w)
            self.mem_biases.append(b)

        # Meta-parameters: learned during training, control test-time updates
        self.lr_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.forget_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def update_memory(self, x: torch.Tensor) -> torch.Tensor:
        """
        Chunk-wise weight updates via gradient descent on surprise.

        For each chunk:
          1. Forward through memory MLP → prediction
          2. Surprise = MSE(prediction, input)
          3. Compute gradients w.r.t. memory weights
          4. Apply forget (weight decay) and learn (gradient step)
          5. Re-forward through updated weights → output

        All ops are functional (no in-place mutation) so autograd works.
        """
        # Under torch.no_grad() (e.g. validation), autograd.grad won't work.
        # Use enable_grad context so the internal weight-update mechanism
        # functions even when called from a no_grad block.
        with torch.enable_grad():
            return self._update_memory_inner(x)

    def _update_memory_inner(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        chunk_size = min(self.chunk_size, L)
        n_layers = len(self.mem_weights)

        # Snapshot current params — clone so we don't mutate originals.
        # requires_grad_(True) makes them differentiable for autograd.grad.
        weights = [w.clone().requires_grad_(True) for w in self.mem_weights]
        biases = [b.clone().requires_grad_(True) for b in self.mem_biases]
        all_params = weights + biases

        outputs = []
        for start in range(0, L, chunk_size):
            chunk = x[:, start:start + chunk_size, :]  # (B, C, D)

            # --- 1. Predict ---
            pred = _mlp_forward(weights, biases, chunk)

            # --- 2. Surprise loss ---
            loss = F.mse_loss(pred, chunk.detach())

            # --- 3. Gradients w.r.t. memory params ---
            grads = torch.autograd.grad(
                loss, all_params, create_graph=self.training,
            )
            w_grads = grads[:n_layers]
            b_grads = grads[n_layers:]

            # --- 4. Learned gates (averaged over batch & tokens in chunk) ---
            effective_lr = self.lr_memory * self.lr_gate(chunk).mean()
            forget = self.forget_gate(chunk.mean(dim=1)).mean()

            # --- 5. Update params: decay + gradient step ---
            weights = [(forget * w - effective_lr * g).requires_grad_(True)
                       for w, g in zip(weights, w_grads)]
            biases = [(forget * b - effective_lr * g).requires_grad_(True)
                      for b, g in zip(biases, b_grads)]
            all_params = weights + biases

            # --- 6. Re-forward through updated memory ---
            updated_out = _mlp_forward(weights, biases, chunk)
            outputs.append(updated_out)

            # Detach between chunks at inference to bound graph depth.
            if not self.training:
                weights = [w.detach().requires_grad_(True) for w in weights]
                biases = [b.detach().requires_grad_(True) for b in biases]
                all_params = weights + biases

        return torch.cat(outputs, dim=1)

    def compute_surprise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token surprise scores (prediction error).

        High surprise = input deviates from what memory has learned.
        Useful for anomaly detection, novelty scoring, etc.

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            surprise: (batch, seq_len, 1) MSE per token
        """
        with torch.no_grad():
            pred = _mlp_forward(
                list(self.mem_weights), list(self.mem_biases), x
            )
        return F.mse_loss(pred, x, reduction='none').mean(dim=-1, keepdim=True)

    def forward(
        self,
        x: torch.Tensor,
        return_surprise: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, d_model) input tokens
            return_surprise: if True, also return per-token surprise scores

        Returns:
            memory_context: (batch, seq_len, d_model) memory output
            surprise: (batch, seq_len, 1) optional surprise scores
        """
        memory_context = self.update_memory(x)

        surprise = None
        if return_surprise:
            surprise = self.compute_surprise(x)

        return memory_context, surprise


class PersistentMemory(nn.Module):
    """
    Persistent Memory: learnable parameters that store task-independent
    knowledge. Fixed after training (not updated at test time).

    These are prepended/appended to the attention input alongside the
    neural memory context, giving the model access to learned "priors."

    Args:
        d_model: Model dimension.
        n_persistent: Number of persistent memory tokens (default: 64).
    """

    def __init__(self, d_model: int, n_persistent: int = 64):
        super().__init__()
        self.persistent_tokens = nn.Parameter(
            torch.randn(1, n_persistent, d_model) * 0.02
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        """Returns persistent memory tokens expanded to batch size."""
        return self.persistent_tokens.expand(batch_size, -1, -1)
