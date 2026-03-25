"""
TITANS Trainer
===============
HuggingFace / Unsloth-style trainer for TITANS models.
Handles training loop, mixed precision, gradient accumulation,
W&B logging, checkpointing, and LR scheduling.

Usage:
    from titans_trainer import TitansConfig, TitansModel, TitansTrainer

    config = TitansConfig(vocab_size=32000, d_model=256, n_layers=4)
    model = TitansModel.from_config(config)
    trainer = TitansTrainer(model, train_dataset, val_dataset, config)
    trainer.train()
"""

import torch
import torch.nn as nn
import time
import math
import signal
from pathlib import Path
from typing import Optional, Dict, Callable
from torch.utils.data import DataLoader, Dataset

# Ensure Ctrl+C works
try:
    signal.signal(signal.SIGINT, signal.default_int_handler)
except (OSError, ValueError):
    pass

torch.backends.cudnn.benchmark = True


class TitansTrainer:
    """
    High-level trainer for TITANS models.

    Supports:
    - Single GPU and DataParallel (multi-GPU)
    - Mixed precision (AMP)
    - Gradient accumulation
    - Cosine LR with warmup
    - W&B logging
    - Mid-epoch validation and checkpointing
    - Fractional epochs (e.g., 1.5)
    - Resume from checkpoint

    Args:
        model: TitansModel or any nn.Module.
        train_dataset: PyTorch Dataset returning {'input_ids', 'labels'} dicts.
        val_dataset: Optional validation dataset.
        config: TitansConfig with training hyperparameters.
        collate_fn: Optional custom collate function.
        callbacks: Optional dict of callback functions:
            'on_step': fn(trainer, step, loss) — called every optimizer step
            'on_epoch_end': fn(trainer, epoch, metrics) — called after each epoch
            'on_val': fn(trainer, metrics) — called after validation
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config=None,
        collate_fn: Optional[Callable] = None,
        callbacks: Optional[Dict[str, Callable]] = None,
    ):
        # Import config here to avoid circular imports
        if config is None:
            from .config import TitansConfig
            config = TitansConfig()

        self.config = config
        self.callbacks = callbacks or {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Multi-GPU
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"Using DataParallel on {n_gpus} GPUs")
            model = nn.DataParallel(model)
        elif n_gpus == 1:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")
            self.device = 'cpu'

        self.model = model.to(self.device)

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # LR scheduler: cosine with warmup
        total_batches = int(len(self.train_loader) * config.epochs)
        warmup = config.warmup_steps

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_batches - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # AMP
        self.use_amp = config.use_amp and self.device != 'cpu'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # State
        self.global_step = 0
        self.batches_done = 0
        self.total_batches = total_batches
        self.best_val_loss = float('inf')
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # W&B
        self.wandb = None
        if config.use_wandb:
            self._init_wandb()

        # Report
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nTitansTrainer ready:")
        print(f"  Parameters: {n_params / 1e6:.1f}M")
        print(f"  Train samples: {len(train_dataset):,}")
        print(f"  Val samples: {len(val_dataset):,}" if val_dataset else "  Val: None")
        print(f"  Epochs: {config.epochs}, Batch: {config.batch_size}")
        print(f"  LR: {config.lr}, AMP: {self.use_amp}")
        print(f"  Output: {self.output_dir}")
        print()

    def _init_wandb(self):
        try:
            import wandb
            self.wandb = wandb
            n_params = sum(p.numel() for p in self.model.parameters())
            wandb.init(
                entity=self.config.wandb_entity,
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    "parameters_M": round(n_params / 1e6, 1),
                    "d_model": self.config.d_model,
                    "n_layers": self.config.n_layers,
                    "n_heads": self.config.n_heads,
                    "architecture": self.config.architecture,
                    "learning_rate": self.config.lr,
                    "batch_size": self.config.batch_size,
                    "epochs": self.config.epochs,
                },
            )
            print(f"W&B: {wandb.run.url}")
        except Exception as e:
            print(f"W&B init failed: {e}")
            self.wandb = None

    def _log(self, metrics: dict):
        if self.wandb is not None:
            self.wandb.log(metrics, step=self.global_step)

    def train(self) -> float:
        """
        Run full training loop. Returns best validation loss.

        The model trains for config.epochs epochs (supports fractional).
        Validates at end of each epoch and optionally mid-epoch.
        Saves checkpoints and tracks best model.
        """
        epochs = math.ceil(self.config.epochs)
        print(f"Starting training for {self.config.epochs} epochs "
              f"({self.total_batches} total batches)\n")

        start_time = time.time()

        try:
            for epoch in range(epochs):
                epoch_start = time.time()
                train_loss = self._train_epoch(epoch)

                # Validate
                val_metrics = {}
                if self.val_loader is not None:
                    val_metrics = self.validate()
                    self._log({
                        "val/loss": val_metrics['val_loss'],
                        "val/accuracy": val_metrics.get('val_accuracy', 0),
                    })

                epoch_time = time.time() - epoch_start
                print(
                    f"\nEpoch {epoch + 1}/{epochs} │ "
                    f"train={train_loss:.4f} │ "
                    f"val={val_metrics.get('val_loss', 0):.4f} │ "
                    f"acc={val_metrics.get('val_accuracy', 0):.3f} │ "
                    f"time={epoch_time:.0f}s\n"
                )

                # Save
                self.save_checkpoint(epoch, {
                    'train_loss': train_loss, **val_metrics
                })

                # Callback
                if 'on_epoch_end' in self.callbacks:
                    self.callbacks['on_epoch_end'](self, epoch, val_metrics)

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            self.save_checkpoint(epoch, {'interrupted': True})

        total_time = time.time() - start_time
        print(f"Training complete! {total_time / 60:.1f} min, "
              f"best val loss: {self.best_val_loss:.4f}")

        if self.wandb is not None:
            self.wandb.finish()

        return self.best_val_loss

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        grad_accum = self.config.grad_accum_steps

        for batch_idx, batch in enumerate(self.train_loader):
            if self.batches_done >= self.total_batches:
                break

            # Move to device
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)

            # Forward
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                out = self.model(input_ids, labels=labels)
                loss = out['loss']
                if isinstance(loss, tuple):
                    loss = loss[0]
                if loss.dim() > 0:
                    loss = loss.mean()
                loss = loss / grad_accum

            # Backward
            self.scaler.scale(loss).backward()

            # Optimizer step
            if (batch_idx + 1) % grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1

                # Log
                if self.global_step % 10 == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    self._log({
                        "train/loss": loss.item() * grad_accum,
                        "train/learning_rate": lr,
                        "train/grad_norm": grad_norm.item()
                            if isinstance(grad_norm, torch.Tensor)
                            else grad_norm,
                        "train/epoch": epoch + batch_idx / len(self.train_loader),
                    })

                # Callback
                if 'on_step' in self.callbacks:
                    self.callbacks['on_step'](
                        self, self.global_step, loss.item() * grad_accum
                    )

            batch_loss = loss.item() * grad_accum
            total_loss += batch_loss
            n_batches += 1
            self.batches_done += 1

            # Mid-epoch validation
            val_every = self.config.val_every_steps
            if (val_every > 0 and self.val_loader is not None
                    and self.global_step > 0
                    and self.global_step % val_every == 0):
                mid_val = self.validate(max_batches=50)
                self.model.train()
                self._log({
                    "val/loss": mid_val['val_loss'],
                    "val/accuracy": mid_val.get('val_accuracy', 0),
                })
                print(f"  val │ loss={mid_val['val_loss']:.4f} │ "
                      f"acc={mid_val.get('val_accuracy', 0):.4f}")
                if 'on_val' in self.callbacks:
                    self.callbacks['on_val'](self, mid_val)

            # Mid-epoch checkpoint
            save_every = self.config.save_every_steps
            if (save_every > 0 and self.global_step > 0
                    and self.global_step % save_every == 0):
                self.save_checkpoint(epoch, {'train_loss': total_loss / n_batches})

            # Console progress
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg = total_loss / n_batches
                lr = self.scheduler.get_last_lr()[0]
                pct = batch_idx / len(self.train_loader) * 100
                print(
                    f"  Epoch {epoch+1} │ "
                    f"{batch_idx+1}/{len(self.train_loader)} ({pct:.0f}%) │ "
                    f"loss={batch_loss:.4f} │ avg={avg:.4f} │ lr={lr:.2e}",
                    end='\r',
                )

        print()  # newline after progress
        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def validate(self, max_batches: int = 0) -> Dict[str, float]:
        """Run validation. Returns dict with val_loss and val_accuracy."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_masked = 0
        n = 0

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                out = self.model(input_ids, labels=labels)
                loss = out['loss']
                if isinstance(loss, tuple):
                    loss = loss[0]
                if loss.dim() > 0:
                    loss = loss.mean()

            total_loss += loss.item()
            preds = out['logits'].argmax(dim=-1)
            mask = labels != -100
            total_correct += (preds[mask] == labels[mask]).sum().item()
            total_masked += mask.sum().item()
            n += 1

            if max_batches > 0 and n >= max_batches:
                break

        return {
            'val_loss': total_loss / max(1, n),
            'val_accuracy': total_correct / max(1, total_masked),
        }

    def save_checkpoint(self, epoch: int, metrics: dict):
        """Save model checkpoint."""
        model_to_save = (
            self.model.module if hasattr(self.model, 'module') else self.model
        )
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
        }

        path = self.output_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(checkpoint, path)

        val_loss = metrics.get('val_loss', float('inf'))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, self.output_dir / "best_model.pt")
            print(f"  New best model! val_loss={val_loss:.4f}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ) -> 'TitansTrainer':
        """Resume training from a checkpoint."""
        from .config import TitansConfig

        ckpt = torch.load(checkpoint_path, weights_only=False)
        config = TitansConfig.from_dict(ckpt.get('config', {}))

        model.load_state_dict(ckpt['model_state_dict'])
        trainer = cls(model, train_dataset, val_dataset, config)
        trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        trainer.global_step = ckpt.get('global_step', 0)
        trainer.best_val_loss = ckpt.get('metrics', {}).get(
            'val_loss', float('inf')
        )

        print(f"Resumed from step {trainer.global_step}")
        return trainer
