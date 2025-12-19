"""Trainer for diffusion backbone model."""

import os
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm


def _to_float(value, name: str) -> float:
    """Convert config value to float with a clear error on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric, got {value!r} ({type(value).__name__})")


def _to_int(value, name: str) -> int:
    """Convert config value to int with a clear error on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be integer-like, got {value!r} ({type(value).__name__})")


class BackboneTrainer:
    """Trainer for discrete masking diffusion backbone."""

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        output_dir: str = 'results',
        step_dir: str = None
    ):
        """Initialize trainer.

        Args:
            model: Diffusion model (backbone + diffusion process).
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader.
            config: Training configuration.
            device: Device for training.
            output_dir: Output directory for shared artifacts (checkpoints).
            step_dir: Step-specific output directory for metrics/figures.
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.step_dir = Path(step_dir) if step_dir else self.output_dir

        # Create output directories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.step_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Optimization config
        opt_config = config.get('optimization', {})
        self.use_amp = opt_config.get('use_amp', False) and device == 'cuda'
        self.compile_model = opt_config.get('compile_model', False)
        self.grad_accum_steps = opt_config.get('gradient_accumulation_steps', 1)

        # Enable cuDNN benchmark for consistent input sizes
        if opt_config.get('cudnn_benchmark', True):
            torch.backends.cudnn.benchmark = True

        # Suppress SequentialLR deprecation warning (PyTorch internal issue)
        warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()`")

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Compile model for faster execution
        if self.compile_model and device == 'cuda':
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # Training config
        train_config = config['training_backbone']
        self.learning_rate = _to_float(train_config['learning_rate'], 'learning_rate')
        self.weight_decay = _to_float(train_config['weight_decay'], 'weight_decay')
        self.warmup_steps = _to_int(train_config['warmup_steps'], 'warmup_steps')
        self.max_steps = _to_int(train_config['max_steps'], 'max_steps')
        self.gradient_clip_norm = _to_float(train_config['gradient_clip_norm'], 'gradient_clip_norm')
        self.eval_every = _to_int(train_config['eval_every'], 'eval_every')
        self.save_every = _to_int(train_config['save_every'], 'save_every')
        self.num_epochs = _to_int(train_config.get('num_epochs', 50), 'num_epochs')

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Initialize scheduler (warmup + cosine decay)
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_steps - self.warmup_steps,
            eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps]
        )

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # GPU memory monitoring
        self.memory_log_interval = opt_config.get('memory_log_interval', 500)
        self.memory_stats = []

    def _log_gpu_memory(self) -> Dict:
        """Log GPU memory usage.

        Returns:
            Dictionary with memory statistics in GB.
        """
        if self.device != 'cuda' or not torch.cuda.is_available():
            return {}

        stats = {
            'step': self.global_step,
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
        }

        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        stats['total_gb'] = total_memory
        stats['free_gb'] = total_memory - stats['reserved_gb']
        stats['utilization_pct'] = (stats['allocated_gb'] / total_memory) * 100

        return stats

    def train(self) -> Dict:
        """Run training loop.

        Returns:
            Training history.
        """
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Train batches: {len(self.train_dataloader)}")
        print(f"Val batches: {len(self.val_dataloader)}")

        for epoch in range(self.num_epochs):
            # Training epoch
            train_loss = self._train_epoch(epoch)

            # Validation
            val_loss = self._validate()

            # Save checkpoint
            self._save_checkpoint(val_loss, epoch)

            print(f"Epoch {epoch+1}/{self.num_epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            if self.global_step >= self.max_steps:
                print(f"Reached max steps ({self.max_steps}), stopping training.")
                break

        # Save final checkpoint
        self._save_checkpoint(val_loss, epoch, final=True)

        # Save training history
        self._save_history()

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss
        }

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            if self.global_step >= self.max_steps:
                break

            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1

            self.train_losses.append(loss)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            pbar.set_postfix({'loss': f'{loss:.4f}'})

            # Periodic validation
            if self.global_step > 0 and self.global_step % self.eval_every == 0:
                val_loss = self._validate()
                self.val_losses.append(val_loss)
                self._save_checkpoint(val_loss, epoch)
                self.model.train()

            # Periodic save
            if self.global_step > 0 and self.global_step % self.save_every == 0:
                self._save_periodic_checkpoint(epoch)

            # GPU memory monitoring
            if self.global_step > 0 and self.global_step % self.memory_log_interval == 0:
                mem_stats = self._log_gpu_memory()
                if mem_stats:
                    self.memory_stats.append(mem_stats)
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'mem': f'{mem_stats["allocated_gb"]:.1f}/{mem_stats["total_gb"]:.0f}GB'
                    })

            self.global_step += 1

        return total_loss / max(num_batches, 1)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step.

        Args:
            batch: Batch of data.

        Returns:
            Loss value.
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        # Forward pass with AMP
        with autocast('cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            outputs = self.model(input_ids, attention_mask)
            loss = outputs['loss'] / self.grad_accum_steps

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()

        # Only update weights every grad_accum_steps
        if (self.global_step + 1) % self.grad_accum_steps == 0:
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm
            )

            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

        return loss.item() * self.grad_accum_steps

    def _validate(self) -> float:
        """Run validation.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                with autocast('cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                    outputs = self.model(input_ids, attention_mask)
                total_loss += outputs['loss'].item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, val_loss: float, epoch: int, final: bool = False):
        """Save model checkpoint.

        Args:
            val_loss: Validation loss.
            epoch: Current epoch.
            final: Whether this is the final checkpoint.
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save best checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, self.checkpoint_dir / 'backbone_best.pt')
            print(f"New best model saved with val_loss: {val_loss:.4f}")

        # Save final checkpoint
        if final:
            torch.save(checkpoint, self.checkpoint_dir / 'backbone_last.pt')

    def _save_periodic_checkpoint(self, epoch: int):
        """Save periodic checkpoint.

        Args:
            epoch: Current epoch.
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_dir / f'backbone_step_{self.global_step}.pt')

    def _save_history(self):
        """Save training history to CSV."""
        # Round floats to 4 decimal places
        rounded_train_losses = [round(loss, 4) for loss in self.train_losses]
        rounded_learning_rates = [round(lr, 8) for lr in self.learning_rates]  # LR needs more precision
        rounded_val_losses = [round(loss, 4) for loss in self.val_losses]

        # Create loss curve CSV
        history = {
            'step': list(range(len(self.train_losses))),
            'train_loss': rounded_train_losses,
            'learning_rate': rounded_learning_rates
        }

        # Add validation losses at eval intervals
        val_steps = list(range(self.eval_every, len(self.train_losses) + 1, self.eval_every))
        history['val_step'] = val_steps[:len(self.val_losses)]
        history['val_loss'] = rounded_val_losses

        # Save as DataFrame
        train_df = pd.DataFrame({
            'step': history['step'],
            'train_loss': history['train_loss'],
            'learning_rate': history['learning_rate']
        })
        train_df.to_csv(self.metrics_dir / 'backbone_loss_curve.csv', index=False)

        if self.val_losses:
            val_df = pd.DataFrame({
                'step': history['val_step'],
                'val_loss': history['val_loss']
            })
            val_df.to_csv(self.metrics_dir / 'backbone_val_loss.csv', index=False)

        # Save memory stats
        if self.memory_stats:
            mem_df = pd.DataFrame(self.memory_stats)
            mem_df.to_csv(self.metrics_dir / 'gpu_memory_stats.csv', index=False)
            print(f"GPU Memory Stats saved. Peak usage: {mem_df['max_allocated_gb'].max():.2f} GB")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Loaded checkpoint from step {self.global_step}")
