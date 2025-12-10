"""Trainer for diffusion backbone model."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm


class BackboneTrainer:
    """Trainer for discrete masking diffusion backbone."""

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        output_dir: str = 'results'
    ):
        """Initialize trainer.

        Args:
            model: Diffusion model (backbone + diffusion process).
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader.
            config: Training configuration.
            device: Device for training.
            output_dir: Output directory.
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)

        # Create output directories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training config
        train_config = config['training_backbone']
        self.learning_rate = train_config['learning_rate']
        self.weight_decay = train_config['weight_decay']
        self.warmup_steps = train_config['warmup_steps']
        self.max_steps = train_config['max_steps']
        self.gradient_clip_norm = train_config['gradient_clip_norm']
        self.eval_every = train_config['eval_every']
        self.save_every = train_config['save_every']
        self.num_epochs = train_config.get('num_epochs', 50)

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

        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(input_ids, attention_mask)
        loss = outputs['loss']

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.gradient_clip_norm
        )

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

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
        # Create loss curve CSV
        history = {
            'step': list(range(len(self.train_losses))),
            'train_loss': self.train_losses,
            'learning_rate': self.learning_rates
        }

        # Add validation losses at eval intervals
        val_steps = list(range(self.eval_every, len(self.train_losses) + 1, self.eval_every))
        history['val_step'] = val_steps[:len(self.val_losses)]
        history['val_loss'] = self.val_losses

        # Save as DataFrame
        train_df = pd.DataFrame({
            'step': history['step'],
            'train_loss': history['train_loss'],
            'learning_rate': history['learning_rate']
        })
        train_df.to_csv(self.output_dir / 'metrics' / 'backbone_loss_curve.csv', index=False)

        if self.val_losses:
            val_df = pd.DataFrame({
                'step': history['val_step'],
                'val_loss': history['val_loss']
            })
            val_df.to_csv(self.output_dir / 'metrics' / 'backbone_val_loss.csv', index=False)

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
