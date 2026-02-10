"""Trainer for diffusion backbone model."""

import warnings
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
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


def _is_cuda_device(device) -> bool:
    """Return True if the provided device resolves to CUDA."""
    try:
        return torch.device(device).type == 'cuda'
    except (TypeError, ValueError):
        return str(device).startswith('cuda')


def _supports_torch_compile(device) -> bool:
    """Return True if torch.compile can safely run on the current GPU."""
    if not _is_cuda_device(device) or not torch.cuda.is_available():
        return False
    try:
        dev = torch.device(device)
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        major, _minor = torch.cuda.get_device_capability(index)
    except Exception:
        return False
    return major >= 7


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
        step_dir: str = None,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        local_rank: Optional[int] = None
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
            distributed: Whether to use DistributedDataParallel.
            rank: Global rank.
            world_size: Total number of ranks.
            local_rank: Local rank (GPU index).
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_main_process = (not self.distributed) or self.rank == 0
        self.output_dir = Path(output_dir)
        self.step_dir = Path(step_dir) if step_dir else self.output_dir

        # Create output directories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.step_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Optimization config
        opt_config = config.get('optimization', {})
        self.use_amp = opt_config.get('use_amp', False) and _is_cuda_device(device)
        self.compile_model = opt_config.get('compile_model', False)
        self.compile_mode = opt_config.get('compile_mode', 'default')
        self.grad_accum_steps = opt_config.get('gradient_accumulation_steps', 1)

        # Enable cuDNN benchmark for consistent input sizes
        if opt_config.get('cudnn_benchmark', True):
            torch.backends.cudnn.benchmark = True

        # Suppress SequentialLR deprecation warning (PyTorch internal issue)
        warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()`")

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Compile model for faster execution (guard against older GPUs)
        if self.compile_model and self.distributed:
            warnings.warn("torch.compile disabled for DDP to avoid compilation issues.")
            self.compile_model = False
        if self.compile_model and _is_cuda_device(device):
            if not _supports_torch_compile(device):
                warnings.warn("torch.compile disabled: GPU compute capability < 7.0")
                self.compile_model = False
            else:
                print(f"Compiling model with torch.compile(mode='{self.compile_mode}')...")
                self.model = torch.compile(self.model, mode=self.compile_mode)
        if self.distributed:
            self.model = self._wrap_ddp(self.model)

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

        ckpt_cfg = config.get('checkpointing', {})
        self.save_best_only = ckpt_cfg.get('save_best_only', True)
        self.save_last = ckpt_cfg.get('save_last', False)
        self.save_periodic = ckpt_cfg.get('save_periodic', False)

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
        self.val_steps = []
        self.learning_rates = []

        # GPU memory monitoring
        self.memory_log_interval = opt_config.get('memory_log_interval', 500)
        self.memory_stats = []

    def _wrap_ddp(self, model: nn.Module) -> nn.Module:
        """Wrap model with DistributedDataParallel when enabled."""
        if not self.distributed or not dist.is_available() or not dist.is_initialized():
            return model
        if _is_cuda_device(self.device):
            device_index = torch.device(self.device).index
            return DDP(model, device_ids=[device_index], output_device=device_index)
        return DDP(model)

    def _get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get a clean state_dict for saving (strip DDP/compile wrappers)."""
        model = self.model
        if isinstance(model, DDP):
            model = model.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model.state_dict()

    def _reduce_mean(self, value: float) -> float:
        """Average a scalar across ranks when using DDP."""
        if not self.distributed or not dist.is_available() or not dist.is_initialized():
            return value
        tensor = torch.tensor(value, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return (tensor / self.world_size).item()

    def _maybe_mark_cudagraph_step_begin(self) -> None:
        """Mark the beginning of a cudagraph step if supported."""
        if not self.compile_model or not _is_cuda_device(self.device):
            return

        compiler_mod = getattr(torch, "compiler", None)
        if compiler_mod is None:
            return

        mark_step = getattr(compiler_mod, "cudagraph_mark_step_begin", None)
        if mark_step is None:
            return

        mark_step()

    def _log_gpu_memory(self) -> Dict:
        """Log GPU memory usage.

        Returns:
            Dictionary with memory statistics in GB.
        """
        if not _is_cuda_device(self.device) or not torch.cuda.is_available():
            return {}

        device_obj = torch.device(self.device)
        device_index = device_obj.index if device_obj.index is not None else torch.cuda.current_device()
        stats = {
            'step': self.global_step,
            'allocated_gb': torch.cuda.memory_allocated(device_index) / 1e9,
            'reserved_gb': torch.cuda.memory_reserved(device_index) / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated(device_index) / 1e9,
        }

        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(device_index).total_memory / 1e9
        stats['total_gb'] = total_memory
        stats['free_gb'] = total_memory - stats['reserved_gb']
        stats['utilization_pct'] = (stats['allocated_gb'] / total_memory) * 100

        return stats

    def train(self) -> Dict:
        """Run training loop.

        Returns:
            Training history.
        """
        if self.is_main_process:
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

            # Barrier after checkpoint to prevent rank drift
            if self.distributed and dist.is_available() and dist.is_initialized():
                dist.barrier()

            if self.is_main_process:
                print(f"Epoch {epoch+1}/{self.num_epochs} - "
                      f"Train Loss: {train_loss:.4f} - "
                      f"Val Loss: {val_loss:.4f} - "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            if self.global_step >= self.max_steps:
                if self.is_main_process:
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

        if self.distributed and hasattr(self.train_dataloader.sampler, "set_epoch"):
            self.train_dataloader.sampler.set_epoch(epoch)
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}", disable=not self.is_main_process)
        for batch_idx, batch in enumerate(pbar):
            if self.global_step >= self.max_steps:
                break

            is_last_batch = (batch_idx + 1) == len(self.train_dataloader)
            hits_step_limit = (self.global_step + 1) >= self.max_steps
            should_step = (
                ((batch_idx + 1) % self.grad_accum_steps == 0) or
                is_last_batch or
                hits_step_limit
            )
            loss = self._train_step(batch, should_step=should_step)
            total_loss += loss
            num_batches += 1

            if self.is_main_process:
                self.train_losses.append(loss)
                self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            if self.is_main_process:
                pbar.set_postfix({'loss': f'{loss:.4f}'})

            # Periodic validation
            if self.global_step > 0 and self.global_step % self.eval_every == 0:
                val_loss = self._validate()
                self.model.train()
                if self.is_main_process:
                    self.val_losses.append(val_loss)
                    self.val_steps.append(self.global_step)
                    self._save_checkpoint(val_loss, epoch)
                if self.distributed and dist.is_available() and dist.is_initialized():
                    dist.barrier()

            # Periodic save
            if (not self.save_best_only and self.save_periodic and self.global_step > 0 and self.global_step % self.save_every == 0):
                self._save_periodic_checkpoint(epoch)
                if self.distributed and dist.is_available() and dist.is_initialized():
                    dist.barrier()

            # GPU memory monitoring
            if self.global_step > 0 and self.global_step % self.memory_log_interval == 0:
                if self.is_main_process:
                    mem_stats = self._log_gpu_memory()
                    if mem_stats:
                        self.memory_stats.append(mem_stats)
                        pbar.set_postfix({
                            'loss': f'{loss:.4f}',
                            'mem': f'{mem_stats["allocated_gb"]:.1f}/{mem_stats["total_gb"]:.0f}GB'
                        })

            self.global_step += 1

        # Barrier before final reduce to ensure all ranks exit loop together
        if self.distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()

        avg_loss = total_loss / max(num_batches, 1)
        return self._reduce_mean(avg_loss)

    def _train_step(self, batch: Dict[str, torch.Tensor], should_step: bool) -> float:
        """Single training step.

        Args:
            batch: Batch of data.
            should_step: Whether to apply optimizer/scheduler updates this micro-batch.

        Returns:
            Loss value.
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        # Forward pass with AMP
        self._maybe_mark_cudagraph_step_begin()
        with autocast('cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            outputs = self.model(input_ids, attention_mask)
            loss = outputs['loss'] / self.grad_accum_steps

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()

        if should_step:
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

                self._maybe_mark_cudagraph_step_begin()
                with autocast('cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                    outputs = self.model(input_ids, attention_mask)
                total_loss += outputs['loss'].item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return self._reduce_mean(avg_loss)

    def _save_checkpoint(self, val_loss: float, epoch: int, final: bool = False):
        """Save model checkpoint.

        Args:
            val_loss: Validation loss.
            epoch: Current epoch.
            final: Whether this is the final checkpoint.
        """
        if not self.is_main_process:
            return

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self._get_model_state(),
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
        if final and not self.save_best_only and self.save_last:
            torch.save(checkpoint, self.checkpoint_dir / 'backbone_last.pt')

    def _save_periodic_checkpoint(self, epoch: int):
        """Save periodic checkpoint.

        Args:
            epoch: Current epoch.
        """
        if self.save_best_only or not self.save_periodic:
            return
        if not self.is_main_process:
            return

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self._get_model_state(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_dir / f'backbone_step_{self.global_step}.pt')

    def _save_history(self):
        """Save training history to CSV."""
        if not self.is_main_process:
            return
        # Round floats to 4 decimal places
        rounded_train_losses = [round(loss, 4) for loss in self.train_losses]
        rounded_learning_rates = [round(lr, 8) for lr in self.learning_rates]  # LR needs more precision
        rounded_val_losses = [round(loss, 4) for loss in self.val_losses]

        # Save as DataFrame
        train_df = pd.DataFrame({
            'step': list(range(len(self.train_losses))),
            'train_loss': rounded_train_losses,
            'learning_rate': rounded_learning_rates
        })
        train_df.to_csv(self.metrics_dir / 'backbone_loss_curve.csv', index=False)

        if self.val_losses and self.val_steps:
            paired_count = min(len(self.val_losses), len(self.val_steps))
            val_df = pd.DataFrame({
                'step': self.val_steps[:paired_count],
                'val_loss': rounded_val_losses[:paired_count]
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model = self.model.module if isinstance(self.model, DDP) else self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Loaded checkpoint from step {self.global_step}")
