#!/usr/bin/env python
"""Step 1: Train diffusion backbone model."""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from src.utils.config import load_config, save_config
from src.utils.plotting import PlotUtils
from src.utils.model_scales import (
    get_model_config, get_training_config, estimate_params,
    get_results_dir, print_model_info
)
from src.data.selfies_tokenizer import SelfiesTokenizer
from src.data.dataset import PolymerDataset, collate_fn
from src.model.backbone import DiffusionBackbone
from src.model.diffusion import DiscreteMaskingDiffusion
from src.training.trainer_backbone import BackboneTrainer
from src.utils.reproducibility import seed_everything, save_run_metadata


def init_distributed():
    """Initialize torch.distributed if launched with torchrun."""
    if not dist.is_available():
        return False, 0, 1, 0, None
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0, None
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    return True, rank, world_size, local_rank, device


def main(args):
    """Main function."""
    # Load config
    config = load_config(args.config)

    distributed, rank, world_size, local_rank, dist_device = init_distributed()
    is_main_process = (not distributed) or rank == 0

    # Set device
    device = dist_device if distributed else ('cuda' if torch.cuda.is_available() else 'cpu')
    if is_main_process:
        print(f"Using device: {device}")

    # Override results_dir if model_size specified
    base_results_dir = config['paths']['results_dir']
    results_dir = Path(get_results_dir(args.model_size, base_results_dir))
    step_dir = results_dir / 'step1_backbone'
    metrics_dir = step_dir / 'metrics'
    figures_dir = step_dir / 'figures'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    seed_info = seed_everything(config['data']['random_seed'])
    if is_main_process:
        save_config(config, step_dir / 'config_used.yaml')
        save_run_metadata(step_dir, args.config, seed_info)

    if is_main_process:
        print("=" * 50)
        print("Step 1: Training Diffusion Backbone")
        print("=" * 50)

    # Get model and training config based on model_size
    backbone_config = get_model_config(args.model_size, config, model_type='sequence')
    if args.model_size:
        training_config = get_training_config(args.model_size, config, model_type='sequence')
        # Override training_backbone config
        config['training_backbone']['batch_size'] = training_config['batch_size']
        config['training_backbone']['learning_rate'] = training_config['learning_rate']
        config['training_backbone']['max_steps'] = training_config['max_steps']
        config['training_backbone']['warmup_steps'] = training_config['warmup_steps']
        config['optimization']['gradient_accumulation_steps'] = training_config['gradient_accumulation_steps']

    # Load tokenizer (from base results dir which has the tokenizer)
    if is_main_process:
        print("\n1. Loading SELFIES tokenizer...")
    tokenizer_path = results_dir / 'tokenizer.json'
    if not tokenizer_path.exists():
        # Fall back to base results dir
        tokenizer_path = Path(base_results_dir) / 'tokenizer.json'
    tokenizer = SelfiesTokenizer.load(tokenizer_path)
    if is_main_process:
        print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Print model info if model_size specified
    if args.model_size and is_main_process:
        print_model_info(args.model_size, backbone_config, training_config,
                        tokenizer.vocab_size, model_type='sequence')

    # Load data (from base results dir which has the data)
    if is_main_process:
        print("\n2. Loading data...")
    train_path = results_dir / 'train_unlabeled.csv'
    val_path = results_dir / 'val_unlabeled.csv'
    if not train_path.exists():
        train_path = Path(base_results_dir) / 'train_unlabeled.csv'
        val_path = Path(base_results_dir) / 'val_unlabeled.csv'
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Get optimization settings
    opt_config = config.get('optimization', {})
    cache_tokenization = opt_config.get('cache_tokenization', False)
    num_workers = opt_config.get('num_workers', 4)
    pin_memory = opt_config.get('pin_memory', True)
    prefetch_factor = opt_config.get('prefetch_factor', 2)

    # Create datasets
    train_dataset = PolymerDataset(train_df, tokenizer, cache_tokenization=cache_tokenization)
    val_dataset = PolymerDataset(val_df, tokenizer, cache_tokenization=cache_tokenization)

    # Create dataloaders
    batch_size = config['training_backbone']['batch_size']
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    if is_main_process:
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

    # Create model
    if is_main_process:
        print("\n3. Creating model...")
    backbone = DiffusionBackbone(
        vocab_size=tokenizer.vocab_size,
        hidden_size=backbone_config['hidden_size'],
        num_layers=backbone_config['num_layers'],
        num_heads=backbone_config['num_heads'],
        ffn_hidden_size=backbone_config['ffn_hidden_size'],
        max_position_embeddings=backbone_config['max_position_embeddings'],
        num_diffusion_steps=config['diffusion']['num_steps'],
        dropout=backbone_config['dropout'],
        pad_token_id=tokenizer.pad_token_id
    )

    model = DiscreteMaskingDiffusion(
        backbone=backbone,
        num_steps=config['diffusion']['num_steps'],
        beta_min=config['diffusion']['beta_min'],
        beta_max=config['diffusion']['beta_max'],
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process:
        print(f"Total parameters: {num_params:,}")
        print(f"Trainable parameters: {num_trainable:,}")

    # Resume from checkpoint if specified
    if args.resume:
        if is_main_process:
            print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        # Handle torch.compile() state dict (keys have _orig_mod. prefix)
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    # Create trainer
    if is_main_process:
        print("\n4. Starting training...")
    trainer = BackboneTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        device=device,
        output_dir=str(step_dir),
        step_dir=str(step_dir),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank
    )

    # Train
    history = trainer.train()

    # Create loss plot
    if is_main_process:
        print("\n5. Creating loss plot...")
        plotter = PlotUtils(
            figure_size=tuple(config['plotting']['figure_size']),
            font_size=config['plotting']['font_size'],
            dpi=config['plotting']['dpi']
        )

        plotter.loss_curve(
            train_losses=history['train_losses'],
            val_losses=history['val_losses'],
            xlabel='Step',
            ylabel='Loss',
            title='Backbone Training Loss',
            save_path=figures_dir / 'backbone_loss_curve.png'
        )

        print("\n" + "=" * 50)
        print("Backbone training complete!")
        print(f"Best validation loss: {history['best_val_loss']:.4f}")
        print(f"Checkpoints saved to: {step_dir / 'checkpoints'}")
        print("=" * 50)

    if distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train diffusion backbone')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_size', type=str, default=None,
                        choices=['small', 'medium', 'large', 'xl'],
                        help='Model size preset (small: ~12M, medium: ~50M, large: ~150M, xl: ~400M)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    main(args)
