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
from torch.utils.data import DataLoader

from src.utils.config import load_config, save_config
from src.utils.plotting import PlotUtils
from src.data.tokenizer import GroupSELFIESTokenizer
from src.data.dataset import PolymerDataset, collate_fn
from src.model.backbone import DiffusionBackbone
from src.model.diffusion import DiscreteMaskingDiffusion
from src.training.trainer_backbone import BackboneTrainer
from src.utils.reproducibility import seed_everything, save_run_metadata


def main(args):
    """Main function."""
    # Load config
    config = load_config(args.config)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create output directories
    results_dir = Path(config['paths']['results_dir'])
    step_dir = results_dir / 'step1_backbone'
    metrics_dir = step_dir / 'metrics'
    figures_dir = step_dir / 'figures'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    seed_info = seed_everything(config['data']['random_seed'])
    save_config(config, step_dir / 'config_used.yaml')
    save_run_metadata(step_dir, args.config, seed_info)

    print("=" * 50)
    print("Step 1: Training Diffusion Backbone")
    print("=" * 50)

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = GroupSELFIESTokenizer.load(results_dir / 'tokenizer.pkl')
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Load data
    print("\n2. Loading data...")
    train_df = pd.read_csv(results_dir / 'train_unlabeled.csv')
    val_df = pd.read_csv(results_dir / 'val_unlabeled.csv')

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\n3. Creating model...")
    backbone_config = config['backbone']
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
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        # Handle torch.compile() state dict (keys have _orig_mod. prefix)
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    # Create trainer
    print("\n4. Starting training...")
    trainer = BackboneTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        device=device,
        output_dir=str(results_dir),
        step_dir=str(step_dir)
    )

    # Train
    history = trainer.train()

    # Create loss plot
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
    print(f"Checkpoints saved to: {results_dir / 'checkpoints'}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train diffusion backbone')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    main(args)
