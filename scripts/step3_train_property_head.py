#!/usr/bin/env python
"""Step 3: Train property prediction heads."""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.plotting import PlotUtils
from src.data.tokenizer import PSmilesTokenizer
from src.data.data_loader import PolymerDataLoader
from src.data.dataset import PropertyDataset, collate_fn
from src.model.backbone import DiffusionBackbone
from src.model.diffusion import DiscreteMaskingDiffusion
from src.model.property_head import PropertyHead, PropertyPredictor
from src.training.trainer_property import PropertyTrainer


def main(args):
    """Main function."""
    # Load config
    config = load_config(args.config)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create output directories
    results_dir = Path(config['paths']['results_dir'])
    step_dir = results_dir / f'step3_{args.property}'
    metrics_dir = step_dir / 'metrics'
    figures_dir = step_dir / 'figures'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print(f"Step 3: Training Property Head for {args.property}")
    print("=" * 50)

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = PSmilesTokenizer.load(results_dir / 'tokenizer.json')

    # Load property data
    print("\n2. Loading property data...")
    data_loader = PolymerDataLoader(config)
    property_data = data_loader.prepare_property_data(args.property)

    train_df = property_data['train']
    val_df = property_data['val']
    test_df = property_data['test']

    # Compute normalization parameters from training data
    mean = train_df[args.property].mean()
    std = train_df[args.property].std()
    print(f"Normalization: mean={mean:.4f}, std={std:.4f}")

    # Create datasets
    train_dataset = PropertyDataset(
        train_df, tokenizer, args.property,
        normalize=True, mean=mean, std=std
    )
    val_dataset = PropertyDataset(
        val_df, tokenizer, args.property,
        normalize=True, mean=mean, std=std
    )
    test_dataset = PropertyDataset(
        test_df, tokenizer, args.property,
        normalize=True, mean=mean, std=std
    )

    # Create dataloaders
    batch_size = config['training_property']['batch_size']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    # Load backbone
    print("\n3. Loading backbone...")
    checkpoint_path = args.backbone_checkpoint or (results_dir / 'checkpoints' / 'backbone_best.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)

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

    # Load backbone weights from diffusion model
    diffusion_model = DiscreteMaskingDiffusion(
        backbone=backbone,
        num_steps=config['diffusion']['num_steps'],
        beta_min=config['diffusion']['beta_min'],
        beta_max=config['diffusion']['beta_max'],
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    diffusion_model.load_state_dict(checkpoint['model_state_dict'])
    backbone = diffusion_model.backbone

    # Create property head
    print("\n4. Creating property head...")
    head_config = config['property_head']
    property_head = PropertyHead(
        input_size=backbone_config['hidden_size'],
        hidden_sizes=head_config['hidden_sizes'],
        dropout=head_config['dropout']
    )

    # Create property predictor
    train_config = config['training_property']
    model = PropertyPredictor(
        backbone=backbone,
        property_head=property_head,
        freeze_backbone=train_config['freeze_backbone'],
        finetune_last_layers=train_config['finetune_last_layers'],
        pooling='mean'
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")

    # Train
    print("\n5. Starting training...")
    trainer = PropertyTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        property_name=args.property,
        config=config,
        device=device,
        output_dir=str(results_dir),
        normalization_params={'mean': mean, 'std': std},
        step_dir=str(step_dir)
    )

    history = trainer.train()

    # Create plots
    print("\n6. Creating plots...")
    plotter = PlotUtils(
        figure_size=tuple(config['plotting']['figure_size']),
        font_size=config['plotting']['font_size'],
        dpi=config['plotting']['dpi']
    )

    # Loss curve
    plotter.loss_curve(
        train_losses=history['train_losses'],
        val_losses=history['val_losses'],
        xlabel='Epoch',
        ylabel='MSE Loss',
        title=f'{args.property} Training Loss',
        save_path=figures_dir / f'{args.property}_loss_curve.png'
    )

    # Parity plot
    test_metrics = history['test_metrics']
    plotter.parity_plot(
        y_true=np.array(test_metrics['labels']),
        y_pred=np.array(test_metrics['predictions']),
        xlabel=f'True {args.property}',
        ylabel=f'Predicted {args.property}',
        title=f'{args.property} Parity Plot',
        save_path=figures_dir / f'{args.property}_parity_plot.png',
        metrics={
            'MAE': test_metrics['MAE'],
            'RMSE': test_metrics['RMSE'],
            'R²': test_metrics['R2']
        }
    )

    # Save data statistics using data_loader.get_statistics() for full stats
    # (includes count, unique_smiles, length_*, sa_*, and property_* stats)
    train_stats = data_loader.get_statistics(train_df, args.property)
    val_stats = data_loader.get_statistics(val_df, args.property)
    test_stats = data_loader.get_statistics(test_df, args.property)

    stats_df = pd.DataFrame([
        {'split': 'train', **train_stats},
        {'split': 'val', **val_stats},
        {'split': 'test', **test_stats}
    ])
    stats_df.to_csv(metrics_dir / f'{args.property}_data_stats.csv', index=False)

    print("\n" + "=" * 50)
    print(f"Property head training complete for {args.property}!")
    print(f"Test MAE: {test_metrics['MAE']:.4f}")
    print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
    print(f"Test R²: {test_metrics['R2']:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train property prediction head')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--property', type=str, required=True,
                        help='Property name (e.g., Tg, Tm)')
    parser.add_argument('--backbone_checkpoint', type=str, default=None,
                        help='Path to backbone checkpoint')
    args = parser.parse_args()
    main(args)
