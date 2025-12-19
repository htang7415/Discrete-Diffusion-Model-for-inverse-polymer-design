#!/usr/bin/env python
"""Step 3: Train property prediction heads using graph backbone."""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from src.utils.config import load_config, save_config
from src.utils.plotting import PlotUtils
from src.data.graph_tokenizer import GraphTokenizer
from src.data.data_loader import PolymerDataLoader
from src.data.dataset import GraphPropertyDataset, graph_collate_fn
from src.model.graph_backbone import GraphDiffusionBackbone, create_graph_backbone
from src.model.graph_diffusion import GraphMaskingDiffusion, create_graph_diffusion
from src.model.graph_property_head import GraphPropertyHead, GraphPropertyPredictor
from src.training.graph_trainer_property import GraphPropertyTrainer
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
    step_dir = results_dir / f'step3_{args.property}'
    metrics_dir = step_dir / 'metrics'
    figures_dir = step_dir / 'figures'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    seed_info = seed_everything(config['data']['random_seed'])
    save_config(config, step_dir / 'config_used.yaml')
    save_run_metadata(step_dir, args.config, seed_info)

    print("=" * 50)
    print(f"Step 3: Training Graph Property Head for {args.property}")
    print("=" * 50)

    # Load graph config
    print("\n1. Loading graph config and tokenizer...")
    graph_config_path = results_dir / 'graph_config.json'
    with open(graph_config_path, 'r') as f:
        graph_config = json.load(f)

    # Load graph tokenizer
    tokenizer = GraphTokenizer.load(results_dir / 'graph_tokenizer.json')
    print(f"Nmax: {graph_config['Nmax']}")
    print(f"Atom vocab size: {graph_config['atom_vocab_size']}")
    print(f"Edge vocab size: {graph_config['edge_vocab_size']}")

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

    # Get optimization settings
    opt_config = config.get('optimization', {})
    cache_graphs = opt_config.get('cache_tokenization', False)
    num_workers = opt_config.get('num_workers', 4)
    pin_memory = opt_config.get('pin_memory', True)
    prefetch_factor = opt_config.get('prefetch_factor', 2)

    # Create datasets
    train_dataset = GraphPropertyDataset(
        train_df, tokenizer, args.property,
        normalize=True, mean=mean, std=std,
        cache_graphs=cache_graphs
    )
    val_dataset = GraphPropertyDataset(
        val_df, tokenizer, args.property,
        normalize=True, mean=mean, std=std,
        cache_graphs=cache_graphs
    )
    test_dataset = GraphPropertyDataset(
        test_df, tokenizer, args.property,
        normalize=True, mean=mean, std=std,
        cache_graphs=cache_graphs
    )

    # Create dataloaders
    batch_size = config['training_property']['batch_size']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=graph_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=graph_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=graph_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    # Load backbone
    print("\n3. Loading graph backbone...")
    checkpoint_path = args.backbone_checkpoint or (results_dir / 'checkpoints' / 'graph_backbone_best.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create backbone
    backbone = create_graph_backbone(config, graph_config)

    # Create diffusion model to load weights
    diffusion_model = create_graph_diffusion(backbone, config, graph_config)

    # Load backbone weights from diffusion model
    # Handle torch.compile() state dict (keys have _orig_mod. prefix)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    diffusion_model.load_state_dict(state_dict)
    backbone = diffusion_model.backbone

    # Create property head
    print("\n4. Creating property head...")
    head_config = config['property_head']
    backbone_config = config.get('backbone', config.get('graph_backbone', {}))
    property_head = GraphPropertyHead(
        input_size=backbone_config.get('hidden_size', 768),
        hidden_sizes=head_config['hidden_sizes'],
        dropout=head_config['dropout']
    )

    # Create property predictor
    train_config = config['training_property']
    default_timestep = train_config.get('default_timestep', 1)
    model = GraphPropertyPredictor(
        backbone=backbone,
        property_head=property_head,
        freeze_backbone=train_config['freeze_backbone'],
        finetune_last_layers=train_config['finetune_last_layers'],
        pooling='mean',
        default_timestep=default_timestep
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")

    # Train
    print("\n5. Starting training...")
    trainer = GraphPropertyTrainer(
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
    parser = argparse.ArgumentParser(description='Train graph property prediction head')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--property', type=str, required=True,
                        help='Property name (e.g., Tg, Tm)')
    parser.add_argument('--backbone_checkpoint', type=str, default=None,
                        help='Path to graph backbone checkpoint')
    args = parser.parse_args()
    main(args)
