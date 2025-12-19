#!/usr/bin/env python
"""Step 1: Train graph diffusion backbone model."""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.utils.config import load_config, save_config
from src.utils.plotting import PlotUtils
from src.data.graph_tokenizer import GraphTokenizer
from src.data.dataset import GraphPolymerDataset, graph_collate_fn
from src.model.graph_backbone import GraphDiffusionBackbone
from src.model.graph_diffusion import GraphMaskingDiffusion
from src.training.graph_trainer_backbone import GraphBackboneTrainer
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

    print("=" * 60)
    print("Step 1: Training Graph Diffusion Backbone")
    print("=" * 60)

    # Load graph configuration (from step0)
    print("\n1. Loading graph configuration...")
    graph_config_path = results_dir / 'graph_config.json'
    with open(graph_config_path, 'r') as f:
        graph_config = json.load(f)

    Nmax = graph_config['Nmax']
    atom_vocab = graph_config['atom_vocab']
    edge_vocab = graph_config['edge_vocab']
    atom_vocab_size = graph_config['atom_vocab_size']
    edge_vocab_size = graph_config['edge_vocab_size']

    print(f"   Nmax: {Nmax}")
    print(f"   Atom vocab size: {atom_vocab_size}")
    print(f"   Edge vocab size: {edge_vocab_size}")

    # Load graph tokenizer
    print("\n2. Loading graph tokenizer...")
    graph_tokenizer = GraphTokenizer.load(results_dir / 'graph_tokenizer.json')

    # Load data
    print("\n3. Loading data...")
    train_df = pd.read_csv(results_dir / 'train_unlabeled.csv')
    val_df = pd.read_csv(results_dir / 'val_unlabeled.csv')

    print(f"   Train samples: {len(train_df)}")
    print(f"   Val samples: {len(val_df)}")

    # Get optimization settings
    opt_config = config.get('optimization', {})
    cache_graphs = opt_config.get('cache_tokenization', False)
    num_workers = opt_config.get('num_workers', 4)
    pin_memory = opt_config.get('pin_memory', True)
    prefetch_factor = opt_config.get('prefetch_factor', 2)

    # Create datasets
    print("\n4. Creating graph datasets...")
    train_dataset = GraphPolymerDataset(
        train_df, graph_tokenizer, cache_graphs=cache_graphs
    )
    val_dataset = GraphPolymerDataset(
        val_df, graph_tokenizer, cache_graphs=cache_graphs
    )

    # Create dataloaders
    batch_size = config['training_backbone']['batch_size']
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

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # Create model
    print("\n5. Creating graph diffusion model...")
    backbone_config = config['backbone']
    diffusion_config = config['diffusion']

    backbone = GraphDiffusionBackbone(
        atom_vocab_size=atom_vocab_size,
        edge_vocab_size=edge_vocab_size,
        Nmax=Nmax,
        hidden_size=backbone_config['hidden_size'],
        num_layers=backbone_config['num_layers'],
        num_heads=backbone_config['num_heads'],
        ffn_hidden_size=backbone_config['ffn_hidden_size'],
        dropout=backbone_config['dropout'],
        num_diffusion_steps=diffusion_config['num_steps']
    )

    model = GraphMaskingDiffusion(
        backbone=backbone,
        num_steps=diffusion_config['num_steps'],
        beta_min=diffusion_config['beta_min'],
        beta_max=diffusion_config['beta_max'],
        node_mask_id=atom_vocab['MASK'],
        edge_mask_id=edge_vocab['MASK'],
        node_pad_id=atom_vocab['PAD'],
        edge_none_id=edge_vocab['NONE'],
        lambda_node=diffusion_config.get('lambda_node', 1.0),
        lambda_edge=diffusion_config.get('lambda_edge', 0.5)
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {num_params:,}")
    print(f"   Trainable parameters: {num_trainable:,}")

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n   Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        state_dict = checkpoint['model_state_dict']
        # Handle torch.compile() state dict
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    # Create trainer
    print("\n6. Starting training...")
    trainer = GraphBackboneTrainer(
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

    # Create loss plots
    print("\n7. Creating loss plots...")
    plotter = PlotUtils(
        figure_size=tuple(config['plotting']['figure_size']),
        font_size=config['plotting']['font_size'],
        dpi=config['plotting']['dpi']
    )

    # Total loss curve
    plotter.loss_curve(
        train_losses=history['train_losses'],
        val_losses=history['val_losses'],
        xlabel='Step',
        ylabel='Loss',
        title='Graph Backbone Training Loss',
        save_path=figures_dir / 'graph_backbone_loss_curve.png'
    )

    # Node vs Edge loss comparison
    if history['train_node_losses'] and history['train_edge_losses']:
        plotter.loss_curve(
            train_losses=history['train_node_losses'],
            val_losses=history['train_edge_losses'],
            xlabel='Step',
            ylabel='Loss',
            title='Node vs Edge Loss',
            save_path=figures_dir / 'node_edge_loss_comparison.png',
            train_label='Node Loss',
            val_label='Edge Loss'
        )

    # Save final summary
    summary = {
        'total_steps': trainer.global_step,
        'best_val_loss': round(history['best_val_loss'], 4),
        'final_train_loss': round(history['train_losses'][-1], 4) if history['train_losses'] else None,
        'final_val_loss': round(history['val_losses'][-1], 4) if history['val_losses'] else None,
        'num_params': num_params,
        'Nmax': Nmax,
        'atom_vocab_size': atom_vocab_size,
        'edge_vocab_size': edge_vocab_size
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(metrics_dir / 'training_summary.csv', index=False)

    print("\n" + "=" * 60)
    print("Graph Backbone Training Complete!")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  Best validation loss: {history['best_val_loss']:.4f}")
    print(f"  Total training steps: {trainer.global_step}")
    print(f"  Checkpoints saved to: {results_dir / 'checkpoints'}")
    print(f"  Metrics saved to: {metrics_dir}")
    print(f"  Figures saved to: {figures_dir}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train graph diffusion backbone')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    main(args)
