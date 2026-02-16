#!/usr/bin/env python
"""Step 1: Train graph diffusion backbone model."""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

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
from src.data.graph_tokenizer import GraphTokenizer
from src.data.dataset import GraphPolymerDataset, graph_collate_fn
from src.model.graph_backbone import GraphDiffusionBackbone
from src.model.graph_diffusion import GraphMaskingDiffusion
from src.training.graph_trainer_backbone import GraphBackboneTrainer
from src.utils.reproducibility import seed_everything, save_run_metadata
from shared.unlabeled_data import require_preprocessed_unlabeled_splits


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

    print("=" * 60)
    print("Step 1: Training Graph Diffusion Backbone")
    print("=" * 60)

    # Get model and training config based on model_size
    backbone_config = get_model_config(args.model_size, config, model_type='graph')
    if args.model_size:
        training_config = get_training_config(args.model_size, config, model_type='graph')
        # Override training_backbone config
        config['training_backbone']['batch_size'] = training_config['batch_size']
        config['training_backbone']['learning_rate'] = training_config['learning_rate']
        config['training_backbone']['max_steps'] = training_config['max_steps']
        config['training_backbone']['warmup_steps'] = training_config['warmup_steps']
        config['optimization']['gradient_accumulation_steps'] = training_config['gradient_accumulation_steps']

    # Load graph configuration (from step0)
    print("\n1. Loading graph configuration...")
    graph_config_path = results_dir / 'graph_config.json'
    if not graph_config_path.exists():
        graph_config_path = Path(base_results_dir) / 'graph_config.json'
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

    # Print model info if model_size specified
    if args.model_size:
        print_model_info(args.model_size, backbone_config, training_config,
                        atom_vocab_size, model_type='graph')

    # Load graph tokenizer
    print("\n2. Loading graph tokenizer...")
    tokenizer_path = results_dir / 'graph_tokenizer.json'
    if not tokenizer_path.exists():
        tokenizer_path = Path(base_results_dir) / 'graph_tokenizer.json'
    graph_tokenizer = GraphTokenizer.load(tokenizer_path)

    # Load data (from base results dir which has the data)
    print("\n3. Loading data...")
    repo_root = Path(__file__).resolve().parents[2]
    train_path, val_path = require_preprocessed_unlabeled_splits(repo_root)
    train_df = pd.read_csv(train_path, usecols=['p_smiles'])
    val_df = pd.read_csv(val_path, usecols=['p_smiles'])

    # Optionally subsample training data (validation always full)
    train_fraction = config.get('data', {}).get('train_fraction', 1.0)
    if train_fraction <= 0 or train_fraction > 1:
        raise ValueError("data.train_fraction must be within (0, 1].")
    if train_fraction < 1.0:
        full_train_count = len(train_df)
        n_train = max(1, int(round(full_train_count * train_fraction)))
        train_df = train_df.sample(
            n=n_train, random_state=config['data']['random_seed']
        ).reset_index(drop=True)
        if is_main_process:
            print(f"Using {n_train}/{full_train_count} train samples ({train_fraction:.2%})")

    # Optionally subsample validation data for faster periodic evaluation.
    train_cfg = config.get('training_backbone', {})
    val_fraction = float(train_cfg.get('val_fraction', 1.0))
    if val_fraction <= 0 or val_fraction > 1:
        raise ValueError("training_backbone.val_fraction must be within (0, 1].")
    if val_fraction < 1.0:
        full_val_count = len(val_df)
        n_val = max(1, int(round(full_val_count * val_fraction)))
        val_df = val_df.sample(
            n=n_val, random_state=config['data']['random_seed']
        ).reset_index(drop=True)
        if is_main_process:
            print(f"Using {n_val}/{full_val_count} val samples ({val_fraction:.2%})")
    val_max_samples = int(train_cfg.get('val_max_samples', 0))
    if val_max_samples > 0 and len(val_df) > val_max_samples:
        full_val_count = len(val_df)
        val_df = val_df.sample(
            n=val_max_samples, random_state=config['data']['random_seed']
        ).reset_index(drop=True)
        if is_main_process:
            print(f"Capping val samples to {val_max_samples}/{full_val_count} for faster eval")

    if is_main_process:
        print(f"   Train samples: {len(train_df)}")
        print(f"   Val samples: {len(val_df)}")

    # Get optimization settings
    opt_config = config.get('optimization', {})
    cache_graphs = opt_config.get('cache_tokenization', False)
    cache_max_samples = int(opt_config.get('cache_tokenization_max_samples', 500000))
    num_workers = int(opt_config.get('step1_num_workers', opt_config.get('num_workers', 4)))
    pin_memory = opt_config.get('pin_memory', True)
    prefetch_factor = opt_config.get('prefetch_factor', 2)
    step1_persistent_workers = bool(
        opt_config.get('step1_persistent_workers', opt_config.get('persistent_workers', False))
    )

    # Bound DataLoader workers to per-rank CPU budget to avoid oversubscription.
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1") or 1)
    slurm_cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
    host_cpus = os.cpu_count() or 1
    if slurm_cpus_per_task > 0:
        per_rank_cpu_budget = max(1, slurm_cpus_per_task // max(1, local_world_size))
    else:
        per_rank_cpu_budget = max(1, host_cpus // max(1, local_world_size))
    per_rank_worker_cap = max(1, per_rank_cpu_budget - 2)
    if num_workers <= 0:
        num_workers = per_rank_worker_cap
        if is_main_process:
            print(
                "Auto-selected DataLoader workers per rank: "
                f"{num_workers} (cpu_budget={per_rank_cpu_budget}, local_world_size={local_world_size})"
            )
    elif num_workers > per_rank_worker_cap:
        if is_main_process:
            print(
                f"Capping num_workers from {num_workers} to {per_rank_worker_cap} "
                f"(cpu_budget={per_rank_cpu_budget}, local_world_size={local_world_size})"
            )
        num_workers = per_rank_worker_cap
    persistent_workers = step1_persistent_workers and num_workers > 0

    # Guard against memory blow-up: graph caching can exceed RAM on large datasets.
    total_samples = len(train_df) + len(val_df)
    if cache_graphs and distributed:
        if is_main_process:
            print("Disabling cache_graphs under DDP to avoid per-rank RAM duplication.")
        cache_graphs = False
    elif cache_graphs and total_samples > cache_max_samples:
        if is_main_process:
            print(
                f"Disabling cache_graphs for {total_samples:,} samples "
                f"(limit={cache_max_samples:,})."
            )
        cache_graphs = False

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
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=graph_collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=graph_collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    if is_main_process:
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   DataLoader workers per rank: {num_workers}")

    # Create model
    print("\n5. Creating graph diffusion model...")
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
        force_clean_t0=diffusion_config.get('force_clean_t0', False),
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
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
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
        output_dir=str(step_dir),
        step_dir=str(step_dir),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank
    )

    # Train
    history = trainer.train()

    if is_main_process:
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
        print(f"  Checkpoints saved to: {step_dir / 'checkpoints'}")
        print(f"  Metrics saved to: {metrics_dir}")
        print(f"  Figures saved to: {figures_dir}")
        print("=" * 60)

    if distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train graph diffusion backbone')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_size', type=str, default=None,
                        choices=['small', 'medium', 'large', 'xl'],
                        help='Model size preset (small: ~14M, medium: ~55M, large: ~140M, xl: ~350M)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    main(args)
