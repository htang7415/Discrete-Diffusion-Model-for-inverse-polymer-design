#!/usr/bin/env python
"""Step 3: Train property prediction heads with optional Optuna hyperparameter tuning."""

import os
import sys
import argparse
import json
import copy
import math
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Optuna for hyperparameter optimization
import optuna
from optuna.pruners import MedianPruner

from src.utils.config import load_config, save_config
from src.utils.plotting import PlotUtils
from src.utils.model_scales import get_model_config, get_results_dir
from src.data.tokenizer import PSmilesTokenizer
from src.data.data_loader import PolymerDataLoader
from src.data.dataset import PropertyDataset, collate_fn
from src.model.backbone import DiffusionBackbone
from src.model.diffusion import DiscreteMaskingDiffusion
from src.model.property_head import PropertyHead, PropertyPredictor
from src.training.trainer_property import PropertyTrainer
from src.utils.reproducibility import seed_everything, save_run_metadata
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_one_epoch(model, train_loader, optimizer, device, scaler=None):
    """Train model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                predictions = model(input_ids, attention_mask)
                loss = nn.functional.mse_loss(predictions.squeeze(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(input_ids, attention_mask)
            loss = nn.functional.mse_loss(predictions.squeeze(), labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def compute_val_r2(model, val_loader, device, normalization_params):
    """Compute R² score on validation set."""
    model.eval()
    all_preds = []
    all_labels = []

    mean = normalization_params['mean']
    std = normalization_params['std']

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                predictions = model(input_ids, attention_mask)

            # Denormalize predictions and labels
            preds = predictions.squeeze().float().cpu().numpy() * std + mean
            labs = labels.cpu().numpy() * std + mean

            all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else [preds])
            all_labels.extend(labs.tolist() if hasattr(labs, 'tolist') else [labs])

    if len(all_preds) < 2:
        return 0.0

    return r2_score(all_labels, all_preds)


def _extract_hidden_sizes(params):
    """Resolve hidden_sizes list from hyperparameter dict."""
    hidden_sizes = params.get('hidden_sizes')
    if hidden_sizes is not None:
        if isinstance(hidden_sizes, tuple):
            return list(hidden_sizes)
        if isinstance(hidden_sizes, list):
            return list(hidden_sizes)
        raise ValueError(
            f"hidden_sizes must be a list or tuple, got {type(hidden_sizes).__name__}"
        )

    num_layers = params.get('num_layers')
    if num_layers is not None:
        num_layers = int(num_layers)
        layer_sizes = [params.get(f'layer_{i}_size') for i in range(num_layers)]
        if all(size is not None for size in layer_sizes):
            return layer_sizes

    if 'neurons' in params and 'num_layers' in params:
        return [params['neurons']] * int(params['num_layers'])

    raise KeyError("Hyperparameters missing hidden layer sizes")


def _resolve_finetune_last_layers(search_space, backbone_config):
    """Resolve finetune_last_layers candidates from ratios or explicit list."""
    ratios = search_space.get('finetune_last_layers_ratios')
    if ratios is None:
        return search_space['finetune_last_layers']

    num_layers = int(backbone_config.get('num_layers', 0))
    if num_layers <= 0:
        raise ValueError("num_layers must be > 0 for finetune_last_layers_ratios")

    candidates = []
    for ratio in ratios:
        ratio_value = float(ratio)
        if ratio_value <= 0:
            candidates.append(0)
            continue
        layers = int(math.ceil(num_layers * ratio_value))
        if layers < 1:
            layers = 1
        if layers > num_layers:
            layers = num_layers
        candidates.append(layers)

    return sorted(set(candidates))


def create_objective(backbone_state_dict, train_dataset, val_dataset, config, device,
                     backbone_config, normalization_params, tokenizer, diffusion_config):
    """Create Optuna objective function for hyperparameter tuning.

    Goal: MAXIMIZE R² on validation set.
    """
    search_space = config['hyperparameter_tuning']['search_space']
    tuning_epochs = config['hyperparameter_tuning']['tuning_epochs']
    tuning_patience = config['hyperparameter_tuning'].get('tuning_patience', 10)
    opt_config = config.get('optimization', {})
    num_workers = opt_config.get('num_workers', 4)
    pin_memory = opt_config.get('pin_memory', True)
    prefetch_factor = opt_config.get('prefetch_factor', 2)

    finetune_candidates = _resolve_finetune_last_layers(search_space, backbone_config)

    def objective(trial):
        # Sample hyperparameters
        lr = trial.suggest_categorical('learning_rate', search_space['learning_rate'])
        dropout = trial.suggest_categorical('dropout', search_space['dropout'])
        finetune_last_layers = trial.suggest_categorical('finetune_last_layers',
                                                          finetune_candidates)
        batch_size = trial.suggest_categorical('batch_size', search_space['batch_size'])

        # Build hidden_sizes: allow per-layer sizes
        num_layers = trial.suggest_categorical('num_layers', search_space['num_layers'])
        hidden_sizes = [
            trial.suggest_categorical(f'layer_{i}_size', search_space['neurons'])
            for i in range(num_layers)
        ]

        # Create fresh backbone for this trial
        backbone = DiffusionBackbone(
            vocab_size=tokenizer.vocab_size,
            hidden_size=backbone_config['hidden_size'],
            num_layers=backbone_config['num_layers'],
            num_heads=backbone_config['num_heads'],
            ffn_hidden_size=backbone_config['ffn_hidden_size'],
            max_position_embeddings=backbone_config['max_position_embeddings'],
            num_diffusion_steps=diffusion_config['num_steps'],
            dropout=backbone_config['dropout'],
            pad_token_id=tokenizer.pad_token_id
        )
        backbone.load_state_dict(backbone_state_dict)

        # Create dataloaders with sampled batch_size
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )

        # Create model with sampled hyperparameters
        property_head = PropertyHead(
            input_size=backbone_config['hidden_size'],
            hidden_sizes=hidden_sizes,
            dropout=dropout
        )
        model = PropertyPredictor(
            backbone=backbone,
            property_head=property_head,
            freeze_backbone=True,
            finetune_last_layers=finetune_last_layers,
            pooling='mean',
            default_timestep=config['training_property'].get('default_timestep', 1)
        )
        model.to(device)

        # Training setup
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=0.01
        )
        scaler = torch.amp.GradScaler('cuda')

        # Train with early stopping
        best_val_r2 = -float('inf')
        patience_counter = 0

        for epoch in range(tuning_epochs):
            # Train one epoch
            train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)

            # Validate and compute R²
            val_r2 = compute_val_r2(model, val_loader, device, normalization_params)

            # Report to Optuna for pruning (maximize R²)
            trial.report(val_r2, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= tuning_patience:
                    break

        return best_val_r2  # Optuna will MAXIMIZE this

    return objective


def main(args):
    """Main function."""
    # Load config
    config = load_config(args.config)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Override results_dir if model_size specified
    base_results_dir = config['paths']['results_dir']
    results_dir = Path(get_results_dir(args.model_size, base_results_dir))

    # Create output directories
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
    print(f"Step 3: Training Property Head for {args.property}")
    if args.model_size:
        print(f"Model Size: {args.model_size}")
    print("=" * 50)

    # Load tokenizer (from base results dir which has the tokenizer)
    print("\n1. Loading tokenizer...")
    tokenizer_path = results_dir / 'tokenizer.json'
    if not tokenizer_path.exists():
        tokenizer_path = Path(base_results_dir) / 'tokenizer.json'
    tokenizer = PSmilesTokenizer.load(tokenizer_path)

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
    cache_tokenization = opt_config.get('cache_tokenization', False)
    num_workers = opt_config.get('num_workers', 4)
    pin_memory = opt_config.get('pin_memory', True)
    prefetch_factor = opt_config.get('prefetch_factor', 2)

    # Create datasets
    train_dataset = PropertyDataset(
        train_df, tokenizer, args.property,
        normalize=True, mean=mean, std=std,
        cache_tokenization=cache_tokenization
    )
    val_dataset = PropertyDataset(
        val_df, tokenizer, args.property,
        normalize=True, mean=mean, std=std,
        cache_tokenization=cache_tokenization
    )
    test_dataset = PropertyDataset(
        test_df, tokenizer, args.property,
        normalize=True, mean=mean, std=std,
        cache_tokenization=cache_tokenization
    )

    # Create dataloaders
    batch_size = config['training_property']['batch_size']
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    # Load backbone
    print("\n3. Loading backbone...")
    checkpoint_path = args.backbone_checkpoint or (results_dir / 'step1_backbone' / 'checkpoints' / 'backbone_best.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get backbone config based on model_size
    backbone_config = get_model_config(args.model_size, config, model_type='sequence')
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
        force_clean_t0=config['diffusion'].get('force_clean_t0', False),
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    # Handle torch.compile() state dict (keys have _orig_mod. prefix)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    diffusion_model.load_state_dict(state_dict)
    backbone = diffusion_model.backbone

    # Save backbone state dict for hyperparameter tuning
    backbone_state_dict = copy.deepcopy(backbone.state_dict())
    normalization_params = {'mean': mean, 'std': std}

    # ============================================================
    # Hyperparameter Tuning with Optuna (if enabled)
    # ============================================================
    enable_tuning = args.tune or config.get('hyperparameter_tuning', {}).get('enabled', False)
    best_hyperparams = None

    if enable_tuning:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING WITH OPTUNA (Maximize Validation R²)")
        print("=" * 60)

        tuning_dir = step_dir / 'tuning'
        tuning_dir.mkdir(parents=True, exist_ok=True)

        tuning_config = config['hyperparameter_tuning']
        n_trials = args.n_trials or tuning_config.get('n_trials', 50)

        # Create Optuna study - MAXIMIZE R²
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        study = optuna.create_study(direction='maximize', pruner=pruner)

        # Create objective function
        objective = create_objective(
            backbone_state_dict=backbone_state_dict,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            device=device,
            backbone_config=backbone_config,
            normalization_params=normalization_params,
            tokenizer=tokenizer,
            diffusion_config=config['diffusion']
        )

        # Run optimization
        print(f"\nRunning {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Get best params
        best_hyperparams = study.best_params
        best_val_r2 = study.best_value

        # ===== Save all trials to txt file =====
        with open(tuning_dir / 'all_trials.txt', 'w') as f:
            f.write(f"Hyperparameter Optimization Results for {args.property}\n")
            f.write(f"Metric: Validation R² (maximized)\n")
            f.write(f"Total trials: {len(study.trials)}\n")
            f.write("=" * 80 + "\n\n")

            for trial in study.trials:
                f.write(f"Trial {trial.number}:\n")
                f.write(f"  Status: {trial.state.name}\n")
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    f.write(f"  Val R²: {trial.value:.6f}\n")
                f.write(f"  Params:\n")
                for key, value in trial.params.items():
                    f.write(f"    {key}: {value}\n")
                f.write("\n")

        # ===== Save best hyperparameters to txt file =====
        hidden_sizes_best = _extract_hidden_sizes(best_hyperparams)
        with open(tuning_dir / 'best_hyperparameters.txt', 'w') as f:
            f.write(f"Best Hyperparameters for {args.property}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Best Validation R²: {best_val_r2:.6f}\n")
            f.write(f"Best Trial: {study.best_trial.number}\n\n")
            f.write("Hyperparameters:\n")
            for key, value in best_hyperparams.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nDerived hidden_sizes: {hidden_sizes_best}\n")

        print(f"\n{'=' * 60}")
        print(f"Best hyperparameters found:")
        for key, value in best_hyperparams.items():
            print(f"  {key}: {value}")
        print(f"Best validation R²: {best_val_r2:.6f}")
        print(f"Results saved to: {tuning_dir}")
        print(f"{'=' * 60}\n")

        print("Proceeding with FULL TRAINING using best hyperparameters...\n")

    # ============================================================
    # Determine hyperparameters (from tuning or config)
    # ============================================================
    if best_hyperparams is not None:
        # Use best hyperparameters from tuning
        hidden_sizes = _extract_hidden_sizes(best_hyperparams)
        head_dropout = best_hyperparams['dropout']
        learning_rate = best_hyperparams['learning_rate']
        finetune_last_layers = best_hyperparams['finetune_last_layers']
        batch_size = best_hyperparams['batch_size']

        # Recreate dataloaders with best batch_size
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )

        # Reload backbone with fresh state for final training
        backbone.load_state_dict(backbone_state_dict)
    else:
        # Use default hyperparameters from config
        head_config = config['property_head']
        hidden_sizes = head_config['hidden_sizes']
        head_dropout = head_config['dropout']
        train_config = config['training_property']
        learning_rate = train_config['learning_rate']
        finetune_last_layers = train_config['finetune_last_layers']
        batch_size = train_config['batch_size']

    # Create property head
    print("\n4. Creating property head...")
    print(f"   hidden_sizes: {hidden_sizes}")
    print(f"   dropout: {head_dropout}")
    print(f"   learning_rate: {learning_rate}")
    print(f"   finetune_last_layers: {finetune_last_layers}")
    print(f"   batch_size: {batch_size}")

    property_head = PropertyHead(
        input_size=backbone_config['hidden_size'],
        hidden_sizes=hidden_sizes,
        dropout=head_dropout
    )

    # Create property predictor
    train_config = config['training_property']
    default_timestep = train_config.get('default_timestep', 1)
    model = PropertyPredictor(
        backbone=backbone,
        property_head=property_head,
        freeze_backbone=train_config['freeze_backbone'],
        finetune_last_layers=finetune_last_layers,
        pooling='mean',
        default_timestep=default_timestep
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")

    # Update config with tuned hyperparameters if tuning was performed
    if best_hyperparams is not None:
        config['training_property']['learning_rate'] = learning_rate
        config['training_property']['batch_size'] = batch_size
        config['property_head']['hidden_sizes'] = hidden_sizes
        config['property_head']['dropout'] = head_dropout

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
        step_dir=str(step_dir),
        # Pass tuned hyperparameters for checkpoint saving
        hidden_sizes=hidden_sizes,
        finetune_last_layers=finetune_last_layers,
        head_dropout=head_dropout,
        best_hyperparams=best_hyperparams
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

    # Get predictions for all splits
    train_preds, train_labels = trainer.get_predictions(train_loader)
    val_preds, val_labels = trainer.get_predictions(val_loader)
    test_metrics = history['test_metrics']

    # Compute train metrics
    train_mae = round(mean_absolute_error(train_labels, train_preds), 4)
    train_rmse = round(np.sqrt(mean_squared_error(train_labels, train_preds)), 4)
    train_r2 = round(r2_score(train_labels, train_preds), 4)

    # Compute val metrics
    val_mae = round(mean_absolute_error(val_labels, val_preds), 4)
    val_rmse = round(np.sqrt(mean_squared_error(val_labels, val_preds)), 4)
    val_r2 = round(r2_score(val_labels, val_preds), 4)

    # Parity plot for train
    plotter.parity_plot(
        y_true=train_labels,
        y_pred=train_preds,
        xlabel=f'True {args.property}',
        ylabel=f'Predicted {args.property}',
        title=f'{args.property} Parity Plot (Train)',
        save_path=figures_dir / f'{args.property}_parity_plot_train.png',
        metrics={'MAE': train_mae, 'RMSE': train_rmse, 'R²': train_r2}
    )

    # Parity plot for validation
    plotter.parity_plot(
        y_true=val_labels,
        y_pred=val_preds,
        xlabel=f'True {args.property}',
        ylabel=f'Predicted {args.property}',
        title=f'{args.property} Parity Plot (Validation)',
        save_path=figures_dir / f'{args.property}_parity_plot_val.png',
        metrics={'MAE': val_mae, 'RMSE': val_rmse, 'R²': val_r2}
    )

    # Parity plot for test
    plotter.parity_plot(
        y_true=np.array(test_metrics['labels']),
        y_pred=np.array(test_metrics['predictions']),
        xlabel=f'True {args.property}',
        ylabel=f'Predicted {args.property}',
        title=f'{args.property} Parity Plot (Test)',
        save_path=figures_dir / f'{args.property}_parity_plot_test.png',
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
    parser.add_argument('--model_size', type=str, default=None,
                        choices=['small', 'medium', 'large', 'xl'],
                        help='Model size preset (small: ~12M, medium: ~50M, large: ~150M, xl: ~400M)')
    parser.add_argument('--property', type=str, required=True,
                        help='Property name (e.g., Tg, Tm)')
    parser.add_argument('--backbone_checkpoint', type=str, default=None,
                        help='Path to backbone checkpoint')
    parser.add_argument('--tune', action='store_true',
                        help='Enable hyperparameter tuning with Optuna')
    parser.add_argument('--n_trials', type=int, default=None,
                        help='Override number of Optuna trials')
    args = parser.parse_args()
    main(args)
