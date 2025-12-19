#!/usr/bin/env python
"""Step 6: Hyperparameter tuning (optional)."""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd

from src.utils.config import load_config, save_config
from src.data.selfies_tokenizer import SelfiesTokenizer
from src.data.data_loader import PolymerDataLoader
from src.data.dataset import PolymerDataset, PropertyDataset
from src.model.backbone import DiffusionBackbone
from src.model.diffusion import DiscreteMaskingDiffusion
from src.training.hyperparameter_tuning import BackboneTuner, PropertyHeadTuner
from src.utils.reproducibility import seed_everything, save_run_metadata


def tune_backbone(args, config, device):
    """Tune backbone hyperparameters."""
    results_dir = Path(config['paths']['results_dir'])
    step_dir = results_dir / 'step6_tuning'
    step_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = SelfiesTokenizer.load(results_dir / 'tokenizer.json')

    # Load data
    train_df = pd.read_csv(results_dir / 'train_unlabeled.csv')
    val_df = pd.read_csv(results_dir / 'val_unlabeled.csv')

    # Create datasets
    train_dataset = PolymerDataset(train_df, tokenizer)
    val_dataset = PolymerDataset(val_df, tokenizer)

    # Create tuner
    tuner = BackboneTuner(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        config=config,
        device=device,
        output_dir=str(step_dir)
    )

    # Define parameter grid
    param_grid = {
        'learning_rate': [3e-5, 1e-4, 3e-4, 5e-4],
        'warmup_steps': [500, 1000, 2000],
        'dropout': [0.0, 0.1, 0.2],
    }

    # Run tuning
    results = tuner.tune_backbone(
        param_grid=param_grid,
        num_steps=args.num_steps,
        batch_size=args.batch_size
    )

    print("\nTop 5 configurations:")
    print(results.head(5).to_string())

    return results


def tune_property_head(args, config, device):
    """Tune property head hyperparameters."""
    results_dir = Path(config['paths']['results_dir'])
    step_dir = results_dir / 'step6_tuning'
    step_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = SelfiesTokenizer.load(results_dir / 'tokenizer.json')

    # Load property data
    data_loader = PolymerDataLoader(config)
    property_data = data_loader.prepare_property_data(args.property)

    train_df = property_data['train']
    val_df = property_data['val']

    # Normalization
    mean = train_df[args.property].mean()
    std = train_df[args.property].std()

    # Create datasets
    train_dataset = PropertyDataset(
        train_df, tokenizer, args.property,
        normalize=True, mean=mean, std=std
    )
    val_dataset = PropertyDataset(
        val_df, tokenizer, args.property,
        normalize=True, mean=mean, std=std
    )

    # Load backbone
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

    diffusion_model = DiscreteMaskingDiffusion(
        backbone=backbone,
        num_steps=config['diffusion']['num_steps'],
        beta_min=config['diffusion']['beta_min'],
        beta_max=config['diffusion']['beta_max'],
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    backbone_ckpt = torch.load(results_dir / 'checkpoints' / 'backbone_best.pt', map_location=device)
    # Handle torch.compile() state dict (keys have _orig_mod. prefix)
    state_dict = backbone_ckpt['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    diffusion_model.load_state_dict(state_dict)
    diffusion_model = diffusion_model.to(device)

    # Create tuner
    tuner = PropertyHeadTuner(
        backbone=diffusion_model.backbone,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
        output_dir=str(step_dir)
    )

    # Define parameter grid
    param_grid = {
        'learning_rate': [1e-5, 3e-5, 1e-4, 3e-4],
        'hidden_sizes': [[256, 64], [128], [512, 128]],
        'dropout': [0.0, 0.1],
    }

    # Run tuning
    results = tuner.tune_property_head(
        property_name=args.property,
        param_grid=param_grid,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )

    print("\nTop 5 configurations:")
    print(results.head(5).to_string())

    return results


def main(args):
    """Main function."""
    # Load config
    config = load_config(args.config)

    results_dir = Path(config['paths']['results_dir'])
    step_dir = results_dir / 'step6_tuning'
    step_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    seed_info = seed_everything(config['data']['random_seed'])
    save_config(config, step_dir / 'config_used.yaml')
    save_run_metadata(step_dir, args.config, seed_info)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("=" * 50)
    print("Hyperparameter Tuning")
    print("=" * 50)

    if args.mode == 'backbone':
        print("\nTuning backbone hyperparameters...")
        results = tune_backbone(args, config, device)
    elif args.mode == 'property':
        if not args.property:
            raise ValueError("--property required for property head tuning")
        print(f"\nTuning property head for {args.property}...")
        results = tune_property_head(args, config, device)
    else:
        raise ValueError(f"Unknown tuning mode: {args.mode}")

    print("\n" + "=" * 50)
    print("Hyperparameter tuning complete!")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter tuning')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--mode', type=str, required=True, choices=['backbone', 'property'],
                        help='Tuning mode')
    parser.add_argument('--property', type=str, default=None,
                        help='Property name for property head tuning')
    parser.add_argument('--num_steps', type=int, default=5000,
                        help='Training steps per backbone trial')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Epochs per property head trial')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    args = parser.parse_args()
    main(args)
