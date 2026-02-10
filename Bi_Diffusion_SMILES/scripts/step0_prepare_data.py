#!/usr/bin/env python
"""Step 0: Prepare data and build vocabulary."""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import random
import pandas as pd
import numpy as np

from src.utils.config import load_config, save_config
from src.utils.plotting import PlotUtils
from src.data.data_loader import PolymerDataLoader
from src.data.tokenizer import PSmilesTokenizer
from src.utils.reproducibility import seed_everything, save_run_metadata
from shared.unlabeled_data import (
    require_preprocessed_unlabeled_splits,
)


def main(args):
    """Main function."""
    # Load config
    config = load_config(args.config)

    # Create output directories
    results_dir = Path(config['paths']['results_dir'])
    step_dir = results_dir / 'step0_data_prep'
    metrics_dir = step_dir / 'metrics'
    figures_dir = step_dir / 'figures'
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    seed_info = seed_everything(config['data']['random_seed'])
    save_config(config, step_dir / 'config_used.yaml')
    save_run_metadata(step_dir, args.config, seed_info)

    # Initialize data loader
    data_loader = PolymerDataLoader(config)

    print("=" * 50)
    print("Step 0: Data Preparation")
    print("=" * 50)

    # Load preprocessed shared unlabeled data
    print("\n1. Loading shared unlabeled train/val data...")
    train_shared_path, val_shared_path = require_preprocessed_unlabeled_splits(repo_root)
    train_df = pd.read_csv(train_shared_path)
    val_df = pd.read_csv(val_shared_path)
    print(f"Using shared train split: {train_shared_path}")
    print(f"Using shared val split: {val_shared_path}")

    # Build tokenizer vocabulary from training data only
    print("\n2. Building tokenizer vocabulary...")
    tokenizer = PSmilesTokenizer(max_length=config['tokenizer']['max_length'])
    vocab = tokenizer.build_vocab(train_df['p_smiles'].tolist())
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Save tokenizer
    tokenizer_path = results_dir / 'tokenizer.json'
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to: {tokenizer_path}")

    # Verify round-trip invertibility
    print("\n3. Verifying tokenization invertibility...")
    train_valid = 0
    train_total = len(train_df)
    for smiles in train_df['p_smiles']:
        if tokenizer.verify_roundtrip(smiles):
            train_valid += 1

    val_valid = 0
    val_total = len(val_df)
    for smiles in val_df['p_smiles']:
        if tokenizer.verify_roundtrip(smiles):
            val_valid += 1

    train_fail = train_total - train_valid
    val_fail = val_total - val_valid
    print(f"Train roundtrip: {train_valid}/{train_total} ({100*train_valid/train_total:.2f}%)")
    print(f"Val roundtrip: {val_valid}/{val_total} ({100*val_valid/val_total:.2f}%)")

    # Save roundtrip results
    roundtrip_df = pd.DataFrame({
        'split': ['train', 'val'],
        'total': [train_total, val_total],
        'valid': [train_valid, val_valid],
        'fail': [train_fail, val_fail],
        'pct': [100*train_valid/train_total, 100*val_valid/val_total]
    })
    roundtrip_df.to_csv(metrics_dir / 'tokenizer_roundtrip.csv', index=False)

    # Save 10 example roundtrips for demonstration
    print("   Saving tokenization examples...")
    random.seed(config['data']['random_seed'])
    sample_smiles = random.sample(train_df['p_smiles'].tolist(), min(10, len(train_df)))

    examples = []
    for smiles in sample_smiles:
        tokens = tokenizer.tokenize(smiles)
        # Create token -> vocab ID hashmap
        token_ids = {tok: tokenizer.vocab.get(tok, tokenizer.unk_token_id) for tok in tokens}
        reconstructed = tokenizer.detokenize(tokens)
        examples.append({
            'original_smiles': smiles,
            'num_tokens': len(tokens),
            'tokens_hashmap': str(token_ids),
            'reconstructed_smiles': reconstructed
        })

    examples_df = pd.DataFrame(examples)
    examples_df.to_csv(metrics_dir / 'tokenizer_examples.csv', index=False)

    # Compute statistics
    print("\n4. Computing statistics...")
    train_stats = data_loader.get_statistics(train_df)
    val_stats = data_loader.get_statistics(val_df)

    # Save statistics
    stats_df = pd.DataFrame([
        {'split': 'train', **train_stats},
        {'split': 'val', **val_stats}
    ])
    stats_df.to_csv(metrics_dir / 'unlabeled_data_stats.csv', index=False)

    # Compute token lengths
    print("\n5. Computing token length distributions...")
    train_lengths = [len(tokenizer.tokenize(s)) for s in train_df['p_smiles']]
    val_lengths = [len(tokenizer.tokenize(s)) for s in val_df['p_smiles']]

    # Length statistics
    length_stats = pd.DataFrame({
        'split': ['train', 'val'],
        'mean': [np.mean(train_lengths), np.mean(val_lengths)],
        'std': [np.std(train_lengths), np.std(val_lengths)],
        'min': [np.min(train_lengths), np.min(val_lengths)],
        'max': [np.max(train_lengths), np.max(val_lengths)],
        'p95': [np.percentile(train_lengths, 95), np.percentile(val_lengths, 95)],
        'p99': [np.percentile(train_lengths, 99), np.percentile(val_lengths, 99)]
    })
    length_stats.to_csv(metrics_dir / 'length_stats.csv', index=False)

    # SA score statistics
    train_sa = train_df['sa_score'].dropna().values
    val_sa = val_df['sa_score'].dropna().values

    sa_stats = pd.DataFrame({
        'split': ['train', 'val'],
        'mean': [np.mean(train_sa), np.mean(val_sa)],
        'std': [np.std(train_sa), np.std(val_sa)],
        'min': [np.min(train_sa), np.min(val_sa)],
        'max': [np.max(train_sa), np.max(val_sa)]
    })
    sa_stats.to_csv(metrics_dir / 'sa_stats.csv', index=False)

    # Create plots
    print("\n6. Creating plots...")
    plotter = PlotUtils(
        figure_size=tuple(config['plotting']['figure_size']),
        font_size=config['plotting']['font_size'],
        dpi=config['plotting']['dpi']
    )

    # Length histogram
    plotter.histogram(
        data=[train_lengths, val_lengths],
        labels=['Train', 'Val'],
        xlabel='Token Length',
        ylabel='Count',
        title='Token Length Distribution',
        save_path=figures_dir / 'length_hist_train_val.png',
        bins=50,
        style='step'
    )

    # SA score histogram
    plotter.histogram(
        data=[train_sa, val_sa],
        labels=['Train', 'Val'],
        xlabel='SA Score',
        ylabel='Count',
        title='SA Score Distribution',
        save_path=figures_dir / 'sa_hist_train_val.png',
        bins=50,
        style='step'
    )

    print("\n7. Using shared split files directly...")
    print(f"  Train split: {train_shared_path}")
    print(f"  Val split: {val_shared_path}")

    print("\n" + "=" * 50)
    print("Data preparation complete!")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data and build vocabulary')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    main(args)
