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
from src.data.selfies_tokenizer import SelfiesTokenizer
from src.utils.reproducibility import seed_everything, save_run_metadata
from src.utils.selfies_utils import (
    verify_roundtrip,
    count_placeholder_in_selfies,
    ensure_selfies_column,
)
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

    # Load preprocessed shared unlabeled data, then derive SELFIES view.
    print("\n1. Loading shared unlabeled train/val data...")
    train_shared_path, val_shared_path = require_preprocessed_unlabeled_splits(repo_root)
    train_df = pd.read_csv(train_shared_path)
    val_df = pd.read_csv(val_shared_path)
    print(f"Using shared train split: {train_shared_path}")
    print(f"Using shared val split: {val_shared_path}")

    print("   Converting shared p-SMILES to SELFIES view for tokenizer/statistics...")
    train_df = ensure_selfies_column(train_df)
    val_df = ensure_selfies_column(val_df)

    # Build SELFIES tokenizer vocabulary from training data only
    print("\n2. Building SELFIES tokenizer vocabulary...")
    tokenizer = SelfiesTokenizer(max_length=config['tokenizer']['max_length'])
    vocab = tokenizer.build_vocab(train_df['selfies'].tolist())
    print(f"SELFIES vocabulary size: {tokenizer.vocab_size}")
    print(f"  (p-SMILES vocab was typically ~50-100, SELFIES is larger due to bracket notation)")

    # Save tokenizer
    tokenizer_path = results_dir / 'tokenizer.json'
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to: {tokenizer_path}")

    # Verify SELFIES tokenization round-trip invertibility
    print("\n3. Verifying SELFIES tokenization invertibility...")
    train_valid = 0
    train_total = len(train_df)
    for selfies in train_df['selfies']:
        if tokenizer.verify_roundtrip(selfies):
            train_valid += 1

    val_valid = 0
    val_total = len(val_df)
    for selfies in val_df['selfies']:
        if tokenizer.verify_roundtrip(selfies):
            val_valid += 1

    train_fail = train_total - train_valid
    val_fail = val_total - val_valid
    print(f"Train SELFIES tokenization roundtrip: {train_valid}/{train_total} ({100*train_valid/train_total:.2f}%)")
    print(f"Val SELFIES tokenization roundtrip: {val_valid}/{val_total} ({100*val_valid/val_total:.2f}%)")

    # Verify p-SMILES <-> SELFIES conversion round-trip (sample of 100)
    print("\n4. Verifying p-SMILES <-> SELFIES conversion roundtrip...")
    sample_size = min(100, len(train_df))
    conversion_success = 0
    placeholder_count_errors = 0

    for idx in range(sample_size):
        success, selfies, recon_psmiles = verify_roundtrip(train_df.iloc[idx]['p_smiles'])
        if success:
            conversion_success += 1
        # Also check placeholder count in SELFIES
        if count_placeholder_in_selfies(train_df.iloc[idx]['selfies']) != 2:
            placeholder_count_errors += 1

    print(f"p-SMILES <-> SELFIES conversion roundtrip: {conversion_success}/{sample_size} ({100*conversion_success/sample_size:.2f}%)")
    print(f"SELFIES with exactly 2 [I+3] placeholders: {sample_size - placeholder_count_errors}/{sample_size} ({100*(sample_size - placeholder_count_errors)/sample_size:.2f}%)")

    # Save standardized tokenizer roundtrip results
    roundtrip_df = pd.DataFrame({
        'split': ['train', 'val'],
        'total': [train_total, val_total],
        'valid': [train_valid, val_valid],
        'fail': [train_fail, val_fail],
        'pct': [100*train_valid/train_total, 100*val_valid/val_total]
    })
    roundtrip_df.to_csv(metrics_dir / 'tokenizer_roundtrip.csv', index=False)

    # Save SELFIES conversion diagnostics separately
    conversion_df = pd.DataFrame({
        'test': ['p-SMILES <-> SELFIES conversion', 'Placeholder count validation'],
        'total': [sample_size, sample_size],
        'valid': [conversion_success, sample_size - placeholder_count_errors],
        'fail': [sample_size - conversion_success, placeholder_count_errors],
        'pct': [100*conversion_success/sample_size, 100*(sample_size - placeholder_count_errors)/sample_size]
    })
    conversion_df.to_csv(metrics_dir / 'selfies_conversion_roundtrip.csv', index=False)

    # Save 10 example roundtrips for demonstration
    print("\n5. Saving tokenization examples...")
    random.seed(config['data']['random_seed'])
    sample_indices = random.sample(range(len(train_df)), min(10, len(train_df)))

    examples = []
    for idx in sample_indices:
        psmiles = train_df.iloc[idx]['p_smiles']
        selfies = train_df.iloc[idx]['selfies']
        tokens = tokenizer.tokenize(selfies)
        # Create token -> vocab ID hashmap (show first 20 tokens to avoid overflow)
        token_ids = {tok: tokenizer.vocab.get(tok, tokenizer.unk_token_id) for tok in tokens[:20]}
        reconstructed = tokenizer.detokenize(tokens)
        examples.append({
            'original_psmiles': psmiles,
            'selfies': selfies,
            'num_tokens': len(tokens),
            'first_20_tokens': str(tokens[:20]),
            'token_ids_sample': str(token_ids),
            'reconstructed_selfies': reconstructed,
            'roundtrip_match': (reconstructed == selfies)
        })

    examples_df = pd.DataFrame(examples)
    examples_df.to_csv(metrics_dir / 'tokenizer_examples.csv', index=False)

    # Compute statistics
    print("\n6. Computing statistics...")
    train_stats = data_loader.get_statistics(train_df)
    val_stats = data_loader.get_statistics(val_df)

    # Save statistics
    stats_df = pd.DataFrame([
        {'split': 'train', **train_stats},
        {'split': 'val', **val_stats}
    ])
    stats_df.to_csv(metrics_dir / 'unlabeled_data_stats.csv', index=False)

    # Compute SELFIES token lengths
    print("\n7. Computing SELFIES token length distributions...")
    train_lengths = [len(tokenizer.tokenize(s)) for s in train_df['selfies']]
    val_lengths = [len(tokenizer.tokenize(s)) for s in val_df['selfies']]

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

    print(f"  Mean token length: train={np.mean(train_lengths):.1f}, val={np.mean(val_lengths):.1f}")
    print(f"  Max token length: train={np.max(train_lengths)}, val={np.max(val_lengths)}")
    print(f"  99th percentile: train={np.percentile(train_lengths, 99):.1f}, val={np.percentile(val_lengths, 99):.1f}")
    if np.max(train_lengths) > config['tokenizer']['max_length']:
        print(f"  WARNING: Some sequences exceed max_length={config['tokenizer']['max_length']}!")
        exceeds = sum(1 for l in train_lengths if l > config['tokenizer']['max_length'])
        print(f"  {exceeds}/{len(train_lengths)} sequences will be truncated")

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
    print("\n8. Creating plots...")
    plotter = PlotUtils(
        figure_size=tuple(config['plotting']['figure_size']),
        font_size=config['plotting']['font_size'],
        dpi=config['plotting']['dpi']
    )

    # SELFIES token length histogram
    plotter.histogram(
        data=[train_lengths, val_lengths],
        labels=['Train', 'Val'],
        xlabel='SELFIES Token Length',
        ylabel='Count',
        title='SELFIES Token Length Distribution',
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

    print("\n9. Using shared split files directly...")
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
