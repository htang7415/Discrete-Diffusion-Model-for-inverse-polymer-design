#!/usr/bin/env python
"""Step 0: Prepare data and build Group SELFIES vocabulary and grammar."""

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
from src.data.tokenizer import GroupSELFIESTokenizer
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
    print("Step 0: Data Preparation (Group SELFIES)")
    print("=" * 50)

    # Load preprocessed shared unlabeled data
    print("\n1. Loading shared unlabeled train/val data...")
    train_shared_path, val_shared_path = require_preprocessed_unlabeled_splits(repo_root)
    train_df = pd.read_csv(train_shared_path)
    val_df = pd.read_csv(val_shared_path)
    print(f"Using shared train split: {train_shared_path}")
    print(f"Using shared val split: {val_shared_path}")

    # Build tokenizer vocabulary and grammar from training data only
    print("\n2. Building Group SELFIES grammar and vocabulary...")
    tokenizer = GroupSELFIESTokenizer(max_length=config['tokenizer']['max_length'])

    # Get group_selfies config
    gs_config = config.get('group_selfies', {})
    max_groups = gs_config.get('max_groups', 20000)
    grammar_sample_size = gs_config.get('grammar_sample_size', 0)  # 0 = use all

    # Get parallelization settings
    parallel_config = gs_config.get('parallel', {})
    num_workers = parallel_config.get('num_workers', 1)
    chunk_size = parallel_config.get('chunk_size', 1000)
    parallel_enabled = parallel_config.get('enabled', False)

    # Auto-detect CPU count if num_workers is 0
    if num_workers == 0:
        import multiprocessing
        num_workers = multiprocessing.cpu_count()
        print(f"Auto-detected {num_workers} CPU cores")

    # Disable parallelization if not enabled
    if not parallel_enabled:
        num_workers = 1
        print("Parallelization disabled in config")

    # Sample for grammar building if configured (much faster for large datasets)
    # IMPORTANT: We sample ONCE and split into grammar and test sets to ensure
    # the test molecules share the same structural distribution as the grammar molecules.
    all_train_smiles = train_df['p_smiles'].tolist()

    if grammar_sample_size > 0 and grammar_sample_size < len(all_train_smiles):
        # Sample more than needed so we can split into grammar + test sets
        roundtrip_test_size = gs_config.get('roundtrip_test_size', 0)
        total_sample_size = grammar_sample_size + roundtrip_test_size

        if total_sample_size > len(all_train_smiles):
            total_sample_size = len(all_train_smiles)

        print(f"   Sampling {total_sample_size:,} molecules total (from {len(all_train_smiles):,})")
        random.seed(config['data']['random_seed'])
        total_sample = random.sample(all_train_smiles, total_sample_size)

        # Split: first part for grammar, rest for roundtrip testing
        grammar_smiles = total_sample[:grammar_sample_size]
        train_roundtrip_sample = total_sample[grammar_sample_size:]  # Held-out for testing

        print(f"   Grammar sample: {len(grammar_smiles):,}")
        print(f"   Held-out for roundtrip testing: {len(train_roundtrip_sample):,}")
    else:
        grammar_smiles = all_train_smiles
        train_roundtrip_sample = None  # Will sample separately
        print(f"   Using all {len(grammar_smiles):,} molecules for grammar building")

    vocab, grammar = tokenizer.build_vocab_and_grammar(
        grammar_smiles,
        max_groups=max_groups,
        num_workers=num_workers,
        chunk_size=chunk_size,
        verbose=True
    )
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Placeholder token: {tokenizer.get_placeholder_token()}")
    print(f"Placeholder token ID: {tokenizer.get_placeholder_token_id()}")

    # Save tokenizer (pickle format for grammar)
    tokenizer_path = results_dir / 'tokenizer.pkl'
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to: {tokenizer_path}")

    # Verify round-trip invertibility (PARALLELIZED with sampling)
    print("\n3. Verifying tokenization invertibility...")

    # Get roundtrip test size from config
    roundtrip_test_size = gs_config.get('roundtrip_test_size', 0)  # 0 = use all

    # Use the held-out sample from grammar building (same distribution)
    # This is crucial: testing on molecules from a DIFFERENT random sample
    # causes 0.01% accuracy because they have different structural diversity.
    val_smiles_for_test = val_df['p_smiles'].tolist()

    if train_roundtrip_sample is not None and len(train_roundtrip_sample) > 0:
        # Use the held-out sample from the same random draw as grammar
        train_smiles_for_test = train_roundtrip_sample
        print(f"   Using {len(train_smiles_for_test):,} held-out molecules for train roundtrip (same distribution as grammar)")
    elif roundtrip_test_size > 0:
        # Fallback: sample from all training data
        train_smiles_for_test = train_df['p_smiles'].tolist()
        if roundtrip_test_size < len(train_smiles_for_test):
            print(f"   Sampling {roundtrip_test_size:,} molecules for train roundtrip (from {len(train_smiles_for_test):,})")
            random.seed(config['data']['random_seed'] + 100)
            train_smiles_for_test = random.sample(train_smiles_for_test, roundtrip_test_size)
    else:
        train_smiles_for_test = train_df['p_smiles'].tolist()

    # Sample validation set
    if roundtrip_test_size > 0:
        val_test_size = min(roundtrip_test_size // 10, len(val_smiles_for_test))  # 10% of train test size for val
        if val_test_size < len(val_smiles_for_test):
            print(f"   Sampling {val_test_size:,} molecules for val roundtrip (from {len(val_smiles_for_test):,})")
            random.seed(config['data']['random_seed'] + 200)
            val_smiles_for_test = random.sample(val_smiles_for_test, val_test_size)

    # Verify train set (parallel now works - grammar recreated in each worker)
    train_valid, train_total, train_failures = tokenizer.parallel_verify_roundtrip(
        train_smiles_for_test,
        num_workers=num_workers,  # Parallel works: workers recreate grammar from group_smiles
        chunk_size=chunk_size,
        verbose=True
    )

    # Verify validation set
    val_valid, val_total, val_failures = tokenizer.parallel_verify_roundtrip(
        val_smiles_for_test,
        num_workers=num_workers,  # Parallel works: workers recreate grammar from group_smiles
        chunk_size=chunk_size,
        verbose=True
    )

    train_fail = train_total - train_valid
    val_fail = val_total - val_valid

    # Save roundtrip results
    roundtrip_df = pd.DataFrame({
        'split': ['train', 'val'],
        'total': [train_total, val_total],
        'valid': [train_valid, val_valid],
        'fail': [train_fail, val_fail],
        'pct': [100*train_valid/train_total, 100*val_valid/val_total]
    })
    roundtrip_df.to_csv(metrics_dir / 'tokenizer_roundtrip.csv', index=False)

    # Save failure diagnostics for quick inspection
    failure_rows = []
    for split, failures in (('train', train_failures), ('val', val_failures)):
        for failure in failures:
            if isinstance(failure, dict):
                row = {'split': split, **failure}
            else:
                row = {'split': split, 'smiles': str(failure), 'error': 'unknown'}
            failure_rows.append(row)
    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(
            metrics_dir / 'tokenizer_roundtrip_failures.csv', index=False
        )

    # Save 10 example roundtrips for demonstration
    print("   Saving tokenization examples...")
    random.seed(config['data']['random_seed'])
    sample_smiles = random.sample(train_df['p_smiles'].tolist(), min(10, len(train_df)))

    examples = []
    for smiles in sample_smiles:
        tokens = tokenizer.tokenize(smiles)
        # Create token -> vocab ID hashmap
        token_ids = {tok: tokenizer.vocab.get(tok, tokenizer.unk_token_id) for tok in tokens}
        decoded = tokenizer.detokenize(tokens)
        examples.append({
            'original_smiles': smiles,
            'num_tokens': len(tokens),
            'group_selfies_tokens': str(tokens),
            'tokens_hashmap': str(token_ids),
            'decoded_smiles': decoded,
            'roundtrip_match': tokenizer.verify_roundtrip(smiles)
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

    # Compute token lengths (PARALLELIZED)
    print("\n5. Computing token length distributions...")
    train_lengths = tokenizer.parallel_get_lengths(
        train_df['p_smiles'].tolist(),
        num_workers=num_workers,
        chunk_size=chunk_size,
        verbose=True
    )
    val_lengths = tokenizer.parallel_get_lengths(
        val_df['p_smiles'].tolist(),
        num_workers=num_workers,
        chunk_size=chunk_size,
        verbose=True
    )

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
        xlabel='Token Length (Group SELFIES)',
        ylabel='Count',
        title='Group SELFIES Token Length Distribution',
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

    # Print summary
    print("\n" + "=" * 50)
    print("Data preparation complete!")
    print("=" * 50)
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Placeholder token: {tokenizer.get_placeholder_token()}")
    print(f"Train roundtrip accuracy: {100*train_valid/train_total:.2f}%")
    print(f"Val roundtrip accuracy: {100*val_valid/val_total:.2f}%")
    print(f"Avg train token length: {np.mean(train_lengths):.2f}")
    print(f"Avg val token length: {np.mean(val_lengths):.2f}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data and build Group SELFIES vocabulary')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    main(args)
