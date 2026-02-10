#!/usr/bin/env python
"""Step 0: Prepare data, build vocabularies, and analyze graph statistics."""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import random
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.utils.config import load_config, save_config
from src.utils.plotting import PlotUtils
from src.data.data_loader import PolymerDataLoader
from src.data.graph_tokenizer import GraphTokenizer, build_atom_vocab_from_data
from src.utils.reproducibility import seed_everything, save_run_metadata
from shared.unlabeled_data import (
    require_preprocessed_unlabeled_splits,
)


def test_graph_invertibility(smiles_list, graph_tokenizer, max_samples=None):
    """Test graph tokenizer invertibility on SMILES list.

    Args:
        smiles_list: List of p-SMILES strings.
        graph_tokenizer: GraphTokenizer instance.
        max_samples: Maximum samples to test (None for all).

    Returns:
        Tuple of (success_count, failures_list).
    """
    if max_samples:
        test_smiles = smiles_list[:max_samples]
    else:
        test_smiles = smiles_list

    success = 0
    failures = []

    for smiles in tqdm(test_smiles, desc="Testing invertibility"):
        if graph_tokenizer.verify_roundtrip(smiles):
            success += 1
        else:
            failures.append(smiles)

    return success, failures


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

    print("=" * 60)
    print("Step 0: Data Preparation (Graph Diffusion)")
    print("=" * 60)

    # Load preprocessed shared unlabeled data
    print("\n1. Loading shared unlabeled train/val data...")
    train_shared_path, val_shared_path = require_preprocessed_unlabeled_splits(repo_root)
    train_df = pd.read_csv(train_shared_path)
    val_df = pd.read_csv(val_shared_path)
    print(f"Using shared train split: {train_shared_path}")
    print(f"Using shared val split: {val_shared_path}")

    print(f"   Train samples: {len(train_df)}")
    print(f"   Val samples: {len(val_df)}")

    # ========== GRAPH ANALYSIS ==========
    print("\n2. Analyzing graph statistics from training data...")

    train_smiles = train_df['p_smiles'].tolist()
    atom_vocab, graph_stats = build_atom_vocab_from_data(train_smiles)

    # Determine Nmax (100th percentile = max)
    Nmax = graph_stats['num_atoms']['p100']
    print(f"   Nmax (100th percentile): {Nmax}")
    print(f"   Atom vocabulary size: {len(atom_vocab)}")
    print(f"   Atom count range: {graph_stats['num_atoms']['min']}-{graph_stats['num_atoms']['max']}")
    print(f"   Atom count mean: {graph_stats['num_atoms']['mean']:.2f} +/- {graph_stats['num_atoms']['std']:.2f}")

    # Default edge vocabulary
    edge_vocab = {
        'NONE': 0,
        'SINGLE': 1,
        'DOUBLE': 2,
        'TRIPLE': 3,
        'AROMATIC': 4,
        'MASK': 5
    }
    print(f"   Edge vocabulary size: {len(edge_vocab)}")

    # Default stereochemistry vocabulary
    stereo_vocab = {
        'STEREONONE': 0,
        'STEREOANY': 1,
        'STEREOZ': 2,
        'STEREOE': 3,
        'STEREOCIS': 4,
        'STEREOTRANS': 5,
        'MASK': 6
    }
    print(f"   Stereo vocabulary size: {len(stereo_vocab)}")

    # Save graph configuration
    print("\n3. Saving graph configuration...")
    graph_config = {
        'Nmax': Nmax,
        'atom_vocab': atom_vocab,
        'edge_vocab': edge_vocab,
        'stereo_vocab': stereo_vocab,
        'atom_vocab_size': len(atom_vocab),
        'edge_vocab_size': len(edge_vocab),
        'stereo_vocab_size': len(stereo_vocab),
        'atom_count_distribution': graph_stats.get('atom_count_distribution', {})
    }

    graph_config_path = results_dir / 'graph_config.json'
    with open(graph_config_path, 'w') as f:
        json.dump(graph_config, f, indent=2)
    print(f"   Saved to: {graph_config_path}")

    # Save graph statistics
    stats_data = {
        'metric': [],
        'value': []
    }
    for key, val in graph_stats['num_atoms'].items():
        stats_data['metric'].append(f'num_atoms_{key}')
        stats_data['value'].append(val)
    for key, val in graph_stats['num_bonds'].items():
        stats_data['metric'].append(f'num_bonds_{key}')
        stats_data['value'].append(val)

    graph_stats_df = pd.DataFrame(stats_data)
    graph_stats_df.to_csv(metrics_dir / 'graph_statistics.csv', index=False)

    # Save atom type distribution
    atom_dist_df = pd.DataFrame([
        {'atom_type': k, 'count': v}
        for k, v in graph_stats['atom_type_distribution'].items()
    ]).sort_values('count', ascending=False)
    atom_dist_df.to_csv(metrics_dir / 'atom_type_distribution.csv', index=False)

    # Save bond type distribution
    bond_dist_df = pd.DataFrame([
        {'bond_type': k, 'count': v}
        for k, v in graph_stats['bond_type_distribution'].items()
    ]).sort_values('count', ascending=False)
    bond_dist_df.to_csv(metrics_dir / 'bond_type_distribution.csv', index=False)

    # ========== BUILD GRAPH TOKENIZER ==========
    print("\n4. Building graph tokenizer...")
    graph_tokenizer = GraphTokenizer(
        atom_vocab=atom_vocab,
        edge_vocab=edge_vocab,
        stereo_vocab=stereo_vocab,
        Nmax=Nmax
    )

    # Save tokenizer
    tokenizer_path = results_dir / 'graph_tokenizer.json'
    graph_tokenizer.save(tokenizer_path)
    print(f"   Saved to: {tokenizer_path}")

    # ========== TEST INVERTIBILITY ==========
    print("\n5. Testing graph tokenizer invertibility...")

    # Test on training data
    train_success, train_failures = test_graph_invertibility(
        train_smiles, graph_tokenizer
    )
    train_total = len(train_smiles)
    train_fail = train_total - train_success
    train_pct = 100 * train_success / len(train_smiles)
    print(f"   Train: {train_success}/{len(train_smiles)} ({train_pct:.2f}%)")

    # Test on validation data
    val_smiles = val_df['p_smiles'].tolist()
    val_success, val_failures = test_graph_invertibility(
        val_smiles, graph_tokenizer
    )
    val_total = len(val_smiles)
    val_fail = val_total - val_success
    val_pct = 100 * val_success / len(val_smiles)
    print(f"   Val: {val_success}/{len(val_smiles)} ({val_pct:.2f}%)")

    # Save standardized tokenizer roundtrip results
    roundtrip_df = pd.DataFrame({
        'split': ['train', 'val'],
        'total': [train_total, val_total],
        'valid': [train_success, val_success],
        'fail': [train_fail, val_fail],
        'pct': [train_pct, val_pct]
    })
    roundtrip_df.to_csv(metrics_dir / 'tokenizer_roundtrip.csv', index=False)

    # Save graph-specific invertibility results
    invert_df = pd.DataFrame({
        'split': ['train', 'val'],
        'total': [train_total, val_total],
        'success': [train_success, val_success],
        'valid': [train_success, val_success],
        'fail': [train_fail, val_fail],
        'pct': [train_pct, val_pct]
    })
    invert_df.to_csv(metrics_dir / 'graph_tokenizer_invertibility.csv', index=False)

    # Save failed examples (for debugging)
    failure_rows = (
        [{'split': 'train', 'smiles': s} for s in train_failures[:100]]
        + [{'split': 'val', 'smiles': s} for s in val_failures[:100]]
    )
    if failure_rows:
        failures_df = pd.DataFrame(failure_rows)
        failures_df.to_csv(metrics_dir / 'graph_tokenizer_failures.csv', index=False)
        failures_df.to_csv(metrics_dir / 'tokenizer_roundtrip_failures.csv', index=False)
        print(
            "   Saved failure examples for debugging "
            f"(train={min(len(train_failures), 100)}, val={min(len(val_failures), 100)})"
        )

    # ========== SAVE TOKENIZATION EXAMPLES ==========
    print("\n6. Saving tokenization examples...")
    random.seed(config['data']['random_seed'])
    sample_smiles = random.sample(train_smiles, min(10, len(train_smiles)))

    examples = []
    for smiles in sample_smiles:
        try:
            graph_data = graph_tokenizer.encode(smiles)
            reconstructed = graph_tokenizer.decode(
                graph_data['X'], graph_data['E'], graph_data['M']
            )
            num_atoms = int(graph_data['M'].sum())

            # Count stars
            star_count = np.sum(graph_data['X'][:num_atoms] == graph_tokenizer.star_id)

            # Count bonds
            E_upper = np.triu(graph_data['E'][:num_atoms, :num_atoms], k=1)
            bond_count = np.sum(E_upper > 0)

            examples.append({
                'original_smiles': smiles,
                'reconstructed_smiles': reconstructed,
                'num_atoms': num_atoms,
                'num_stars': int(star_count),
                'num_bonds': int(bond_count),
                # Strict match to avoid inflating roundtrip quality metrics.
                'match': smiles == reconstructed
            })
        except Exception as e:
            examples.append({
                'original_smiles': smiles,
                'reconstructed_smiles': None,
                'num_atoms': 0,
                'num_stars': 0,
                'num_bonds': 0,
                'match': False
            })

    examples_df = pd.DataFrame(examples)
    examples_df.to_csv(metrics_dir / 'graph_tokenizer_examples.csv', index=False)
    examples_df.to_csv(metrics_dir / 'tokenizer_examples.csv', index=False)

    # ========== COMPUTE ADDITIONAL STATISTICS ==========
    print("\n7. Computing additional statistics...")

    # Get atom counts for plotting
    atom_counts = []
    for smiles in tqdm(train_smiles, desc="Counting atoms"):
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                atom_counts.append(mol.GetNumAtoms())
        except:
            pass

    # SA score statistics
    train_sa = train_df['sa_score'].dropna().values
    val_sa = val_df['sa_score'].dropna().values

    # Standardized dataset-level statistics
    train_stats = data_loader.get_statistics(train_df)
    val_stats = data_loader.get_statistics(val_df)
    unlabeled_stats_df = pd.DataFrame([
        {'split': 'train', **train_stats},
        {'split': 'val', **val_stats}
    ])
    unlabeled_stats_df.to_csv(metrics_dir / 'unlabeled_data_stats.csv', index=False)

    # Standardized length statistics (p-SMILES length for cross-method comparability)
    train_lengths = train_df['p_smiles'].str.len().to_numpy()
    val_lengths = val_df['p_smiles'].str.len().to_numpy()
    length_stats_df = pd.DataFrame({
        'split': ['train', 'val'],
        'mean': [np.mean(train_lengths), np.mean(val_lengths)],
        'std': [np.std(train_lengths), np.std(val_lengths)],
        'min': [np.min(train_lengths), np.min(val_lengths)],
        'max': [np.max(train_lengths), np.max(val_lengths)],
        'p95': [np.percentile(train_lengths, 95), np.percentile(val_lengths, 95)],
        'p99': [np.percentile(train_lengths, 99), np.percentile(val_lengths, 99)],
    })
    length_stats_df.to_csv(metrics_dir / 'length_stats.csv', index=False)

    sa_stats = pd.DataFrame({
        'split': ['train', 'val'],
        'mean': [np.mean(train_sa), np.mean(val_sa)],
        'std': [np.std(train_sa), np.std(val_sa)],
        'min': [np.min(train_sa), np.min(val_sa)],
        'max': [np.max(train_sa), np.max(val_sa)]
    })
    sa_stats.to_csv(metrics_dir / 'sa_stats.csv', index=False)

    # ========== CREATE PLOTS ==========
    print("\n8. Creating plots...")
    plotter = PlotUtils(
        figure_size=tuple(config['plotting']['figure_size']),
        font_size=config['plotting']['font_size'],
        dpi=config['plotting']['dpi']
    )

    # Atom count histogram
    if atom_counts:
        plotter.histogram(
            data=[atom_counts],
            labels=['Train'],
            xlabel='Number of Atoms',
            ylabel='Count',
            title='Molecule Size Distribution',
            save_path=figures_dir / 'num_atoms_histogram.png',
            bins=50,
            style='step'
        )

    # Atom type bar plot
    if not atom_dist_df.empty:
        top_atoms = atom_dist_df.head(20)
        plotter.bar_chart(
            categories=top_atoms['atom_type'].tolist(),
            values=top_atoms['count'].tolist(),
            xlabel='Atom Type',
            ylabel='Count',
            title='Top 20 Atom Types',
            save_path=figures_dir / 'atom_type_barplot.png'
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

    # ========== SHARED DATA PATHS ==========
    print("\n9. Using shared split files directly...")
    print(f"  Train split: {train_shared_path}")
    print(f"  Val split: {val_shared_path}")

    # ========== SUMMARY ==========
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"\nDataset:")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    print(f"\nGraph Configuration:")
    print(f"  Nmax (max atoms): {Nmax}")
    print(f"  Atom vocab size: {len(atom_vocab)}")
    print(f"  Edge vocab size: {len(edge_vocab)}")
    print(f"\nInvertibility:")
    print(f"  Train: {train_pct:.2f}%")
    print(f"  Val: {val_pct:.2f}%")
    print(f"\nOutput files:")
    print(f"  Graph config: {graph_config_path}")
    print(f"  Graph tokenizer: {tokenizer_path}")
    print(f"  Metrics: {metrics_dir}")
    print(f"  Figures: {figures_dir}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data and build graph vocabulary')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    main(args)
