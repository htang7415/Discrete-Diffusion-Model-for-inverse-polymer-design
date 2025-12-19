#!/usr/bin/env python
"""Step 2: Sample from graph backbone and evaluate generative metrics."""

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
from collections import Counter

from src.utils.config import load_config, save_config
from src.utils.plotting import PlotUtils
from src.utils.chemistry import compute_sa_score, count_stars
from src.data.graph_tokenizer import GraphTokenizer
from src.model.graph_backbone import GraphDiffusionBackbone
from src.model.graph_diffusion import GraphMaskingDiffusion
from src.sampling.graph_sampler import GraphSampler
from src.evaluation.generative_metrics import GenerativeEvaluator
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
    step_dir = results_dir / 'step2_sampling'
    metrics_dir = step_dir / 'metrics'
    figures_dir = step_dir / 'figures'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    seed_info = seed_everything(config['data']['random_seed'])
    save_config(config, step_dir / 'config_used.yaml')
    save_run_metadata(step_dir, args.config, seed_info)

    print("=" * 60)
    print("Step 2: Graph Sampling and Generative Evaluation")
    print("=" * 60)

    # Load graph configuration
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

    # Load training data for novelty computation
    print("\n3. Loading training data...")
    train_df = pd.read_csv(results_dir / 'train_unlabeled.csv')
    training_smiles = set(train_df['p_smiles'].tolist())
    print(f"   Training set size: {len(training_smiles)}")

    # Load model
    print("\n4. Loading model...")
    checkpoint_path = args.checkpoint or (results_dir / 'checkpoints' / 'graph_backbone_best.pt')
    print(f"   Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

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

    # Handle torch.compile() state dict
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Create graph sampler
    print("\n5. Creating graph sampler...")
    sampler = GraphSampler(
        backbone=model.backbone,
        graph_tokenizer=graph_tokenizer,
        num_steps=diffusion_config['num_steps'],
        device=device,
        atom_count_distribution=graph_config.get('atom_count_distribution')
    )

    # Sample
    num_samples = args.num_samples or config['sampling']['num_samples']
    batch_size = args.batch_size or config['sampling']['batch_size']
    temperature = args.temperature or config['sampling']['temperature']

    print(f"\n6. Sampling {num_samples} polymers (batch_size={batch_size}, temp={temperature})...")

    generated_smiles = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        current_batch = min(batch_size, num_samples - len(generated_smiles))
        batch_smiles = sampler.sample(
            batch_size=current_batch,
            temperature=temperature,
            show_progress=(i == 0)  # Only show progress for first batch
        )
        generated_smiles.extend(batch_smiles)

        if (i + 1) % 10 == 0:
            print(f"   Sampled {len(generated_smiles)}/{num_samples}")

    print(f"   Total sampled: {len(generated_smiles)}")

    # Filter out None values
    valid_generated = [s for s in generated_smiles if s is not None]
    print(f"   Successfully decoded: {len(valid_generated)}/{len(generated_smiles)}")

    # Save generated samples
    samples_df = pd.DataFrame({'smiles': generated_smiles})
    samples_df.to_csv(metrics_dir / 'generated_samples.csv', index=False)
    print(f"   Saved generated samples to: {metrics_dir / 'generated_samples.csv'}")

    # Evaluate
    print("\n7. Evaluating generative metrics...")
    evaluator = GenerativeEvaluator(training_smiles)
    metrics = evaluator.evaluate(
        generated_smiles,
        sample_id=f'graph_uncond_{num_samples}',
        show_progress=True
    )

    # Save metrics
    metrics_csv = evaluator.format_metrics_csv(metrics)
    metrics_csv.to_csv(metrics_dir / 'sampling_generative_metrics.csv', index=False)

    # Print metrics
    print("\n   Generative Metrics:")
    print(f"   Validity: {metrics['validity']:.4f}")
    print(f"   Uniqueness: {metrics['uniqueness']:.4f}")
    print(f"   Novelty: {metrics['novelty']:.4f}")
    print(f"   Diversity: {metrics['avg_diversity']:.4f}")
    print(f"   Frac star=2: {metrics['frac_star_eq_2']:.4f}")
    print(f"   Mean SA: {metrics['mean_sa']:.4f}")
    print(f"   Std SA: {metrics['std_sa']:.4f}")

    # Create plots
    print("\n8. Creating plots...")
    plotter = PlotUtils(
        figure_size=tuple(config['plotting']['figure_size']),
        font_size=config['plotting']['font_size'],
        dpi=config['plotting']['dpi']
    )

    # Get valid samples with exactly 2 stars
    valid_smiles = evaluator.get_valid_samples(generated_smiles, require_two_stars=True)
    print(f"   Valid with 2 stars: {len(valid_smiles)}")

    # SA histogram: train vs generated
    print("   Creating SA histogram...")
    train_sa = [compute_sa_score(s) for s in list(training_smiles)[:5000]]
    train_sa = [s for s in train_sa if s is not None]
    gen_sa = [compute_sa_score(s) for s in valid_smiles[:5000]]
    gen_sa = [s for s in gen_sa if s is not None]

    if train_sa and gen_sa:
        plotter.histogram(
            data=[train_sa, gen_sa],
            labels=['Train', 'Generated'],
            xlabel='SA Score',
            ylabel='Count',
            title='SA Score: Train vs Generated',
            save_path=figures_dir / 'sa_hist_train_vs_graph.png',
            bins=50,
            style='step'
        )

    # Length histogram: train vs generated
    print("   Creating length histogram...")
    train_lengths = [len(s) for s in list(training_smiles)[:5000]]
    gen_lengths = [len(s) for s in valid_smiles[:5000] if s]

    if train_lengths and gen_lengths:
        plotter.histogram(
            data=[train_lengths, gen_lengths],
            labels=['Train', 'Generated'],
            xlabel='SMILES Length',
            ylabel='Count',
            title='Length: Train vs Generated',
            save_path=figures_dir / 'length_hist_train_vs_graph.png',
            bins=50,
            style='step'
        )

    # Star count histogram
    print("   Creating star count histogram...")
    star_counts = [count_stars(s) for s in valid_smiles if s]
    star_counter = Counter(star_counts)

    if star_counter:
        plotter.bar_chart(
            categories=[str(k) for k in sorted(star_counter.keys())],
            values=[star_counter[k] for k in sorted(star_counter.keys())],
            xlabel='Star Count',
            ylabel='Count',
            title='Star Count Distribution (Graph)',
            save_path=figures_dir / 'star_count_hist_graph.png'
        )

    # Save summary
    summary = {
        'num_generated': len(generated_smiles),
        'num_decoded': len(valid_generated),
        'num_valid_2stars': len(valid_smiles),
        'validity': round(metrics['validity'], 4),
        'uniqueness': round(metrics['uniqueness'], 4),
        'novelty': round(metrics['novelty'], 4),
        'diversity': round(metrics['avg_diversity'], 4),
        'frac_star_eq_2': round(metrics['frac_star_eq_2'], 4),
        'mean_sa': round(metrics['mean_sa'], 4),
        'std_sa': round(metrics['std_sa'], 4)
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(metrics_dir / 'sampling_summary.csv', index=False)

    print("\n" + "=" * 60)
    print("Graph Sampling and Evaluation Complete!")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  Generated: {len(generated_smiles)}")
    print(f"  Valid (2 stars): {len(valid_smiles)}")
    print(f"  Validity: {metrics['validity']:.4f}")
    print(f"  Uniqueness: {metrics['uniqueness']:.4f}")
    print(f"  Novelty: {metrics['novelty']:.4f}")
    print(f"\nOutput files:")
    print(f"  Samples: {metrics_dir / 'generated_samples.csv'}")
    print(f"  Metrics: {metrics_dir / 'sampling_generative_metrics.csv'}")
    print(f"  Figures: {figures_dir}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample and evaluate graph generative model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to generate (default: from config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for sampling (default: from config)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature (default: from config)')
    args = parser.parse_args()
    main(args)
