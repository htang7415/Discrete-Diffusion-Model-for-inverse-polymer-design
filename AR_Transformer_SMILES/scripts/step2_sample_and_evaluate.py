#!/usr/bin/env python
"""Step 2: Sample from backbone and evaluate generative metrics."""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
import numpy as np

from src.utils.config import load_config, save_config
from src.utils.plotting import PlotUtils
from src.utils.chemistry import compute_sa_score
from src.utils.model_scales import get_model_config, get_results_dir
from src.data.tokenizer import PSmilesTokenizer
from src.model.backbone import DiffusionBackbone
from src.model.autoregressive import AutoregressiveLM
from src.sampling.sampler import ConstrainedSampler
from src.evaluation.generative_metrics import GenerativeEvaluator
from src.utils.reproducibility import seed_everything, save_run_metadata


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
    step_dir = results_dir / 'step2_sampling'
    metrics_dir = step_dir / 'metrics'
    figures_dir = step_dir / 'figures'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    seed_info = seed_everything(config['data']['random_seed'])
    save_config(config, step_dir / 'config_used.yaml')
    save_run_metadata(step_dir, args.config, seed_info)

    print("=" * 50)
    print("Step 2: Sampling and Generative Evaluation")
    if args.model_size:
        print(f"Model Size: {args.model_size}")
    print("=" * 50)

    # Load tokenizer (from base results dir which has the tokenizer)
    print("\n1. Loading tokenizer...")
    tokenizer_path = results_dir / 'tokenizer.json'
    if not tokenizer_path.exists():
        tokenizer_path = Path(base_results_dir) / 'tokenizer.json'
    tokenizer = PSmilesTokenizer.load(tokenizer_path)

    # Load training data for novelty computation (from base results dir)
    print("\n2. Loading training data...")
    train_path = results_dir / 'train_unlabeled.csv'
    if not train_path.exists():
        train_path = Path(base_results_dir) / 'train_unlabeled.csv'
    train_df = pd.read_csv(train_path)
    training_smiles = set(train_df['p_smiles'].tolist())
    print(f"Training set size: {len(training_smiles)}")

    # Load model
    print("\n3. Loading model...")
    checkpoint_path = args.checkpoint or (results_dir / 'step1_backbone' / 'checkpoints' / 'backbone_best.pt')
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

    model = AutoregressiveLM(
        backbone=backbone,
        pad_token_id=tokenizer.pad_token_id
    )

    # Handle torch.compile() state dict (keys have _orig_mod. prefix)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Create sampler
    print("\n4. Creating sampler...")
    sampler = ConstrainedSampler(
        diffusion_model=model,
        tokenizer=tokenizer,
        num_steps=config['diffusion']['num_steps'],
        temperature=config['sampling']['temperature'],
        use_constraints=config['sampling'].get('use_constraints', True),
        device=device,
        top_k=config['sampling'].get('top_k', 0),
        top_p=config['sampling'].get('top_p', 1.0),
        max_length=config['sampling'].get('max_length')
    )

    # Sample
    batch_size = args.batch_size or config['sampling']['batch_size']
    print(f"\n5. Sampling {args.num_samples} polymers (batch_size={batch_size})...")
    if args.variable_length:
        print(f"   Using variable length sampling (range: {args.min_length}-{args.max_length})")
        _, generated_smiles = sampler.sample_variable_length(
            num_samples=args.num_samples,
            length_range=(args.min_length, args.max_length),
            batch_size=batch_size,
            samples_per_length=args.samples_per_length,
            show_progress=True
        )
    else:
        # Sample lengths from training distribution (token length + BOS/EOS)
        replace = args.num_samples > len(train_df)
        sampled = train_df['p_smiles'].sample(
            n=args.num_samples,
            replace=replace,
            random_state=config['data']['random_seed']
        )
        lengths = [
            min(len(tokenizer.tokenize(s)) + 2, tokenizer.max_length)
            for s in sampled.tolist()
        ]
        print(f"   Using training length distribution (min={min(lengths)}, max={max(lengths)})")
        _, generated_smiles = sampler.sample_batch(
            num_samples=args.num_samples,
            seq_length=tokenizer.max_length,
            batch_size=batch_size,
            show_progress=True,
            lengths=lengths
        )

    # Save generated samples
    samples_df = pd.DataFrame({'smiles': generated_smiles})
    samples_df.to_csv(metrics_dir / 'generated_samples.csv', index=False)
    print(f"Saved {len(generated_smiles)} generated samples")

    # Evaluate
    print("\n6. Evaluating generative metrics...")
    evaluator = GenerativeEvaluator(training_smiles)
    metrics = evaluator.evaluate(
        generated_smiles,
        sample_id=f'uncond_{args.num_samples}_best_checkpoint',
        show_progress=True
    )

    # Save metrics
    metrics_csv = evaluator.format_metrics_csv(metrics)
    metrics_csv.to_csv(metrics_dir / 'sampling_generative_metrics.csv', index=False)

    # Print metrics
    print("\nGenerative Metrics:")
    print(f"  Validity: {metrics['validity']:.4f}")
    print(f"  Uniqueness: {metrics['uniqueness']:.4f}")
    print(f"  Novelty: {metrics['novelty']:.4f}")
    print(f"  Diversity: {metrics['avg_diversity']:.4f}")
    print(f"  Frac star=2: {metrics['frac_star_eq_2']:.4f}")
    print(f"  Mean SA: {metrics['mean_sa']:.4f}")
    print(f"  Std SA: {metrics['std_sa']:.4f}")

    # Create plots
    print("\n7. Creating plots...")
    plotter = PlotUtils(
        figure_size=tuple(config['plotting']['figure_size']),
        font_size=config['plotting']['font_size'],
        dpi=config['plotting']['dpi']
    )

    # Get valid samples
    valid_smiles = evaluator.get_valid_samples(generated_smiles, require_two_stars=True)

    # SA histogram: train vs generated
    train_sa = [compute_sa_score(s) for s in list(training_smiles)[:5000]]
    train_sa = [s for s in train_sa if s is not None]
    gen_sa = [compute_sa_score(s) for s in valid_smiles[:5000]]
    gen_sa = [s for s in gen_sa if s is not None]

    plotter.histogram(
        data=[train_sa, gen_sa],
        labels=['Train', 'Generated'],
        xlabel='SA Score',
        ylabel='Count',
        title='SA Score: Train vs Generated',
        save_path=figures_dir / 'sa_hist_train_vs_uncond.png',
        bins=50,
        style='step'
    )

    # Length histogram: train vs generated
    train_lengths = [len(s) for s in list(training_smiles)[:5000]]
    gen_lengths = [len(s) for s in valid_smiles[:5000]]

    plotter.histogram(
        data=[train_lengths, gen_lengths],
        labels=['Train', 'Generated'],
        xlabel='SMILES Length',
        ylabel='Count',
        title='Length: Train vs Generated',
        save_path=figures_dir / 'length_hist_train_vs_uncond.png',
        bins=50,
        style='step'
    )

    # Star count histogram
    from collections import Counter
    from src.utils.chemistry import count_stars

    star_counts = [count_stars(s) for s in valid_smiles]
    star_counter = Counter(star_counts)

    plotter.bar_chart(
        categories=[str(k) for k in sorted(star_counter.keys())],
        values=[star_counter[k] for k in sorted(star_counter.keys())],
        xlabel='Star Count',
        ylabel='Count',
        title='Star Count Distribution',
        save_path=figures_dir / 'star_count_hist_uncond.png'
    )

    print("\n" + "=" * 50)
    print("Sampling and evaluation complete!")
    print(f"Results saved to: {metrics_dir}")
    print(f"Figures saved to: {figures_dir}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample and evaluate generative model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_size', type=str, default=None,
                        choices=['small', 'medium', 'large', 'xl'],
                        help='Model size preset (small: ~12M, medium: ~50M, large: ~150M, xl: ~400M)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for sampling (default: from config)')
    parser.add_argument('--variable_length', action='store_true',
                        help='Enable variable length sampling')
    parser.add_argument('--min_length', type=int, default=20,
                        help='Minimum sequence length for variable length sampling')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum sequence length for variable length sampling')
    parser.add_argument('--samples_per_length', type=int, default=16,
                        help='Samples per length in variable length mode (controls diversity)')
    args = parser.parse_args()
    main(args)
