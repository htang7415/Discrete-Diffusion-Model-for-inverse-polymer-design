#!/usr/bin/env python
"""Step 2: Sample from graph backbone and evaluate generative metrics."""

import os
import sys
import json
import argparse
import re
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import torch
import pandas as pd
import numpy as np
from collections import Counter

from src.utils.config import load_config, save_config
from src.utils.plotting import PlotUtils
from src.utils.chemistry import compute_sa_score, count_stars
from src.utils.model_scales import get_model_config, get_results_dir
from src.data.graph_tokenizer import GraphTokenizer
from src.model.graph_backbone import GraphDiffusionBackbone
from src.model.graph_diffusion import GraphMaskingDiffusion
from src.sampling.graph_sampler import GraphSampler
from src.evaluation.generative_metrics import GenerativeEvaluator
from src.utils.reproducibility import seed_everything, save_run_metadata
from shared.unlabeled_data import require_preprocessed_unlabeled_splits



# Constraint logging helpers
BOND_CHARS = set(['-', '=', '#', '/', '\\'])


def _smiles_constraint_violations(smiles: str) -> dict:
    if not smiles:
        return {
            "star_count": True,
            "bond_placement": True,
            "paren_balance": True,
            "empty_parens": True,
            "ring_closure": True,
        }

    star_violation = count_stars(smiles) != 2
    empty_parens = "()" in smiles

    # Parenthesis balance
    depth = 0
    paren_violation = False
    for ch in smiles:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                paren_violation = True
                break
    if depth != 0:
        paren_violation = True

    # Bond placement (heuristic)
    bond_violation = False
    prev = None
    for ch in smiles:
        if ch in BOND_CHARS:
            if prev is None or prev in BOND_CHARS or prev in '()':
                bond_violation = True
                break
        if ch.strip() == "":
            continue
        prev = ch

    # Ring closure (digits and %nn tokens must appear exactly twice)
    ring_tokens = re.findall(r'%\d{2}', smiles)
    no_percent = re.sub(r'%\d{2}', '', smiles)
    ring_tokens += re.findall(r'\d', no_percent)
    ring_violation = False
    if ring_tokens:
        counts = Counter(ring_tokens)
        ring_violation = any(c != 2 for c in counts.values())

    return {
        "star_count": star_violation,
        "bond_placement": bond_violation,
        "paren_balance": paren_violation,
        "empty_parens": empty_parens,
        "ring_closure": ring_violation,
    }


def compute_smiles_constraint_metrics(smiles_list, method, representation, model_size):
    total = len(smiles_list)
    violations = {
        "star_count": 0,
        "bond_placement": 0,
        "paren_balance": 0,
        "empty_parens": 0,
        "ring_closure": 0,
    }

    for smiles in smiles_list:
        flags = _smiles_constraint_violations(smiles)
        for key, violated in flags.items():
            if violated:
                violations[key] += 1

    rows = []
    for constraint, count in violations.items():
        rate = count / total if total > 0 else 0.0
        rows.append({
            "method": method,
            "representation": representation,
            "model_size": model_size,
            "constraint": constraint,
            "total": total,
            "violations": count,
            "violation_rate": round(rate, 4),
        })
    return rows
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

    # Get model config
    backbone_config = get_model_config(args.model_size, config, model_type='graph')

    # Load graph configuration
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

    # Load graph tokenizer
    print("\n2. Loading graph tokenizer...")
    tokenizer_path = results_dir / 'graph_tokenizer.json'
    if not tokenizer_path.exists():
        tokenizer_path = Path(base_results_dir) / 'graph_tokenizer.json'
    graph_tokenizer = GraphTokenizer.load(tokenizer_path)

    # Load training data for novelty computation
    print("\n3. Loading training data...")
    repo_root = Path(__file__).resolve().parents[2]
    train_path, _ = require_preprocessed_unlabeled_splits(repo_root)
    train_df = pd.read_csv(train_path)
    training_smiles = set(train_df['p_smiles'].tolist())
    print(f"   Training set size: {len(training_smiles)}")

    # Load model
    print("\n4. Loading model...")
    checkpoint_path = args.checkpoint or (results_dir / 'step1_backbone' / 'checkpoints' / 'graph_backbone_best.pt')
    print(f"   Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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
        atom_count_distribution=graph_config.get('atom_count_distribution'),
        use_constraints=config['sampling'].get('use_constraints', True)
    )

    # Sample
    num_samples = args.num_samples or config['sampling']['num_samples']
    batch_size = args.batch_size or config['sampling']['batch_size']
    temperature = args.temperature or config['sampling']['temperature']

    print(f"\n6. Sampling {num_samples} polymers (batch_size={batch_size}, temp={temperature})...")

    generated_smiles = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    sampling_start = time.time()

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

    sampling_time_sec = time.time() - sampling_start

    # Save generated samples
    samples_df = pd.DataFrame({'smiles': generated_smiles})
    samples_df.to_csv(metrics_dir / 'generated_samples.csv', index=False)
    print(f"   Saved generated samples to: {metrics_dir / 'generated_samples.csv'}")

    # Evaluate
    print("\n7. Evaluating generative metrics...")
    method_name = "Bi_Diffusion"
    representation_name = "Graph"
    model_size_label = args.model_size or "base"
    evaluator = GenerativeEvaluator(training_smiles)
    metrics = evaluator.evaluate(
        generated_smiles,
        sample_id=f'graph_uncond_{num_samples}',
        show_progress=True,
        sampling_time_sec=sampling_time_sec,
        method=method_name,
        representation=representation_name,
        model_size=model_size_label
    )

    # Save metrics
    metrics_csv = evaluator.format_metrics_csv(metrics)
    metrics_csv.to_csv(metrics_dir / 'sampling_generative_metrics.csv', index=False)

    constraint_rows = compute_smiles_constraint_metrics(generated_smiles, method_name, representation_name, model_size_label)
    pd.DataFrame(constraint_rows).to_csv(metrics_dir / 'constraint_metrics.csv', index=False)

    if args.evaluate_ood:
        foundation_dir = Path(args.foundation_results_dir)
        d1_path = foundation_dir / "embeddings_d1.npy"
        d2_path = foundation_dir / "embeddings_d2.npy"
        gen_path = Path(args.generated_embeddings_path) if args.generated_embeddings_path else None
        if d1_path.exists() and d2_path.exists():
            try:
                from shared.ood_metrics import compute_ood_metrics_from_files
                ood_metrics = compute_ood_metrics_from_files(d1_path, d2_path, gen_path, k=args.ood_k)
                ood_row = {
                    "method": method_name,
                    "representation": representation_name,
                    "model_size": model_size_label,
                    **ood_metrics
                }
                pd.DataFrame([ood_row]).to_csv(metrics_dir / "metrics_ood.csv", index=False)
            except Exception as exc:
                print(f"OOD evaluation failed: {exc}")
        else:
            print("OOD embeddings not found; skipping OOD evaluation.")

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

    if star_counts:
        plotter.star_count_bar(
            star_counts=star_counts,
            expected_count=2,
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
    print(f"  Validity (star=2): {metrics['validity_two_stars']:.4f}")
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
    parser.add_argument('--model_size', type=str, default=None,
                        choices=['small', 'medium', 'large', 'xl'],
                        help='Model size preset')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to generate (default: 10000)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for sampling (default: from config)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature (default: from config)')
    parser.add_argument("--evaluate_ood", action="store_true",
                        help="Compute OOD metrics if embeddings are available")
    parser.add_argument("--foundation_results_dir", type=str,
                        default="../Multi_View_Foundation/results",
                        help="Path to Multi_View_Foundation results directory")
    parser.add_argument("--generated_embeddings_path", type=str, default=None,
                        help="Optional path to generated embeddings (.npy)")
    parser.add_argument("--ood_k", type=int, default=1,
                        help="k for nearest-neighbor distance in OOD metrics")

    args = parser.parse_args()
    main(args)
