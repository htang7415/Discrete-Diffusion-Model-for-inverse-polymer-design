#!/usr/bin/env python
"""Step 4: Property-guided inverse design."""

import os
import sys
import argparse
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.utils.config import load_config, save_config
from src.utils.plotting import PlotUtils
from src.utils.chemistry import compute_sa_score
from src.utils.model_scales import get_model_config, get_results_dir
from src.data.selfies_tokenizer import SelfiesTokenizer
from src.model.backbone import DiffusionBackbone
from src.model.autoregressive import AutoregressiveLM
from src.model.property_head import PropertyHead, PropertyPredictor
from src.sampling.sampler import ConstrainedSampler
from src.evaluation.inverse_design import InverseDesigner
from src.utils.reproducibility import seed_everything, save_run_metadata
from shared.unlabeled_data import require_preprocessed_unlabeled_splits
from src.utils.selfies_utils import sample_selfies_from_dataframe
from shared.rerank_utils import compute_rerank_metrics


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
    step_dir = results_dir / f'step4_{args.property}'
    metrics_dir = step_dir / 'metrics'
    figures_dir = step_dir / 'figures'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    seed_info = seed_everything(config['data']['random_seed'])
    save_config(config, step_dir / 'config_used.yaml')
    save_run_metadata(step_dir, args.config, seed_info)

    print("=" * 50)
    print(f"Step 4: Inverse Design for {args.property}")
    print("=" * 50)

    # Get model config
    backbone_config = get_model_config(args.model_size, config, model_type='sequence')

    # Load tokenizer (from base results dir which has the tokenizer)
    print("\n1. Loading SELFIES tokenizer...")
    tokenizer_path = results_dir / 'tokenizer.json'
    if not tokenizer_path.exists():
        tokenizer_path = Path(base_results_dir) / 'tokenizer.json'
    tokenizer = SelfiesTokenizer.load(tokenizer_path)

    # Load training data for novelty (use p_smiles for comparison)
    print("\n2. Loading training data...")
    repo_root = Path(__file__).resolve().parents[2]
    train_path, _ = require_preprocessed_unlabeled_splits(repo_root)
    train_df = pd.read_csv(train_path)
    training_smiles = set(train_df['p_smiles'].tolist())  # p-SMILES for novelty computation

    # Load backbone model
    print("\n3. Loading backbone model...")
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

    backbone_ckpt = torch.load(results_dir / 'step1_backbone' / 'checkpoints' / 'backbone_best.pt', map_location=device)
    # Handle torch.compile() state dict (keys have _orig_mod. prefix)
    state_dict = backbone_ckpt['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    if any(k.startswith('backbone.') for k in state_dict.keys()):
        model.load_state_dict(state_dict)
    else:
        model.backbone.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Create sampler
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

    # Load property predictor
    print("\n4. Loading property predictor...")
    property_ckpt = torch.load(
        results_dir / 'checkpoints' / f'{args.property}_best.pt',
        map_location=device
    )

    step3_config_path = results_dir / f'step3_{args.property}' / 'config_used.yaml'
    trained_config = config
    if step3_config_path.exists():
        trained_config = load_config(step3_config_path)
        if trained_config.get('property_head') != config.get('property_head'):
            print(f"Using property head settings from {step3_config_path} to match checkpoint.")

    # Get hyperparameters from checkpoint (if tuned) or config
    head_config = trained_config.get('property_head', config['property_head'])
    if 'hidden_sizes' in property_ckpt and property_ckpt['hidden_sizes'] is not None:
        hidden_sizes = property_ckpt['hidden_sizes']
        dropout = property_ckpt.get('dropout', head_config['dropout'])
    else:
        hidden_sizes = head_config['hidden_sizes']
        dropout = head_config['dropout']

    property_head = PropertyHead(
        input_size=backbone_config['hidden_size'],
        hidden_sizes=hidden_sizes,
        dropout=dropout
    )

    default_timestep = trained_config.get('training_property', {}).get(
        'default_timestep',
        config['training_property'].get('default_timestep', 1)
    )
    property_predictor = PropertyPredictor(
        backbone=model.backbone,
        property_head=property_head,
        freeze_backbone=True,
        pooling='mean',
        default_timestep=default_timestep
    )
    property_predictor.load_property_head(results_dir / 'checkpoints' / f'{args.property}_best.pt')
    property_predictor = property_predictor.to(device)
    property_predictor.eval()

    # Get normalization parameters
    norm_params = property_ckpt.get('normalization_params', {'mean': 0.0, 'std': 1.0})

    # Create inverse designer
    designer = InverseDesigner(
        sampler=sampler,
        property_predictor=property_predictor,
        tokenizer=tokenizer,
        training_smiles=training_smiles,
        device=device,
        normalization_params=norm_params
    )

    def default_targets_for_property(property_name: str):
        presets = {
            'Tg': [350.0],
            'Tm': [450.0],
            'Td': [550.0],
            'Eg': [8.0],
        }
        return presets.get(property_name)

    def default_epsilon_for_property(property_name: str) -> float:
        presets = {
            'Tg': 30.0,
            'Tm': 30.0,
            'Td': 30.0,
            'Eg': 0.5,
        }
        return presets.get(property_name, 10.0)

    if args.epsilon is None:
        args.epsilon = default_epsilon_for_property(args.property)

    # Parse target values
    if args.targets:
        target_values = [float(t) for t in args.targets.split(',')]
    else:
        preset_targets = default_targets_for_property(args.property)
        if preset_targets is not None:
            target_values = preset_targets
        else:
            # Default targets based on property data statistics (from step3)
            step3_metrics = results_dir / f'step3_{args.property}' / 'metrics'
            property_df = pd.read_csv(step3_metrics / f'{args.property}_data_stats.csv')
            # Stats columns are named {property}_mean, {property}_std from get_statistics()
            mean_col = f'{args.property}_mean'
            std_col = f'{args.property}_std'
            mean_val = property_df.loc[property_df['split'] == 'train', mean_col].values[0]
            std_val = property_df.loc[property_df['split'] == 'train', std_col].values[0]
            target_values = [
                mean_val - std_val,
                mean_val,
                mean_val + std_val
            ]

    print(f"\n5. Running inverse design for targets: {target_values}")
    print(f"   Epsilon: {args.epsilon}")
    print(f"   Candidates per target: {args.num_candidates}")

    # Sample lengths from training distribution
    total_candidates = args.num_candidates * len(target_values)
    sampled = sample_selfies_from_dataframe(
        train_df,
        num_samples=total_candidates,
        random_seed=config['data']['random_seed'],
    )
    lengths = [
        min(len(tokenizer.tokenize(s)) + 2, tokenizer.max_length)
        for s in sampled
    ]
    print(f"   Using training length distribution (min={min(lengths)}, max={max(lengths)})")

    # Run design
    raw_results = []
    for i, target in enumerate(tqdm(target_values, desc="Targets")):
        start_idx = i * args.num_candidates
        end_idx = start_idx + args.num_candidates
        target_lengths = lengths[start_idx:end_idx]
        results = designer.design(
            target_value=target,
            epsilon=args.epsilon,
            num_candidates=args.num_candidates,
            seq_length=tokenizer.max_length,
            batch_size=config['sampling']['batch_size'],
            show_progress=False,
            lengths=target_lengths
        )
        raw_results.append(results)

    drop_keys = {
        "valid_smiles",
        "valid_selfies",
        "predictions",
        "hits_smiles",
        "hits_selfies",
        "hits_predictions"
    }
    results_df = pd.DataFrame(
        [{k: v for k, v in results.items() if k not in drop_keys} for results in raw_results]
    )

    # Save results
    results_df.to_csv(metrics_dir / f'{args.property}_design.csv', index=False)

    # Print summary
    print("\nInverse Design Results:")
    print(results_df[['target_value', 'n_valid', 'n_hits', 'success_rate', 'pred_mean_hits']].to_string())

    # Create plots
    print("\n6. Creating plots...")
    plotter = PlotUtils(
        figure_size=tuple(config['plotting']['figure_size']),
        font_size=config['plotting']['font_size'],
        dpi=config['plotting']['dpi']
    )

    # Property distribution plots
    for i, results in enumerate(raw_results, start=1):
        predictions = results.get("predictions", [])
        target_value = results["target_value"]
        plotter.property_distribution_plot(
            predictions=predictions,
            target_value=target_value,
            xlabel=f"Predicted {args.property}",
            ylabel="Count",
            title=f"{args.property} Distribution (target={target_value})",
            save_path=figures_dir / f"{args.property}_distribution_target_{i}.png"
        )

    print("\n" + "=" * 50)
    print("Inverse design complete!")
    print(f"Results saved to: {metrics_dir / f'{args.property}_design.csv'}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Property-guided inverse design')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_size', type=str, default=None,
                        choices=['small', 'medium', 'large', 'xl'],
                        help='Model size preset')
    parser.add_argument('--property', type=str, required=True,
                        help='Property name (e.g., Tg, Tm)')
    parser.add_argument('--targets', type=str, default=None,
                        help='Comma-separated target values')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Tolerance for property matching (default uses property-specific preset)')
    parser.add_argument('--num_candidates', type=int, default=10000,
                        help='Number of candidates per target')
    parser.add_argument("--rerank_strategy", type=str, default="none",
                        choices=["none", "property_error", "external", "d2_distance", "consistency", "retrieval"],
                        help="Reranking strategy for foundation-enhanced inverse design")
    parser.add_argument("--rerank_scores_path", type=str, default=None,
                        help="Path to rerank scores (.csv or .npy)")
    parser.add_argument("--rerank_key", type=str, default=None,
                        help="Key column in rerank scores CSV (smiles/selfies)")
    parser.add_argument("--rerank_top_k", type=int, default=1000,
                        help="Top-k candidates to evaluate after reranking")

    args = parser.parse_args()
    main(args)
