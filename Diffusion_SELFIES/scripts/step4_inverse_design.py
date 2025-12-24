#!/usr/bin/env python
"""Step 4: Property-guided inverse design."""

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
from src.data.selfies_tokenizer import SelfiesTokenizer
from src.model.backbone import DiffusionBackbone
from src.model.diffusion import DiscreteMaskingDiffusion
from src.model.property_head import PropertyHead, PropertyPredictor
from src.sampling.sampler import ConstrainedSampler
from src.evaluation.inverse_design import InverseDesigner
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

    # Load tokenizer
    print("\n1. Loading SELFIES tokenizer...")
    tokenizer = SelfiesTokenizer.load(results_dir / 'tokenizer.json')

    # Load training data for novelty (use p_smiles for comparison)
    print("\n2. Loading training data...")
    train_df = pd.read_csv(results_dir / 'train_unlabeled.csv')
    training_smiles = set(train_df['p_smiles'].tolist())  # p-SMILES for novelty computation

    # Load diffusion model
    print("\n3. Loading diffusion model...")
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
    diffusion_model.eval()

    # Create sampler
    sampler = ConstrainedSampler(
        diffusion_model=diffusion_model,
        tokenizer=tokenizer,
        num_steps=config['diffusion']['num_steps'],
        temperature=config['sampling']['temperature'],
        device=device
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

    head_config = trained_config.get('property_head', config['property_head'])
    property_head = PropertyHead(
        input_size=backbone_config['hidden_size'],
        hidden_sizes=head_config['hidden_sizes'],
        dropout=head_config['dropout']
    )

    default_timestep = trained_config.get('training_property', {}).get(
        'default_timestep',
        config['training_property'].get('default_timestep', 1)
    )
    property_predictor = PropertyPredictor(
        backbone=diffusion_model.backbone,
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

    # Parse target values
    if args.targets:
        target_values = [float(t) for t in args.targets.split(',')]
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

    # Run design
    results_df = designer.design_multiple_targets(
        target_values=target_values,
        epsilon=args.epsilon,
        num_candidates_per_target=args.num_candidates,
        seq_length=tokenizer.max_length,
        batch_size=config['sampling']['batch_size'],
        show_progress=True
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

    # Calibration plot
    plotter.calibration_plot(
        target_values=results_df['target_value'].tolist(),
        mean_predictions=results_df['pred_mean_hits'].tolist(),
        std_predictions=results_df['pred_std_hits'].tolist(),
        xlabel=f'Target {args.property}',
        ylabel=f'Mean Predicted {args.property}',
        title=f'{args.property} Calibration',
        save_path=figures_dir / f'{args.property}_calibration.png'
    )

    # Run detailed design for best target
    print("\n7. Running detailed analysis for middle target...")
    middle_target = target_values[len(target_values) // 2]
    detailed_results = designer.design(
        target_value=middle_target,
        epsilon=args.epsilon,
        num_candidates=args.num_candidates * 2,
        seq_length=tokenizer.max_length,
        batch_size=config['sampling']['batch_size'],
        show_progress=True
    )

    # Histogram of predictions around target
    if detailed_results['predictions']:
        predictions = np.array(detailed_results['predictions'])
        plotter.histogram(
            data=[predictions],
            labels=['Predictions'],
            xlabel=f'Predicted {args.property}',
            ylabel='Count',
            title=f'Predictions near target={middle_target}',
            save_path=figures_dir / f'{args.property}_pred_hist_target_{int(middle_target)}.png',
            bins=50
        )

    # SA histogram: train vs hits
    if detailed_results['hits_smiles']:
        train_sa = [compute_sa_score(s) for s in list(training_smiles)[:3000]]
        train_sa = [s for s in train_sa if s is not None]
        hits_sa = [compute_sa_score(s) for s in detailed_results['hits_smiles']]
        hits_sa = [s for s in hits_sa if s is not None]

        if hits_sa:
            plotter.histogram(
                data=[train_sa, hits_sa],
                labels=['Train', 'Hits'],
                xlabel='SA Score',
                ylabel='Count',
                title=f'SA Score: Train vs Hits (target={int(middle_target)})',
                save_path=figures_dir / f'{args.property}_sa_hist_train_vs_hits.png',
                bins=50,
                style='step'
            )

    print("\n" + "=" * 50)
    print("Inverse design complete!")
    print(f"Results saved to: {metrics_dir / f'{args.property}_design.csv'}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Property-guided inverse design')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--property', type=str, required=True,
                        help='Property name (e.g., Tg, Tm)')
    parser.add_argument('--targets', type=str, default=None,
                        help='Comma-separated target values')
    parser.add_argument('--epsilon', type=float, default=10.0,
                        help='Tolerance for property matching')
    parser.add_argument('--num_candidates', type=int, default=10000,
                        help='Number of candidates per target')
    args = parser.parse_args()
    main(args)
