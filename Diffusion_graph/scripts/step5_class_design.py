#!/usr/bin/env python
"""Step 5: Polymer class-guided design using graph diffusion."""

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

from src.utils.config import load_config, save_config
from src.utils.plotting import PlotUtils
from src.utils.chemistry import compute_sa_score
from src.data.graph_tokenizer import GraphTokenizer
from src.model.graph_backbone import create_graph_backbone
from src.model.graph_diffusion import create_graph_diffusion
from src.model.graph_property_head import GraphPropertyHead, GraphPropertyPredictor
from src.sampling.graph_sampler import create_graph_sampler
from src.evaluation.polymer_class import PolymerClassifier, GraphClassGuidedDesigner
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
    step_name = f'step5_{args.polymer_class}'
    if args.property:
        step_name += f'_{args.property}'
    step_dir = results_dir / step_name
    metrics_dir = step_dir / 'metrics'
    figures_dir = step_dir / 'figures'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    seed_info = seed_everything(config['data']['random_seed'])
    save_config(config, step_dir / 'config_used.yaml')
    save_run_metadata(step_dir, args.config, seed_info)

    print("=" * 50)
    print(f"Step 5: Graph Class-Guided Design for {args.polymer_class}")
    print("=" * 50)

    # Load graph config
    print("\n1. Loading graph config and tokenizer...")
    graph_config_path = results_dir / 'graph_config.json'
    with open(graph_config_path, 'r') as f:
        graph_config = json.load(f)

    tokenizer = GraphTokenizer.load(results_dir / 'graph_tokenizer.json')
    print(f"Nmax: {graph_config['Nmax']}")

    # Load training data for novelty
    print("\n2. Loading training data...")
    train_df = pd.read_csv(results_dir / 'train_unlabeled.csv')
    training_smiles = set(train_df['p_smiles'].tolist())

    # Create classifier
    print("\n3. Creating polymer classifier...")
    classifier = PolymerClassifier(config.get('polymer_classes', None))
    print(f"Available classes: {list(classifier.patterns.keys())}")

    # Load graph diffusion model
    print("\n4. Loading graph diffusion model...")
    backbone = create_graph_backbone(config, graph_config)
    diffusion_model = create_graph_diffusion(backbone, config, graph_config)

    backbone_ckpt = torch.load(
        results_dir / 'checkpoints' / 'graph_backbone_best.pt',
        map_location=device,
        weights_only=False
    )
    # Handle torch.compile() state dict (keys have _orig_mod. prefix)
    state_dict = backbone_ckpt['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    diffusion_model.load_state_dict(state_dict)
    diffusion_model = diffusion_model.to(device)
    diffusion_model.eval()

    # Create sampler
    sampler = create_graph_sampler(
        diffusion_model=diffusion_model,
        tokenizer=tokenizer,
        config=config,
        graph_config=graph_config
    )

    # Load property predictor if specified
    property_predictor = None
    norm_params = {'mean': 0.0, 'std': 1.0}
    if args.property:
        print(f"\n5. Loading property predictor for {args.property}...")
        head_config = config['property_head']
        backbone_config = config.get('backbone', config.get('graph_backbone', {}))
        property_head = GraphPropertyHead(
            input_size=backbone_config.get('hidden_size', 768),
            hidden_sizes=head_config['hidden_sizes'],
            dropout=head_config['dropout']
        )

        property_predictor = GraphPropertyPredictor(
            backbone=diffusion_model.backbone,
            property_head=property_head,
            freeze_backbone=True,
            pooling='mean',
            default_timestep=config['training_property'].get('default_timestep', 1)
        )
        property_predictor.load_property_head(results_dir / 'checkpoints' / f'{args.property}_best.pt')
        property_predictor = property_predictor.to(device)
        property_predictor.eval()

        # Load normalization parameters
        property_ckpt = torch.load(
            results_dir / 'checkpoints' / f'{args.property}_best.pt',
            map_location=device,
            weights_only=False
        )
        norm_params = property_ckpt.get('normalization_params', {'mean': 0.0, 'std': 1.0})

    # Create class-guided designer
    designer = GraphClassGuidedDesigner(
        sampler=sampler,
        tokenizer=tokenizer,
        classifier=classifier,
        training_smiles=training_smiles,
        property_predictor=property_predictor,
        device=device,
        normalization_params=norm_params
    )

    # Run class-only design
    print(f"\n6. Running class-guided design for {args.polymer_class}...")
    class_results = designer.design_by_class(
        target_class=args.polymer_class,
        num_candidates=args.num_candidates,
        batch_size=config['sampling']['batch_size'],
        show_progress=True
    )

    # Save class results
    class_metrics = {k: v for k, v in class_results.items() if not isinstance(v, list)}
    class_df = pd.DataFrame([class_metrics])
    class_df.to_csv(metrics_dir / f'{args.polymer_class}_class_design.csv', index=False)

    # Save matched samples
    if class_results['class_matches_smiles']:
        samples_df = pd.DataFrame({'smiles': class_results['class_matches_smiles']})
        samples_df.to_csv(metrics_dir / f'{args.polymer_class}_samples.csv', index=False)

    print(f"\nClass-only Results for {args.polymer_class}:")
    print(f"  Valid candidates: {class_results['n_valid']}")
    print(f"  Class matches: {class_results['n_class_matches']}")
    print(f"  Class success rate: {class_results['class_success_rate']:.4f}")

    # Run joint design if property specified
    joint_results = None
    if args.property and args.target_value is not None:
        print(f"\n7. Running joint design: {args.polymer_class} + {args.property}={args.target_value}...")
        joint_results = designer.design_joint(
            target_class=args.polymer_class,
            target_value=args.target_value,
            epsilon=args.epsilon,
            num_candidates=args.num_candidates,
            batch_size=config['sampling']['batch_size'],
            show_progress=True
        )

        # Save joint results
        joint_metrics = {k: v for k, v in joint_results.items() if not isinstance(v, list)}
        joint_df = pd.DataFrame([joint_metrics])
        joint_df.to_csv(metrics_dir / f'{args.polymer_class}_{args.property}_joint_design.csv', index=False)

        # Save joint hits
        if joint_results['joint_hits_smiles']:
            hits_df = pd.DataFrame({'smiles': joint_results['joint_hits_smiles']})
            hits_df.to_csv(metrics_dir / f'{args.polymer_class}_{args.property}_joint_hits.csv', index=False)

        print(f"\nJoint Design Results:")
        print(f"  Valid candidates: {joint_results['n_valid']}")
        print(f"  Class matches: {joint_results['n_class_matches']}")
        print(f"  Joint hits: {joint_results['n_joint_hits']}")
        print(f"  Class success rate: {joint_results['class_success_rate']:.4f}")
        print(f"  Joint success rate: {joint_results['joint_success_rate']:.4f}")

    # Create plots
    print("\n8. Creating plots...")
    plotter = PlotUtils(
        figure_size=tuple(config['plotting']['figure_size']),
        font_size=config['plotting']['font_size'],
        dpi=config['plotting']['dpi']
    )

    # SA histogram: train vs class matches
    if class_results['class_matches_smiles']:
        train_sa = [compute_sa_score(s) for s in list(training_smiles)[:3000]]
        train_sa = [s for s in train_sa if s is not None]
        class_sa = [compute_sa_score(s) for s in class_results['class_matches_smiles']]
        class_sa = [s for s in class_sa if s is not None]

        if class_sa:
            plotter.histogram(
                data=[train_sa, class_sa],
                labels=['Train', args.polymer_class],
                xlabel='SA Score',
                ylabel='Count',
                title=f'SA Score: Train vs {args.polymer_class}',
                save_path=figures_dir / f'{args.polymer_class}_sa_hist.png',
                bins=50,
                style='step'
            )

    # Joint SA histogram if applicable
    if args.property and args.target_value is not None and joint_results and joint_results.get('joint_hits_smiles'):
        joint_sa = [compute_sa_score(s) for s in joint_results['joint_hits_smiles']]
        joint_sa = [s for s in joint_sa if s is not None]

        if joint_sa and class_sa:
            plotter.histogram(
                data=[train_sa, class_sa, joint_sa],
                labels=['Train', f'{args.polymer_class}', 'Joint Hits'],
                xlabel='SA Score',
                ylabel='Count',
                title=f'SA: Train vs {args.polymer_class} vs Joint',
                save_path=figures_dir / f'{args.polymer_class}_{args.property}_joint_sa_hist.png',
                bins=50,
                style='step'
            )

    print("\n" + "=" * 50)
    print("Class-guided design complete!")
    print(f"Results saved to: {metrics_dir}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph polymer class-guided design')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--polymer_class', type=str, required=True,
                        help='Target polymer class (e.g., polyimide, polyester)')
    parser.add_argument('--property', type=str, default=None,
                        help='Property for joint design (optional)')
    parser.add_argument('--target_value', type=float, default=None,
                        help='Target property value for joint design')
    parser.add_argument('--epsilon', type=float, default=10.0,
                        help='Tolerance for property matching')
    parser.add_argument('--num_candidates', type=int, default=10000,
                        help='Number of candidates to generate')
    args = parser.parse_args()
    main(args)
