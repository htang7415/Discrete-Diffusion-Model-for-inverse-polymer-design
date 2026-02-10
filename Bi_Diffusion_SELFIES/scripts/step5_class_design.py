#!/usr/bin/env python
"""Step 5: Polymer class-guided design."""

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
from src.data.selfies_tokenizer import SelfiesTokenizer
from src.model.backbone import DiffusionBackbone
from src.model.diffusion import DiscreteMaskingDiffusion
from src.model.property_head import PropertyHead, PropertyPredictor
from src.sampling.sampler import ConstrainedSampler
from src.evaluation.polymer_class import PolymerClassifier, ClassGuidedDesigner
from src.utils.reproducibility import seed_everything, save_run_metadata
from src.utils.selfies_utils import sample_selfies_from_dataframe


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

    def default_epsilon_for_property(property_name: str) -> float:
        presets = {
            'Tg': 30.0,
            'Tm': 30.0,
            'Td': 30.0,
            'Eg': 0.5,
        }
        return presets.get(property_name, 10.0)

    if args.property and args.epsilon is None:
        args.epsilon = default_epsilon_for_property(args.property)

    print("=" * 50)
    print(f"Step 5: Class-Guided Design for {args.polymer_class}")
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
    shared_train_path = repo_root / 'Data' / 'Polymer' / 'train_unlabeled.csv.gz'
    if not shared_train_path.exists():
        shared_train_path = repo_root / 'Data' / 'Polymer' / 'train_unlabeled.csv'
    train_path = shared_train_path if shared_train_path.exists() else results_dir / 'train_unlabeled.csv'
    if not train_path.exists():
        train_path = Path(base_results_dir) / 'train_unlabeled.csv'
    train_df = pd.read_csv(train_path)
    training_smiles = set(train_df['p_smiles'].tolist())  # p-SMILES for novelty computation

    # Create classifier
    print("\n3. Creating polymer classifier...")
    classifier = PolymerClassifier(config.get('polymer_classes', None))
    print(f"Available classes: {list(classifier.patterns.keys())}")

    # Load diffusion model
    print("\n4. Loading diffusion model...")
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
        force_clean_t0=config['diffusion'].get('force_clean_t0', False),
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    backbone_ckpt = torch.load(results_dir / 'step1_backbone' / 'checkpoints' / 'backbone_best.pt', map_location=device)
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
        use_constraints=config['sampling'].get('use_constraints', True),
        device=device
    )

    # Load property predictor if specified
    property_predictor = None
    norm_params = {'mean': 0.0, 'std': 1.0}
    if args.property:
        print(f"\n5. Loading property predictor for {args.property}...")

        # Load checkpoint first to get hyperparameters
        property_ckpt = torch.load(
            results_dir / 'checkpoints' / f'{args.property}_best.pt',
            map_location=device
        )
        norm_params = property_ckpt.get('normalization_params', {'mean': 0.0, 'std': 1.0})

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
            backbone=diffusion_model.backbone,
            property_head=property_head,
            freeze_backbone=True,
            pooling='mean',
            default_timestep=default_timestep
        )
        property_predictor.load_property_head(results_dir / 'checkpoints' / f'{args.property}_best.pt')
        property_predictor = property_predictor.to(device)
        property_predictor.eval()

    # Create class-guided designer
    designer = ClassGuidedDesigner(
        sampler=sampler,
        tokenizer=tokenizer,
        classifier=classifier,
        training_smiles=training_smiles,
        property_predictor=property_predictor,
        device=device,
        normalization_params=norm_params
    )

    # Sample lengths from training distribution
    sampled = sample_selfies_from_dataframe(
        train_df,
        num_samples=args.num_candidates,
        random_seed=config['data']['random_seed'],
    )
    lengths = [
        min(len(tokenizer.tokenize(s)) + 2, tokenizer.max_length)
        for s in sampled
    ]
    print(f"   Using training length distribution (min={min(lengths)}, max={max(lengths)})")

    # Run class-only design
    print(f"\n6. Running class-guided design for {args.polymer_class}...")
    class_results = designer.design_by_class(
        target_class=args.polymer_class,
        num_candidates=args.num_candidates,
        seq_length=tokenizer.max_length,
        batch_size=config['sampling']['batch_size'],
        show_progress=True,
        lengths=lengths
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
    if args.property and args.target_value is not None:
        print(f"\n7. Running joint design: {args.polymer_class} + {args.property}={args.target_value}...")
        # Sample new lengths for joint design
        joint_sampled = sample_selfies_from_dataframe(
            train_df,
            num_samples=args.num_candidates,
            random_seed=config['data']['random_seed'] + 1,
        )
        joint_lengths = [
            min(len(tokenizer.tokenize(s)) + 2, tokenizer.max_length)
            for s in joint_sampled
        ]
        joint_results = designer.design_joint(
            target_class=args.polymer_class,
            target_value=args.target_value,
            epsilon=args.epsilon,
            num_candidates=args.num_candidates,
            seq_length=tokenizer.max_length,
            batch_size=config['sampling']['batch_size'],
            show_progress=True,
            lengths=joint_lengths
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
    if args.property and args.target_value is not None and joint_results.get('joint_hits_smiles'):
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
    parser = argparse.ArgumentParser(description='Polymer class-guided design')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_size', type=str, default=None,
                        choices=['small', 'medium', 'large', 'xl'],
                        help='Model size preset')
    parser.add_argument('--polymer_class', type=str, required=True,
                        help='Target polymer class (e.g., polyimide, polyester)')
    parser.add_argument('--property', type=str, default=None,
                        help='Property for joint design (optional)')
    parser.add_argument('--target_value', type=float, default=None,
                        help='Target property value for joint design')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Tolerance for property matching (default uses property-specific preset)')
    parser.add_argument('--num_candidates', type=int, default=10000,
                        help='Number of candidates to generate')
    args = parser.parse_args()
    main(args)
