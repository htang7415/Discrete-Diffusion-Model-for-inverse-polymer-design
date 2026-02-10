#!/usr/bin/env python
"""Run full scaling pipeline with timing and metrics logging.

This script runs steps 1-6 sequentially for a given model size,
logging timing and metrics for each step to scaling_results.json.
"""

import os
import sys
import json
import argparse
import subprocess
import shlex
import shutil
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.model_scales import get_model_config, get_training_config, estimate_params, get_results_dir
from src.utils.scaling_logger import ScalingLogger


def run_step(script_name: str, model_size: str, extra_args: str = "") -> tuple:
    """Run a step script with the given model_size.

    Args:
        script_name: Name of the script (e.g., 'step1_train_backbone.py')
        model_size: Model size preset
        extra_args: Additional command-line arguments

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    cmd = f"python scripts/{script_name} --model_size {model_size} {extra_args}"
    print(f"\nRunning: {cmd}")
    # Use shlex.split for security (avoid shell=True)
    args = shlex.split(cmd)
    result = subprocess.run(args, capture_output=True, text=True)
    # Print output in real-time style
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(f"STDERR:\n{result.stderr}")
    return result.returncode, result.stdout, result.stderr


def check_backbone_checkpoint(results_dir: Path) -> bool:
    """Check if backbone checkpoint exists."""
    checkpoint_path = results_dir / 'step1_backbone' / 'checkpoints' / 'backbone_best.pt'
    return checkpoint_path.exists()


def check_property_head_checkpoint(results_dir: Path, property_name: str) -> bool:
    """Check if property head checkpoint exists."""
    checkpoint_path = results_dir / 'checkpoints' / f'{property_name}_best.pt'
    return checkpoint_path.exists()


def extract_step1_metrics(results_dir: Path) -> dict:
    """Extract metrics from step 1 (backbone training)."""
    metrics = {}
    metrics_file = results_dir / 'step1_backbone' / 'metrics' / 'backbone_training_history.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            history = json.load(f)
        if 'val_losses' in history and history['val_losses']:
            metrics['final_val_loss'] = history['val_losses'][-1]
            metrics['best_val_loss'] = min(history['val_losses'])
        if 'train_losses' in history and history['train_losses']:
            metrics['final_train_loss'] = history['train_losses'][-1]
    return metrics


def extract_step2_metrics(results_dir: Path) -> dict:
    """Extract metrics from step 2 (sampling)."""
    metrics = {}
    import pandas as pd
    metrics_file = results_dir / 'step2_sampling' / 'metrics' / 'sampling_generative_metrics.csv'
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        if len(df) > 0:
            row = df.iloc[0]
            metrics['validity'] = row.get('validity', None)
            metrics['uniqueness'] = row.get('uniqueness', None)
            metrics['novelty'] = row.get('novelty', None)
            metrics['diversity'] = row.get('avg_diversity', None)
    return metrics


def extract_step3_metrics(results_dir: Path, property_name: str) -> dict:
    """Extract metrics from step 3 (property head training)."""
    metrics = {}
    import pandas as pd
    metrics_file = results_dir / f'step3_{property_name}' / 'metrics' / f'{property_name}_test_metrics.csv'
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        if len(df) > 0:
            row = df.iloc[0]
            metrics['test_mae'] = row.get('MAE', None)
            metrics['test_rmse'] = row.get('RMSE', None)
            metrics['test_r2'] = row.get('R2', None)
    return metrics


def extract_step4_metrics(results_dir: Path, property_name: str) -> dict:
    """Extract metrics from step 4 (inverse design)."""
    metrics = {}
    import pandas as pd
    metrics_file = results_dir / f'step4_{property_name}' / 'metrics' / f'{property_name}_design.csv'
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        if len(df) > 0:
            # Calculate success rate and average error
            if 'hit_rate' in df.columns:
                metrics['avg_hit_rate'] = df['hit_rate'].mean()
            if 'mean_pred_hits' in df.columns and 'target_value' in df.columns:
                df['abs_error'] = abs(df['mean_pred_hits'] - df['target_value'])
                metrics['avg_abs_error'] = df['abs_error'].mean()
    return metrics


def extract_step5_metrics(results_dir: Path, polymer_class: str) -> dict:
    """Extract metrics from step 5 (class design)."""
    metrics = {}
    import pandas as pd
    metrics_file = results_dir / f'step5_{polymer_class}' / 'metrics' / f'{polymer_class}_class_design.csv'
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        if len(df) > 0:
            row = df.iloc[0]
            metrics['class_hit_rate'] = row.get('class_hit_rate', None)
            metrics['class_valid_rate'] = row.get('valid_rate', None)
    return metrics


def extract_step6_metrics(results_dir: Path, polymer_class: str, property_name: str) -> dict:
    """Extract metrics from step 6 (class-property joint design)."""
    metrics = {}
    import pandas as pd
    metrics_file = results_dir / f'step5_{polymer_class}_{property_name}' / 'metrics' / f'{polymer_class}_{property_name}_joint_design.csv'
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        if len(df) > 0:
            row = df.iloc[0]
            metrics['joint_hit_rate'] = row.get('joint_hit_rate', None)
            metrics['class_hit_rate'] = row.get('class_hit_rate', None)
            metrics['property_hit_rate'] = row.get('property_hit_rate', None)
            metrics['valid_rate'] = row.get('valid_rate', None)
    return metrics


def build_results_tag(args: argparse.Namespace) -> str:
    """Build a results tag based on which steps run and the target property/class."""
    if not args.skip_step6:
        return f"joint_{args.polymer_class}_{args.property}"
    if not args.skip_step5 and args.skip_step3 and args.skip_step4 and args.skip_step6:
        return f"class_{args.polymer_class}"
    if not args.skip_step3 or not args.skip_step4:
        return f"prop_{args.property}"
    return "base"


def archive_scaling_results(results_dir: Path, tag: str) -> None:
    """Archive scaling_results.json with a property/class tag."""
    src = results_dir / 'scaling_results.json'
    if not src.exists():
        return
    safe_tag = tag.replace('/', '_')
    dest = results_dir / f"scaling_results_{safe_tag}.json"
    if dest.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dest = results_dir / f"scaling_results_{safe_tag}_{timestamp}.json"
    shutil.copy2(src, dest)
    print(f"Archived scaling results to: {dest}")


def main():
    parser = argparse.ArgumentParser(description='Run full scaling pipeline with logging')
    parser.add_argument('--model_size', type=str, required=True,
                        choices=['small', 'medium', 'large', 'xl'],
                        help='Model size preset')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--property', type=str, default='Tg',
                        help='Property name for steps 3-5')
    parser.add_argument('--target', type=str, default=None,
                        help='Target value for inverse design (default uses property-specific preset)')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Tolerance for property matching (default uses property-specific preset)')
    parser.add_argument('--polymer_class', type=str, default='polyimide',
                        help='Polymer class for step 5')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples for step 2')
    parser.add_argument('--num_candidates', type=int, default=10000,
                        help='Number of candidates for steps 4-5')
    parser.add_argument('--skip_step1', action='store_true',
                        help='Skip step 1 (backbone training)')
    parser.add_argument('--skip_step2', action='store_true',
                        help='Skip step 2 (sampling)')
    parser.add_argument('--skip_step3', action='store_true',
                        help='Skip step 3 (property head)')
    parser.add_argument('--skip_step4', action='store_true',
                        help='Skip step 4 (inverse design)')
    parser.add_argument('--skip_step5', action='store_true',
                        help='Skip step 5 (class design)')
    parser.add_argument('--skip_step6', action='store_true',
                        help='Skip step 6 (class-property joint design)')
    args = parser.parse_args()

    def default_target_for_property(property_name: str) -> str:
        presets = {
            'Tg': '350',
            'Tm': '450',
            'Td': '550',
            'Eg': '8',
        }
        return presets.get(property_name, '300')

    def default_epsilon_for_property(property_name: str) -> float:
        presets = {
            'Tg': 30.0,
            'Tm': 30.0,
            'Td': 30.0,
            'Eg': 0.5,
        }
        return presets.get(property_name, 10.0)

    target_value = args.target if args.target is not None else default_target_for_property(args.property)
    epsilon_value = args.epsilon if args.epsilon is not None else default_epsilon_for_property(args.property)

    # Load config
    config = load_config(args.config)

    # Get results directory
    results_dir = Path(get_results_dir(args.model_size, config['paths']['results_dir']))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = ScalingLogger(results_dir, args.model_size)

    # Log model config
    model_config = get_model_config(args.model_size, config, model_type='sequence')
    training_config = get_training_config(args.model_size, config, model_type='sequence')

    # Load tokenizer to get vocab_size for param estimation
    tokenizer_path = Path(config['paths']['results_dir']) / 'tokenizer.json'
    vocab_size = 100  # Default estimate
    if tokenizer_path.exists():
        with open(tokenizer_path, 'r') as f:
            tokenizer_data = json.load(f)
            # SMILES tokenizer has 'token_to_id' dict
            if 'token_to_id' in tokenizer_data:
                vocab_size = len(tokenizer_data['token_to_id'])
            elif isinstance(tokenizer_data, dict):
                vocab_size = len(tokenizer_data)
    else:
        print(f"Warning: Tokenizer not found at {tokenizer_path}, using default vocab_size=100")

    num_params = estimate_params(model_config, vocab_size, model_type='sequence')
    logger.log_model_config(model_config, training_config, num_params)

    print("=" * 60)
    print(f"SCALING EXPERIMENT: {args.model_size.upper()}")
    print("=" * 60)
    print(f"Parameters: ~{num_params:,}")
    print(f"Results dir: {results_dir}")
    print(f"Property: {args.property}")
    print(f"Target: {target_value}")
    print(f"Epsilon: {epsilon_value}")
    print(f"Polymer class: {args.polymer_class}")
    print("=" * 60)

    # Step 1: Train backbone
    if not args.skip_step1:
        logger.start_step('step1_backbone')
        ret, stdout, stderr = run_step('step1_train_backbone.py', args.model_size)
        if ret != 0:
            logger.log_error('step1_backbone', f'Step failed with return code {ret}\nSTDERR: {stderr}')
            logger.finalize()
            return 1
        metrics = extract_step1_metrics(results_dir)
        logger.end_step('step1_backbone', metrics)
    else:
        logger.end_step('step1_backbone', status='skipped')

    # Step 2: Sample and evaluate
    if not args.skip_step2:
        # Check dependency: backbone checkpoint must exist
        if not check_backbone_checkpoint(results_dir):
            print(f"ERROR: Backbone checkpoint not found in {results_dir}/step1_backbone/checkpoints/")
            print("Run step 1 first or ensure checkpoint exists.")
            logger.log_error('step2_sampling', 'Missing backbone checkpoint')
            logger.finalize()
            return 1
        logger.start_step('step2_sampling')
        ret, stdout, stderr = run_step('step2_sample_and_evaluate.py', args.model_size,
                      f'--num_samples {args.num_samples}')
        if ret != 0:
            logger.log_error('step2_sampling', f'Step failed with return code {ret}\nSTDERR: {stderr}')
            logger.finalize()
            return 1
        metrics = extract_step2_metrics(results_dir)
        logger.end_step('step2_sampling', metrics)
    else:
        logger.end_step('step2_sampling', status='skipped')

    # Step 3: Train property head
    if not args.skip_step3:
        # Check dependency: backbone checkpoint must exist
        if not check_backbone_checkpoint(results_dir):
            print(f"ERROR: Backbone checkpoint not found in {results_dir}/step1_backbone/checkpoints/")
            print("Run step 1 first or ensure checkpoint exists.")
            logger.log_error('step3_property', 'Missing backbone checkpoint')
            logger.finalize()
            return 1
        logger.start_step('step3_property')
        ret, stdout, stderr = run_step('step3_train_property_head.py', args.model_size,
                      f'--property {args.property}')
        if ret != 0:
            logger.log_error('step3_property', f'Step failed with return code {ret}\nSTDERR: {stderr}')
            logger.finalize()
            return 1
        metrics = extract_step3_metrics(results_dir, args.property)
        logger.end_step('step3_property', metrics)
    else:
        logger.end_step('step3_property', status='skipped')

    # Step 4: Inverse design
    if not args.skip_step4:
        # Check dependency: property head checkpoint must exist
        if not check_property_head_checkpoint(results_dir, args.property):
            print(f"ERROR: Property head checkpoint not found for '{args.property}'")
            print("Run step 3 first or ensure checkpoint exists.")
            logger.log_error('step4_inverse_design', f'Missing {args.property} property head checkpoint')
            logger.finalize()
            return 1
        logger.start_step('step4_inverse_design')
        ret, stdout, stderr = run_step('step4_inverse_design.py', args.model_size,
                      f'--property {args.property} --targets {target_value} --epsilon {epsilon_value} '
                      f'--num_candidates {args.num_candidates}')
        if ret != 0:
            logger.log_error('step4_inverse_design', f'Step failed with return code {ret}\nSTDERR: {stderr}')
            logger.finalize()
            return 1
        metrics = extract_step4_metrics(results_dir, args.property)
        logger.end_step('step4_inverse_design', metrics)
    else:
        logger.end_step('step4_inverse_design', status='skipped')

    # Step 5: Class-guided design
    if not args.skip_step5:
        # Check dependency: backbone checkpoint must exist
        if not check_backbone_checkpoint(results_dir):
            print(f"ERROR: Backbone checkpoint not found in {results_dir}/step1_backbone/checkpoints/")
            print("Run step 1 first or ensure checkpoint exists.")
            logger.log_error('step5_class_design', 'Missing backbone checkpoint')
            logger.finalize()
            return 1
        logger.start_step('step5_class_design')
        ret, stdout, stderr = run_step('step5_class_design.py', args.model_size,
                      f'--polymer_class {args.polymer_class} --num_candidates {args.num_candidates}')
        if ret != 0:
            logger.log_error('step5_class_design', f'Step failed with return code {ret}\nSTDERR: {stderr}')
            logger.finalize()
            return 1
        metrics = extract_step5_metrics(results_dir, args.polymer_class)
        logger.end_step('step5_class_design', metrics)
    else:
        logger.end_step('step5_class_design', status='skipped')

    # Step 6: Class-property joint design
    if not args.skip_step6:
        # Check dependency: backbone checkpoint must exist
        if not check_backbone_checkpoint(results_dir):
            print(f"ERROR: Backbone checkpoint not found in {results_dir}/step1_backbone/checkpoints/")
            print("Run step 1 first or ensure checkpoint exists.")
            logger.log_error('step6_joint_design', 'Missing backbone checkpoint')
            logger.finalize()
            return 1
        # Check dependency: property head checkpoint must exist
        if not check_property_head_checkpoint(results_dir, args.property):
            print(f"ERROR: Property head checkpoint not found for '{args.property}'")
            print("Run step 3 first or ensure checkpoint exists.")
            logger.log_error('step6_joint_design', f'Missing {args.property} property head checkpoint')
            logger.finalize()
            return 1
        logger.start_step('step6_joint_design')
        ret, stdout, stderr = run_step('step5_class_design.py', args.model_size,
                      f'--polymer_class {args.polymer_class} --property {args.property} '
                      f'--target_value {target_value} --epsilon {epsilon_value} --num_candidates {args.num_candidates}')
        if ret != 0:
            logger.log_error('step6_joint_design', f'Step failed with return code {ret}\nSTDERR: {stderr}')
            logger.finalize()
            return 1
        metrics = extract_step6_metrics(results_dir, args.polymer_class, args.property)
        logger.end_step('step6_joint_design', metrics)
    else:
        logger.end_step('step6_joint_design', status='skipped')

    # Finalize
    logger.finalize()
    archive_scaling_results(results_dir, build_results_tag(args))

    return 0


if __name__ == '__main__':
    sys.exit(main())
