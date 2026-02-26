#!/usr/bin/env python
"""Step 1: Train diffusion backbone model."""

import os
import sys
import argparse
import math
import time
from functools import partial
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import torch
import pandas as pd
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from src.utils.config import load_config, save_config
from src.utils.plotting import PlotUtils
from src.utils.model_scales import (
    get_model_config, get_training_config, estimate_params,
    get_results_dir, print_model_info
)
from src.data.selfies_tokenizer import SelfiesTokenizer
from src.data.dataset import PolymerDataset, collate_fn, dynamic_collate_fn
from src.data.samplers import LengthBucketBatchSampler
from src.model.backbone import DiffusionBackbone
from src.model.diffusion import DiscreteMaskingDiffusion
from src.training.trainer_backbone import BackboneTrainer
from src.utils.reproducibility import seed_everything, save_run_metadata
from shared.unlabeled_data import require_preprocessed_unlabeled_splits
from shared.step1_recommendations import (
    apply_auto_step1_recommendations,
    maybe_apply_cuda_oom_env,
)
from src.utils.selfies_utils import ensure_selfies_column


def init_distributed():
    """Initialize torch.distributed if launched with torchrun."""
    if not dist.is_available():
        return False, 0, 1, 0, None
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0, None
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    return True, rank, world_size, local_rank, device


def main(args):
    """Main function."""
    # Load config
    config = load_config(args.config)
    cuda_alloc_conf = maybe_apply_cuda_oom_env(config)

    distributed, rank, world_size, local_rank, dist_device = init_distributed()
    is_main_process = (not distributed) or rank == 0

    # Set device
    device = dist_device if distributed else ('cuda' if torch.cuda.is_available() else 'cpu')
    if is_main_process:
        print(f"Using device: {device}")
        if cuda_alloc_conf:
            print(f"CUDA allocator config: PYTORCH_CUDA_ALLOC_CONF={cuda_alloc_conf}")

    # Override results_dir if model_size specified
    base_results_dir = config['paths']['results_dir']
    results_dir = Path(get_results_dir(args.model_size, base_results_dir))
    step_dir = results_dir / 'step1_backbone'
    metrics_dir = step_dir / 'metrics'
    figures_dir = step_dir / 'figures'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    seed_info = seed_everything(config['data']['random_seed'])
    if is_main_process:
        save_config(config, step_dir / 'config_used.yaml')
        save_run_metadata(step_dir, args.config, seed_info)

    if is_main_process:
        print("=" * 50)
        print("Step 1: Training Diffusion Backbone")
        print("=" * 50)

    # Get model and training config based on model_size
    backbone_config = get_model_config(args.model_size, config, model_type='sequence')
    if args.model_size:
        training_config = get_training_config(args.model_size, config, model_type='sequence')
        # Override training_backbone config
        config['training_backbone']['batch_size'] = training_config['batch_size']
        config['training_backbone']['learning_rate'] = training_config['learning_rate']
        config['training_backbone']['max_steps'] = training_config['max_steps']
        config['training_backbone']['warmup_steps'] = training_config['warmup_steps']
        config['optimization']['gradient_accumulation_steps'] = training_config['gradient_accumulation_steps']

    opt_config = config.setdefault('optimization', {})
    auto_recommend_summary = apply_auto_step1_recommendations(
        config=config,
        pipeline_kind='sequence',
        world_size=world_size,
        model_num_layers=int(backbone_config['num_layers']),
        model_config=backbone_config,
    )
    if auto_recommend_summary and is_main_process:
        gpu_info = auto_recommend_summary['gpu']
        system_info = auto_recommend_summary.get('system', {})
        gpu_mem = gpu_info.get('min_memory_gb', gpu_info.get('memory_gb'))
        print(
            "Applied system/GPU-based Step1 recommendations: "
            f"pipeline={auto_recommend_summary['pipeline_kind']}, "
            f"world_size={auto_recommend_summary['world_size']}, "
            f"reference_world_size={auto_recommend_summary['reference_world_size']}, "
            f"visible_gpus={gpu_info.get('device_count')}, "
            f"gpu={gpu_info.get('name')}, capability={gpu_info.get('capability')}, "
            f"min_gpu_mem_gb={gpu_mem}, "
            f"cpu_count={system_info.get('cpu_count')}, "
            f"avail_ram_gb={system_info.get('available_ram_gb')}, "
            f"override_existing={auto_recommend_summary['override_existing']}"
        )
        if auto_recommend_summary['applied_optimization']:
            print(
                "Optimization overrides applied: "
                f"{auto_recommend_summary['applied_optimization']}"
            )
        else:
            print("Auto recommendations enabled, but no optimization keys changed.")
        if auto_recommend_summary.get('applied_training_backbone'):
            print(
                "Training backbone overrides applied: "
                f"{auto_recommend_summary['applied_training_backbone']}"
            )
        memory_meta = auto_recommend_summary.get('memory_aware_batch_meta', {})
        if memory_meta.get('enabled'):
            original_b = memory_meta.get('original_per_rank_batch_size')
            recommended_b = memory_meta.get('recommended_per_rank_batch_size')
            if original_b != recommended_b:
                print(
                    "Memory-aware batch adjustment: "
                    f"per_rank_batch {original_b} -> {recommended_b} "
                    f"(gpu_mem_gb={memory_meta.get('gpu_memory_gb')}, "
                    f"seq_len={memory_meta.get('seq_len')})"
                )
        cpu_meta = auto_recommend_summary.get('cpu_oom_guard_meta', {})
        if cpu_meta.get('enabled'):
            print(
                "CPU OOM guard: "
                f"workers={cpu_meta.get('recommended_workers')}, "
                f"prefetch_factor={cpu_meta.get('recommended_prefetch_factor')}, "
                f"per_rank_cpu_budget={cpu_meta.get('per_rank_cpu_budget')}"
            )

    # Optional: preserve global batch size when scaling to more GPUs.
    target_global_batch = opt_config.get('target_global_batch_size')
    if target_global_batch is not None:
        target_global_batch = int(target_global_batch)
        if target_global_batch <= 0:
            raise ValueError("optimization.target_global_batch_size must be > 0 when set.")
        per_rank_batch = int(config['training_backbone']['batch_size'])
        micro_batch_global = per_rank_batch * max(1, world_size)
        grad_accum_steps = max(1, (target_global_batch + micro_batch_global - 1) // micro_batch_global)
        opt_config['gradient_accumulation_steps'] = grad_accum_steps
        if is_main_process:
            achieved_global_batch = micro_batch_global * grad_accum_steps
            print(
                "Adjusted gradient accumulation for world-size scaling: "
                f"target_global_batch={target_global_batch}, "
                f"micro_batch_global={micro_batch_global}, "
                f"grad_accum={grad_accum_steps}, "
                f"achieved_global_batch={achieved_global_batch}"
            )

    if is_main_process:
        per_rank_batch = int(config['training_backbone']['batch_size'])
        grad_accum_steps = int(config.get('optimization', {}).get('gradient_accumulation_steps', 1))
        effective_global_batch = per_rank_batch * max(1, world_size) * grad_accum_steps
        print(
            "Step1 effective global batch size: "
            f"{effective_global_batch} "
            f"(per_rank_batch={per_rank_batch}, world_size={world_size}, grad_accum={grad_accum_steps})"
        )

    # Load tokenizer (from base results dir which has the tokenizer)
    if is_main_process:
        print("\n1. Loading SELFIES tokenizer...")
    tokenizer_path = results_dir / 'tokenizer.json'
    if not tokenizer_path.exists():
        # Fall back to base results dir
        tokenizer_path = Path(base_results_dir) / 'tokenizer.json'
    tokenizer = SelfiesTokenizer.load(tokenizer_path)
    if is_main_process:
        print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Print model info if model_size specified
    if args.model_size and is_main_process:
        print_model_info(args.model_size, backbone_config, training_config,
                        tokenizer.vocab_size, model_type='sequence')

    # Load data (from base results dir which has the data)
    if is_main_process:
        print("\n2. Loading data...")
    repo_root = Path(__file__).resolve().parents[2]
    train_path, val_path = require_preprocessed_unlabeled_splits(repo_root)
    # Keep only columns Step1 can consume to limit per-rank host RAM.
    needed_cols = {"p_smiles", "selfies"}
    train_df = pd.read_csv(train_path, usecols=lambda col: col in needed_cols)
    val_df = pd.read_csv(val_path, usecols=lambda col: col in needed_cols)
    if not ({"p_smiles", "selfies"} & set(train_df.columns)):
        raise ValueError("Train split must contain at least one of: p_smiles, selfies")
    if not ({"p_smiles", "selfies"} & set(val_df.columns)):
        raise ValueError("Val split must contain at least one of: p_smiles, selfies")

    # Optionally subsample training data (validation always full)
    train_fraction = config.get('data', {}).get('train_fraction', 1.0)
    if train_fraction <= 0 or train_fraction > 1:
        raise ValueError("data.train_fraction must be within (0, 1].")
    if train_fraction < 1.0:
        full_train_count = len(train_df)
        n_train = max(1, int(round(full_train_count * train_fraction)))
        train_df = train_df.sample(
            n=n_train, random_state=config['data']['random_seed']
        ).reset_index(drop=True)
        if is_main_process:
            print(f"Using {n_train}/{full_train_count} train samples ({train_fraction:.2%})")

    # Optionally subsample validation data before SELFIES conversion.
    # This reduces conversion overhead during periodic evaluation.
    train_cfg = config.get('training_backbone', {})
    val_fraction = float(train_cfg.get('val_fraction', 1.0))
    if val_fraction <= 0 or val_fraction > 1:
        raise ValueError("training_backbone.val_fraction must be within (0, 1].")
    if val_fraction < 1.0:
        full_val_count = len(val_df)
        n_val = max(1, int(round(full_val_count * val_fraction)))
        val_df = val_df.sample(
            n=n_val, random_state=config['data']['random_seed']
        ).reset_index(drop=True)
        if is_main_process:
            print(f"Using {n_val}/{full_val_count} val samples ({val_fraction:.2%})")
    val_max_samples = int(train_cfg.get('val_max_samples', 0))
    if val_max_samples > 0 and len(val_df) > val_max_samples:
        full_val_count = len(val_df)
        val_df = val_df.sample(
            n=val_max_samples, random_state=config['data']['random_seed']
        ).reset_index(drop=True)
        if is_main_process:
            print(f"Capping val samples to {val_max_samples}/{full_val_count} for faster eval")

    # Backward compatibility: Step0 now exports p_smiles/sa_score only.
    had_train_selfies = 'selfies' in train_df.columns
    had_val_selfies = 'selfies' in val_df.columns
    train_before = len(train_df)
    val_before = len(val_df)
    train_df = ensure_selfies_column(train_df)
    val_df = ensure_selfies_column(val_df)
    if is_main_process and (not had_train_selfies or not had_val_selfies):
        print("SELFIES column missing in unlabeled CSVs; regenerated from p_smiles.")
    if is_main_process and (len(train_df) != train_before or len(val_df) != val_before):
        print(
            f"Dropped {train_before - len(train_df)} train and {val_before - len(val_df)} val "
            "rows due to failed p_smiles->SELFIES conversion."
        )
    # Keep only model-required sequence text to avoid carrying unused metadata.
    train_df = train_df[['selfies']].copy()
    val_df = val_df[['selfies']].copy()

    # Get optimization settings
    opt_config = config.get('optimization', {})
    cache_tokenization = opt_config.get('cache_tokenization', False)
    cache_max_samples = int(opt_config.get('cache_tokenization_max_samples', 500000))
    num_workers = int(opt_config.get('step1_num_workers', opt_config.get('num_workers', 4)))
    pin_memory = opt_config.get('pin_memory', True)
    prefetch_factor = opt_config.get('prefetch_factor', 2)
    dynamic_padding = bool(opt_config.get('dynamic_padding', False))
    length_bucket_sampler = bool(opt_config.get('length_bucket_sampler', False))
    bucket_size_multiplier = int(opt_config.get('bucket_size_multiplier', 50))
    if bucket_size_multiplier <= 0:
        raise ValueError("optimization.bucket_size_multiplier must be > 0.")
    step1_persistent_workers = bool(
        opt_config.get('step1_persistent_workers', opt_config.get('persistent_workers', False))
    )

    # Bound DataLoader workers to per-rank CPU budget to avoid oversubscription.
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1") or 1)
    slurm_cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
    host_cpus = os.cpu_count() or 1
    if slurm_cpus_per_task > 0:
        per_rank_cpu_budget = max(1, slurm_cpus_per_task // max(1, local_world_size))
    else:
        per_rank_cpu_budget = max(1, host_cpus // max(1, local_world_size))
    per_rank_worker_cap = max(1, per_rank_cpu_budget - 2)
    if num_workers <= 0:
        num_workers = per_rank_worker_cap
        if is_main_process:
            print(
                "Auto-selected DataLoader workers per rank: "
                f"{num_workers} (cpu_budget={per_rank_cpu_budget}, local_world_size={local_world_size})"
            )
    elif num_workers > per_rank_worker_cap:
        if is_main_process:
            print(
                f"Capping num_workers from {num_workers} to {per_rank_worker_cap} "
                f"(cpu_budget={per_rank_cpu_budget}, local_world_size={local_world_size})"
            )
        num_workers = per_rank_worker_cap
    persistent_workers = step1_persistent_workers and num_workers > 0

    # Guard against memory blow-up: full-cache can be too large for multi-million datasets.
    total_samples = len(train_df) + len(val_df)
    if cache_tokenization and distributed:
        if is_main_process:
            print("Disabling cache_tokenization under DDP to avoid per-rank RAM duplication.")
        cache_tokenization = False
    elif cache_tokenization and total_samples > cache_max_samples:
        if is_main_process:
            print(
                f"Disabling cache_tokenization for {total_samples:,} samples "
                f"(limit={cache_max_samples:,})."
            )
        cache_tokenization = False

    # Create datasets
    train_dataset = PolymerDataset(
        train_df,
        tokenizer,
        cache_tokenization=cache_tokenization,
        pad_to_max_length=not dynamic_padding,
    )
    val_dataset = PolymerDataset(
        val_df,
        tokenizer,
        cache_tokenization=cache_tokenization,
        pad_to_max_length=not dynamic_padding,
    )

    active_collate_fn = collate_fn
    if dynamic_padding:
        active_collate_fn = partial(dynamic_collate_fn, pad_token_id=tokenizer.pad_token_id)

    # Create dataloaders
    batch_size = config['training_backbone']['batch_size']
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    if length_bucket_sampler:
        train_lengths = train_df['selfies'].astype(str).str.len().tolist()
        train_batch_sampler = LengthBucketBatchSampler(
            lengths=train_lengths,
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
            seed=config['data']['random_seed'],
            bucket_size_multiplier=bucket_size_multiplier,
            num_replicas=world_size if distributed else 1,
            rank=rank if distributed else 0,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=active_collate_fn,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
    else:
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=active_collate_fn,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=active_collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    if is_main_process:
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        if dynamic_padding:
            print("Using dynamic batch padding for Step1 dataloaders.")
        if length_bucket_sampler:
            print(
                f"Using length-bucket batching (bucket_size_multiplier={bucket_size_multiplier})."
            )
        print(f"DataLoader workers per rank: {num_workers}")

    # Create model
    if is_main_process:
        print("\n3. Creating model...")
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

    model = DiscreteMaskingDiffusion(
        backbone=backbone,
        num_steps=config['diffusion']['num_steps'],
        beta_min=config['diffusion']['beta_min'],
        force_clean_t0=config['diffusion'].get('force_clean_t0', False),
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process:
        print(f"Total parameters: {num_params:,}")
        print(f"Trainable parameters: {num_trainable:,}")

    # Resume from checkpoint if specified
    if args.resume:
        if is_main_process:
            print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        # Handle torch.compile() state dict (keys have _orig_mod. prefix)
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    # Create trainer
    if is_main_process:
        print("\n4. Starting training...")
    trainer = BackboneTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        device=device,
        output_dir=str(step_dir),
        step_dir=str(step_dir),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank
    )

    # Train
    train_start_time = time.perf_counter()
    history = trainer.train()
    training_time_sec = time.perf_counter() - train_start_time

    # Create loss plot
    if is_main_process:
        print("\n5. Creating loss plot...")
        plotter = PlotUtils(
            figure_size=tuple(config['plotting']['figure_size']),
            font_size=config['plotting']['font_size'],
            dpi=config['plotting']['dpi']
        )

        plotter.loss_curve(
            train_losses=history['train_losses'],
            val_losses=history['val_losses'],
            xlabel='Step',
            ylabel='Loss',
            title='Backbone Training Loss',
            save_path=figures_dir / 'backbone_loss_curve.png'
        )

        epoch_train_losses = history.get('epoch_train_losses', [])
        epoch_val_losses = history.get('epoch_val_losses', [])
        if epoch_train_losses:
            train_bpb = [loss / math.log(2.0) for loss in epoch_train_losses]
            val_bpb = [loss / math.log(2.0) for loss in epoch_val_losses] if epoch_val_losses else None
            plotter.loss_curve(
                train_losses=train_bpb,
                val_losses=val_bpb,
                xlabel='Epoch',
                ylabel='BPB',
                title='Backbone Training BPB',
                save_path=figures_dir / 'backbone_bpb_curve.png'
            )

        summary = {
            'total_steps': int(trainer.global_step),
            'best_val_loss': round(float(history['best_val_loss']), 4),
            'final_train_loss': round(float(history['train_losses'][-1]), 4) if history.get('train_losses') else None,
            'final_val_loss': round(float(history['val_losses'][-1]), 4) if history.get('val_losses') else None,
            'training_time_sec': round(float(training_time_sec), 2),
            'training_time_hr': round(float(training_time_sec) / 3600.0, 4),
            'num_params': int(num_params),
            'num_trainable_params': int(num_trainable),
        }
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(metrics_dir / 'training_summary.csv', index=False)

        print("\n" + "=" * 50)
        print("Backbone training complete!")
        print(f"Best validation loss: {history['best_val_loss']:.4f}")
        print(f"Training time: {training_time_sec:.2f} sec ({training_time_sec / 3600.0:.4f} hr)")
        print(f"Checkpoints saved to: {step_dir / 'checkpoints'}")
        print(f"Summary saved to: {metrics_dir / 'training_summary.csv'}")
        print("=" * 50)

    if distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train diffusion backbone')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_size', type=str, default=None,
                        choices=['small', 'medium', 'large', 'xl'],
                        help='Model size preset (small: ~12M, medium: ~50M, large: ~150M, xl: ~400M)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    main(args)
