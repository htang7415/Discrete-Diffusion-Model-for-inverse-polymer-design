"""Hyperparameter tuning utilities."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import product
from tqdm import tqdm
import json


class BackboneTuner:
    """Hyperparameter tuning for backbone using proxy model."""

    def __init__(
        self,
        train_dataset,
        val_dataset,
        tokenizer,
        config: Dict,
        device: str = 'cuda',
        output_dir: str = 'results/tuning'
    ):
        """Initialize tuner.

        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            tokenizer: Tokenizer instance.
            config: Base configuration.
            device: Device for training.
            output_dir: Output directory.
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tune_backbone(
        self,
        param_grid: Optional[Dict] = None,
        num_steps: int = 5000,
        batch_size: int = 64
    ) -> pd.DataFrame:
        """Tune backbone hyperparameters.

        Args:
            param_grid: Parameter grid for search.
            num_steps: Training steps per trial.
            batch_size: Batch size.

        Returns:
            DataFrame with tuning results.
        """
        if param_grid is None:
            param_grid = {
                'learning_rate': [3e-5, 1e-4, 3e-4, 5e-4],
                'warmup_steps': [500, 1000, 2000],
                'dropout': [0.0, 0.1, 0.2],
            }

        # Use subset for faster training
        train_subset = Subset(self.train_dataset, range(min(10000, len(self.train_dataset))))
        val_subset = Subset(self.val_dataset, range(min(2000, len(self.val_dataset))))

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        results = []

        for i, params in enumerate(tqdm(combinations, desc="Tuning trials")):
            param_dict = dict(zip(param_names, params))
            print(f"\nTrial {i+1}/{len(combinations)}: {param_dict}")

            try:
                val_loss = self._run_trial(
                    train_loader, val_loader,
                    param_dict, num_steps
                )

                results.append({
                    **param_dict,
                    'val_loss': val_loss,
                    'trial': i
                })
            except Exception as e:
                print(f"Trial failed: {e}")
                results.append({
                    **param_dict,
                    'val_loss': float('inf'),
                    'trial': i
                })

        # Save results
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('val_loss')
        results_df.to_csv(self.output_dir / 'backbone_tuning_results.csv', index=False)

        # Save best config
        best_params = results_df.iloc[0].to_dict()
        with open(self.output_dir / 'best_backbone_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)

        print(f"\nBest parameters: {best_params}")
        return results_df

    def _run_trial(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        params: Dict,
        num_steps: int
    ) -> float:
        """Run a single tuning trial.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            params: Hyperparameters.
            num_steps: Number of training steps.

        Returns:
            Final validation loss.
        """
        from ..model.backbone import DiffusionBackbone
        from ..model.diffusion import DiscreteMaskingDiffusion

        # Create proxy model (smaller)
        proxy_config = self.config['proxy_backbone']
        backbone = DiffusionBackbone(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=proxy_config['hidden_size'],
            num_layers=proxy_config['num_layers'],
            num_heads=proxy_config['num_heads'],
            ffn_hidden_size=proxy_config['ffn_hidden_size'],
            max_position_embeddings=proxy_config['max_position_embeddings'],
            num_diffusion_steps=self.config['diffusion']['num_steps'],
            dropout=params.get('dropout', 0.1),
            pad_token_id=self.tokenizer.pad_token_id
        )

        model = DiscreteMaskingDiffusion(
            backbone=backbone,
            num_steps=self.config['diffusion']['num_steps'],
            beta_min=self.config['diffusion']['beta_min'],
            beta_max=self.config['diffusion']['beta_max'],
            mask_token_id=self.tokenizer.mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        ).to(self.device)

        # Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=0.01
        )

        # Training loop
        model.train()
        step = 0
        warmup_steps = params.get('warmup_steps', 1000)

        for _ in range(100):  # Max epochs
            for batch in train_loader:
                if step >= num_steps:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = outputs['loss']
                loss.backward()

                # Learning rate warmup
                if step < warmup_steps:
                    lr_scale = (step + 1) / warmup_steps
                    for pg in optimizer.param_groups:
                        pg['lr'] = params['learning_rate'] * lr_scale

                optimizer.step()
                step += 1

            if step >= num_steps:
                break

        # Evaluate
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = model(input_ids, attention_mask)
                total_loss += outputs['loss'].item()
                num_batches += 1

        return total_loss / max(num_batches, 1)


class PropertyHeadTuner:
    """Hyperparameter tuning for property heads."""

    def __init__(
        self,
        backbone,
        train_dataset,
        val_dataset,
        config: Dict,
        device: str = 'cuda',
        output_dir: str = 'results/tuning'
    ):
        """Initialize tuner.

        Args:
            backbone: Pretrained backbone model.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            config: Base configuration.
            device: Device.
            output_dir: Output directory.
        """
        self.backbone = backbone
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tune_property_head(
        self,
        property_name: str,
        param_grid: Optional[Dict] = None,
        num_epochs: int = 20,
        batch_size: int = 32
    ) -> pd.DataFrame:
        """Tune property head hyperparameters.

        Args:
            property_name: Property name.
            param_grid: Parameter grid.
            num_epochs: Epochs per trial.
            batch_size: Batch size.

        Returns:
            DataFrame with results.
        """
        if param_grid is None:
            param_grid = {
                'learning_rate': [1e-5, 3e-5, 1e-4, 3e-4],
                'hidden_sizes': [[256, 64], [128], [512, 128]],
                'dropout': [0.0, 0.1],
            }

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

        # Generate combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        results = []

        for i, params in enumerate(tqdm(combinations, desc="Tuning trials")):
            param_dict = dict(zip(param_names, params))
            print(f"\nTrial {i+1}/{len(combinations)}: {param_dict}")

            try:
                val_loss = self._run_trial(
                    train_loader, val_loader,
                    param_dict, num_epochs
                )

                results.append({
                    **{k: str(v) for k, v in param_dict.items()},
                    'val_loss': val_loss,
                    'trial': i
                })
            except Exception as e:
                print(f"Trial failed: {e}")
                results.append({
                    **{k: str(v) for k, v in param_dict.items()},
                    'val_loss': float('inf'),
                    'trial': i
                })

        # Save results
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('val_loss')
        results_df.to_csv(self.output_dir / f'{property_name}_tuning_results.csv', index=False)

        # Save best config
        best_params = results_df.iloc[0].to_dict()
        with open(self.output_dir / f'{property_name}_best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)

        print(f"\nBest parameters: {best_params}")
        return results_df

    def _run_trial(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        params: Dict,
        num_epochs: int
    ) -> float:
        """Run a single trial.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            params: Hyperparameters.
            num_epochs: Number of epochs.

        Returns:
            Final validation loss.
        """
        from ..model.property_head import PropertyHead, PropertyPredictor

        hidden_sizes = params['hidden_sizes']
        if isinstance(hidden_sizes, str):
            hidden_sizes = eval(hidden_sizes)

        # Create property head
        head = PropertyHead(
            input_size=self.backbone.hidden_size,
            hidden_sizes=hidden_sizes,
            dropout=params.get('dropout', 0.1)
        )

        model = PropertyPredictor(
            backbone=self.backbone,
            property_head=head,
            freeze_backbone=True,
            pooling='mean'
        ).to(self.device)

        # Optimizer
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=params['learning_rate'],
            weight_decay=0.01
        )

        # Training
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            model.train()
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = model.compute_loss(input_ids, labels, attention_mask)
                outputs['loss'].backward()
                optimizer.step()

            # Validation
            model.eval()
            total_loss = 0.0
            num_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    outputs = model.compute_loss(input_ids, labels, attention_mask)
                    total_loss += outputs['loss'].item()
                    num_batches += 1

            val_loss = total_loss / max(num_batches, 1)
            best_val_loss = min(best_val_loss, val_loss)

        return best_val_loss
