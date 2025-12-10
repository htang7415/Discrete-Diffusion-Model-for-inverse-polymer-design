"""Trainer for property prediction heads."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class PropertyTrainer:
    """Trainer for property prediction heads."""

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        property_name: str,
        config: Dict,
        device: str = 'cuda',
        output_dir: str = 'results',
        normalization_params: Optional[Dict] = None,
        step_dir: str = None
    ):
        """Initialize trainer.

        Args:
            model: Property predictor model.
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader.
            test_dataloader: Test data loader.
            property_name: Name of the property.
            config: Training configuration.
            device: Device for training.
            output_dir: Output directory for shared artifacts (checkpoints).
            normalization_params: Normalization parameters (mean, std).
            step_dir: Step-specific output directory for metrics/figures.
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.property_name = property_name
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.step_dir = Path(step_dir) if step_dir else self.output_dir
        self.normalization_params = normalization_params or {'mean': 0.0, 'std': 1.0}

        # Create output directories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.step_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Training config
        train_config = config['training_property']
        self.learning_rate = train_config['learning_rate']
        self.weight_decay = train_config['weight_decay']
        self.num_epochs = train_config['num_epochs']
        self.patience = train_config['patience']

        # Initialize optimizer
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Initialize scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=1e-6
        )

        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

    def train(self) -> Dict:
        """Run training loop.

        Returns:
            Training history and test metrics.
        """
        print(f"Training property head for {self.property_name}")
        print(f"Train samples: {len(self.train_dataloader.dataset)}")
        print(f"Val samples: {len(self.val_dataloader.dataset)}")
        print(f"Test samples: {len(self.test_dataloader.dataset)}")

        for epoch in range(self.num_epochs):
            # Training epoch
            train_loss = self._train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validation
            val_loss = self._validate()
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            self.scheduler.step()

            # Save checkpoint
            improved = self._save_checkpoint(val_loss, epoch)

            print(f"Epoch {epoch+1}/{self.num_epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Early stopping
            if not improved:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                self.patience_counter = 0

        # Load best model for evaluation
        self._load_best_checkpoint()

        # Evaluate on test set
        test_metrics = self._evaluate_test()

        # Save history and results
        self._save_results(test_metrics)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'test_metrics': test_metrics,
            'best_val_loss': self.best_val_loss
        }

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss:.4f}'})

        return total_loss / max(num_batches, 1)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step.

        Args:
            batch: Batch of data.

        Returns:
            Loss value.
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model.compute_loss(input_ids, labels, attention_mask)
        loss = outputs['loss']

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _validate(self) -> float:
        """Run validation.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model.compute_loss(input_ids, labels, attention_mask)
                total_loss += outputs['loss'].item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _evaluate_test(self) -> Dict:
        """Evaluate on test set.

        Returns:
            Dictionary with test metrics.
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                preds = self.model.predict(input_ids, attention_mask)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        # Denormalize if needed
        mean = self.normalization_params['mean']
        std = self.normalization_params['std']
        all_preds = np.array(all_preds) * std + mean
        all_labels = np.array(all_labels) * std + mean

        # Compute metrics (round to 4 decimal places)
        mae = round(mean_absolute_error(all_labels, all_preds), 4)
        rmse = round(np.sqrt(mean_squared_error(all_labels, all_preds)), 4)
        r2 = round(r2_score(all_labels, all_preds), 4)

        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'predictions': [round(p, 4) for p in all_preds.tolist()],
            'labels': [round(l, 4) for l in all_labels.tolist()]
        }

    def _save_checkpoint(self, val_loss: float, epoch: int) -> bool:
        """Save model checkpoint.

        Args:
            val_loss: Validation loss.
            epoch: Current epoch.

        Returns:
            Whether the model improved.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'normalization_params': self.normalization_params
        }

        improved = False
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, self.checkpoint_dir / f'{self.property_name}_best.pt')
            improved = True

        # Always save last checkpoint
        torch.save(checkpoint, self.checkpoint_dir / f'{self.property_name}_last.pt')

        return improved

    def _load_best_checkpoint(self):
        """Load best checkpoint."""
        checkpoint_path = self.checkpoint_dir / f'{self.property_name}_best.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best checkpoint with val_loss: {checkpoint['val_loss']:.4f}")

    def _save_results(self, test_metrics: Dict):
        """Save training results.

        Args:
            test_metrics: Test set metrics.
        """
        # Save loss curves
        loss_df = pd.DataFrame({
            'epoch': list(range(1, len(self.train_losses) + 1)),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        })
        loss_df.to_csv(self.metrics_dir / f'{self.property_name}_loss_curve.csv', index=False)

        # Save test metrics
        metrics_df = pd.DataFrame([{
            'property': self.property_name,
            'MAE': test_metrics['MAE'],
            'RMSE': test_metrics['RMSE'],
            'R2': test_metrics['R2'],
            'best_val_loss': self.best_val_loss
        }])
        metrics_df.to_csv(self.metrics_dir / f'{self.property_name}_test_metrics.csv', index=False)

        # Save predictions for parity plot
        pred_df = pd.DataFrame({
            'true': test_metrics['labels'],
            'predicted': test_metrics['predictions']
        })
        pred_df.to_csv(self.metrics_dir / f'{self.property_name}_predictions.csv', index=False)

        print(f"\nTest Results for {self.property_name}:")
        print(f"  MAE: {test_metrics['MAE']:.4f}")
        print(f"  RMSE: {test_metrics['RMSE']:.4f}")
        print(f"  R²: {test_metrics['R2']:.4f}")

    def get_predictions(
        self,
        dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions for a dataloader.

        Args:
            dataloader: Data loader.

        Returns:
            Tuple of (predictions, labels).
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                preds = self.model.predict(input_ids, attention_mask)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        # Denormalize
        mean = self.normalization_params['mean']
        std = self.normalization_params['std']
        all_preds = np.array(all_preds) * std + mean
        all_labels = np.array(all_labels) * std + mean

        return all_preds, all_labels
