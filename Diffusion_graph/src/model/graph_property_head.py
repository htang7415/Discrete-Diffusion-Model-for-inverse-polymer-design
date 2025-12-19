"""Graph-based property prediction head for regression tasks."""

import torch
import torch.nn as nn
from typing import Optional, Dict, List


class GraphPropertyHead(nn.Module):
    """Regression head for property prediction from graph embeddings.

    Takes pooled hidden states from graph backbone and predicts a scalar property.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [256, 64],
        dropout: float = 0.1
    ):
        """Initialize property head.

        Args:
            input_size: Input feature dimension (backbone hidden size).
            hidden_sizes: List of hidden layer sizes.
            dropout: Dropout rate.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        # Build MLP layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Pooled hidden states of shape [batch, hidden_size].

        Returns:
            Predictions of shape [batch, 1].
        """
        return self.mlp(hidden_states)


class GraphPropertyPredictor(nn.Module):
    """Full property predictor combining graph backbone and head."""

    def __init__(
        self,
        backbone: nn.Module,
        property_head: GraphPropertyHead,
        freeze_backbone: bool = True,
        finetune_last_layers: int = 0,
        pooling: str = 'mean',
        default_timestep: int = 1
    ):
        """Initialize graph property predictor.

        Args:
            backbone: Pretrained graph backbone model (GraphDiffusionBackbone).
            property_head: Property prediction head.
            freeze_backbone: Whether to freeze backbone weights.
            finetune_last_layers: Number of last layers to finetune (if freeze_backbone=True).
            pooling: Pooling method ('mean', 'sum', 'max').
            default_timestep: Default timestep for backbone (0 = no masking).
        """
        super().__init__()

        self.backbone = backbone
        self.property_head = property_head
        self.pooling = pooling
        self.default_timestep = default_timestep

        # Freeze/unfreeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Optionally finetune last few layers
            if finetune_last_layers > 0 and hasattr(self.backbone, 'blocks'):
                for block in self.backbone.blocks[-finetune_last_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True
                # Also unfreeze final norm
                if hasattr(self.backbone, 'ln_final'):
                    for param in self.backbone.ln_final.parameters():
                        param.requires_grad = True

    def forward(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        M: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            X: Node tokens of shape [batch, Nmax].
            E: Edge tokens of shape [batch, Nmax, Nmax].
            M: Node mask of shape [batch, Nmax].
            timesteps: Timesteps (uses default if None).

        Returns:
            Predictions of shape [batch, 1].
        """
        batch_size = X.shape[0]
        device = X.device

        # Use default timestep if not provided
        if timesteps is None:
            timesteps = torch.full(
                (batch_size,), self.default_timestep,
                device=device, dtype=torch.long
            )

        # Get pooled output from backbone
        pooled = self.backbone.get_node_embeddings(
            X, E, timesteps, M, pooling=self.pooling
        )

        # Predict property
        predictions = self.property_head(pooled)

        return predictions

    def compute_loss(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        M: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute MSE loss.

        Args:
            X: Node tokens.
            E: Edge tokens.
            M: Node mask.
            labels: Target property values of shape [batch] or [batch, 1].

        Returns:
            Dictionary with 'loss' and 'predictions'.
        """
        predictions = self.forward(X, E, M)

        # Ensure labels have correct shape
        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)

        loss = nn.MSELoss()(predictions, labels)

        return {
            'loss': loss,
            'predictions': predictions
        }

    def predict(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        M: torch.Tensor
    ) -> torch.Tensor:
        """Make predictions (inference mode).

        Args:
            X: Node tokens.
            E: Edge tokens.
            M: Node mask.

        Returns:
            Predictions of shape [batch].
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(X, E, M)
        return predictions.squeeze(-1)

    def save_pretrained(self, path: str, include_backbone: bool = False):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
            include_backbone: Whether to include backbone weights.
        """
        checkpoint = {
            'property_head_state_dict': self.property_head.state_dict(),
            'pooling': self.pooling,
            'default_timestep': self.default_timestep,
            'input_size': self.property_head.input_size,
            'hidden_sizes': self.property_head.hidden_sizes
        }

        if include_backbone:
            checkpoint['backbone_state_dict'] = self.backbone.state_dict()

        torch.save(checkpoint, path)

    def load_property_head(self, path: str):
        """Load property head weights.

        Args:
            path: Path to checkpoint.
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.property_head.load_state_dict(checkpoint['property_head_state_dict'])
