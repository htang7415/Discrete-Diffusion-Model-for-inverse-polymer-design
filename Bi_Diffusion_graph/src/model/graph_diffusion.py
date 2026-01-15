"""Graph-based discrete masking diffusion model.

Handles dual masking for nodes (X) and edges (E) with symmetric edge constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class GraphMaskingDiffusion(nn.Module):
    """Discrete masking diffusion for graphs.

    Forward process: gradually mask nodes and edges
    Reverse process: predict original nodes and edges from masked graph
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_steps: int = 100,
        beta_min: float = 0.05,
        beta_max: float = 0.95,
        node_mask_id: int = None,
        edge_mask_id: int = 5,
        node_pad_id: int = None,
        edge_none_id: int = 0,
        lambda_node: float = 1.0,
        lambda_edge: float = 0.5
    ):
        """Initialize graph diffusion model.

        Args:
            backbone: Graph transformer backbone model.
            num_steps: Number of diffusion steps T.
            beta_min: Minimum mask probability.
            beta_max: Maximum mask probability.
            node_mask_id: ID of node MASK token.
            edge_mask_id: ID of edge MASK token (default 5).
            node_pad_id: ID of node PAD token.
            edge_none_id: ID of edge NONE token (default 0).
            lambda_node: Weight for node loss.
            lambda_edge: Weight for edge loss.
        """
        super().__init__()

        self.backbone = backbone
        self.num_steps = num_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.node_mask_id = node_mask_id if node_mask_id is not None else backbone.atom_vocab_size - 2
        self.edge_mask_id = edge_mask_id
        self.node_pad_id = node_pad_id if node_pad_id is not None else backbone.atom_vocab_size - 1
        self.edge_none_id = edge_none_id
        self.lambda_node = lambda_node
        self.lambda_edge = lambda_edge

        # Precompute mask schedule
        self.register_buffer(
            'mask_schedule',
            self._compute_mask_schedule()
        )

    def _compute_mask_schedule(self) -> torch.Tensor:
        """Compute linear mask probability schedule.

        Returns:
            Tensor of shape [num_steps + 1] with mask probabilities.
        """
        steps = torch.arange(self.num_steps + 1, dtype=torch.float32)
        schedule = self.beta_min + (self.beta_max - self.beta_min) * steps / self.num_steps
        return schedule

    def get_mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """Get mask probability for timestep t.

        Args:
            t: Timestep tensor.

        Returns:
            Mask probability.
        """
        return self.mask_schedule[t]

    def forward_process(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        M: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply forward noising (masking) to nodes and edges.

        Args:
            X: Clean node tokens of shape [batch, Nmax].
            E: Clean edge tokens of shape [batch, Nmax, Nmax].
            M: Node mask of shape [batch, Nmax] (1 for real, 0 for padding).
            t: Timesteps of shape [batch].

        Returns:
            Tuple of (X_noisy, E_noisy, node_mask_indicator, edge_mask_indicator).
        """
        B, N = X.shape
        device = X.device

        # Get mask probabilities for each sample
        mask_probs = self.get_mask_prob(t)  # [batch]
        mask_probs = mask_probs.view(B, 1)  # [batch, 1] for broadcasting

        # ========== Mask Nodes ==========
        random_node = torch.rand(B, N, device=device)
        node_should_mask = (random_node < mask_probs) & (M == 1)

        X_noisy = X.clone()
        X_noisy[node_should_mask] = self.node_mask_id

        # ========== Mask Edges ==========
        random_edge = torch.rand(B, N, N, device=device)
        mask_probs_2d = mask_probs.view(B, 1, 1)

        # Valid edges are between real nodes
        valid_edge_mask = M.unsqueeze(2) * M.unsqueeze(1)  # [B, N, N]

        edge_should_mask = (random_edge < mask_probs_2d) & (valid_edge_mask == 1)

        # Enforce symmetry: if (i,j) is masked, (j,i) must also be masked
        edge_should_mask = edge_should_mask | edge_should_mask.transpose(1, 2)

        E_noisy = E.clone()
        E_noisy[edge_should_mask] = self.edge_mask_id

        return X_noisy, E_noisy, node_should_mask, edge_should_mask

    def forward(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        M: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            X: Clean node tokens of shape [batch, Nmax].
            E: Clean edge tokens of shape [batch, Nmax, Nmax].
            M: Node mask of shape [batch, Nmax].
            timesteps: Optional fixed timesteps (random if None).

        Returns:
            Dictionary with 'loss', 'node_loss', 'edge_loss', and other info.
        """
        B = X.shape[0]
        device = X.device

        # Sample random timesteps if not provided
        if timesteps is None:
            timesteps = torch.randint(1, self.num_steps + 1, (B,), device=device)

        # Apply forward process (masking)
        X_noisy, E_noisy, node_mask_ind, edge_mask_ind = self.forward_process(X, E, M, timesteps)

        # Get model predictions
        node_logits, edge_logits = self.backbone(X_noisy, E_noisy, timesteps, M)

        # Compute losses
        node_loss = self._compute_node_loss(node_logits, X, node_mask_ind)
        edge_loss = self._compute_edge_loss(edge_logits, E, edge_mask_ind, M)

        # Weighted total loss
        total_loss = self.lambda_node * node_loss + self.lambda_edge * edge_loss

        return {
            'loss': total_loss,
            'node_loss': node_loss,
            'edge_loss': edge_loss,
            'node_logits': node_logits,
            'edge_logits': edge_logits,
            'X_noisy': X_noisy,
            'E_noisy': E_noisy,
            'node_mask_ind': node_mask_ind,
            'edge_mask_ind': edge_mask_ind,
            'timesteps': timesteps
        }

    def _compute_node_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask_indicator: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss for node predictions.

        Args:
            logits: Predicted logits of shape [batch, Nmax, atom_vocab_size].
            targets: Target node tokens of shape [batch, Nmax].
            mask_indicator: Boolean tensor indicating masked positions.

        Returns:
            Scalar loss value.
        """
        if not mask_indicator.any():
            return torch.tensor(0.0, device=logits.device)

        # Select only masked positions
        masked_logits = logits[mask_indicator]  # [num_masked, vocab_size]
        masked_targets = targets[mask_indicator]  # [num_masked]

        loss = F.cross_entropy(masked_logits, masked_targets, reduction='mean')
        return loss

    def _compute_edge_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask_indicator: torch.Tensor,
        M: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss for edge predictions.

        Only considers upper triangle to avoid double-counting symmetric edges.

        Args:
            logits: Predicted logits of shape [batch, Nmax, Nmax, edge_vocab_size].
            targets: Target edge tokens of shape [batch, Nmax, Nmax].
            mask_indicator: Boolean tensor indicating masked edges.
            M: Node mask.

        Returns:
            Scalar loss value.
        """
        B, N, _, V = logits.shape
        device = logits.device

        # Get upper triangle indices
        upper_i, upper_j = torch.triu_indices(N, N, offset=1, device=device)

        # Extract upper triangle
        logits_upper = logits[:, upper_i, upper_j, :]  # [B, num_upper, V]
        targets_upper = targets[:, upper_i, upper_j]  # [B, num_upper]
        mask_upper = mask_indicator[:, upper_i, upper_j]  # [B, num_upper]

        # Also check that both nodes are valid
        valid_i = M[:, upper_i]  # [B, num_upper]
        valid_j = M[:, upper_j]  # [B, num_upper]
        valid_mask = mask_upper & (valid_i == 1) & (valid_j == 1)

        if not valid_mask.any():
            return torch.tensor(0.0, device=device)

        # Select valid positions
        masked_logits = logits_upper[valid_mask]  # [num_valid, V]
        masked_targets = targets_upper[valid_mask]  # [num_valid]

        loss = F.cross_entropy(masked_logits, masked_targets, reduction='mean')
        return loss

    def sample_step(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        M: torch.Tensor,
        t: int,
        temperature: float = 1.0,
        top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one reverse diffusion step.

        Args:
            X: Current noisy node tokens.
            E: Current noisy edge tokens.
            M: Node mask.
            t: Current timestep.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Tuple of (updated X, updated E).
        """
        B = X.shape[0]
        device = X.device

        # Create timestep tensor
        timesteps = torch.full((B,), t, device=device, dtype=torch.long)

        # Get model predictions
        with torch.no_grad():
            node_logits, edge_logits = self.backbone(X, E, timesteps, M)

        # Apply temperature
        node_logits = node_logits / temperature
        edge_logits = edge_logits / temperature

        # Sample nodes
        node_probs = F.softmax(node_logits, dim=-1)
        X_new = X.clone()

        is_node_masked = X == self.node_mask_id
        if is_node_masked.any():
            masked_probs = node_probs[is_node_masked]
            sampled_nodes = torch.multinomial(masked_probs, 1).squeeze(-1)
            X_new[is_node_masked] = sampled_nodes

        # Sample edges
        edge_probs = F.softmax(edge_logits, dim=-1)
        E_new = E.clone()

        is_edge_masked = E == self.edge_mask_id
        if is_edge_masked.any():
            masked_probs = edge_probs[is_edge_masked]
            sampled_edges = torch.multinomial(masked_probs, 1).squeeze(-1)
            E_new[is_edge_masked] = sampled_edges

        # Enforce edge symmetry
        E_new = self._enforce_edge_symmetry(E_new)

        return X_new, E_new

    def _enforce_edge_symmetry(self, E: torch.Tensor) -> torch.Tensor:
        """Ensure edge matrix is symmetric.

        Args:
            E: Edge tokens of shape [batch, Nmax, Nmax].

        Returns:
            Symmetric edge tokens.
        """
        # Take upper triangle and mirror to lower
        upper = torch.triu(E, diagonal=1)
        lower = upper.transpose(1, 2)
        diag = torch.diag_embed(torch.diagonal(E, dim1=1, dim2=2))
        return upper + lower + diag

    def get_loss_at_timestep(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        M: torch.Tensor,
        timestep: int
    ) -> Dict[str, torch.Tensor]:
        """Compute loss at a specific timestep (for evaluation).

        Args:
            X: Clean node tokens.
            E: Clean edge tokens.
            M: Node mask.
            timestep: Specific timestep.

        Returns:
            Dictionary with losses.
        """
        B = X.shape[0]
        device = X.device
        timesteps = torch.full((B,), timestep, device=device, dtype=torch.long)

        result = self.forward(X, E, M, timesteps)
        return {
            'loss': result['loss'],
            'node_loss': result['node_loss'],
            'edge_loss': result['edge_loss']
        }


def create_graph_diffusion(
    backbone: nn.Module,
    config: Dict,
    graph_config: Dict
) -> GraphMaskingDiffusion:
    """Create GraphMaskingDiffusion from config.

    Args:
        backbone: Graph transformer backbone.
        config: Main configuration.
        graph_config: Graph configuration with vocab info.

    Returns:
        GraphMaskingDiffusion instance.
    """
    diffusion_config = config.get('diffusion', config.get('graph_diffusion', {}))

    return GraphMaskingDiffusion(
        backbone=backbone,
        num_steps=diffusion_config.get('num_steps', 100),
        beta_min=diffusion_config.get('beta_min', 0.05),
        beta_max=diffusion_config.get('beta_max', 0.95),
        node_mask_id=graph_config['atom_vocab']['MASK'],
        edge_mask_id=graph_config['edge_vocab']['MASK'],
        node_pad_id=graph_config['atom_vocab']['PAD'],
        edge_none_id=graph_config['edge_vocab']['NONE'],
        lambda_node=diffusion_config.get('lambda_node', 1.0),
        lambda_edge=diffusion_config.get('lambda_edge', 0.5)
    )
