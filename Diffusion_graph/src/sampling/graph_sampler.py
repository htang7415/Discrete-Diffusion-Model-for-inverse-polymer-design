"""Graph Sampler with hard constraints for polymer generation.

Implements reverse diffusion sampling with:
- Exactly 2 star (*) attachment points
- Star bond constraints (no star-star, only SINGLE allowed)
- Star degree = 1 constraint
- Edge symmetry enforcement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class GraphSampler:
    """Constrained sampler for graph diffusion.

    Applies hard constraints during reverse diffusion to ensure
    valid polymer structures with exactly 2 attachment points.
    """

    def __init__(
        self,
        backbone: nn.Module,
        graph_tokenizer,
        num_steps: int = 100,
        star_id: Optional[int] = None,
        mask_id: Optional[int] = None,
        pad_id: Optional[int] = None,
        edge_none_id: int = 0,
        edge_single_id: int = 1,
        edge_mask_id: int = 5,
        device: str = 'cuda',
        atom_count_distribution: Optional[Dict[int, int]] = None
    ):
        """Initialize GraphSampler.

        Args:
            backbone: Graph transformer backbone.
            graph_tokenizer: GraphTokenizer instance for decoding.
            num_steps: Number of diffusion steps.
            star_id: ID of STAR token in atom vocab.
            mask_id: ID of MASK token in atom vocab.
            pad_id: ID of PAD token in atom vocab.
            edge_none_id: ID of NONE in edge vocab.
            edge_single_id: ID of SINGLE in edge vocab.
            edge_mask_id: ID of MASK in edge vocab.
            device: Device for sampling.
        """
        self.backbone = backbone
        self.tokenizer = graph_tokenizer
        self.num_steps = num_steps
        self.device = device

        # Get token IDs from tokenizer if not provided
        self.star_id = star_id if star_id is not None else graph_tokenizer.star_id
        self.mask_id = mask_id if mask_id is not None else graph_tokenizer.mask_id
        self.pad_id = pad_id if pad_id is not None else graph_tokenizer.pad_id

        self.edge_none_id = edge_none_id
        self.edge_single_id = edge_single_id
        self.edge_mask_id = edge_mask_id

        self.Nmax = graph_tokenizer.Nmax
        self.atom_count_values = None
        self.atom_count_probs = None

        if atom_count_distribution:
            values = []
            weights = []
            for key, count in atom_count_distribution.items():
                try:
                    value = int(key)
                except (TypeError, ValueError):
                    continue
                if value <= 0:
                    continue
                values.append(value)
                weights.append(float(count))

            if values and sum(weights) > 0:
                order = np.argsort(values)
                values = np.array(values)[order]
                weights = np.array(weights, dtype=np.float64)[order]
                self.atom_count_values = values
                self.atom_count_probs = weights / weights.sum()

    def _sample_num_atoms(self, batch_size: int) -> List[int]:
        """Sample number of atoms per graph from the empirical distribution."""
        if self.atom_count_values is None:
            return [self.Nmax] * batch_size

        samples = np.random.choice(
            self.atom_count_values,
            size=batch_size,
            p=self.atom_count_probs
        )
        samples = np.clip(samples, 1, self.Nmax)
        return samples.tolist()

    def _apply_node_special_token_constraints(
        self,
        node_logits: torch.Tensor,
        X_current: torch.Tensor,
        M: torch.Tensor
    ) -> torch.Tensor:
        """Forbid MASK/PAD tokens at valid masked node positions."""
        B, N, _ = node_logits.shape
        node_logits = node_logits.clone()

        for b in range(B):
            mask_positions = (X_current[b] == self.mask_id) & (M[b] == 1)
            node_logits[b, mask_positions, self.mask_id] = -float('inf')
            node_logits[b, mask_positions, self.pad_id] = -float('inf')

        return node_logits

    def _apply_star_constraints(
        self,
        node_logits: torch.Tensor,
        X_current: torch.Tensor,
        M: torch.Tensor,
        is_final_step: bool = False
    ) -> torch.Tensor:
        """Apply star count constraints to node logits.

        Rules:
        - At most 2 stars during sampling
        - Exactly 2 stars at final step

        Args:
            node_logits: (B, N, V) node logits.
            X_current: (B, N) current node tokens.
            M: (B, N) node mask.
            is_final_step: Whether this is the final step.

        Returns:
            Modified node logits.
        """
        B, N, V = node_logits.shape
        node_logits = node_logits.clone()

        for b in range(B):
            # Count existing non-masked stars
            valid_mask = (M[b] == 1) & (X_current[b] != self.mask_id)
            num_stars = (X_current[b][valid_mask] == self.star_id).sum().item()

            if num_stars >= 2:
                # Forbid more stars at masked positions
                mask_positions = (X_current[b] == self.mask_id) & (M[b] == 1)
                node_logits[b, mask_positions, self.star_id] = -float('inf')

            if is_final_step:
                # Force exactly 2 stars
                valid_nodes = M[b] == 1

                # Count current stars (including those just sampled)
                current_stars = (X_current[b][valid_nodes] == self.star_id).sum().item()

                if current_stars < 2:
                    # Need to force more stars
                    # Get star probabilities for valid non-star nodes
                    star_probs = F.softmax(node_logits[b, :, self.star_id], dim=0)
                    star_probs = star_probs * valid_nodes.float()
                    star_probs[X_current[b] == self.star_id] = 0  # Don't count existing stars

                    # Select top-(2-current_stars) positions
                    needed = 2 - int(current_stars)
                    if needed > 0 and star_probs.sum() > 0:
                        _, top_indices = torch.topk(star_probs, k=min(needed, int(valid_nodes.sum())))
                        # Strongly favor star at these positions
                        node_logits[b, top_indices, self.star_id] = 100.0

                elif current_stars > 2:
                    # Too many stars, forbid more
                    mask_positions = (X_current[b] == self.mask_id) & (M[b] == 1)
                    node_logits[b, mask_positions, self.star_id] = -float('inf')

        return node_logits

    def _apply_edge_constraints(
        self,
        edge_logits: torch.Tensor,
        X_current: torch.Tensor,
        M: torch.Tensor
    ) -> torch.Tensor:
        """Apply star bond constraints to edge logits.

        Rules:
        - No star-star bonds (force NONE)
        - Star edges can only be NONE or SINGLE
        - Non-star edges allow all bond types

        Args:
            edge_logits: (B, N, N, V_edge) edge logits.
            X_current: (B, N) current node tokens.
            M: (B, N) node mask.

        Returns:
            Modified edge logits.
        """
        B, N, _, V = edge_logits.shape
        edge_logits = edge_logits.clone()

        for b in range(B):
            # Find star positions
            star_mask = (X_current[b] == self.star_id) & (M[b] == 1)
            star_indices = torch.where(star_mask)[0]

            if len(star_indices) == 0:
                continue

            for star_idx in star_indices:
                # Forbid star-star bonds
                for other_star_idx in star_indices:
                    if star_idx != other_star_idx:
                        # Force NONE for star-star edges
                        edge_logits[b, star_idx, other_star_idx, self.edge_single_id:] = -float('inf')

                # Star edges can only be NONE or SINGLE
                # Forbid DOUBLE (2), TRIPLE (3), AROMATIC (4)
                edge_logits[b, star_idx, :, 2:self.edge_mask_id] = -float('inf')
                edge_logits[b, :, star_idx, 2:self.edge_mask_id] = -float('inf')

        return edge_logits

    def _enforce_star_degree(
        self,
        E_sampled: torch.Tensor,
        X_current: torch.Tensor,
        M: torch.Tensor,
        edge_logits: torch.Tensor
    ) -> torch.Tensor:
        """Post-sampling: ensure each star has exactly 1 neighbor.

        Args:
            E_sampled: (B, N, N) sampled edge tokens.
            X_current: (B, N) current node tokens.
            M: (B, N) node mask.
            edge_logits: (B, N, N, V) for tie-breaking.

        Returns:
            Modified edge tokens.
        """
        B, N, _ = E_sampled.shape
        E_sampled = E_sampled.clone()

        for b in range(B):
            star_mask = (X_current[b] == self.star_id) & (M[b] == 1)
            star_indices = torch.where(star_mask)[0]

            for star_idx in star_indices:
                star_idx = star_idx.item()

                # Get neighbors (edges == SINGLE)
                neighbors = torch.where(
                    (E_sampled[b, star_idx, :] == self.edge_single_id) & (M[b] == 1)
                )[0]

                # Exclude self and other stars
                other_stars = torch.where(
                    (X_current[b] == self.star_id) & (M[b] == 1)
                )[0]
                valid_neighbors = neighbors[~torch.isin(neighbors, other_stars)]
                valid_neighbors = valid_neighbors[valid_neighbors != star_idx]

                if len(valid_neighbors) == 0:
                    # No neighbors, need to pick one
                    # Get SINGLE probabilities
                    single_probs = F.softmax(edge_logits[b, star_idx, :, self.edge_single_id], dim=0)
                    single_probs = single_probs * (M[b] == 1).float()
                    single_probs[X_current[b] == self.star_id] = 0  # Exclude stars
                    single_probs[star_idx] = 0  # Exclude self

                    if single_probs.sum() > 0:
                        best_neighbor = torch.argmax(single_probs).item()
                        E_sampled[b, star_idx, best_neighbor] = self.edge_single_id
                        E_sampled[b, best_neighbor, star_idx] = self.edge_single_id

                elif len(valid_neighbors) > 1:
                    # Too many neighbors, keep only best one
                    probs = F.softmax(
                        edge_logits[b, star_idx, valid_neighbors, self.edge_single_id],
                        dim=0
                    )
                    best_idx = torch.argmax(probs).item()
                    best_neighbor = valid_neighbors[best_idx].item()

                    # Set all star edges to NONE
                    E_sampled[b, star_idx, :] = self.edge_none_id
                    E_sampled[b, :, star_idx] = self.edge_none_id

                    # Set only best neighbor to SINGLE
                    E_sampled[b, star_idx, best_neighbor] = self.edge_single_id
                    E_sampled[b, best_neighbor, star_idx] = self.edge_single_id

        return E_sampled

    def _enforce_edge_symmetry(self, E: torch.Tensor) -> torch.Tensor:
        """Ensure edge matrix is symmetric.

        Args:
            E: (B, N, N) edge tokens.

        Returns:
            Symmetric edge tokens.
        """
        # Take upper triangle and mirror
        upper = torch.triu(E, diagonal=1)
        lower = upper.transpose(1, 2)
        diag = torch.zeros_like(E)
        return upper + lower + diag

    def sample(
        self,
        batch_size: int,
        temperature: float = 1.0,
        show_progress: bool = True,
        num_atoms: Optional[int] = None
    ) -> List[Optional[str]]:
        """Generate molecules via reverse diffusion sampling.

        Args:
            batch_size: Number of molecules to generate.
            temperature: Sampling temperature.
            show_progress: Whether to show progress bar.
            num_atoms: Fixed number of atoms (if None, use Nmax).

        Returns:
            List of p-SMILES strings (None for failed conversions).
        """
        device = next(self.backbone.parameters()).device
        N = self.Nmax
        T = self.num_steps

        # Determine atom counts for each sample
        if num_atoms is None:
            num_atoms_list = self._sample_num_atoms(batch_size)
        else:
            num_atoms_list = [int(num_atoms)] * batch_size

        # Node mask
        M = torch.zeros((batch_size, N), dtype=torch.float, device=device)
        for b, n_atoms in enumerate(num_atoms_list):
            n_atoms = min(max(int(n_atoms), 1), N)
            M[b, :n_atoms] = 1.0

        # Initialize nodes: PAD everywhere, MASK for valid nodes
        X = torch.full((batch_size, N), self.pad_id, dtype=torch.long, device=device)
        X[M == 1] = self.mask_id

        # Initialize edges: NONE everywhere, MASK for valid node pairs (no self-loops)
        E = torch.full((batch_size, N, N), self.edge_none_id, dtype=torch.long, device=device)
        valid_edge_mask = (M.unsqueeze(2) * M.unsqueeze(1)).bool()
        diag = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        valid_edge_mask = valid_edge_mask & (~diag)
        E[valid_edge_mask] = self.edge_mask_id

        # Progress bar
        iterator = range(T, 0, -1)
        if show_progress:
            iterator = tqdm(iterator, desc="Sampling")

        self.backbone.eval()

        for t in iterator:
            is_final = (t == 1)
            timesteps = torch.full((batch_size,), t, dtype=torch.long, device=device)

            # Get model predictions
            with torch.no_grad():
                node_logits, edge_logits = self.backbone(X, E, timesteps, M)

            # Apply temperature
            node_logits = node_logits / temperature
            edge_logits = edge_logits / temperature

            # Apply constraints
            node_logits = self._apply_star_constraints(node_logits, X, M, is_final)
            node_logits = self._apply_node_special_token_constraints(node_logits, X, M)
            edge_logits = self._apply_edge_constraints(edge_logits, X, M)
            edge_logits[:, :, :, self.edge_mask_id][valid_edge_mask] = -float('inf')

            # Enforce symmetry in edge logits
            edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2

            # Sample nodes (only update masked positions)
            node_probs = F.softmax(node_logits, dim=-1)
            is_node_masked = (X == self.mask_id) & (M == 1)

            if is_node_masked.any():
                masked_probs = node_probs[is_node_masked]
                # Clamp to avoid numerical issues
                masked_probs = masked_probs.clamp(min=1e-10)
                masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
                sampled_nodes = torch.multinomial(masked_probs, 1).squeeze(-1)
                X = X.clone()
                X[is_node_masked] = sampled_nodes

            # Sample edges (only update masked positions)
            edge_probs = F.softmax(edge_logits, dim=-1)
            is_edge_masked = (E == self.edge_mask_id) & valid_edge_mask

            if is_edge_masked.any():
                masked_probs = edge_probs[is_edge_masked]
                masked_probs = masked_probs.clamp(min=1e-10)
                masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
                sampled_edges = torch.multinomial(masked_probs, 1).squeeze(-1)
                E = E.clone()
                E[is_edge_masked] = sampled_edges

            # Enforce symmetry
            E = self._enforce_edge_symmetry(E)

            # Post-process at final step
            if is_final:
                E = self._enforce_star_degree(E, X, M, edge_logits)

        # Decode to SMILES
        X_np = X.cpu().numpy()
        E_np = E.cpu().numpy()
        M_np = M.cpu().numpy()

        smiles_list = []
        for i in range(batch_size):
            try:
                smiles = self.tokenizer.decode(X_np[i], E_np[i], M_np[i])
                smiles_list.append(smiles)
            except Exception as e:
                smiles_list.append(None)

        return smiles_list

    def sample_with_graphs(
        self,
        batch_size: int,
        temperature: float = 1.0,
        show_progress: bool = True
    ) -> Tuple[List[Optional[str]], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate molecules and return both SMILES and graph tensors.

        Args:
            batch_size: Number of molecules to generate.
            temperature: Sampling temperature.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (smiles_list, X, E, M).
        """
        device = next(self.backbone.parameters()).device
        N = self.Nmax
        T = self.num_steps

        # Determine atom counts for each sample
        num_atoms_list = self._sample_num_atoms(batch_size)

        # Node mask
        M = torch.zeros((batch_size, N), dtype=torch.float, device=device)
        for b, n_atoms in enumerate(num_atoms_list):
            n_atoms = min(max(int(n_atoms), 1), N)
            M[b, :n_atoms] = 1.0

        # Initialize nodes: PAD everywhere, MASK for valid nodes
        X = torch.full((batch_size, N), self.pad_id, dtype=torch.long, device=device)
        X[M == 1] = self.mask_id

        # Initialize edges: NONE everywhere, MASK for valid node pairs (no self-loops)
        E = torch.full((batch_size, N, N), self.edge_none_id, dtype=torch.long, device=device)
        valid_edge_mask = (M.unsqueeze(2) * M.unsqueeze(1)).bool()
        diag = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        valid_edge_mask = valid_edge_mask & (~diag)
        E[valid_edge_mask] = self.edge_mask_id

        iterator = range(T, 0, -1)
        if show_progress:
            iterator = tqdm(iterator, desc="Sampling")

        self.backbone.eval()

        for t in iterator:
            is_final = (t == 1)
            timesteps = torch.full((batch_size,), t, dtype=torch.long, device=device)

            with torch.no_grad():
                node_logits, edge_logits = self.backbone(X, E, timesteps, M)

            node_logits = node_logits / temperature
            edge_logits = edge_logits / temperature

            node_logits = self._apply_star_constraints(node_logits, X, M, is_final)
            node_logits = self._apply_node_special_token_constraints(node_logits, X, M)
            edge_logits = self._apply_edge_constraints(edge_logits, X, M)
            edge_logits[:, :, :, self.edge_mask_id][valid_edge_mask] = -float('inf')
            edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2

            node_probs = F.softmax(node_logits, dim=-1)
            is_node_masked = (X == self.mask_id) & (M == 1)

            if is_node_masked.any():
                masked_probs = node_probs[is_node_masked].clamp(min=1e-10)
                masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
                sampled_nodes = torch.multinomial(masked_probs, 1).squeeze(-1)
                X = X.clone()
                X[is_node_masked] = sampled_nodes

            edge_probs = F.softmax(edge_logits, dim=-1)
            is_edge_masked = (E == self.edge_mask_id) & valid_edge_mask

            if is_edge_masked.any():
                masked_probs = edge_probs[is_edge_masked].clamp(min=1e-10)
                masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
                sampled_edges = torch.multinomial(masked_probs, 1).squeeze(-1)
                E = E.clone()
                E[is_edge_masked] = sampled_edges

            E = self._enforce_edge_symmetry(E)

            if is_final:
                E = self._enforce_star_degree(E, X, M, edge_logits)

        # Decode
        X_np = X.cpu().numpy()
        E_np = E.cpu().numpy()
        M_np = M.cpu().numpy()

        smiles_list = []
        for i in range(batch_size):
            try:
                smiles = self.tokenizer.decode(X_np[i], E_np[i], M_np[i])
                smiles_list.append(smiles)
            except:
                smiles_list.append(None)

        return smiles_list, X, E, M


def create_graph_sampler(
    backbone: Optional[nn.Module] = None,
    graph_tokenizer=None,
    config: Optional[Dict] = None,
    diffusion_model: Optional[nn.Module] = None,
    tokenizer=None,
    graph_config: Optional[Dict] = None
) -> GraphSampler:
    """Create GraphSampler from config.

    Args:
        backbone: Graph transformer backbone.
        graph_tokenizer: GraphTokenizer instance.
        config: Configuration dictionary.
        diffusion_model: Optional diffusion model wrapper (uses .backbone).
        tokenizer: Optional alias for graph_tokenizer.
        graph_config: Optional graph configuration (for atom count distribution).

    Returns:
        GraphSampler instance.
    """
    if diffusion_model is not None:
        backbone = diffusion_model.backbone
    if tokenizer is not None:
        graph_tokenizer = tokenizer
    if backbone is None or graph_tokenizer is None:
        raise ValueError("backbone and graph_tokenizer must be provided")

    diffusion_config = {}
    if config is not None:
        diffusion_config = config.get('diffusion', config.get('graph_diffusion', {}))

    atom_count_distribution = None
    if graph_config is not None:
        atom_count_distribution = graph_config.get('atom_count_distribution')

    return GraphSampler(
        backbone=backbone,
        graph_tokenizer=graph_tokenizer,
        num_steps=diffusion_config.get('num_steps', 100),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        atom_count_distribution=atom_count_distribution
    )
