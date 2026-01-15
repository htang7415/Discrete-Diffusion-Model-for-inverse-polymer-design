"""Graph Sampler with optional constraints for polymer generation.

Implements reverse diffusion sampling with optional constraints (when enabled):
- Exactly 2 star (*) attachment points
- Star bond constraints (no star-star, only SINGLE allowed)
- Star degree = 1 constraint
- Valence/connectivity checks
Edge symmetry and edge-mask handling are always enforced.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class GraphSampler:
    """Sampler for graph diffusion with optional chemistry constraints."""

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
        atom_count_distribution: Optional[Dict[int, int]] = None,
        use_constraints: bool = True
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
            use_constraints: Whether to apply chemistry constraints during sampling.
        """
        self.backbone = backbone
        self.tokenizer = graph_tokenizer
        self.num_steps = num_steps
        self.device = device
        self.use_constraints = use_constraints

        # Get token IDs from tokenizer if not provided
        self.star_id = star_id if star_id is not None else graph_tokenizer.star_id
        self.mask_id = mask_id if mask_id is not None else graph_tokenizer.mask_id
        self.pad_id = pad_id if pad_id is not None else graph_tokenizer.pad_id

        self.edge_none_id = edge_none_id
        self.edge_single_id = edge_single_id
        self.edge_mask_id = edge_mask_id
        self.edge_double_id = graph_tokenizer.edge_vocab.get('DOUBLE', 2)
        self.edge_triple_id = graph_tokenizer.edge_vocab.get('TRIPLE', 3)
        self.edge_aromatic_id = graph_tokenizer.edge_vocab.get('AROMATIC', 4)

        self.Nmax = graph_tokenizer.Nmax
        self.atom_count_values = None
        self.atom_count_probs = None
        self.bond_order = self._build_bond_order_vector(len(graph_tokenizer.edge_vocab))
        self.max_valence = self._build_max_valence_vector(len(graph_tokenizer.atom_vocab))

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

    def _build_bond_order_vector(self, edge_vocab_size: int) -> torch.Tensor:
        """Build bond order weights for edge types."""
        bond_order = torch.zeros(edge_vocab_size, dtype=torch.float32)
        bond_order[self.edge_none_id] = 0.0
        bond_order[self.edge_single_id] = 1.0
        if self.edge_double_id is not None and self.edge_double_id < edge_vocab_size:
            bond_order[self.edge_double_id] = 2.0
        if self.edge_triple_id is not None and self.edge_triple_id < edge_vocab_size:
            bond_order[self.edge_triple_id] = 3.0
        if self.edge_aromatic_id is not None and self.edge_aromatic_id < edge_vocab_size:
            bond_order[self.edge_aromatic_id] = 1.5
        if self.edge_mask_id is not None and self.edge_mask_id < edge_vocab_size:
            bond_order[self.edge_mask_id] = 0.0
        return bond_order

    def _build_max_valence_vector(self, atom_vocab_size: int) -> torch.Tensor:
        """Build max valence lookup tensor for atom vocab."""
        valence_map = {
            'C': 4,
            'N': 3,
            'O': 2,
            'F': 1,
            'Cl': 1,
            'S': 6,
            'P': 5,
            'Si': 4,
            'STAR': 1
        }
        max_valence = torch.full((atom_vocab_size,), float('inf'), dtype=torch.float32)
        for atom_id, symbol in self.tokenizer.id_to_atom.items():
            if symbol in valence_map:
                max_valence[atom_id] = float(valence_map[symbol])
        return max_valence

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

    def _apply_valence_constraints(
        self,
        edge_logits: torch.Tensor,
        X_current: torch.Tensor,
        E_current: torch.Tensor,
        M: torch.Tensor
    ) -> torch.Tensor:
        """Mask edge logits that would exceed max valence for known atoms."""
        B, N, _, V = edge_logits.shape
        device = edge_logits.device
        edge_logits = edge_logits.clone()

        # Max valence lookup (inf for unknown or masked atoms)
        max_valence = self.max_valence.to(device)[X_current]
        max_valence = max_valence.masked_fill(M == 0, float('inf'))

        # Current bond order per node (ignore masked edges)
        bond_order_vec = self.bond_order.to(device)
        bond_order = bond_order_vec[E_current]
        valid_edge_mask = (M.unsqueeze(2) * M.unsqueeze(1)).float()
        bond_order = bond_order * valid_edge_mask
        current_valence = bond_order.sum(dim=2)

        remaining = max_valence - current_valence
        remaining = torch.clamp(remaining, min=0.0)
        max_allowed = torch.minimum(remaining.unsqueeze(2), remaining.unsqueeze(1))

        allowed = bond_order_vec.view(1, 1, 1, V) <= max_allowed.unsqueeze(-1)
        edge_logits = edge_logits.masked_fill(~allowed, -float('inf'))

        return edge_logits

    def _fix_star_count(
        self,
        X_current: torch.Tensor,
        M: torch.Tensor,
        node_logits: torch.Tensor
    ) -> torch.Tensor:
        """Force exactly two stars by adjusting node tokens at final step."""
        X_fixed = X_current.clone()
        B, N, _ = node_logits.shape

        for b in range(B):
            valid_nodes = M[b] == 1
            star_mask = (X_fixed[b] == self.star_id) & valid_nodes
            num_stars = int(star_mask.sum().item())

            if num_stars == 2:
                continue

            star_logits = node_logits[b, :, self.star_id]

            if num_stars < 2:
                candidates = torch.where(valid_nodes & ~star_mask)[0]
                if candidates.numel() == 0:
                    continue
                needed = 2 - num_stars
                topk = torch.topk(star_logits[candidates], k=min(needed, candidates.numel()))
                chosen = candidates[topk.indices]
                X_fixed[b, chosen] = self.star_id
            else:
                star_indices = torch.where(star_mask)[0]
                remove_k = num_stars - 2
                if remove_k <= 0:
                    continue
                bottom = torch.topk(star_logits[star_indices], k=remove_k, largest=False)
                to_replace = star_indices[bottom.indices]

                for idx in to_replace:
                    logits = node_logits[b, idx].clone()
                    for tok in (self.star_id, self.mask_id, self.pad_id):
                        if 0 <= tok < logits.numel():
                            logits[tok] = -float('inf')
                    replacement = torch.argmax(logits).item()
                    X_fixed[b, idx] = replacement

        return X_fixed

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

    def _enforce_connectivity(
        self,
        E_sampled: torch.Tensor,
        X_current: torch.Tensor,
        M: torch.Tensor,
        edge_logits: torch.Tensor
    ) -> torch.Tensor:
        """Ensure graphs are connected by adding valid edges between components."""
        B, N, _ = E_sampled.shape
        V = edge_logits.shape[-1]
        device = E_sampled.device
        E_sampled = E_sampled.clone()

        bond_order_vec = self.bond_order.to(device)
        max_valence_vec = self.max_valence.to(device)

        for b in range(B):
            valid_nodes = (M[b] == 1)
            valid_idx = torch.where(valid_nodes)[0].tolist()
            if len(valid_idx) <= 1:
                continue

            edges = E_sampled[b]
            connected = (edges != self.edge_none_id) & (edges != self.edge_mask_id)

            # Build components with BFS
            unvisited = set(valid_idx)
            components = []
            while unvisited:
                start = unvisited.pop()
                stack = [start]
                comp = [start]
                while stack:
                    node = stack.pop()
                    neighbors = torch.where(connected[node] & valid_nodes)[0].tolist()
                    for nbr in neighbors:
                        if nbr in unvisited:
                            unvisited.remove(nbr)
                            stack.append(nbr)
                            comp.append(nbr)
                components.append(comp)

            if len(components) <= 1:
                continue

            atom_ids = X_current[b]
            max_valence = max_valence_vec[atom_ids].clone()
            max_valence = torch.where(valid_nodes, max_valence, torch.tensor(float('inf'), device=device))

            bond_order = bond_order_vec[edges] * (valid_nodes.unsqueeze(0) & valid_nodes.unsqueeze(1)).float()
            current_valence = bond_order.sum(dim=1)
            remaining = (max_valence - current_valence).clamp(min=0.0)

            def best_edge_between(comp_a, comp_b):
                best_score = -float('inf')
                best_edge = None
                for i in comp_a:
                    for j in comp_b:
                        logits = edge_logits[b, i, j].clone()
                        if self.edge_none_id < V:
                            logits[self.edge_none_id] = -float('inf')
                        if self.edge_mask_id < V:
                            logits[self.edge_mask_id] = -float('inf')

                        i_is_star = atom_ids[i].item() == self.star_id
                        j_is_star = atom_ids[j].item() == self.star_id
                        if i_is_star and j_is_star:
                            continue
                        if i_is_star or j_is_star:
                            for edge_id in range(V):
                                if edge_id != self.edge_single_id:
                                    logits[edge_id] = -float('inf')

                        max_allowed = min(remaining[i].item(), remaining[j].item())
                        valid_edge_types = bond_order_vec <= max_allowed + 1e-6
                        logits[~valid_edge_types] = -float('inf')

                        score, edge_id = logits.max(dim=0)
                        if score.item() > best_score and edge_id.item() != self.edge_none_id:
                            best_score = score.item()
                            best_edge = (i, j, int(edge_id.item()))
                return best_edge, best_score

            # Connect star components first if needed
            star_indices = torch.where((atom_ids == self.star_id) & valid_nodes)[0].tolist()
            if len(star_indices) >= 2:
                star_comps = []
                for idx in star_indices[:2]:
                    for comp_i, comp in enumerate(components):
                        if idx in comp:
                            star_comps.append(comp_i)
                            break
                if len(star_comps) == 2 and star_comps[0] != star_comps[1]:
                    comp_a = components[star_comps[0]]
                    comp_b = components[star_comps[1]]
                    best_edge, _ = best_edge_between(comp_a, comp_b)
                    if best_edge is not None:
                        i, j, edge_id = best_edge
                        E_sampled[b, i, j] = edge_id
                        E_sampled[b, j, i] = edge_id
                        bo = bond_order_vec[edge_id].item()
                        remaining[i] = max(remaining[i] - bo, 0.0)
                        remaining[j] = max(remaining[j] - bo, 0.0)
                        merged = comp_a + comp_b
                        keep = min(star_comps)
                        drop = max(star_comps)
                        components[keep] = merged
                        del components[drop]

            # Connect remaining components greedily
            while len(components) > 1:
                best_pair = None
                best_edge = None
                best_score = -float('inf')

                for i in range(len(components)):
                    for j in range(i + 1, len(components)):
                        edge, score = best_edge_between(components[i], components[j])
                        if edge is not None and score > best_score:
                            best_score = score
                            best_pair = (i, j)
                            best_edge = edge

                if best_edge is None or best_pair is None:
                    break

                i_idx, j_idx = best_pair
                i_node, j_node, edge_id = best_edge
                E_sampled[b, i_node, j_node] = edge_id
                E_sampled[b, j_node, i_node] = edge_id
                bo = bond_order_vec[edge_id].item()
                remaining[i_node] = max(remaining[i_node] - bo, 0.0)
                remaining[j_node] = max(remaining[j_node] - bo, 0.0)

                merged = components[i_idx] + components[j_idx]
                components[i_idx] = merged
                del components[j_idx]

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
        upper_triangle = torch.triu(
            torch.ones((N, N), device=device, dtype=torch.bool),
            diagonal=1
        )

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
            if self.use_constraints:
                node_logits = self._apply_star_constraints(node_logits, X, M, is_final)
                edge_logits = self._apply_edge_constraints(edge_logits, X, M)
                edge_logits = self._apply_valence_constraints(edge_logits, X, E, M)
            node_logits = self._apply_node_special_token_constraints(node_logits, X, M)
            edge_logits[:, :, :, self.edge_mask_id][valid_edge_mask] = -float('inf')

            # Enforce symmetry in edge logits
            edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2

            # Sample nodes (progressive unmasking)
            node_probs = F.softmax(node_logits, dim=-1)
            is_node_masked = (X == self.mask_id) & (M == 1)
            unmask_prob = 1.0 / t

            if is_node_masked.any():
                for b in range(batch_size):
                    masked_pos = torch.where(is_node_masked[b])[0]
                    if masked_pos.numel() == 0:
                        continue
                    num_unmask = max(1, int(masked_pos.numel() * unmask_prob))
                    select = masked_pos[torch.randperm(masked_pos.numel(), device=device)[:num_unmask]]
                    masked_probs = node_probs[b, select].clamp(min=1e-10)
                    masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
                    sampled_nodes = torch.multinomial(masked_probs, 1).squeeze(-1)
                    X[b, select] = sampled_nodes

            if is_final and self.use_constraints:
                X = self._fix_star_count(X, M, node_logits)
                edge_logits = self._apply_edge_constraints(edge_logits, X, M)
                edge_logits = self._apply_valence_constraints(edge_logits, X, E, M)
                edge_logits[:, :, :, self.edge_mask_id][valid_edge_mask] = -float('inf')
                edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2

            # Sample edges (progressive unmasking, upper triangle only)
            edge_probs = F.softmax(edge_logits, dim=-1)
            is_edge_masked = (E == self.edge_mask_id) & valid_edge_mask & upper_triangle

            if is_edge_masked.any():
                for b in range(batch_size):
                    edge_pos = torch.where(is_edge_masked[b])
                    if edge_pos[0].numel() == 0:
                        continue
                    num_unmask = max(1, int(edge_pos[0].numel() * unmask_prob))
                    select = torch.randperm(edge_pos[0].numel(), device=device)[:num_unmask]
                    i_idx = edge_pos[0][select]
                    j_idx = edge_pos[1][select]
                    masked_probs = edge_probs[b, i_idx, j_idx].clamp(min=1e-10)
                    masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
                    sampled_edges = torch.multinomial(masked_probs, 1).squeeze(-1)
                    E[b, i_idx, j_idx] = sampled_edges
                    E[b, j_idx, i_idx] = sampled_edges

            # Enforce symmetry
            E = self._enforce_edge_symmetry(E)

            # Post-process at final step
            if is_final and self.use_constraints:
                E = self._enforce_star_degree(E, X, M, edge_logits)
                E = self._enforce_connectivity(E, X, M, edge_logits)
                E = self._enforce_connectivity(E, X, M, edge_logits)

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

    def sample_batch(
        self,
        num_samples: int,
        batch_size: int = 256,
        show_progress: bool = True,
        temperature: float = 1.0,
        num_atoms: Optional[int] = None
    ) -> List[Optional[str]]:
        """Sample multiple batches of graphs.

        Args:
            num_samples: Total number of samples to generate.
            batch_size: Batch size for sampling.
            show_progress: Whether to show progress.
            temperature: Sampling temperature.
            num_atoms: Fixed number of atoms (if None, sample per-graph).

        Returns:
            List of p-SMILES strings (None for failed conversions).
        """
        all_smiles: List[Optional[str]] = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        batch_iter = range(num_batches)
        if show_progress:
            batch_iter = tqdm(batch_iter, desc="Batch sampling")

        for _ in batch_iter:
            current_batch_size = min(batch_size, num_samples - len(all_smiles))
            if current_batch_size <= 0:
                break
            smiles = self.sample(
                current_batch_size,
                temperature=temperature,
                show_progress=False,
                num_atoms=num_atoms
            )
            all_smiles.extend(smiles)

        return all_smiles

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
        upper_triangle = torch.triu(
            torch.ones((N, N), device=device, dtype=torch.bool),
            diagonal=1
        )

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

            if self.use_constraints:
                node_logits = self._apply_star_constraints(node_logits, X, M, is_final)
                edge_logits = self._apply_edge_constraints(edge_logits, X, M)
                edge_logits = self._apply_valence_constraints(edge_logits, X, E, M)
            node_logits = self._apply_node_special_token_constraints(node_logits, X, M)
            edge_logits[:, :, :, self.edge_mask_id][valid_edge_mask] = -float('inf')
            edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2

            node_probs = F.softmax(node_logits, dim=-1)
            is_node_masked = (X == self.mask_id) & (M == 1)
            unmask_prob = 1.0 / t

            if is_node_masked.any():
                for b in range(batch_size):
                    masked_pos = torch.where(is_node_masked[b])[0]
                    if masked_pos.numel() == 0:
                        continue
                    num_unmask = max(1, int(masked_pos.numel() * unmask_prob))
                    select = masked_pos[torch.randperm(masked_pos.numel(), device=device)[:num_unmask]]
                    masked_probs = node_probs[b, select].clamp(min=1e-10)
                    masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
                    sampled_nodes = torch.multinomial(masked_probs, 1).squeeze(-1)
                    X[b, select] = sampled_nodes

            if is_final and self.use_constraints:
                X = self._fix_star_count(X, M, node_logits)
                edge_logits = self._apply_edge_constraints(edge_logits, X, M)
                edge_logits = self._apply_valence_constraints(edge_logits, X, E, M)
                edge_logits[:, :, :, self.edge_mask_id][valid_edge_mask] = -float('inf')
                edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2

            edge_probs = F.softmax(edge_logits, dim=-1)
            is_edge_masked = (E == self.edge_mask_id) & valid_edge_mask & upper_triangle

            if is_edge_masked.any():
                for b in range(batch_size):
                    edge_pos = torch.where(is_edge_masked[b])
                    if edge_pos[0].numel() == 0:
                        continue
                    num_unmask = max(1, int(edge_pos[0].numel() * unmask_prob))
                    select = torch.randperm(edge_pos[0].numel(), device=device)[:num_unmask]
                    i_idx = edge_pos[0][select]
                    j_idx = edge_pos[1][select]
                    masked_probs = edge_probs[b, i_idx, j_idx].clamp(min=1e-10)
                    masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
                    sampled_edges = torch.multinomial(masked_probs, 1).squeeze(-1)
                    E[b, i_idx, j_idx] = sampled_edges
                    E[b, j_idx, i_idx] = sampled_edges

            E = self._enforce_edge_symmetry(E)

            if is_final and self.use_constraints:
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
    sampling_config = config.get('sampling', {}) if config is not None else {}
    use_constraints = sampling_config.get('use_constraints', True)

    atom_count_distribution = None
    if graph_config is not None:
        atom_count_distribution = graph_config.get('atom_count_distribution')

    return GraphSampler(
        backbone=backbone,
        graph_tokenizer=graph_tokenizer,
        num_steps=diffusion_config.get('num_steps', 100),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        atom_count_distribution=atom_count_distribution,
        use_constraints=use_constraints
    )
