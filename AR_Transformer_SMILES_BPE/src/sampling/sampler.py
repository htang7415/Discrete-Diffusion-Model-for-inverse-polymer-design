"""Constrained sampler for polymer generation."""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np
from tqdm import tqdm


class ConstrainedSampler:
    """Sampler for polymer generation with optional constraints.

    When use_constraints is True, enforces SMILES syntax constraints and
    exactly two '*' tokens. Special tokens are always forbidden at MASK positions.
    """

    def __init__(
        self,
        diffusion_model,
        tokenizer,
        num_steps: int = 100,
        temperature: float = 1.0,
        use_constraints: bool = True,
        device: str = 'cuda',
        top_k: int = 0,
        top_p: float = 1.0,
        max_length: Optional[int] = None
    ):
        """Initialize sampler.

        Args:
            diffusion_model: Trained autoregressive model wrapper.
            tokenizer: Tokenizer instance.
            num_steps: Unused (kept for backward compatibility).
            temperature: Sampling temperature.
            use_constraints: Whether to apply chemistry constraints during sampling.
            device: Device for computation.
            top_k: Top-k sampling (0 disables).
            top_p: Top-p (nucleus) sampling (1.0 disables).
            max_length: Optional max sequence length for generation.
        """
        self.diffusion_model = diffusion_model
        self.tokenizer = tokenizer
        self.num_steps = num_steps
        self.temperature = temperature
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        self.use_constraints = use_constraints
        self.device = device
        self.top_k = top_k
        self.top_p = top_p
        self.max_length = max_length

        # Get special token IDs
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.star_id = tokenizer.get_star_token_id()

        # Token categories for syntax constraints
        self.open_paren_id = tokenizer.vocab.get('(', -1)
        self.close_paren_id = tokenizer.vocab.get(')', -1)
        self.bond_ids = {tokenizer.vocab.get(b, -1) for b in ['-', '=', '#', '/', '\\']} - {-1}
        self.ring_digit_ids = {tokenizer.vocab.get(str(d), -1) for d in range(10)} - {-1}
        self.ring_percent_ids = {v for k, v in tokenizer.vocab.items() if k.startswith('%')}

        # Build set of bracket tokens (tokens that start with '[' and end with ']')
        self.bracket_token_ids = {
            v for k, v in tokenizer.vocab.items()
            if k.startswith('[') and k.endswith(']') and k not in tokenizer.SPECIAL_TOKENS
        }

        # Atom tokens (for determining what can follow bonds)
        self.atom_ids = {
            tokenizer.vocab.get(a, -1) for a in
            ['B', 'C', 'N', 'O', 'F', 'P', 'S', 'I', 'H', 'c', 'n', 'o', 's', 'p',
             'Cl', 'Br', 'Si', 'Na', 'Li', 'Ca', 'Mg', 'Al', '*']
        } - {-1}
        self.atom_ids.update(self.bracket_token_ids)  # Bracket tokens are also atoms

    def _count_stars(self, ids: torch.Tensor) -> torch.Tensor:
        """Count '*' tokens in each sequence.

        Args:
            ids: Token IDs of shape [batch, seq_len].

        Returns:
            Counts of shape [batch].
        """
        return (ids == self.star_id).sum(dim=1)

    def _analyze_syntax_state(self, seq: torch.Tensor) -> dict:
        """Analyze syntax state of partially generated sequence.

        Examines unmasked tokens to determine current SMILES syntax state.
        Used to determine which tokens are valid for remaining masked positions.

        Args:
            seq: Token IDs of shape [seq_len].

        Returns:
            Dictionary with:
            - paren_depth: int (unclosed parentheses count)
            - open_rings: set (ring numbers currently open)
            - has_unclosed_structure: bool (any unclosed parens)
        """
        paren_depth = 0
        open_rings = set()

        for i in range(len(seq)):
            token_id = seq[i].item()

            # Skip masked and special tokens
            if token_id in [self.mask_id, self.pad_id, self.bos_id, self.eos_id]:
                continue

            # Track parentheses
            if token_id == self.open_paren_id:
                paren_depth += 1
            elif token_id == self.close_paren_id:
                paren_depth = max(0, paren_depth - 1)

            # Track ring closures (digits 0-9)
            if token_id in self.ring_digit_ids:
                if token_id in open_rings:
                    open_rings.remove(token_id)
                else:
                    open_rings.add(token_id)

            # Track ring closures (%10, %11, etc.)
            if token_id in self.ring_percent_ids:
                if token_id in open_rings:
                    open_rings.remove(token_id)
                else:
                    open_rings.add(token_id)

        return {
            'paren_depth': paren_depth,
            'open_rings': open_rings,
            'has_unclosed_structure': paren_depth > 0
        }

    def _get_forbidden_tokens(self, syntax_state: dict, num_masked: int) -> set:
        """Return token IDs that are forbidden given current syntax state.

        Args:
            syntax_state: Dictionary from _analyze_syntax_state.
            num_masked: Number of remaining masked positions.

        Returns:
            Set of token IDs that should be forbidden.
        """
        forbidden = set()
        paren_depth = syntax_state['paren_depth']

        # Rule 1: Cannot close more parentheses than are open
        if paren_depth == 0:
            if self.close_paren_id >= 0:
                forbidden.add(self.close_paren_id)

        # Rule 2: Cannot open '(' if not enough positions left to close existing parens
        # Need at least 1 position for each unclosed paren
        if num_masked <= paren_depth:
            if self.open_paren_id >= 0:
                forbidden.add(self.open_paren_id)

        # Rule 3: Limit total open parens to half of remaining positions
        # Each '(' needs a matching ')', so max opens = num_masked // 2
        max_allowed_opens = num_masked // 2
        if paren_depth >= max_allowed_opens:
            if self.open_paren_id >= 0:
                forbidden.add(self.open_paren_id)

        # Rule 4: Forbid special tokens from being sampled
        forbidden.add(self.mask_id)
        forbidden.add(self.pad_id)
        forbidden.add(self.bos_id)
        forbidden.add(self.eos_id)

        return forbidden

    def _apply_special_token_constraints(
        self,
        logits: torch.Tensor,
        current_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forbid special tokens from being sampled at MASK positions."""
        is_masked = current_ids == self.mask_id
        for tok in [self.mask_id, self.pad_id, self.bos_id, self.eos_id]:
            if tok >= 0:
                logits[:, :, tok].masked_fill_(is_masked, float('-inf'))
        return logits

    def _adapt_ar_logits_for_constraints(
        self,
        ar_logits: torch.Tensor,
        prefix_ids: torch.Tensor,
        max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adapt AR next-token logits to diffusion-style format for constraint reuse."""
        batch_size, prefix_len = prefix_ids.shape
        vocab_size = ar_logits.shape[-1]

        full_ids = torch.full(
            (batch_size, max_length),
            self.mask_id,
            dtype=prefix_ids.dtype,
            device=prefix_ids.device
        )
        full_ids[:, :prefix_len] = prefix_ids

        full_logits = torch.zeros(
            (batch_size, max_length, vocab_size),
            dtype=ar_logits.dtype,
            device=ar_logits.device
        )
        full_logits[:, prefix_len, :] = ar_logits

        return full_ids, full_logits

    def _filter_logits_top_k_top_p(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> torch.Tensor:
        """Apply top-k and/or top-p filtering to logits."""
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            min_values = values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_values,
                torch.full_like(logits, float('-inf')),
                logits
            )

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits

    def _apply_syntax_constraints(
        self,
        logits: torch.Tensor,
        current_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply SMILES syntax constraints to logits.

        For each sequence in batch:
        1. Analyze current unmasked tokens to get syntax state
        2. Determine forbidden tokens for remaining masked positions
        3. Set logits of forbidden tokens to -inf

        Args:
            logits: Logits of shape [batch, seq_len, vocab_size].
            current_ids: Current token IDs of shape [batch, seq_len].

        Returns:
            Modified logits with invalid tokens masked out.
        """
        batch_size, seq_len, vocab_size = logits.shape

        for i in range(batch_size):
            # Count masked positions
            num_masked = (current_ids[i] == self.mask_id).sum().item()

            # Analyze syntax state from unmasked tokens
            syntax_state = self._analyze_syntax_state(current_ids[i])

            # Get forbidden tokens
            forbidden = self._get_forbidden_tokens(syntax_state, num_masked)

            # Apply to all masked positions
            mask_positions = current_ids[i] == self.mask_id
            for token_id in forbidden:
                if token_id >= 0:
                    logits[i, mask_positions, token_id] = float('-inf')

        return logits

    def _apply_position_aware_paren_constraints(
        self,
        logits: torch.Tensor,
        current_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply position-aware parenthesis constraints to logits.

        Unlike global constraints, this method considers the POSITION of each
        masked token to determine valid parenthesis placements:
        - ')' can only be placed if there's an unclosed '(' to its LEFT
        - '(' can only be placed if there's room to close it to its RIGHT

        Args:
            logits: Logits of shape [batch, seq_len, vocab_size].
            current_ids: Current token IDs of shape [batch, seq_len].

        Returns:
            Modified logits with invalid tokens masked out.
        """
        if self.open_paren_id < 0 and self.close_paren_id < 0:
            return logits

        is_masked = current_ids == self.mask_id
        if not is_masked.any():
            return logits

        open_mask = current_ids == self.open_paren_id
        close_mask = current_ids == self.close_paren_id
        depth_delta = open_mask.int() - close_mask.int()
        depth_prefix = depth_delta.cumsum(dim=1)
        depth_at_pos = torch.cat(
            [torch.zeros_like(depth_prefix[:, :1]), depth_prefix[:, :-1]],
            dim=1
        )

        masked_from_pos = torch.flip(
            torch.cumsum(torch.flip(is_masked.int(), dims=[1]), dim=1),
            dims=[1]
        )

        if self.close_paren_id >= 0:
            invalid_close = is_masked & (depth_at_pos <= 0)
            logits[:, :, self.close_paren_id].masked_fill_(invalid_close, float('-inf'))

        if self.open_paren_id >= 0:
            invalid_open = is_masked & (depth_at_pos + 2 > masked_from_pos)
            logits[:, :, self.open_paren_id].masked_fill_(invalid_open, float('-inf'))

        for tok in [self.mask_id, self.pad_id, self.bos_id, self.eos_id]:
            if tok >= 0:
                logits[:, :, tok].masked_fill_(is_masked, float('-inf'))

        return logits

    def _apply_ring_constraints(
        self,
        logits: torch.Tensor,
        current_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply ring closure constraints to logits.

        Ensures each ring digit (1-9, %10-%99) appears exactly twice.
        - Forbids ring digits that are already closed (appeared twice)
        - If too many open rings, forbids opening new ones

        Args:
            logits: Logits of shape [batch, seq_len, vocab_size].
            current_ids: Current token IDs of shape [batch, seq_len].

        Returns:
            Modified logits with invalid ring tokens masked out.
        """
        all_ring_ids = sorted(self.ring_digit_ids | self.ring_percent_ids)
        if not all_ring_ids:
            return logits

        ring_ids = torch.tensor(all_ring_ids, device=current_ids.device, dtype=torch.long)
        ring_mask = current_ids.unsqueeze(-1) == ring_ids
        ring_counts = ring_mask.sum(dim=1)

        open_rings = ring_counts == 1
        closed_rings = ring_counts >= 2

        num_masked = (current_ids == self.mask_id).sum(dim=1)
        open_counts = open_rings.sum(dim=1)
        forbid_open_new = num_masked <= open_counts

        forbidden_ring = closed_rings | (forbid_open_new.unsqueeze(1) & ~open_rings)

        masked_positions = current_ids == self.mask_id
        ring_logits = logits.index_select(2, ring_ids)
        forbidden = masked_positions.unsqueeze(-1) & forbidden_ring.unsqueeze(1)
        ring_logits = ring_logits.masked_fill(forbidden, float('-inf'))
        logits.index_copy_(2, ring_ids, ring_logits)

        return logits

    def _apply_bond_placement_constraints(
        self,
        logits: torch.Tensor,
        current_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply bond placement constraints to logits.

        Bonds (=, #, /, \) must be preceded by an atom, not by:
        - Open parenthesis (
        - Another bond
        - Start of sequence
        - Close parenthesis )

        Args:
            logits: Logits of shape [batch, seq_len, vocab_size].
            current_ids: Current token IDs of shape [batch, seq_len].

        Returns:
            Modified logits with invalid bond placements masked out.
        """
        if not self.bond_ids:
            return logits

        is_masked = current_ids == self.mask_id
        if not is_masked.any():
            return logits

        batch_size, seq_len = current_ids.shape
        indices = torch.arange(seq_len, device=current_ids.device).unsqueeze(0).expand(batch_size, -1)
        valid_idx = torch.where(is_masked, torch.full_like(indices, -1), indices)
        prev_idx = torch.cummax(valid_idx, dim=1).values

        prev_idx_clamped = prev_idx.clamp(min=0)
        prev_ids = current_ids.gather(1, prev_idx_clamped)
        prev_ids = torch.where(prev_idx >= 0, prev_ids, torch.full_like(prev_ids, -1))

        invalid_prev = prev_idx < 0
        if self.bos_id >= 0:
            invalid_prev = invalid_prev | (prev_ids == self.bos_id)
        if self.eos_id >= 0:
            invalid_prev = invalid_prev | (prev_ids == self.eos_id)
        if self.pad_id >= 0:
            invalid_prev = invalid_prev | (prev_ids == self.pad_id)
        if self.open_paren_id >= 0:
            invalid_prev = invalid_prev | (prev_ids == self.open_paren_id)
        if self.close_paren_id >= 0:
            invalid_prev = invalid_prev | (prev_ids == self.close_paren_id)

        bond_ids = sorted(self.bond_ids)
        bond_ids_tensor = torch.tensor(bond_ids, device=current_ids.device, dtype=current_ids.dtype)
        prev_is_bond = (prev_ids.unsqueeze(-1) == bond_ids_tensor).any(dim=-1)
        invalid_prev = invalid_prev | prev_is_bond

        invalid_bond = is_masked & invalid_prev
        bond_logits = logits.index_select(2, bond_ids_tensor)
        bond_logits = bond_logits.masked_fill(invalid_bond.unsqueeze(-1), float('-inf'))
        logits.index_copy_(2, bond_ids_tensor, bond_logits)

        if self.open_paren_id >= 0 and self.close_paren_id >= 0:
            invalid_close = is_masked & (prev_ids == self.open_paren_id)
            logits[:, :, self.close_paren_id].masked_fill_(invalid_close, float('-inf'))

        return logits

    def _apply_star_constraint(
        self,
        logits: torch.Tensor,
        current_ids: torch.Tensor,
        max_stars: int = 2
    ) -> torch.Tensor:
        """Apply constraint to limit number of '*' tokens.

        Args:
            logits: Logits of shape [batch, seq_len, vocab_size].
            current_ids: Current token IDs of shape [batch, seq_len].
            max_stars: Maximum allowed '*' tokens.

        Returns:
            Modified logits.
        """
        if self.star_id < 0:
            return logits

        # Count current stars (excluding MASK positions)
        non_mask = current_ids != self.mask_id
        current_stars = ((current_ids == self.star_id) & non_mask).sum(dim=1)

        # For sequences with >= max_stars, set star logit to -inf at MASK positions
        blocked = current_stars >= max_stars
        if blocked.any():
            mask_positions = current_ids == self.mask_id
            logits[:, :, self.star_id].masked_fill_(mask_positions & blocked.unsqueeze(1), float('-inf'))

        return logits

    def _fix_star_count(
        self,
        ids: torch.Tensor,
        logits: torch.Tensor,
        target_stars: int = 2
    ) -> torch.Tensor:
        """Fix the number of '*' tokens in final sequences.

        Args:
            ids: Token IDs of shape [batch, seq_len].
            logits: Final logits of shape [batch, seq_len, vocab_size].
            target_stars: Target number of '*' tokens.

        Returns:
            Fixed token IDs.
        """
        batch_size, seq_len = ids.shape
        fixed_ids = ids.clone()

        for i in range(batch_size):
            star_mask = fixed_ids[i] == self.star_id
            num_stars = star_mask.sum().item()

            if num_stars > target_stars:
                # Keep only the top-k most probable star positions
                star_positions = torch.where(star_mask)[0]
                star_probs = logits[i, star_positions, self.star_id]

                # Get indices of stars to keep (highest probability)
                _, keep_indices = torch.topk(star_probs, target_stars)
                keep_positions = star_positions[keep_indices]

                # Replace extra stars with second-best token
                for pos in star_positions:
                    if pos not in keep_positions:
                        # Get second-best token (excluding star)
                        pos_logits = logits[i, pos].clone()
                        pos_logits[self.star_id] = float('-inf')
                        pos_logits[self.mask_id] = float('-inf')
                        pos_logits[self.pad_id] = float('-inf')
                        best_token = pos_logits.argmax()
                        fixed_ids[i, pos] = best_token

            elif num_stars < target_stars:
                # Find best positions to add stars
                needed = target_stars - num_stars

                # Get star probabilities at all non-special positions
                valid_mask = (
                    (fixed_ids[i] != self.bos_id) &
                    (fixed_ids[i] != self.eos_id) &
                    (fixed_ids[i] != self.pad_id) &
                    (fixed_ids[i] != self.star_id)
                )
                valid_positions = torch.where(valid_mask)[0]

                if len(valid_positions) >= needed:
                    star_probs = logits[i, valid_positions, self.star_id]
                    _, best_indices = torch.topk(star_probs, needed)
                    best_positions = valid_positions[best_indices]

                    for pos in best_positions:
                        fixed_ids[i, pos] = self.star_id

        return fixed_ids

    def _fix_paren_balance(
        self,
        ids: torch.Tensor,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """Fix unbalanced parentheses in final sequences.

        Two-pass algorithm:
        1. Left-to-right: Replace any ')' that has no matching '(' to its left
        2. Right-to-left: Replace any '(' that has no matching ')' to its right

        Args:
            ids: Token IDs of shape [batch, seq_len].
            logits: Final logits of shape [batch, seq_len, vocab_size].

        Returns:
            Fixed token IDs with balanced parentheses.
        """
        batch_size, seq_len = ids.shape
        fixed_ids = ids.clone()

        for i in range(batch_size):
            # First pass (left-to-right): remove ')' that have no matching '('
            depth = 0
            for j in range(seq_len):
                token_id = fixed_ids[i, j].item()
                if token_id == self.open_paren_id:
                    depth += 1
                elif token_id == self.close_paren_id:
                    if depth > 0:
                        depth -= 1
                    else:
                        # No matching '(' to the left - replace with best alternative
                        pos_logits = logits[i, j].clone()
                        pos_logits[self.close_paren_id] = float('-inf')
                        pos_logits[self.open_paren_id] = float('-inf')
                        pos_logits[self.mask_id] = float('-inf')
                        pos_logits[self.pad_id] = float('-inf')
                        pos_logits[self.bos_id] = float('-inf')
                        pos_logits[self.eos_id] = float('-inf')
                        fixed_ids[i, j] = pos_logits.argmax()

            # Second pass (right-to-left): remove unclosed '('
            depth = 0
            for j in range(seq_len - 1, -1, -1):
                token_id = fixed_ids[i, j].item()
                if token_id == self.close_paren_id:
                    depth += 1
                elif token_id == self.open_paren_id:
                    if depth > 0:
                        depth -= 1
                    else:
                        # No matching ')' to the right - replace with best alternative
                        pos_logits = logits[i, j].clone()
                        pos_logits[self.close_paren_id] = float('-inf')
                        pos_logits[self.open_paren_id] = float('-inf')
                        pos_logits[self.mask_id] = float('-inf')
                        pos_logits[self.pad_id] = float('-inf')
                        pos_logits[self.bos_id] = float('-inf')
                        pos_logits[self.eos_id] = float('-inf')
                        fixed_ids[i, j] = pos_logits.argmax()

        return fixed_ids

    def _fix_ring_closures(
        self,
        ids: torch.Tensor,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """Fix unpaired ring closures in final sequences.

        Scans each sequence and replaces ring digits that appear an odd number
        of times (unpaired) with the next-best non-ring token.

        Args:
            ids: Token IDs of shape [batch, seq_len].
            logits: Final logits of shape [batch, seq_len, vocab_size].

        Returns:
            Fixed token IDs with paired ring closures.
        """
        batch_size, seq_len = ids.shape
        fixed_ids = ids.clone()

        # Combine all ring token IDs
        all_ring_ids = self.ring_digit_ids | self.ring_percent_ids

        for i in range(batch_size):
            # Count occurrences and positions of each ring digit
            ring_positions = {}  # ring_id -> list of positions
            for j in range(seq_len):
                token_id = fixed_ids[i, j].item()
                if token_id in all_ring_ids:
                    if token_id not in ring_positions:
                        ring_positions[token_id] = []
                    ring_positions[token_id].append(j)

            # Fix rings with odd counts (replace last occurrence)
            for ring_id, positions in ring_positions.items():
                if len(positions) % 2 != 0:  # Odd count - unpaired
                    # Replace last occurrence with best non-ring alternative
                    last_pos = positions[-1]
                    pos_logits = logits[i, last_pos].clone()

                    # Forbid all ring digits
                    for r_id in all_ring_ids:
                        pos_logits[r_id] = float('-inf')
                    # Forbid special tokens
                    for tok in [self.mask_id, self.pad_id, self.bos_id, self.eos_id]:
                        if tok >= 0:
                            pos_logits[tok] = float('-inf')
                    # Forbid parentheses (to not break paren balance)
                    if self.open_paren_id >= 0:
                        pos_logits[self.open_paren_id] = float('-inf')
                    if self.close_paren_id >= 0:
                        pos_logits[self.close_paren_id] = float('-inf')

                    fixed_ids[i, last_pos] = pos_logits.argmax()

        return fixed_ids

    def _sample_from_ids(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, List[str]]:
        """Run left-to-right autoregressive sampling starting from provided token IDs.

        Args:
            ids: Initial token IDs of shape [batch, seq_len].
            attention_mask: Attention mask of shape [batch, seq_len].
            fixed_mask: Boolean mask marking fixed (non-sampled) positions.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (token_ids, smiles_strings).
        """
        self.diffusion_model.eval()
        backbone = self.diffusion_model.backbone
        batch_size, max_length = ids.shape

        final_logits = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        steps = range(1, max_length)
        if show_progress:
            steps = tqdm(steps, desc="Sampling")

        for pos in steps:
            if finished.all():
                break

            pos_fixed = fixed_mask[:, pos] if fixed_mask is not None else torch.zeros_like(finished)
            sample_mask = (~pos_fixed) & (~finished)

            prefix_ids = ids[:, :pos]
            prefix_mask = torch.ones_like(prefix_ids)

            with torch.no_grad():
                logits = backbone(prefix_ids, prefix_mask)[:, -1, :]

            logits = logits / self.temperature
            raw_logits = logits.clone()

            full_ids, full_logits = self._adapt_ar_logits_for_constraints(
                logits, prefix_ids, max_length
            )

            if self.use_constraints:
                full_logits = self._apply_star_constraint(full_logits, full_ids, max_stars=2)
                full_logits = self._apply_position_aware_paren_constraints(full_logits, full_ids)
                full_logits = self._apply_ring_constraints(full_logits, full_ids)
                full_logits = self._apply_bond_placement_constraints(full_logits, full_ids)
            full_logits = self._apply_special_token_constraints(full_logits, full_ids)

            logits = full_logits[:, pos, :]

            # Allow EOS only after star count is satisfied
            star_count = self._count_stars(prefix_ids)
            allow_eos = star_count >= 2
            if self.eos_id >= 0:
                eos_logits = raw_logits[:, self.eos_id]
                logits[:, self.eos_id] = torch.where(
                    allow_eos,
                    eos_logits,
                    torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype)
                )

            next_tokens = ids[:, pos].clone()

            if sample_mask.any():
                filtered = self._filter_logits_top_k_top_p(
                    logits[sample_mask], self.top_k, self.top_p
                )

                all_inf = torch.isneginf(filtered).all(dim=-1)
                if all_inf.any():
                    filtered[all_inf] = 0.0
                    for tok in [self.mask_id, self.pad_id, self.bos_id]:
                        if tok >= 0:
                            filtered[all_inf, tok] = float('-inf')

                probs = F.softmax(filtered, dim=-1)
                sampled = torch.multinomial(probs, 1).squeeze(-1)
                next_tokens[sample_mask] = sampled

            # Pad out finished sequences unless position is fixed
            to_pad = finished & (~pos_fixed)
            if to_pad.any() and self.pad_id >= 0:
                next_tokens[to_pad] = self.pad_id

            ids[:, pos] = next_tokens

            finished = finished | (next_tokens == self.eos_id)

            if final_logits is None:
                final_logits = torch.zeros(
                    (batch_size, max_length, logits.shape[-1]),
                    dtype=logits.dtype,
                    device=logits.device
                )
            final_logits[:, pos, :] = full_logits[:, pos, :]

        if self.use_constraints and final_logits is not None:
            ids = self._fix_star_count(ids, final_logits, target_stars=2)
            ids = self._fix_paren_balance(ids, final_logits)
            ids = self._fix_ring_closures(ids, final_logits)

        smiles_list = self.tokenizer.batch_decode(ids.cpu().tolist(), skip_special_tokens=True)
        return ids, smiles_list

    def sample_with_lengths(
        self,
        lengths: List[int],
        max_length: Optional[int] = None,
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, List[str]]:
        """Sample polymers with per-sample lengths (includes BOS/EOS).

        Args:
            lengths: Sequence lengths INCLUDING BOS/EOS for each sample.
            max_length: Optional hard cap for sequence length.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (token_ids, smiles_strings).
        """
        if not lengths:
            return torch.empty((0, 0), dtype=torch.long, device=self.device), []

        lengths = [max(2, int(l)) for l in lengths]
        seq_length = max(lengths)
        if max_length is not None and seq_length > max_length:
            raise ValueError(f"Max length {max_length} is smaller than required {seq_length}")

        batch_size = len(lengths)
        ids = torch.full(
            (batch_size, seq_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device
        )
        attention_mask = torch.zeros_like(ids)
        fixed_mask = torch.zeros_like(ids, dtype=torch.bool)

        ids[:, 0] = self.bos_id
        fixed_mask[:, 0] = True

        for i, length in enumerate(lengths):
            eos_pos = length - 1
            ids[i, eos_pos] = self.eos_id
            fixed_mask[i, eos_pos] = True
            attention_mask[i, :length] = 1

            if length < seq_length:
                ids[i, length:] = self.pad_id
                fixed_mask[i, length:] = True

        return self._sample_from_ids(ids, attention_mask, fixed_mask, show_progress)

    def sample(
        self,
        batch_size: int,
        seq_length: int,
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, List[str]]:
        """Sample new polymers with exactly two '*' tokens.

        Args:
            batch_size: Number of samples to generate.
            seq_length: Sequence length.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (token_ids, smiles_strings).
        """
        self.diffusion_model.eval()

        # Initialize with fully masked sequence (except BOS/EOS)
        ids = torch.full(
            (batch_size, seq_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device
        )
        ids[:, 0] = self.bos_id
        ids[:, -1] = self.eos_id

        # Create attention mask
        attention_mask = torch.ones_like(ids)

        fixed_mask = ids != self.mask_id

        return self._sample_from_ids(ids, attention_mask, fixed_mask, show_progress)

    def sample_batch(
        self,
        num_samples: int,
        seq_length: int,
        batch_size: int = 256,
        show_progress: bool = True,
        lengths: Optional[List[int]] = None
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Sample multiple batches of polymers.

        Args:
            num_samples: Total number of samples.
            seq_length: Sequence length.
            batch_size: Batch size for sampling.
            show_progress: Whether to show progress.
            lengths: Optional list of per-sample lengths (includes BOS/EOS).

        Returns:
            Tuple of (all_ids, all_smiles).
        """
        all_ids = []
        all_smiles = []

        if lengths is not None and len(lengths) != num_samples:
            raise ValueError("lengths must match num_samples")

        num_batches = (num_samples + batch_size - 1) // batch_size
        sample_idx = 0

        for batch_idx in tqdm(range(num_batches), desc="Batch sampling", disable=not show_progress):
            current_batch_size = min(batch_size, num_samples - sample_idx)

            if lengths is None:
                ids, smiles = self.sample(
                    current_batch_size,
                    seq_length,
                    show_progress=False
                )
            else:
                batch_lengths = lengths[sample_idx:sample_idx + current_batch_size]
                ids, smiles = self.sample_with_lengths(
                    batch_lengths,
                    max_length=seq_length,
                    show_progress=False
                )

            all_ids.append(ids)
            all_smiles.extend(smiles)
            sample_idx += current_batch_size

        return all_ids, all_smiles

    def sample_variable_length(
        self,
        num_samples: int,
        length_range: Tuple[int, int] = (20, 100),
        batch_size: int = 256,
        samples_per_length: int = 16,
        show_progress: bool = True
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Sample polymers with variable sequence lengths.

        Each small batch uses a different randomly chosen sequence length
        from the specified range, producing diverse SMILES lengths.

        Args:
            num_samples: Total number of samples.
            length_range: (min_length, max_length) for sequence lengths.
            batch_size: Maximum batch size for GPU memory.
            samples_per_length: Number of samples per length (controls diversity).
                               Smaller values = more length diversity.
            show_progress: Whether to show progress.

        Returns:
            Tuple of (all_ids, all_smiles).
        """
        all_ids = []
        all_smiles = []

        min_len, max_len = length_range
        # Use smaller internal batch size for more length diversity
        internal_batch_size = min(batch_size, samples_per_length)
        num_batches = (num_samples + internal_batch_size - 1) // internal_batch_size

        for batch_idx in tqdm(range(num_batches), desc="Variable length sampling", disable=not show_progress):
            current_batch_size = min(internal_batch_size, num_samples - len(all_smiles))

            # Random sequence length for this batch
            seq_length = np.random.randint(min_len, max_len + 1)

            ids, smiles = self.sample(
                current_batch_size,
                seq_length,
                show_progress=False
            )

            all_ids.append(ids)
            all_smiles.extend(smiles)

        return all_ids, all_smiles

    def sample_conditional(
        self,
        batch_size: int,
        seq_length: int,
        prefix_ids: Optional[torch.Tensor] = None,
        suffix_ids: Optional[torch.Tensor] = None,
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, List[str]]:
        """Sample with optional prefix/suffix conditioning.

        Args:
            batch_size: Number of samples.
            seq_length: Sequence length.
            prefix_ids: Fixed prefix tokens.
            suffix_ids: Fixed suffix tokens.
            show_progress: Whether to show progress.

        Returns:
            Tuple of (token_ids, smiles_strings).
        """
        # Initialize
        ids = torch.full(
            (batch_size, seq_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device
        )
        ids[:, 0] = self.bos_id
        ids[:, -1] = self.eos_id

        # Apply prefix/suffix constraints
        fixed_mask = torch.zeros_like(ids, dtype=torch.bool)
        fixed_mask[:, 0] = True  # BOS
        fixed_mask[:, -1] = True  # EOS

        if prefix_ids is not None:
            prefix_len = prefix_ids.shape[1]
            ids[:, 1:1+prefix_len] = prefix_ids
            fixed_mask[:, 1:1+prefix_len] = True

        if suffix_ids is not None:
            suffix_len = suffix_ids.shape[1]
            ids[:, -1-suffix_len:-1] = suffix_ids
            fixed_mask[:, -1-suffix_len:-1] = True

        attention_mask = torch.ones_like(ids)

        return self._sample_from_ids(ids, attention_mask, fixed_mask, show_progress)
