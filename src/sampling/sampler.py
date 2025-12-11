"""Constrained sampler for polymer generation."""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np
from tqdm import tqdm


class ConstrainedSampler:
    """Constrained sampler ensuring exactly two '*' tokens in generated polymers.

    Implements reverse diffusion with constraints:
    - During sampling: limits '*' tokens to at most 2
    - At final step: ensures exactly 2 '*' tokens
    """

    def __init__(
        self,
        diffusion_model,
        tokenizer,
        num_steps: int = 100,
        temperature: float = 1.0,
        device: str = 'cuda'
    ):
        """Initialize sampler.

        Args:
            diffusion_model: Trained discrete masking diffusion model.
            tokenizer: Tokenizer instance.
            num_steps: Number of diffusion steps.
            temperature: Sampling temperature.
            device: Device for computation.
        """
        self.diffusion_model = diffusion_model
        self.tokenizer = tokenizer
        self.num_steps = num_steps
        self.temperature = temperature
        self.device = device

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
        batch_size, seq_len, vocab_size = logits.shape

        for i in range(batch_size):
            seq = current_ids[i]

            # Precompute depth from LEFT for each position
            # depth_at_pos[j] = net (opens - closes) from unmasked tokens in positions 0..j-1
            depth_at_pos = []
            depth = 0
            for j in range(seq_len):
                depth_at_pos.append(depth)
                token_id = seq[j].item()
                # Only count unmasked parentheses
                if token_id == self.open_paren_id:
                    depth += 1
                elif token_id == self.close_paren_id:
                    depth -= 1
                # masked tokens don't affect depth yet

            # Count masked positions from each index to end (including that index)
            masked_from_pos = []
            count = 0
            for j in range(seq_len - 1, -1, -1):
                if seq[j].item() == self.mask_id:
                    count += 1
                masked_from_pos.insert(0, count)

            # Apply position-specific constraints
            for pos in range(seq_len):
                if seq[pos] != self.mask_id:
                    continue

                d_left = depth_at_pos[pos]  # Depth from unmasked tokens to the left
                n_right = masked_from_pos[pos]  # Masked positions from here to end

                # Rule 1: Cannot place ')' if no unclosed '(' to the left
                if d_left <= 0:
                    if self.close_paren_id >= 0:
                        logits[i, pos, self.close_paren_id] = float('-inf')

                # Rule 2: Cannot place '(' if not enough room to close it
                # If we place '(' here, we need at least one position to close it
                # d_left + 1 (new open) must be <= n_right - 1 (remaining after this)
                # Simplified: d_left + 1 > n_right means we can't close
                if d_left + 1 > n_right:
                    if self.open_paren_id >= 0:
                        logits[i, pos, self.open_paren_id] = float('-inf')

                # Rule 3: Forbid special tokens
                for tok in [self.mask_id, self.pad_id, self.bos_id, self.eos_id]:
                    if tok >= 0:
                        logits[i, pos, tok] = float('-inf')

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
        batch_size, seq_len, vocab_size = logits.shape

        # Combine all ring token IDs
        all_ring_ids = self.ring_digit_ids | self.ring_percent_ids

        for i in range(batch_size):
            seq = current_ids[i]

            # Count occurrences of each ring token in unmasked positions
            ring_counts = {}
            for j in range(seq_len):
                token_id = seq[j].item()
                if token_id in all_ring_ids:
                    ring_counts[token_id] = ring_counts.get(token_id, 0) + 1

            # Identify open rings (count == 1) and closed rings (count >= 2)
            open_rings = {r for r, c in ring_counts.items() if c == 1}
            closed_rings = {r for r, c in ring_counts.items() if c >= 2}

            # Count masked positions
            num_masked = (seq == self.mask_id).sum().item()

            # For each masked position, apply constraints
            for pos in range(seq_len):
                if seq[pos] != self.mask_id:
                    continue

                # Rule 1: Forbid ring digits that are already closed (count >= 2)
                for ring_id in closed_rings:
                    logits[i, pos, ring_id] = float('-inf')

                # Rule 2: If too many open rings, forbid opening new ones
                # Each open ring needs 1 more position to close
                # Only allow opening new if: num_masked > len(open_rings)
                if num_masked <= len(open_rings):
                    # Forbid all ring digits except those that are open (can close them)
                    for ring_id in all_ring_ids:
                        if ring_id not in open_rings:
                            logits[i, pos, ring_id] = float('-inf')

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
        batch_size, seq_len, vocab_size = logits.shape

        for i in range(batch_size):
            seq = current_ids[i]

            # For each masked position, find previous unmasked token
            for pos in range(seq_len):
                if seq[pos] != self.mask_id:
                    continue

                # Find previous unmasked token
                prev_token_id = None
                for j in range(pos - 1, -1, -1):
                    if seq[j] != self.mask_id:
                        prev_token_id = seq[j].item()
                        break

                # Forbid bonds if:
                # 1. At start (no previous token)
                if prev_token_id is None:
                    for bond_id in self.bond_ids:
                        logits[i, pos, bond_id] = float('-inf')

                # 2. Previous is open parenthesis
                elif prev_token_id == self.open_paren_id:
                    for bond_id in self.bond_ids:
                        logits[i, pos, bond_id] = float('-inf')
                    # Also forbid close paren immediately after open paren (prevents empty parens)
                    logits[i, pos, self.close_paren_id] = float('-inf')

                # 3. Previous is close parenthesis (conservative)
                elif prev_token_id == self.close_paren_id:
                    for bond_id in self.bond_ids:
                        logits[i, pos, bond_id] = float('-inf')

                # 4. Previous is a bond (no consecutive bonds)
                elif prev_token_id in self.bond_ids:
                    for bond_id in self.bond_ids:
                        logits[i, pos, bond_id] = float('-inf')

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
        batch_size, seq_len, vocab_size = logits.shape

        # Count current stars (excluding MASK positions)
        non_mask = current_ids != self.mask_id
        current_stars = ((current_ids == self.star_id) & non_mask).sum(dim=1)

        # For sequences with >= max_stars, set star logit to -inf at MASK positions
        for i in range(batch_size):
            if current_stars[i] >= max_stars:
                mask_positions = current_ids[i] == self.mask_id
                logits[i, mask_positions, self.star_id] = float('-inf')

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
        backbone = self.diffusion_model.backbone

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

        # Store logits for final fixing
        final_logits = None

        # Reverse diffusion
        steps = range(self.num_steps, 0, -1)
        if show_progress:
            steps = tqdm(steps, desc="Sampling")

        for t in steps:
            timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            with torch.no_grad():
                logits = backbone(ids, timesteps, attention_mask)

            # Apply temperature
            logits = logits / self.temperature

            # Apply star constraint
            logits = self._apply_star_constraint(logits, ids, max_stars=2)

            # Apply position-aware parenthesis constraints
            logits = self._apply_position_aware_paren_constraints(logits, ids)

            # Apply ring closure constraints
            logits = self._apply_ring_constraints(logits, ids)

            # Apply bond placement constraints
            logits = self._apply_bond_placement_constraints(logits, ids)

            # Sample from masked positions
            probs = F.softmax(logits, dim=-1)

            # Only update masked positions
            is_masked = ids == self.mask_id

            # Determine which masked tokens to unmask at this step
            # Unmask proportionally based on schedule
            unmask_prob = 1.0 / t  # Simple linear unmasking

            for i in range(batch_size):
                masked_pos = torch.where(is_masked[i])[0]
                if len(masked_pos) == 0:
                    continue

                # Randomly select positions to unmask
                num_unmask = max(1, int(len(masked_pos) * unmask_prob))
                unmask_indices = torch.randperm(len(masked_pos))[:num_unmask]
                unmask_positions = masked_pos[unmask_indices]

                # Sample tokens for these positions SEQUENTIALLY with constraint updates
                # This prevents race conditions where multiple positions sample conflicting tokens
                for pos in unmask_positions:
                    sampled = torch.multinomial(probs[i, pos], 1)
                    ids[i, pos] = sampled

                    # Update constraints dynamically based on what was just sampled
                    sampled_token = sampled.item()

                    # If we sampled a star, update star constraint for remaining positions
                    if sampled_token == self.star_id:
                        # Count current stars (excluding remaining MASK positions)
                        non_mask = ids[i] != self.mask_id
                        current_stars = ((ids[i] == self.star_id) & non_mask).sum().item()

                        # If we've reached the limit, forbid stars at remaining masked positions
                        if current_stars >= 2:
                            remaining_mask = ids[i] == self.mask_id
                            logits[i, remaining_mask, self.star_id] = float('-inf')
                            probs[i] = F.softmax(logits[i], dim=-1)

                    # If we sampled a bond, forbid bonds at the next position
                    elif sampled_token in self.bond_ids:
                        # Find next position in the sequence (not necessarily next in unmask order)
                        next_pos = pos + 1
                        if next_pos < len(ids[i]) and ids[i, next_pos] == self.mask_id:
                            for bond_id in self.bond_ids:
                                logits[i, next_pos, bond_id] = float('-inf')
                            probs[i] = F.softmax(logits[i], dim=-1)

                    # If we sampled an open paren, forbid close paren at the next position
                    elif sampled_token == self.open_paren_id:
                        next_pos = pos + 1
                        if next_pos < len(ids[i]) and ids[i, next_pos] == self.mask_id:
                            logits[i, next_pos, self.close_paren_id] = float('-inf')
                            probs[i] = F.softmax(logits[i], dim=-1)

            # Store logits for final step
            if t == 1:
                final_logits = logits

        # Fix star count in final sequences
        ids = self._fix_star_count(ids, final_logits, target_stars=2)

        # Fix parenthesis balance (safety net)
        ids = self._fix_paren_balance(ids, final_logits)

        # Fix ring closures (safety net)
        ids = self._fix_ring_closures(ids, final_logits)

        # Decode to SMILES
        smiles_list = self.tokenizer.batch_decode(ids.cpu().tolist(), skip_special_tokens=True)

        return ids, smiles_list

    def sample_batch(
        self,
        num_samples: int,
        seq_length: int,
        batch_size: int = 256,
        show_progress: bool = True
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Sample multiple batches of polymers.

        Args:
            num_samples: Total number of samples.
            seq_length: Sequence length.
            batch_size: Batch size for sampling.
            show_progress: Whether to show progress.

        Returns:
            Tuple of (all_ids, all_smiles).
        """
        all_ids = []
        all_smiles = []

        num_batches = (num_samples + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Batch sampling", disable=not show_progress):
            current_batch_size = min(batch_size, num_samples - len(all_smiles))

            ids, smiles = self.sample(
                current_batch_size,
                seq_length,
                show_progress=False
            )

            all_ids.append(ids)
            all_smiles.extend(smiles)

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
        self.diffusion_model.eval()
        backbone = self.diffusion_model.backbone

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
        final_logits = None

        steps = range(self.num_steps, 0, -1)
        if show_progress:
            steps = tqdm(steps, desc="Sampling")

        for t in steps:
            timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            with torch.no_grad():
                logits = backbone(ids, timesteps, attention_mask)

            logits = logits / self.temperature
            logits = self._apply_star_constraint(logits, ids, max_stars=2)
            logits = self._apply_position_aware_paren_constraints(logits, ids)
            logits = self._apply_ring_constraints(logits, ids)
            logits = self._apply_bond_placement_constraints(logits, ids)

            probs = F.softmax(logits, dim=-1)
            is_masked = (ids == self.mask_id) & (~fixed_mask)

            unmask_prob = 1.0 / t

            for i in range(batch_size):
                masked_pos = torch.where(is_masked[i])[0]
                if len(masked_pos) == 0:
                    continue

                num_unmask = max(1, int(len(masked_pos) * unmask_prob))
                unmask_indices = torch.randperm(len(masked_pos))[:num_unmask]
                unmask_positions = masked_pos[unmask_indices]

                # Sample tokens for these positions SEQUENTIALLY with constraint updates
                # This prevents race conditions where multiple positions sample conflicting tokens
                for pos in unmask_positions:
                    sampled = torch.multinomial(probs[i, pos], 1)
                    ids[i, pos] = sampled

                    # Update constraints dynamically based on what was just sampled
                    sampled_token = sampled.item()

                    # If we sampled a star, update star constraint for remaining positions
                    if sampled_token == self.star_id:
                        # Count current stars (excluding remaining MASK positions)
                        non_mask = ids[i] != self.mask_id
                        current_stars = ((ids[i] == self.star_id) & non_mask).sum().item()

                        # If we've reached the limit, forbid stars at remaining masked positions
                        if current_stars >= 2:
                            remaining_mask = ids[i] == self.mask_id
                            logits[i, remaining_mask, self.star_id] = float('-inf')
                            probs[i] = F.softmax(logits[i], dim=-1)

                    # If we sampled a bond, forbid bonds at the next position
                    elif sampled_token in self.bond_ids:
                        # Find next position in the sequence (not necessarily next in unmask order)
                        next_pos = pos + 1
                        if next_pos < len(ids[i]) and ids[i, next_pos] == self.mask_id:
                            for bond_id in self.bond_ids:
                                logits[i, next_pos, bond_id] = float('-inf')
                            probs[i] = F.softmax(logits[i], dim=-1)

                    # If we sampled an open paren, forbid close paren at the next position
                    elif sampled_token == self.open_paren_id:
                        next_pos = pos + 1
                        if next_pos < len(ids[i]) and ids[i, next_pos] == self.mask_id:
                            logits[i, next_pos, self.close_paren_id] = float('-inf')
                            probs[i] = F.softmax(logits[i], dim=-1)

            if t == 1:
                final_logits = logits

        ids = self._fix_star_count(ids, final_logits, target_stars=2)
        ids = self._fix_paren_balance(ids, final_logits)
        ids = self._fix_ring_closures(ids, final_logits)
        smiles_list = self.tokenizer.batch_decode(ids.cpu().tolist(), skip_special_tokens=True)

        return ids, smiles_list
