"""Constrained sampler for SELFIES-based polymer generation."""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np
from tqdm import tqdm


class ConstrainedSampler:
    """Sampler for SELFIES polymer generation with optional constraints.

    When use_constraints is True, enforces exactly two '[I+3]' placeholder tokens.
    Special tokens are always forbidden at MASK positions.
    """

    def __init__(
        self,
        diffusion_model,
        tokenizer,
        num_steps: int = 100,
        temperature: float = 1.0,
        use_constraints: bool = True,
        device: str = 'cuda'
    ):
        """Initialize sampler.

        Args:
            diffusion_model: Trained discrete masking diffusion model.
            tokenizer: SelfiesTokenizer instance.
            num_steps: Number of diffusion steps.
            temperature: Sampling temperature.
            use_constraints: Whether to apply chemistry constraints during sampling.
            device: Device for computation.
        """
        self.diffusion_model = diffusion_model
        self.tokenizer = tokenizer
        self.num_steps = num_steps
        self.temperature = temperature
        self.use_constraints = use_constraints
        self.device = device

        # Get special token IDs
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.placeholder_id = tokenizer.get_placeholder_token_id()

        # No SMILES-specific token categories needed for SELFIES!
        # SELFIES grammar is valid by construction, so we only need
        # to track the placeholder token '[I+3]' for polymer attachment points.

    def _count_placeholders(self, ids: torch.Tensor) -> torch.Tensor:
        """Count '[I+3]' placeholder tokens in each sequence.

        Args:
            ids: Token IDs of shape [batch, seq_len].

        Returns:
            Counts of shape [batch].
        """
        return (ids == self.placeholder_id).sum(dim=1)

    def _apply_placeholder_constraint(
        self,
        logits: torch.Tensor,
        current_ids: torch.Tensor,
        max_placeholders: int = 2
    ) -> torch.Tensor:
        """Apply constraint to limit number of '[I+3]' placeholder tokens.

        Vectorized implementation for better GPU utilization.

        Args:
            logits: Logits of shape [batch, seq_len, vocab_size].
            current_ids: Current token IDs of shape [batch, seq_len].
            max_placeholders: Maximum allowed '[I+3]' tokens (default: 2 for polymer repeat units).

        Returns:
            Modified logits.
        """
        # Count current placeholders (excluding MASK positions)
        non_mask = current_ids != self.mask_id
        current_placeholders = ((current_ids == self.placeholder_id) & non_mask).sum(dim=1)  # [batch]

        # Find sequences that have reached the placeholder limit [batch]
        exceed_limit = current_placeholders >= max_placeholders

        # Find mask positions [batch, seq_len]
        mask_positions = current_ids == self.mask_id

        # Combined mask: sequences that exceed limit AND are mask positions [batch, seq_len]
        should_forbid = exceed_limit.unsqueeze(1) & mask_positions

        # Set placeholder logit to -inf where should_forbid is True (vectorized)
        logits[:, :, self.placeholder_id] = torch.where(
            should_forbid,
            torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype),
            logits[:, :, self.placeholder_id]
        )

        # Urgency-based boosting: boost placeholder probability when running low on masked positions
        batch_size = logits.shape[0]
        for b in range(batch_size):
            valid_mask = current_ids[b] != self.mask_id
            num_placeholders = ((current_ids[b] == self.placeholder_id) & valid_mask).sum().item()
            num_masked = (current_ids[b] == self.mask_id).sum().item()

            needed = 2 - int(num_placeholders)

            # If we still need placeholders and running low on masked positions, boost
            # Urgency threshold: boost when num_masked <= 5 * needed
            if needed > 0 and num_masked > 0 and num_masked <= 5 * needed:
                mask_positions = (current_ids[b] == self.mask_id)
                placeholder_probs = F.softmax(logits[b, :, self.placeholder_id], dim=0)
                placeholder_probs = placeholder_probs * mask_positions.float()

                if placeholder_probs.sum() > 0:
                    k = min(needed, int(mask_positions.sum()))
                    _, top_indices = torch.topk(placeholder_probs, k=k)
                    logits[b, top_indices, self.placeholder_id] = 100.0  # Strong boost

        return logits

    def _fix_placeholder_count(
        self,
        ids: torch.Tensor,
        logits: torch.Tensor,
        target_placeholders: int = 2
    ) -> torch.Tensor:
        """Fix the number of '[I+3]' placeholder tokens in final sequences.

        Args:
            ids: Token IDs of shape [batch, seq_len].
            logits: Final logits of shape [batch, seq_len, vocab_size].
            target_placeholders: Target number of '[I+3]' tokens (default: 2 for polymer repeat units).

        Returns:
            Fixed token IDs.
        """
        batch_size, seq_len = ids.shape
        fixed_ids = ids.clone()

        for i in range(batch_size):
            placeholder_mask = fixed_ids[i] == self.placeholder_id
            num_placeholders = placeholder_mask.sum().item()

            if num_placeholders > target_placeholders:
                # Keep only the top-k most probable placeholder positions
                placeholder_positions = torch.where(placeholder_mask)[0]
                placeholder_probs = logits[i, placeholder_positions, self.placeholder_id]

                # Get indices of placeholders to keep (highest probability)
                _, keep_indices = torch.topk(placeholder_probs, target_placeholders)
                keep_positions = placeholder_positions[keep_indices]

                # Replace extra placeholders with second-best token
                for pos in placeholder_positions:
                    if pos not in keep_positions:
                        # Get second-best token (excluding placeholder)
                        pos_logits = logits[i, pos].clone()
                        pos_logits[self.placeholder_id] = float('-inf')
                        pos_logits[self.mask_id] = float('-inf')
                        pos_logits[self.pad_id] = float('-inf')
                        best_token = pos_logits.argmax()
                        fixed_ids[i, pos] = best_token

            elif num_placeholders < target_placeholders:
                # Find best positions to add placeholders
                needed = target_placeholders - num_placeholders

                # Get placeholder probabilities at all non-special positions
                valid_mask = (
                    (fixed_ids[i] != self.bos_id) &
                    (fixed_ids[i] != self.eos_id) &
                    (fixed_ids[i] != self.pad_id) &
                    (fixed_ids[i] != self.placeholder_id)
                )
                valid_positions = torch.where(valid_mask)[0]

                if len(valid_positions) >= needed:
                    placeholder_probs = logits[i, valid_positions, self.placeholder_id]
                    _, best_indices = torch.topk(placeholder_probs, needed)
                    best_positions = valid_positions[best_indices]

                    for pos in best_positions:
                        fixed_ids[i, pos] = self.placeholder_id

        return fixed_ids

    def _update_eos_termination(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_idx: int,
        eos_pos: int
    ) -> None:
        """Update sequence after EOS is sampled: mark remaining positions as PAD.

        Args:
            ids: Token IDs of shape [batch, seq_len].
            attention_mask: Attention mask of shape [batch, seq_len].
            batch_idx: Index of the sequence in the batch.
            eos_pos: Position where EOS was sampled.
        """
        seq_len = ids.shape[1]
        # Mark all positions after EOS as PAD
        if eos_pos + 1 < seq_len:
            ids[batch_idx, eos_pos + 1:] = self.pad_id
            attention_mask[batch_idx, eos_pos + 1:] = 0

    def sample(
        self,
        batch_size: int,
        seq_length: int,
        show_progress: bool = True,
        allow_natural_eos: bool = False
    ) -> Tuple[torch.Tensor, List[str]]:
        """Sample new polymers with exactly two '[I+3]' placeholder tokens.

        SELFIES grammar is valid by construction, so we only need to enforce
        the placeholder count constraint. No parenthesis, ring, or bond
        constraints are needed.

        Args:
            batch_size: Number of samples to generate.
            seq_length: Sequence length.
            show_progress: Whether to show progress bar.
            allow_natural_eos: If True, allow EOS to be sampled naturally.
                              If False (default), fix EOS at last position.
                              Note: Natural EOS requires model training with
                              variable-length sequences. Use lengths parameter
                              in sample_batch() for best results.

        Returns:
            Tuple of (token_ids, selfies_strings).
        """
        self.diffusion_model.eval()
        backbone = self.diffusion_model.backbone

        # Initialize with fully masked sequence (except BOS)
        ids = torch.full(
            (batch_size, seq_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device
        )
        ids[:, 0] = self.bos_id

        # Track which sequences have terminated (sampled EOS)
        has_eos = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        if not allow_natural_eos:
            # Legacy behavior: fix EOS at last position
            ids[:, -1] = self.eos_id
            has_eos[:] = True

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

            if self.use_constraints:
                # Apply placeholder constraint (max 2 [I+3] tokens)
                # This is the ONLY constraint needed for SELFIES!
                logits = self._apply_placeholder_constraint(logits, ids, max_placeholders=2)

            # Forbid special tokens at masked positions
            is_masked = ids == self.mask_id
            # Always forbid MASK, PAD, BOS at masked positions
            for tok in [self.mask_id, self.pad_id, self.bos_id]:
                logits[:, :, tok] = torch.where(
                    is_masked.unsqueeze(-1).expand_as(logits[:, :, tok:tok+1]).squeeze(-1),
                    torch.tensor(float('-inf'), device=logits.device),
                    logits[:, :, tok]
                )

            # For EOS: forbid if already has EOS, otherwise allow natural termination
            if allow_natural_eos:
                # Forbid EOS only for sequences that already have EOS
                eos_forbidden = has_eos.unsqueeze(1) & is_masked
                logits[:, :, self.eos_id] = torch.where(
                    eos_forbidden,
                    torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype),
                    logits[:, :, self.eos_id]
                )
            else:
                # Legacy: forbid EOS everywhere
                logits[:, :, self.eos_id] = torch.where(
                    is_masked.unsqueeze(-1).expand_as(logits[:, :, self.eos_id:self.eos_id+1]).squeeze(-1),
                    torch.tensor(float('-inf'), device=logits.device),
                    logits[:, :, self.eos_id]
                )

            # Sample from masked positions
            probs = F.softmax(logits, dim=-1)

            # Determine which masked tokens to unmask at this step
            # Unmask proportionally based on schedule
            unmask_prob = 1.0 / t  # Simple linear unmasking

            for i in range(batch_size):
                masked_pos = torch.where(is_masked[i])[0]
                if len(masked_pos) == 0:
                    continue

                # Randomly select positions to unmask
                num_unmask = max(1, int(len(masked_pos) * unmask_prob))
                unmask_indices = torch.randperm(len(masked_pos), device=self.device)[:num_unmask]
                unmask_positions = masked_pos[unmask_indices]

                # Sort positions so we process left-to-right (for proper EOS handling)
                unmask_positions, _ = torch.sort(unmask_positions)

                # Sample tokens for these positions SEQUENTIALLY with constraint updates
                for pos in unmask_positions:
                    # Skip if this position was already filled (by EOS termination)
                    if ids[i, pos] != self.mask_id:
                        continue

                    sampled = torch.multinomial(probs[i, pos], 1)
                    ids[i, pos] = sampled

                    sampled_token = sampled.item()

                    # If we sampled EOS, terminate the sequence
                    if sampled_token == self.eos_id:
                        has_eos[i] = True
                        self._update_eos_termination(ids, attention_mask, i, pos.item())
                        # No need to process remaining positions for this sequence
                        break

                    if self.use_constraints:
                        # If we sampled a placeholder, update constraint for remaining positions
                        if sampled_token == self.placeholder_id:
                            # Count current placeholders (excluding remaining MASK positions)
                            non_mask = ids[i] != self.mask_id
                            current_placeholders = ((ids[i] == self.placeholder_id) & non_mask).sum().item()

                            # If we've reached the limit, forbid placeholders at remaining masked positions
                            if current_placeholders >= 2:
                                remaining_mask = ids[i] == self.mask_id
                                logits[i, remaining_mask, self.placeholder_id] = float('-inf')
                                probs[i] = F.softmax(logits[i], dim=-1)

            # Store logits for final step
            if t == 1:
                final_logits = logits

        # Ensure all sequences have EOS (add at end if needed)
        for i in range(batch_size):
            if not has_eos[i]:
                # Find last non-PAD position
                non_pad = ids[i] != self.pad_id
                if non_pad.any():
                    last_pos = torch.where(non_pad)[0][-1].item()
                    # If last position is MASK, replace with EOS; otherwise append EOS
                    if ids[i, last_pos] == self.mask_id:
                        ids[i, last_pos] = self.eos_id
                    elif last_pos + 1 < seq_length:
                        ids[i, last_pos + 1] = self.eos_id
                        if last_pos + 2 < seq_length:
                            ids[i, last_pos + 2:] = self.pad_id
                            attention_mask[i, last_pos + 2:] = 0
                has_eos[i] = True

        if self.use_constraints:
            # Fix placeholder count in final sequences (ensure exactly 2)
            ids = self._fix_placeholder_count(ids, final_logits, target_placeholders=2)

        # Decode to SELFIES
        selfies_list = self.tokenizer.batch_decode(ids.cpu().tolist(), skip_special_tokens=True)

        return ids, selfies_list

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
            seq_length: Default sequence length (used if lengths is None).
            batch_size: Batch size for sampling.
            show_progress: Whether to show progress.
            lengths: Optional list of sequence lengths for each sample.
                    If provided, samples are grouped by length for efficient batching.

        Returns:
            Tuple of (all_ids, all_selfies).
        """
        all_ids = []
        all_smiles = []

        if lengths is None:
            # Use fixed length for all samples
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
        else:
            # Group samples by length for efficient batching
            from collections import defaultdict
            length_groups = defaultdict(list)
            for idx, length in enumerate(lengths):
                length_groups[length].append(idx)

            # Create result arrays
            result_smiles = [None] * len(lengths)

            # Process each length group
            sorted_lengths = sorted(length_groups.keys())
            pbar = tqdm(total=len(lengths), desc="Length-distributed sampling", disable=not show_progress)

            for length in sorted_lengths:
                indices = length_groups[length]
                group_size = len(indices)

                # Sample in batches for this length
                group_smiles = []
                num_group_batches = (group_size + batch_size - 1) // batch_size

                for batch_idx in range(num_group_batches):
                    current_batch_size = min(batch_size, group_size - len(group_smiles))

                    ids, smiles = self.sample(
                        current_batch_size,
                        length,
                        show_progress=False
                    )

                    all_ids.append(ids)
                    group_smiles.extend(smiles)
                    pbar.update(current_batch_size)

                # Assign results back to original indices
                for local_idx, global_idx in enumerate(indices):
                    result_smiles[global_idx] = group_smiles[local_idx]

            pbar.close()
            all_smiles = result_smiles

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
        from the specified range, producing diverse SELFIES lengths.

        Args:
            num_samples: Total number of samples.
            length_range: (min_length, max_length) for sequence lengths.
            batch_size: Maximum batch size for GPU memory.
            samples_per_length: Number of samples per length (controls diversity).
                               Smaller values = more length diversity.
            show_progress: Whether to show progress.

        Returns:
            Tuple of (all_ids, all_selfies).
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
        show_progress: bool = True,
        allow_natural_eos: bool = False
    ) -> Tuple[torch.Tensor, List[str]]:
        """Sample with optional prefix/suffix conditioning.

        SELFIES grammar is valid by construction, so we only need to enforce
        the placeholder count constraint.

        Args:
            batch_size: Number of samples.
            seq_length: Sequence length.
            prefix_ids: Fixed prefix tokens.
            suffix_ids: Fixed suffix tokens.
            show_progress: Whether to show progress.
            allow_natural_eos: If True, allow EOS to be sampled naturally.
                              If False (default), fix EOS at last position.
                              Ignored if suffix_ids is provided (EOS fixed at end).

        Returns:
            Tuple of (token_ids, selfies_strings).
        """
        self.diffusion_model.eval()
        backbone = self.diffusion_model.backbone

        # If suffix is provided, we need fixed EOS at end
        use_natural_eos = allow_natural_eos and suffix_ids is None

        # Initialize
        ids = torch.full(
            (batch_size, seq_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device
        )
        ids[:, 0] = self.bos_id

        # Track which sequences have terminated (sampled EOS)
        has_eos = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Apply prefix/suffix constraints
        fixed_mask = torch.zeros_like(ids, dtype=torch.bool)
        fixed_mask[:, 0] = True  # BOS

        if not use_natural_eos:
            # Fixed EOS at last position
            ids[:, -1] = self.eos_id
            fixed_mask[:, -1] = True
            has_eos[:] = True

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

            if self.use_constraints:
                # Apply placeholder constraint (max 2 [I+3] tokens)
                # This is the ONLY constraint needed for SELFIES!
                logits = self._apply_placeholder_constraint(logits, ids, max_placeholders=2)

            # Forbid special tokens at masked positions
            is_masked = (ids == self.mask_id) & (~fixed_mask)

            # Always forbid MASK, PAD, BOS at masked positions
            for tok in [self.mask_id, self.pad_id, self.bos_id]:
                logits[:, :, tok] = torch.where(
                    is_masked.unsqueeze(-1).expand_as(logits[:, :, tok:tok+1]).squeeze(-1),
                    torch.tensor(float('-inf'), device=logits.device),
                    logits[:, :, tok]
                )

            # For EOS: allow natural termination if enabled
            if use_natural_eos:
                # Forbid EOS only for sequences that already have EOS
                eos_forbidden = has_eos.unsqueeze(1) & is_masked
                logits[:, :, self.eos_id] = torch.where(
                    eos_forbidden,
                    torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype),
                    logits[:, :, self.eos_id]
                )
            else:
                # Fixed EOS: forbid everywhere
                logits[:, :, self.eos_id] = torch.where(
                    is_masked.unsqueeze(-1).expand_as(logits[:, :, self.eos_id:self.eos_id+1]).squeeze(-1),
                    torch.tensor(float('-inf'), device=logits.device),
                    logits[:, :, self.eos_id]
                )

            probs = F.softmax(logits, dim=-1)

            unmask_prob = 1.0 / t

            for i in range(batch_size):
                masked_pos = torch.where(is_masked[i])[0]
                if len(masked_pos) == 0:
                    continue

                num_unmask = max(1, int(len(masked_pos) * unmask_prob))
                unmask_indices = torch.randperm(len(masked_pos), device=self.device)[:num_unmask]
                unmask_positions = masked_pos[unmask_indices]

                # Sort positions so we process left-to-right (for proper EOS handling)
                unmask_positions, _ = torch.sort(unmask_positions)

                # Sample tokens for these positions SEQUENTIALLY with constraint updates
                for pos in unmask_positions:
                    # Skip if this position was already filled (by EOS termination)
                    if ids[i, pos] != self.mask_id:
                        continue

                    sampled = torch.multinomial(probs[i, pos], 1)
                    ids[i, pos] = sampled

                    sampled_token = sampled.item()

                    # If we sampled EOS, terminate the sequence
                    if sampled_token == self.eos_id:
                        has_eos[i] = True
                        self._update_eos_termination(ids, attention_mask, i, pos.item())
                        break

                    if self.use_constraints:
                        # If we sampled a placeholder, update constraint for remaining positions
                        if sampled_token == self.placeholder_id:
                            # Count current placeholders (excluding remaining MASK positions)
                            non_mask = ids[i] != self.mask_id
                            current_placeholders = ((ids[i] == self.placeholder_id) & non_mask).sum().item()

                            # If we've reached the limit, forbid placeholders at remaining masked positions
                            if current_placeholders >= 2:
                                remaining_mask = ids[i] == self.mask_id
                                logits[i, remaining_mask, self.placeholder_id] = float('-inf')
                                probs[i] = F.softmax(logits[i], dim=-1)

            if t == 1:
                final_logits = logits

        # Ensure all sequences have EOS (add at end if needed)
        if use_natural_eos:
            for i in range(batch_size):
                if not has_eos[i]:
                    non_pad = ids[i] != self.pad_id
                    if non_pad.any():
                        last_pos = torch.where(non_pad)[0][-1].item()
                        if ids[i, last_pos] == self.mask_id:
                            ids[i, last_pos] = self.eos_id
                        elif last_pos + 1 < seq_length:
                            ids[i, last_pos + 1] = self.eos_id
                            if last_pos + 2 < seq_length:
                                ids[i, last_pos + 2:] = self.pad_id
                                attention_mask[i, last_pos + 2:] = 0
                    has_eos[i] = True

        if self.use_constraints:
            # Fix placeholder count in final sequences (ensure exactly 2)
            ids = self._fix_placeholder_count(ids, final_logits, target_placeholders=2)

        # Decode to SELFIES
        selfies_list = self.tokenizer.batch_decode(ids.cpu().tolist(), skip_special_tokens=True)

        return ids, selfies_list
