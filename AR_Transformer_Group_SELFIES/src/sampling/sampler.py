"""Constrained sampler for Group SELFIES polymer generation."""

import re
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import numpy as np
from tqdm import tqdm

from ..utils.chemistry import count_stars, check_validity


class ConstrainedSampler:
    """Sampler for Group SELFIES polymer generation with optional constraints.

    When use_constraints is True, enforces exactly two placeholder tokens.
    Special tokens are always forbidden at MASK positions.
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
            tokenizer: GroupSELFIESTokenizer instance.
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
        self.unk_id = tokenizer.unk_token_id

        # Get placeholder token ID (represents '*' in Group SELFIES)
        # This is the token that becomes [I+3] placeholder
        self.placeholder_id = tokenizer.get_placeholder_token_id()
        if self.placeholder_id is None:
            # Fallback: try to find placeholder token in vocab
            self._find_placeholder_id()
        if self.placeholder_id is None:
            raise ValueError("Placeholder token not found in tokenizer vocabulary; rebuild tokenizer with the placeholder settings.")

        # For backward compatibility with code that uses star_id
        self.star_id = self.placeholder_id

        # Build set of special tokens to forbid during sampling
        self.special_token_ids = {
            self.mask_id, self.pad_id, self.bos_id, self.eos_id, self.unk_id
        } - {None}

        # Build mapping of ALL placeholder-bearing tokens to their contribution count
        # This includes both direct placeholder token and group tokens containing [I+3]
        self.placeholder_contributions = self._build_placeholder_token_map()

    def _find_placeholder_id(self):
        """Try to find placeholder token ID in vocabulary."""
        # Look for tokens containing 'I' and '+3' (placeholder pattern)
        for token, token_id in self.tokenizer.vocab.items():
            if 'I' in token and '+3' in token:
                self.placeholder_id = token_id
                return

        self.placeholder_id = None

    def _build_placeholder_token_map(self) -> Dict[int, int]:
        """Build mapping of token ID -> number of placeholders it contributes.

        This finds ALL tokens that produce placeholder atoms when decoded:
        - Direct placeholder token (e.g., [IH0+3])
        - Group tokens that contain [I+3] in their SMILES representation

        Returns:
            Dict mapping token_id -> placeholder count for that token.
        """
        placeholder_contributions = {}

        # Direct placeholder token contributes 1
        if self.placeholder_id is not None:
            placeholder_contributions[self.placeholder_id] = 1

        # Check for group tokens containing [I+3]
        # Group tokens have format like [:/0G100] and decode to group SMILES
        if hasattr(self.tokenizer, 'group_smiles') and self.tokenizer.group_smiles:
            for token, idx in self.tokenizer.vocab.items():
                # Match group reference tokens like [:/0G100]
                match = re.match(r'\[:/(\d+)G(\d+)\]', token)
                if match:
                    group_idx = int(match.group(2))
                    if group_idx < len(self.tokenizer.group_smiles):
                        group_smiles = self.tokenizer.group_smiles[group_idx]
                        # Count [I+3] occurrences in this group's SMILES
                        count = group_smiles.count('[I+3]')
                        if count > 0:
                            placeholder_contributions[idx] = count

        return placeholder_contributions

    def _count_placeholders(self, ids: torch.Tensor) -> torch.Tensor:
        """Count total placeholder atoms in each sequence.

        Uses placeholder_contributions to sum contributions from all
        placeholder-bearing tokens (direct placeholder + group tokens).

        Args:
            ids: Token IDs of shape [batch, seq_len].

        Returns:
            Counts of shape [batch].
        """
        if not self.placeholder_contributions:
            return torch.zeros(ids.shape[0], device=ids.device, dtype=torch.long)

        # Sum contributions from all placeholder-bearing tokens
        counts = torch.zeros(ids.shape[0], device=ids.device, dtype=torch.long)
        for token_id, contribution in self.placeholder_contributions.items():
            counts += (ids == token_id).sum(dim=1) * contribution

        return counts

    def _apply_placeholder_constraint(
        self,
        logits: torch.Tensor,
        current_ids: torch.Tensor,
        max_placeholders: int = 2
    ) -> torch.Tensor:
        """Apply constraint to limit TOTAL number of placeholder atoms.

        Enforces a sum constraint across ALL placeholder-bearing tokens:
        - Direct placeholder token (e.g., [IH0+3])
        - Group tokens that contain [I+3] in their decoded SMILES

        Args:
            logits: Logits of shape [batch, seq_len, vocab_size].
            current_ids: Current token IDs of shape [batch, seq_len].
            max_placeholders: Maximum allowed placeholder atoms.

        Returns:
            Modified logits.
        """
        if not self.placeholder_contributions:
            return logits

        # Count current total placeholders (sum of contributions, excluding MASK positions)
        non_mask = current_ids != self.mask_id
        current_total = torch.zeros(current_ids.shape[0], device=current_ids.device, dtype=torch.long)
        for token_id, contribution in self.placeholder_contributions.items():
            current_total += ((current_ids == token_id) & non_mask).sum(dim=1) * contribution

        # Find sequences that have reached the placeholder limit [batch]
        exceed_limit = current_total >= max_placeholders

        # Find mask positions [batch, seq_len]
        mask_positions = current_ids == self.mask_id

        # Combined mask: sequences that exceed limit AND are mask positions [batch, seq_len]
        should_forbid = exceed_limit.unsqueeze(1) & mask_positions

        # Forbid ALL placeholder-bearing tokens at masked positions (vectorized)
        neg_inf = torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype)
        for token_id in self.placeholder_contributions:
            logits[:, :, token_id] = torch.where(
                should_forbid,
                neg_inf,
                logits[:, :, token_id]
            )

        return logits

    def _apply_special_token_constraints(
        self,
        logits: torch.Tensor,
        current_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forbid special tokens from being sampled at MASK positions.

        Vectorized implementation for better GPU utilization.

        Args:
            logits: Logits of shape [batch, seq_len, vocab_size].
            current_ids: Current token IDs of shape [batch, seq_len].

        Returns:
            Modified logits.
        """
        # Find mask positions [batch, seq_len]
        mask_positions = current_ids == self.mask_id

        # Forbid all special tokens at masked positions (vectorized)
        for token_id in self.special_token_ids:
            if token_id is not None and token_id >= 0:
                logits[:, :, token_id] = torch.where(
                    mask_positions,
                    torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype),
                    logits[:, :, token_id]
                )

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

    def _sample_from_ids(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        show_progress: bool = True,
        allow_natural_eos: bool = False
    ) -> Tuple[torch.Tensor, List[str]]:
        """Run left-to-right autoregressive sampling starting from provided token IDs."""
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
                full_logits = self._apply_placeholder_constraint(full_logits, full_ids, max_placeholders=2)
            full_logits = self._apply_special_token_constraints(full_logits, full_ids)

            logits = full_logits[:, pos, :]

            if self.eos_id >= 0:
                if allow_natural_eos:
                    placeholder_count = self._count_placeholders(prefix_ids)
                    allow_eos = placeholder_count >= 2
                    eos_logits = raw_logits[:, self.eos_id]
                    logits[:, self.eos_id] = torch.where(
                        allow_eos,
                        eos_logits,
                        torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype)
                    )
                else:
                    logits[:, self.eos_id] = float('-inf')

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

        if allow_natural_eos and self.eos_id >= 0:
            for i in range(batch_size):
                if (ids[i] == self.eos_id).any():
                    continue
                non_pad = ids[i] != self.pad_id
                if non_pad.any():
                    last_pos = torch.where(non_pad)[0][-1].item()
                else:
                    last_pos = max_length - 1
                ids[i, last_pos] = self.eos_id
                if last_pos + 1 < max_length:
                    ids[i, last_pos + 1:] = self.pad_id

        if self.use_constraints and final_logits is not None:
            ids = self._fix_placeholder_count(ids, final_logits, target_placeholders=2)

        selfies_list = self.tokenizer.batch_decode(ids.cpu().tolist(), skip_special_tokens=True)
        return ids, selfies_list

    def _fix_placeholder_count(
        self,
        ids: torch.Tensor,
        logits: torch.Tensor,
        target_placeholders: int = 2
    ) -> torch.Tensor:
        """Fix the number of placeholder tokens in final sequences.

        Args:
            ids: Token IDs of shape [batch, seq_len].
            logits: Final logits of shape [batch, seq_len, vocab_size].
            target_placeholders: Target number of placeholder tokens.

        Returns:
            Fixed token IDs.
        """
        if self.placeholder_id is None:
            return ids

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
                        # Get second-best token (excluding placeholder and special tokens)
                        pos_logits = logits[i, pos].clone()
                        pos_logits[self.placeholder_id] = float('-inf')
                        for tok_id in self.special_token_ids:
                            if tok_id is not None and tok_id >= 0:
                                pos_logits[tok_id] = float('-inf')
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

    def sample(
        self,
        batch_size: int,
        seq_length: int,
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, List[str]]:
        """Sample new polymers with exactly two placeholder tokens.

        Args:
            batch_size: Number of samples to generate.
            seq_length: Sequence length.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (token_ids, smiles_strings).
        """
        ids = torch.full(
            (batch_size, seq_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device
        )
        ids[:, 0] = self.bos_id
        ids[:, -1] = self.eos_id

        attention_mask = torch.ones_like(ids)
        fixed_mask = torch.zeros_like(ids, dtype=torch.bool)
        fixed_mask[:, 0] = True
        fixed_mask[:, -1] = True

        return self._sample_from_ids(
            ids=ids,
            attention_mask=attention_mask,
            fixed_mask=fixed_mask,
            show_progress=show_progress,
            allow_natural_eos=False
        )

        # Legacy diffusion sampler retained for reference.
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

            if self.use_constraints:
                # Apply placeholder constraint (limit to 2)
                logits = self._apply_placeholder_constraint(logits, ids, max_placeholders=2)

            # Apply special token constraints
            logits = self._apply_special_token_constraints(logits, ids)

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
                for pos in unmask_positions:
                    sampled = torch.multinomial(probs[i, pos], 1)
                    ids[i, pos] = sampled

                    # Update constraints dynamically based on what was just sampled
                    sampled_token = sampled.item()

                    if self.use_constraints:
                        # If we sampled ANY placeholder-bearing token, update constraint
                        if sampled_token in self.placeholder_contributions:
                            # Count current total placeholders using contribution map
                            non_mask = ids[i] != self.mask_id
                            current_total = 0
                            for token_id, contribution in self.placeholder_contributions.items():
                                current_total += ((ids[i] == token_id) & non_mask).sum().item() * contribution

                            # If we've reached the limit, forbid ALL placeholder tokens at remaining positions
                            if current_total >= 2:
                                remaining_mask = ids[i] == self.mask_id
                                for token_id in self.placeholder_contributions:
                                    logits[i, remaining_mask, token_id] = float('-inf')
                                probs[i] = F.softmax(logits[i], dim=-1)

            # Store logits for final step
            if t == 1:
                final_logits = logits

        if self.use_constraints:
            # Fix placeholder count in final sequences
            ids = self._fix_placeholder_count(ids, final_logits, target_placeholders=2)

        # Decode to SMILES (Group SELFIES tokenizer decodes to p-SMILES)
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

    def sample_with_rejection(
        self,
        target_samples: int,
        seq_length: int,
        batch_size: int = 256,
        max_attempts: int = 10,
        target_stars: int = 2,
        require_validity: bool = True,
        show_progress: bool = True
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Sample with rejection to guarantee exactly target_stars.

        Uses oversampling with rejection to ensure all returned samples
        have exactly the target number of star tokens and are RDKit-valid.

        Args:
            target_samples: Exact number of valid samples to return.
            seq_length: Sequence length for generation.
            batch_size: Batch size for sampling.
            max_attempts: Maximum number of oversampling rounds.
            target_stars: Required number of star tokens (default: 2).
            require_validity: Also require RDKit validity (default: True).
            show_progress: Whether to show progress.

        Returns:
            Tuple of (valid_ids, valid_smiles) with exactly target_samples entries.
        """
        valid_ids = []
        valid_smiles = []

        # Start with initial estimate of acceptance rate
        initial_batch = min(target_samples * 2, batch_size * 4)
        acceptance_rate = 0.5  # Will be updated based on actual results

        attempt = 0
        pbar = tqdm(total=target_samples, desc="Rejection sampling", disable=not show_progress)

        while len(valid_smiles) < target_samples and attempt < max_attempts:
            # Calculate how many samples to generate based on current acceptance rate
            remaining = target_samples - len(valid_smiles)
            oversample_factor = max(1.5, 1.0 / max(acceptance_rate, 0.1))
            num_to_generate = min(int(remaining * oversample_factor), batch_size * 10)

            # Generate samples
            _, smiles_list = self.sample_batch(
                num_to_generate, seq_length, batch_size, show_progress=False
            )

            # Filter valid samples
            new_valid = 0
            for smiles in smiles_list:
                if len(valid_smiles) >= target_samples:
                    break

                stars = count_stars(smiles)
                is_valid = not require_validity or check_validity(smiles)

                if stars == target_stars and is_valid:
                    valid_smiles.append(smiles)
                    new_valid += 1

            # Update acceptance rate estimate
            if len(smiles_list) > 0:
                batch_rate = new_valid / len(smiles_list)
                # Exponential moving average
                acceptance_rate = 0.7 * acceptance_rate + 0.3 * batch_rate

            pbar.update(new_valid)
            attempt += 1

        pbar.close()

        if len(valid_smiles) < target_samples:
            print(f"Warning: Only found {len(valid_smiles)}/{target_samples} valid samples "
                  f"after {max_attempts} attempts (acceptance rate: {acceptance_rate:.2%})")

        return [], valid_smiles[:target_samples]

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
        ids = torch.full(
            (batch_size, seq_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device
        )
        ids[:, 0] = self.bos_id
        ids[:, -1] = self.eos_id

        fixed_mask = torch.zeros_like(ids, dtype=torch.bool)
        fixed_mask[:, 0] = True
        fixed_mask[:, -1] = True

        if prefix_ids is not None:
            prefix_len = prefix_ids.shape[1]
            ids[:, 1:1+prefix_len] = prefix_ids
            fixed_mask[:, 1:1+prefix_len] = True

        if suffix_ids is not None:
            suffix_len = suffix_ids.shape[1]
            ids[:, -1-suffix_len:-1] = suffix_ids
            fixed_mask[:, -1-suffix_len:-1] = True

        attention_mask = torch.ones_like(ids)
        return self._sample_from_ids(
            ids=ids,
            attention_mask=attention_mask,
            fixed_mask=fixed_mask,
            show_progress=show_progress,
            allow_natural_eos=False
        )

        # Legacy diffusion sampler retained for reference.
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
            if self.use_constraints:
                logits = self._apply_placeholder_constraint(logits, ids, max_placeholders=2)
            logits = self._apply_special_token_constraints(logits, ids)

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

                for pos in unmask_positions:
                    sampled = torch.multinomial(probs[i, pos], 1)
                    ids[i, pos] = sampled

                    sampled_token = sampled.item()

                    if self.use_constraints:
                        if sampled_token == self.placeholder_id:
                            non_mask = ids[i] != self.mask_id
                            current_placeholders = ((ids[i] == self.placeholder_id) & non_mask).sum().item()

                            if current_placeholders >= 2:
                                remaining_mask = ids[i] == self.mask_id
                                logits[i, remaining_mask, self.placeholder_id] = float('-inf')
                                probs[i] = F.softmax(logits[i], dim=-1)

            if t == 1:
                final_logits = logits

        if self.use_constraints:
            ids = self._fix_placeholder_count(ids, final_logits, target_placeholders=2)
        smiles_list = self.tokenizer.batch_decode(ids.cpu().tolist(), skip_special_tokens=True)

        return ids, smiles_list

    # Backward compatibility aliases
    def _count_stars(self, ids: torch.Tensor) -> torch.Tensor:
        """Alias for _count_placeholders for backward compatibility."""
        return self._count_placeholders(ids)

    def _apply_star_constraint(
        self,
        logits: torch.Tensor,
        current_ids: torch.Tensor,
        max_stars: int = 2
    ) -> torch.Tensor:
        """Alias for _apply_placeholder_constraint for backward compatibility."""
        return self._apply_placeholder_constraint(logits, current_ids, max_stars)

    def _fix_star_count(
        self,
        ids: torch.Tensor,
        logits: torch.Tensor,
        target_stars: int = 2
    ) -> torch.Tensor:
        """Alias for _fix_placeholder_count for backward compatibility."""
        return self._fix_placeholder_count(ids, logits, target_stars)
