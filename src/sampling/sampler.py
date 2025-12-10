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

    def _count_stars(self, ids: torch.Tensor) -> torch.Tensor:
        """Count '*' tokens in each sequence.

        Args:
            ids: Token IDs of shape [batch, seq_len].

        Returns:
            Counts of shape [batch].
        """
        return (ids == self.star_id).sum(dim=1)

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

                # Sample tokens for these positions
                for pos in unmask_positions:
                    sampled = torch.multinomial(probs[i, pos], 1)
                    ids[i, pos] = sampled

            # Store logits for final step
            if t == 1:
                final_logits = logits

        # Fix star count in final sequences
        ids = self._fix_star_count(ids, final_logits, target_stars=2)

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

            if t == 1:
                final_logits = logits

        ids = self._fix_star_count(ids, final_logits, target_stars=2)
        smiles_list = self.tokenizer.batch_decode(ids.cpu().tolist(), skip_special_tokens=True)

        return ids, smiles_list
