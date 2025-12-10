"""Discrete masking diffusion model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


class DiscreteMaskingDiffusion(nn.Module):
    """Discrete masking diffusion process.

    Forward process: gradually mask tokens
    Reverse process: predict original tokens from masked sequence
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_steps: int = 100,
        beta_min: float = 0.05,
        beta_max: float = 0.95,
        mask_token_id: int = 1,
        pad_token_id: int = 0
    ):
        """Initialize diffusion model.

        Args:
            backbone: Transformer backbone model.
            num_steps: Number of diffusion steps T.
            beta_min: Minimum mask probability.
            beta_max: Maximum mask probability.
            mask_token_id: ID of the MASK token.
            pad_token_id: ID of the PAD token.
        """
        super().__init__()

        self.backbone = backbone
        self.num_steps = num_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

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
        # beta_t increases linearly from beta_min to beta_max
        # t=0 means no masking, t=T means full masking
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
        input_ids: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward noising (masking) process.

        Args:
            input_ids: Clean token IDs of shape [batch, seq_len].
            t: Timesteps of shape [batch].
            attention_mask: Attention mask of shape [batch, seq_len].

        Returns:
            Tuple of (noisy_ids, mask_indicator) where mask_indicator shows which positions were masked.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get mask probabilities for each sample in batch
        mask_probs = self.get_mask_prob(t)  # [batch]

        # Sample which tokens to mask
        random_probs = torch.rand(batch_size, seq_len, device=device)
        mask_probs_expanded = mask_probs.unsqueeze(1).expand(-1, seq_len)
        should_mask = random_probs < mask_probs_expanded

        # Don't mask padding tokens
        if attention_mask is not None:
            should_mask = should_mask & (attention_mask == 1)
        else:
            should_mask = should_mask & (input_ids != self.pad_token_id)

        # Apply masking
        noisy_ids = input_ids.clone()
        noisy_ids[should_mask] = self.mask_token_id

        return noisy_ids, should_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            input_ids: Clean token IDs of shape [batch, seq_len].
            attention_mask: Attention mask of shape [batch, seq_len].
            timesteps: Optional fixed timesteps (random if None).

        Returns:
            Dictionary with 'loss', 'logits', and other info.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Sample random timesteps if not provided
        if timesteps is None:
            timesteps = torch.randint(1, self.num_steps + 1, (batch_size,), device=device)

        # Apply forward process
        noisy_ids, mask_indicator = self.forward_process(input_ids, timesteps, attention_mask)

        # Get model predictions
        logits = self.backbone(noisy_ids, timesteps, attention_mask)

        # Compute loss only on masked positions
        loss = self._compute_loss(logits, input_ids, mask_indicator, attention_mask)

        return {
            'loss': loss,
            'logits': logits,
            'noisy_ids': noisy_ids,
            'mask_indicator': mask_indicator,
            'timesteps': timesteps
        }

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask_indicator: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: Predicted logits of shape [batch, seq_len, vocab_size].
            targets: Target token IDs of shape [batch, seq_len].
            mask_indicator: Boolean tensor indicating masked positions.
            attention_mask: Attention mask.

        Returns:
            Scalar loss value.
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten for cross-entropy
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        # Compute loss for all positions
        loss_all = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        loss_all = loss_all.view(batch_size, seq_len)

        # Create mask for valid positions (masked and not padding)
        valid_mask = mask_indicator.float()
        if attention_mask is not None:
            valid_mask = valid_mask * attention_mask.float()

        # Compute mean loss over valid positions
        loss = (loss_all * valid_mask).sum() / valid_mask.sum().clamp(min=1e-9)

        return loss

    def sample_step(
        self,
        noisy_ids: torch.Tensor,
        t: int,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Perform one reverse diffusion step.

        Args:
            noisy_ids: Current noisy token IDs.
            t: Current timestep.
            attention_mask: Attention mask.
            temperature: Sampling temperature.

        Returns:
            Less noisy token IDs.
        """
        batch_size = noisy_ids.shape[0]
        device = noisy_ids.device

        # Create timestep tensor
        timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Get model predictions
        with torch.no_grad():
            logits = self.backbone(noisy_ids, timesteps, attention_mask)

        # Apply temperature
        logits = logits / temperature

        # Sample new tokens
        probs = F.softmax(logits, dim=-1)
        sampled_ids = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(batch_size, -1)

        # Only update masked positions
        is_masked = noisy_ids == self.mask_token_id
        output_ids = torch.where(is_masked, sampled_ids, noisy_ids)

        return output_ids

    def get_loss_at_timestep(
        self,
        input_ids: torch.Tensor,
        timestep: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss at a specific timestep (for evaluation).

        Args:
            input_ids: Clean token IDs.
            timestep: Specific timestep.
            attention_mask: Attention mask.

        Returns:
            Loss value.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        timesteps = torch.full((batch_size,), timestep, device=device, dtype=torch.long)

        result = self.forward(input_ids, attention_mask, timesteps)
        return result['loss']
