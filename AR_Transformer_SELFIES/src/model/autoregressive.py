"""Autoregressive language model wrapper for causal Transformer backbone."""

from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoregressiveLM(nn.Module):
    """Wraps a causal backbone with next-token prediction loss."""

    def __init__(self, backbone: nn.Module, pad_token_id: int = 0):
        super().__init__()
        self.backbone = backbone
        self.pad_token_id = pad_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with next-token prediction loss."""
        logits = self.backbone(input_ids, attention_mask)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        if attention_mask is not None:
            shift_mask = attention_mask[:, 1:].contiguous()
            shift_labels = shift_labels.masked_fill(shift_mask == 0, self.pad_token_id)

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.pad_token_id
        )

        return {'loss': loss, 'logits': logits}
