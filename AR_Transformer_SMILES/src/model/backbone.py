"""GPT-2-small-like Transformer backbone for autoregressive modeling."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional causal masking."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        causal: bool = False
    ):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.causal = causal

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size].
            attention_mask: Mask of shape [batch, seq_len].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_size].
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask to prevent attention to future tokens
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=attn_scores.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        # Apply attention mask (mask padding only)
        if attention_mask is not None:
            # Expand mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return self.out_proj(out)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        dropout: float = 0.1,
        causal: bool = False
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout, causal=causal)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, ffn_hidden_size, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), attention_mask)
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class DiffusionBackbone(nn.Module):
    """GPT-2-small-like Transformer backbone for autoregressive modeling.

    Architecture:
    - Learned token embeddings
    - Learned positional embeddings
    - Stack of Transformer blocks with causal attention
    - Final layer norm and output projection
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_hidden_size: int = 3072,
        max_position_embeddings: int = 256,
        num_diffusion_steps: int = 100,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        causal: bool = True
    ):
        """Initialize backbone.

        Args:
            vocab_size: Size of vocabulary.
            hidden_size: Hidden dimension.
            num_layers: Number of Transformer layers.
            num_heads: Number of attention heads.
            ffn_hidden_size: FFN intermediate dimension.
            max_position_embeddings: Maximum sequence length.
            num_diffusion_steps: Unused (kept for backward compatibility).
            dropout: Dropout rate.
            pad_token_id: ID of padding token.
            causal: Whether to apply causal masking.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.num_diffusion_steps = num_diffusion_steps
        self.causal = causal

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)

        self.embedding_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ffn_hidden_size, dropout, causal=causal)
            for _ in range(num_layers)
        ])

        # Output
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape [batch, seq_len].
            attention_mask: Attention mask of shape [batch, seq_len].

        Returns:
            Logits of shape [batch, seq_len, vocab_size].
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_emb = self.token_embedding(input_ids)

        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)

        # Combine embeddings
        x = token_emb + pos_emb
        x = self.embedding_dropout(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Output projection
        x = self.final_norm(x)
        logits = self.output_proj(x)

        return logits

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """Get hidden states from a specific layer.

        Args:
            input_ids: Token IDs of shape [batch, seq_len].
            attention_mask: Attention mask.
            layer_idx: Index of layer to get hidden states from (-1 for final).

        Returns:
            Hidden states of shape [batch, seq_len, hidden_size].
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb
        x = self.embedding_dropout(x)

        # Go through layers up to layer_idx
        num_layers = len(self.layers) if layer_idx == -1 else layer_idx + 1
        for i in range(num_layers):
            x = self.layers[i](x, attention_mask)

        if layer_idx == -1:
            x = self.final_norm(x)

        return x

    def get_pooled_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = 'mean'
    ) -> torch.Tensor:
        """Get pooled representation for the sequence.

        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.
            pooling: Pooling method ('mean', 'cls', 'max').

        Returns:
            Pooled representation of shape [batch, hidden_size].
        """
        hidden_states = self.get_hidden_states(input_ids, attention_mask)

        if pooling == 'mean':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                hidden_states = hidden_states * mask
                return hidden_states.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return hidden_states.mean(dim=1)
        elif pooling == 'cls':
            return hidden_states[:, 0]
        elif pooling == 'max':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                hidden_states = hidden_states * mask + (1 - mask) * (-1e9)
            return hidden_states.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

    def save_pretrained(self, path: str):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'pad_token_id': self.pad_token_id,
                'num_diffusion_steps': self.num_diffusion_steps,
                'causal': self.causal
            }
        }, path)

    @classmethod
    def load_pretrained(cls, path: str, device: str = 'cpu', **kwargs) -> 'DiffusionBackbone':
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint.
            device: Device to load model to.
            **kwargs: Additional arguments to override config.

        Returns:
            Loaded model.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint['config']
        config.update(kwargs)

        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'])

        return model
