"""Graph Transformer Backbone for Discrete Diffusion.

Implements GraphGPS-style architecture with:
- Edge-aware multi-head attention
- Dual prediction heads (node and edge)
- Timestep conditioning
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class EdgeAwareAttention(nn.Module):
    """Multi-head attention with edge bias.

    Computes attention as:
        attn_scores[i,j] = (Q_i · K_j) / sqrt(d_k) + edge_bias[i,j]

    where edge_bias is derived from edge embeddings.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        edge_vocab_size: int,
        dropout: float = 0.1
    ):
        """Initialize EdgeAwareAttention.

        Args:
            hidden_size: Hidden dimension.
            num_heads: Number of attention heads.
            edge_vocab_size: Size of edge vocabulary.
            dropout: Dropout rate.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # Edge embeddings -> attention bias (one bias per head)
        self.edge_embedding = nn.Embedding(edge_vocab_size, num_heads)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        E: torch.Tensor,
        M: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, N, H) node features.
            E: (B, N, N) edge tokens.
            M: (B, N) node mask (1 for real, 0 for padding).

        Returns:
            (B, N, H) updated node features.
        """
        B, N, H = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (B, num_heads, N, head_dim)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # Shape: (B, num_heads, N, N)

        # Add edge bias
        edge_bias = self.edge_embedding(E)  # (B, N, N, num_heads)
        edge_bias = edge_bias.permute(0, 3, 1, 2)  # (B, num_heads, N, N)
        scores = scores + edge_bias

        # Mask padding nodes (set attention to -inf for padding)
        # M: (B, N) -> (B, 1, 1, N) for broadcasting
        mask = M.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(mask == 0, -1e9)

        # Also mask source padding
        mask_src = M.unsqueeze(1).unsqueeze(-1)
        scores = scores.masked_fill(mask_src == 0, -1e9)

        # Softmax and apply
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values
        out = torch.matmul(attn, V)  # (B, num_heads, N, head_dim)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, N, H)
        out = self.o_proj(out)

        return out


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, hidden_size: int, ffn_hidden_size: int, dropout: float = 0.1):
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


class GraphTransformerBlock(nn.Module):
    """Single Graph Transformer block with pre-norm architecture."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        dropout: float,
        edge_vocab_size: int
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = EdgeAwareAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            edge_vocab_size=edge_vocab_size,
            dropout=dropout
        )

        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        E: torch.Tensor,
        M: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, N, H) node features.
            E: (B, N, N) edge tokens.
            M: (B, N) node mask.

        Returns:
            (B, N, H) updated node features.
        """
        # Pre-norm attention with residual
        x = x + self.attn(self.ln1(x), E, M)

        # Pre-norm FFN with residual
        x = x + self.ffn(self.ln2(x))

        return x


class GraphDiffusionBackbone(nn.Module):
    """Graph Transformer backbone for discrete diffusion.

    Architecture:
    - Node embedding + timestep embedding
    - Stack of GraphTransformerBlocks with edge-aware attention
    - Dual heads for node and edge prediction
    """

    def __init__(
        self,
        atom_vocab_size: int,
        edge_vocab_size: int,
        Nmax: int,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_hidden_size: int = 3072,
        dropout: float = 0.1,
        num_diffusion_steps: int = 100
    ):
        """Initialize GraphDiffusionBackbone.

        Args:
            atom_vocab_size: Size of atom vocabulary.
            edge_vocab_size: Size of edge vocabulary.
            Nmax: Maximum number of atoms.
            hidden_size: Hidden dimension.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            ffn_hidden_size: FFN hidden dimension.
            dropout: Dropout rate.
            num_diffusion_steps: Number of diffusion timesteps.
        """
        super().__init__()

        self.atom_vocab_size = atom_vocab_size
        self.edge_vocab_size = edge_vocab_size
        self.Nmax = Nmax
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Node embedding
        self.node_embedding = nn.Embedding(atom_vocab_size, hidden_size)

        # Timestep embedding
        self.time_embedding = nn.Embedding(num_diffusion_steps + 1, hidden_size)

        # Input dropout
        self.input_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GraphTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_hidden_size=ffn_hidden_size,
                dropout=dropout,
                edge_vocab_size=edge_vocab_size
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_final = nn.LayerNorm(hidden_size)

        # Node prediction head
        self.node_head = nn.Linear(hidden_size, atom_vocab_size, bias=False)

        # Edge prediction head (from pairwise node features)
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, edge_vocab_size)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        t: torch.Tensor,
        M: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            X: (B, Nmax) node tokens.
            E: (B, Nmax, Nmax) edge tokens.
            t: (B,) timesteps.
            M: (B, Nmax) node mask (1 for real, 0 for padding).

        Returns:
            node_logits: (B, Nmax, atom_vocab_size) node predictions.
            edge_logits: (B, Nmax, Nmax, edge_vocab_size) edge predictions (symmetric).
        """
        B, N = X.shape

        # Embed nodes and add timestep embedding
        x = self.node_embedding(X)  # (B, N, H)
        x = x + self.time_embedding(t).unsqueeze(1)  # (B, 1, H) broadcast

        # Apply dropout
        x = self.input_dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, E, M)

        # Final layer norm
        x = self.ln_final(x)

        # Node prediction head
        node_logits = self.node_head(x)  # (B, N, atom_vocab)

        # Edge prediction head: pairwise features
        x_i = x.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, H)
        x_j = x.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, H)
        x_ij = torch.cat([x_i, x_j], dim=-1)  # (B, N, N, 2H)

        edge_logits = self.edge_head(x_ij)  # (B, N, N, edge_vocab)

        # Enforce symmetry (average upper and lower triangles)
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2

        return node_logits, edge_logits

    def get_node_embeddings(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        t: torch.Tensor,
        M: torch.Tensor,
        pooling: str = 'mean'
    ) -> torch.Tensor:
        """Get pooled node embeddings (for property prediction).

        Args:
            X: (B, Nmax) node tokens.
            E: (B, Nmax, Nmax) edge tokens.
            t: (B,) timesteps (use 0 for clean data).
            M: (B, Nmax) node mask.
            pooling: Pooling method ('mean', 'sum', 'max').

        Returns:
            (B, H) pooled graph embeddings.
        """
        B, N = X.shape

        # Embed nodes and add timestep embedding
        x = self.node_embedding(X)
        x = x + self.time_embedding(t).unsqueeze(1)
        x = self.input_dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, E, M)

        # Final layer norm
        x = self.ln_final(x)

        # Pool over nodes (masked)
        M_expanded = M.unsqueeze(-1)  # (B, N, 1)

        if pooling == 'mean':
            x = (x * M_expanded).sum(dim=1) / M.sum(dim=1, keepdim=True).clamp(min=1)
        elif pooling == 'sum':
            x = (x * M_expanded).sum(dim=1)
        elif pooling == 'max':
            x = x.masked_fill(M_expanded == 0, -1e9).max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        return x  # (B, H)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_graph_backbone(config: Dict, graph_config: Dict) -> GraphDiffusionBackbone:
    """Create GraphDiffusionBackbone from config dictionaries.

    Args:
        config: Main configuration with 'backbone' section.
        graph_config: Graph configuration with 'Nmax', 'atom_vocab_size', 'edge_vocab_size'.

    Returns:
        GraphDiffusionBackbone instance.
    """
    backbone_config = config.get('backbone', config.get('graph_backbone', {}))
    diffusion_config = config.get('diffusion', config.get('graph_diffusion', {}))

    return GraphDiffusionBackbone(
        atom_vocab_size=graph_config['atom_vocab_size'],
        edge_vocab_size=graph_config['edge_vocab_size'],
        Nmax=graph_config['Nmax'],
        hidden_size=backbone_config.get('hidden_size', 768),
        num_layers=backbone_config.get('num_layers', 12),
        num_heads=backbone_config.get('num_heads', 12),
        ffn_hidden_size=backbone_config.get('ffn_hidden_size', 3072),
        dropout=backbone_config.get('dropout', 0.1),
        num_diffusion_steps=diffusion_config.get('num_steps', 100)
    )
