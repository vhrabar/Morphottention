from typing import Any

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """
    Default PyTorch implementation of Multi-Head Self-Attention
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, **_: Any) -> None:
        super().__init__()
        self.name = "attention"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_bias: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Base forward pass, split into benchmarking parts.
        :param attn_bias: optional per-key (B, N) bias
        """
        B, N, C = q.shape

        # QKV projection
        qkv = self.qkv(q)

        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # QK^T
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn = attn + attn_bias.view(B, 1, 1, N)

        # Softmax
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, C)

        # Output projection
        out = self.proj(out)

        return out, attn
