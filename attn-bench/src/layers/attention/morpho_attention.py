from typing import cast

import torch
from numpy import sqrt
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """
    wrapper for custom Multi-Head Self-Attention implemntations
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.name = "attention"
        self.num_heads = num_heads

        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _project(self) -> tuple[torch.Tensor]:
        raise NotImplementedError("CUDA extension not implemented yet")


    def _attention(self) -> torch.Tensor:
        raise NotImplementedError("CUDA extension not implemented yet")

    def _invert_out(self, out: torch.Tensor) -> torch.Tensor:
        BH, N, D = out.shape
        B = BH // self.num_heads
        C = self.num_heads * D
        out = out.view(B, self.num_heads, N, D)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return cast(torch.Tensor, self.dropout(out))

    def forward(self) -> tuple[torch.Tensor, None]:
        raise NotImplementedError("CUDA extension not implemented yet")
