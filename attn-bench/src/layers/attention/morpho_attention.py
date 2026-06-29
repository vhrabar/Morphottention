from typing import Any, cast

import torch
from torch import nn

from morphottention import MorphoAttention


class MultiHeadSelfAttention(nn.Module):
    """
    wrapper for custom Multi-Head Self-Attention implemntations
    """

    HEAD_DIM = 64
    CUBE_M = 64

    # fmt: off
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0,*, cube_m: int = CUBE_M,
                 scale: float | None = None, causal: bool = False, **_: Any) -> None:
        # fmt: on
        super().__init__()
        self.name = "morpho_attention"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.cube_m = cube_m

        self.scale = scale if scale is not None else cube_m**-0.5
        self.causal = causal

        self.attn = MorphoAttention(dim, num_heads, cube_m, scale=self.scale, causal=causal)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None]:
        orig_dtype = x.dtype
        out = self.attn(x.to(torch.float16)).to(orig_dtype)
        out = self.proj(out)
        return cast(torch.Tensor, self.dropout(out)), None
