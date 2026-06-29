"""
Autograd wrapper and nn.Module around the compiled Morphottention CUDA kernels.
"""

from __future__ import annotations

import torch
from torch import nn

from . import _C


class MorphoAttentionFunction(torch.autograd.Function):
    """
    Bridges the CUDA attention kernels into autograd.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        W_phi: torch.Tensor,
        gate_q: torch.Tensor,
        gate_k: torch.Tensor,
        W_V: torch.Tensor,
        H: int,
        cube_m: int,
        scale: float,
        causal: bool,
    ) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("MorphoAttention expects a CUDA tensor")

        x = x.contiguous()
        out, lse = _C.forward(x, W_phi, gate_q, gate_k, W_V, H, cube_m, scale, causal)

        ctx.save_for_backward(x, W_phi, gate_q, gate_k, W_V, lse)
        ctx.H = H  # type: ignore[attr-defined]
        ctx.cube_m = cube_m  # type: ignore[attr-defined]
        ctx.scale = scale  # type: ignore[attr-defined]
        ctx.causal = causal  # type: ignore[attr-defined]
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        raise NotImplementedError("MorphoAttention BWD has not been implemented yet.")


class MorphoAttention(nn.Module):
    """
    Morphological hypercube attention
    :param dim: model dimension ``D`` (must be divisible by ``num_heads``).
    :param num_heads: number of attention heads ``H``.
    :param cube_m: hypercube width per head ``m`` (the score-GEMM contraction dim).
    :param scale: softmax temperature.
    :param causal: whether to apply causal masking.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cube_m: int,
        *,
        scale: float = 1.0,
        causal: bool = False,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.cube_m = cube_m
        self.scale = scale
        self.causal = causal

        head_dim_v = dim // num_heads
        self.W_phi = nn.Parameter(torch.empty(dim, num_heads * cube_m, dtype=dtype, device=device))
        self.gate_q = nn.Parameter(torch.empty(num_heads, cube_m, dtype=dtype, device=device))
        self.gate_k = nn.Parameter(torch.empty(num_heads, cube_m, dtype=dtype, device=device))
        self.W_V = nn.Parameter(torch.empty(dim, num_heads * head_dim_v, dtype=dtype, device=device))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = self.dim**-0.5
        with torch.no_grad():
            for w in (self.W_phi, self.W_V):
                w.copy_(torch.empty(w.shape, dtype=torch.float32, device=w.device).normal_(0.0, std))
            self.gate_q.fill_(1.0)
            self.gate_k.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MorphoAttentionFunction.apply(  # type: ignore[no-any-return, no-untyped-call]
            x, self.W_phi, self.gate_q, self.gate_k, self.W_V, self.num_heads, self.cube_m, self.scale, self.causal
        )


def morpho_attention(
    x: torch.Tensor,
    W_phi: torch.Tensor,
    gate_q: torch.Tensor,
    gate_k: torch.Tensor,
    W_V: torch.Tensor,
    num_heads: int,
    cube_m: int,
    *,
    scale: float = 1.0,
    causal: bool = False,
) -> torch.Tensor:
    """
    Apply the Morphottention attention mechanism with autograd support.
    :param: x: input activations of shape (B, N, D) on a CUDA device.
    :returns: the attention output, differentiable w.r.t. ``x``.
    """
    return MorphoAttentionFunction.apply(  # type: ignore[no-any-return, no-untyped-call]
        x, W_phi, gate_q, gate_k, W_V, num_heads, cube_m, scale, causal
    )
