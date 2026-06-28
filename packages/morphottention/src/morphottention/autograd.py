"""
Autograd wrapper around the compiled Morphottention CUDA kernels.
"""

from __future__ import annotations

import torch
from . import _C



class MorphoAttentionFunction(torch.autograd.Function):
    """
    Bridges the CUDA attention kernels into autograd.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if not x.is_cuda:
            raise ValueError("MorphoAttention expects a CUDA tensor")

        x = x.contiguous()
        outputs = _C.forward(x)
        out, *saved = outputs

        ctx.save_for_backward(x, *saved)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        x = ctx.saved_tensors[0]
        grad_out = grad_out.contiguous()
        grads = _C.backward(grad_out, x)
        return grads[0]


def attention(x: torch.Tensor) -> torch.Tensor:
    """
    Apply the Morphottention attention mechanism with autograd support.
    :param: x: input activations of shape (B, N, D) on a CUDA device.
    :returns: The attention output, differentiable w.r.t. x.
    """
    return MorphoAttentionFunction.apply(x)
