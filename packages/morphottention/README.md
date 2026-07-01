# Morphottention
Mathematical Morphology-based self-attention module for PyTorch using Flash-style kernel fusion.

## Install

Prebuilt wheels are published for CPython 3.14 on Linux (x86_64, aarch64) and
Windows (x86_64). A working CUDA-enabled PyTorch (`torch >= 2.12`) must already
be installed in the environment.

```bash
pip install morphottention
```

## Usage

The package exposes an `nn.Module` (`MorphoAttention`), a functional entry point
(`morpho_attention`), and the raw autograd bridge (`MorphoAttentionFunction`).
All inputs must be CUDA tensors; the module defaults to `float16`.

### As an `nn.Module`

```python
import torch
from morphottention import MorphoAttention

attn = MorphoAttention(
    dim=256,       # model dimension D
    num_heads=8,   # number of attention heads H
    cube_m=16,     # hypercube width per head
    scale=1.0,     # softmax temperature
    causal=False,  # casual masking flag
    device="cuda",
)

x = torch.randn(2, 128, 256, dtype=torch.float16, device="cuda")  # (B, N, D)
out = attn(x)          # (B, N, D)
out.sum().backward()
```

### Functional form

```python
from morphottention import morpho_attention

out = morpho_attention(
    x, 
    W_phi, 
    gate_q, 
    gate_k, 
    W_V,
    num_heads=8, cube_m=16, scale=1.0, 
    causal=False,
)
```

### Raw autograd bridge

```python
import torch
from morphottention import MorphoAttentionFunction

B, N, D, H, cube_m = 2, 128, 256, 8, 16

x = torch.randn(B, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
W_phi = torch.randn(D, H * cube_m, dtype=torch.float16, device="cuda", requires_grad=True)
gate_q = torch.ones(H, cube_m, dtype=torch.float16, device="cuda", requires_grad=True)
gate_k = torch.ones(H, cube_m, dtype=torch.float16, device="cuda", requires_grad=True)
W_V = torch.randn(D, D, dtype=torch.float16, device="cuda", requires_grad=True)

out = MorphoAttentionFunction.apply(
    x, W_phi, gate_q, gate_k, W_V,
    H, cube_m, 1.0, False,  # num_heads, cube_m, scale, causal
)  # (B, N, D)
out.sum().backward()
```

`W_phi` has shape `(D, H * cube_m)`, `W_V` has shape `(D, D)`, and `gate_q` /
`gate_k` each have shape `(H, cube_m)`.

## Building from source

Requires the CUDA 13.X toolkit (`nvcc`) and a matching `torch` build:

```bash
uv sync --package morphottention --no-dev --group build
uv build --package morphottention --wheel --no-build-isolation
```

## License

MIT

Copyright © 2026 Vedran Hrabar.
