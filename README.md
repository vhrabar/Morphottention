# Morphottention

![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A52.12-ee4c2c?style=for-the-badge&logo=pytorch)
![CUDA](https://img.shields.io/badge/cuda-13.X-76b900?style=for-the-badge&logo=nvidia)
![PyPI Version](https://img.shields.io/pypi/v/morphottention?style=for-the-badge&logo=pypi&logoColor=orange)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge&logo=opensourceinitiative)


Mathematical Morphology-based self-attention module for PyTorch (CUDA) using Flash-style kernel fusion.

## Layout

This is a [uv](https://docs.astral.sh/uv/) workspace:

- `packages/morphottention` - the published kernel package ([README](packages/morphottention/README.md)).
- `attn-bench` - benchmarks and dataset harnesses.

## Install

Prebuilt wheels (CPython 3.12–3.14; Linux x86_64/aarch64, Windows x86_64) require a
CUDA-enabled `torch >= 2.12` already installed:

```bash
pip install morphottention
```

## Usage

Drop-in self-attention module. Inputs must be CUDA tensors; the module defaults to `float16`.

```python
import torch
from morphottention import MorphoAttention

attn = MorphoAttention(
    dim=256,       # model dimension D
    num_heads=8,   # number of attention heads H
    cube_m=16,     # hypercube width per head
    scale=1.0,     # softmax temperature
    causal=False,  # casual masking flag
    device="cuda"
)
x = torch.randn(2, 128, 256, dtype=torch.float16, device="cuda")  # (B, N, D)
out = attn(x)          # (B, N, D)
out.sum().backward()
```

## Develop

Set up the workspace:
```bash
uv sync
```

Building the CUDA extension from source needs the CUDA 13.X toolkit (`nvcc`):

```bash
uv sync --package morphottention --no-dev --group build
uv build --package morphottention --wheel --no-build-isolation
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

Copyright © 2026 Vedran Hrabar
