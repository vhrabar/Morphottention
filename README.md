# Morphottention

![Python](https://img.shields.io/badge/python-3.14-blue)
![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A52.12-ee4c2c)
![CUDA](https://img.shields.io/badge/cuda-13.X-76b900)
![License](https://img.shields.io/badge/license-MIT-green)

Mathematical Morphology-based self-attention module for PyTorch (CUDA) using Flash-style kernel fusion.

## Layout

This is a [uv](https://docs.astral.sh/uv/) workspace:

- `packages/morphottention` - the published kernel package ([README](packages/morphottention/README.md)).
- `attn-bench` - benchmarks and dataset harnesses.

## Install

Prebuilt wheels (CPython 3.14; Linux x86_64/aarch64, Windows x86_64) require a
CUDA-enabled `torch >= 2.12` already installed:

```bash
pip install morphottention
```

## Develop
Setup the workspace:
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
