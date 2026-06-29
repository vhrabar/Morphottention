# Morphottention
Mathematical Morphology-based self-attention module for PyTorch using Flash-style kernel fusion.

## Install

Prebuilt wheels are published for CPython 3.14 on Linux (x86_64, aarch64) and
Windows (x86_64). A working CUDA-enabled PyTorch (`torch >= 2.12`) must already
be installed in the environment.

```bash
pip install morphottention
```

## Building from source

Requires the CUDA 13.X toolkit (`nvcc`) and a matching `torch` build:

```bash
uv sync --package morphottention --no-dev --group build
uv build --package morphottention --wheel --no-build-isolation
```

## License

MIT

Copyright © 2026 Vedran Hrabar.
