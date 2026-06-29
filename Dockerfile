# syntax=docker/dockerfile:1.7
#


ARG CUDA_VERSION=13.3.0
ARG KERNEL_PKG=morphottention

###############################################################################
# Builder
###############################################################################
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder
ARG KERNEL_PKG

# install build ytool & clear space
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# add uv pm
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

ENV UV_PYTHON_INSTALL_DIR=/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="12.0" \
    MAX_JOBS=4

WORKDIR /app

# deps
COPY pyproject.toml ./
COPY packages/${KERNEL_PKG}/pyproject.toml  packages/${KERNEL_PKG}/
COPY attn-bench/pyproject.toml attn-bench/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --package "${KERNEL_PKG}" --no-dev --group build --no-install-project --no-editable

# src + build
COPY packages/${KERNEL_PKG}/CMakeLists.txt packages/${KERNEL_PKG}/
COPY packages/${KERNEL_PKG}/README.md packages/${KERNEL_PKG}/
COPY packages/${KERNEL_PKG}/csrc/ packages/${KERNEL_PKG}/csrc/
COPY packages/${KERNEL_PKG}/src/  packages/${KERNEL_PKG}/src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --package "${KERNEL_PKG}" --no-dev --group build --no-editable

###############################################################################
# Runtime
###############################################################################
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime
ARG KERNEL_PKG

# copy python from builder
COPY --from=builder /python /python
COPY --from=builder /app/.venv /app/.venv

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN userdel --remove ubuntu 2>/dev/null || true; \
    useradd --create-home --uid 1000 app
USER app
WORKDIR /home/app

RUN python -c "import torch, ${KERNEL_PKG} as k; print(k.__name__, 'import OK')"

ENTRYPOINT ["bash"]