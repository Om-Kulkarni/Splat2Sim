# syntax=docker/dockerfile:1.7
# sam3 service: SAM 3.1 (Object Multiplex) video predictor -> masks/. For RTX 5090 (sm_120): torch 2.8.0+cu128.
# Base python:3.12-slim (SAM 3 needs python >= 3.12; r2s-3dgrut is 3.11). torch/CUDA libs come from the wheels, so no
# CUDA base image is needed. The uv cache mount shares its id with docker/3dgrut/Dockerfile, so the torch 2.8.0+cu128
# wheels fetched for that image are reused (workstation pull bandwidth is ~3 MB/s).
# Weights are gated (facebook/sam3.1): NOT in the image. Downloaded at run time with $HF_TOKEN (compose env_file)
# into the mounted HF cache /cache/hf. The package (splat2sim/) is bind-mounted by compose; no rebuild on code edits.
FROM python:3.12-slim

# main @ 2026-10-14 (includes SAM 3.1 multiplex predictor)
ARG SAM3_COMMIT=7f473cfd983d5eb43040ff608f66bff0d7bdcb53
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0

ENV UV_HTTP_TIMEOUT=900 UV_HTTP_RETRIES=10 PIP_DEFAULT_TIMEOUT=900 UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/opt/r2s \
    HF_HOME=/cache/hf HOME=/tmp/home XDG_CACHE_HOME=/tmp/home/.cache \
    TRITON_CACHE_DIR=/tmp/home/.triton TORCHINDUCTOR_CACHE_DIR=/tmp/home/.inductor \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --index-url https://download.pytorch.org/whl/cu128 \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" && \
    uv pip install --system "numpy<2" "opencv-python-headless>=4.9" "pyyaml>=6" "pytest>=8" "psutil" "einops" "pycocotools" \
        "sam3 @ https://github.com/facebookresearch/sam3/archive/${SAM3_COMMIT}.tar.gz"

# Smoke test (no GPU at build time): imports, tokenizer asset shipped, sm_120 in the torch build.
RUN python -c "import torch, sam3, os, importlib.resources as r; \
from sam3.model_builder import build_sam3_predictor; \
assert (r.files('sam3')/'assets'/'bpe_simple_vocab_16e6.txt.gz').is_file(); \
print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())" && \
    mkdir -p /cache/hf /tmp/home && chmod -R a+rwX /cache /tmp/home

WORKDIR /opt/r2s
