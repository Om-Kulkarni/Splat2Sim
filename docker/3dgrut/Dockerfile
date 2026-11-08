# syntax=docker/dockerfile:1.7
# 3DGRUT (3DGUT trainer + OpenUSD ParticleField export) for RTX 3090 Ti (sm_86) and RTX 5090 (sm_120).
# Base is the CUDA 13 *base* image already on the workstation; the CUDA 12.8 toolkit (nvcc + headers,
# needed for torch JIT of the tracers) is installed from NVIDIA's apt repo instead of pulling the 3 GB
# `-devel` image. PyTorch 2.8.0+cu128 comes from the PyTorch wheel index (Blackwell-capable).
FROM nvidia/cuda:13.0.0-base-ubuntu24.04

ARG GRUT_REPO=https://github.com/nv-tlabs/3dgrut.git
# main @ 2026-09-22; pyproject version 2.0.0 (no v2.0.0 tag exists upstream, latest tag is v1.1.0)
ARG GRUT_COMMIT=e6f4552b408af020a96de09ab5757f9b57430aaa
ARG CUDA_APT=12-8
ARG TORCH_VERSION=2.8.0
ARG SLANGC_VERSION=2026.5.2

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    EGL_PLATFORM=surfaceless \
    CUDA_HOME=/usr/local/cuda-12.8 \
    CUDA_VERSION=12.8 \
    TORCH_CUDA_ARCH_LIST="8.6;12.0" \
    TCNN_CUDA_ARCHITECTURES="86;120" \
    FORCE_CUDA=1 \
    UV_HTTP_TIMEOUT=900 UV_HTTP_RETRIES=10 PIP_DEFAULT_TIMEOUT=900 \
    UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    PATH=/opt/3dgrut/.venv/bin:/usr/local/cuda-12.8/bin:/usr/local/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 \
    LIBRARY_PATH=/usr/local/cuda-12.8/lib64/stubs

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates git curl wget build-essential ninja-build \
        libgl1 libglib2.0-0 libegl1 libglvnd0 \
        cuda-nvcc-${CUDA_APT} cuda-cudart-dev-${CUDA_APT} cuda-nvrtc-dev-${CUDA_APT} cuda-driver-dev-${CUDA_APT} \
        cuda-profiler-api-${CUDA_APT} cuda-cccl-${CUDA_APT} cuda-nvtx-${CUDA_APT} \
        libcublas-dev-${CUDA_APT} libcusparse-dev-${CUDA_APT} libcusolver-dev-${CUDA_APT} \
        libcurand-dev-${CUDA_APT} libcufft-dev-${CUDA_APT}

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

WORKDIR /opt/3dgrut
RUN git init -q . && git remote add origin ${GRUT_REPO} && git fetch -q --depth 1 origin ${GRUT_COMMIT} \
    && git checkout -q FETCH_HEAD && git submodule update --init --recursive --depth 1

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv .venv --python 3.11 --prompt 3dgrut && \
    export VIRTUAL_ENV=/opt/3dgrut/.venv UV_PROJECT_ENVIRONMENT=/opt/3dgrut/.venv && \
    echo "torch==${TORCH_VERSION}" > .venv/constraints.txt && export UV_CONSTRAINT=$PWD/.venv/constraints.txt && \
    export UV_INDEX="pytorch=https://download.pytorch.org/whl/cu128" && \
    uv pip install -e . && \
    uv pip install --no-build-isolation ./thirdparty/tiny-cuda-nn/bindings/torch && \
    uv pip install --no-build-isolation -r requirements_extra.txt && \
    curl -fsSL https://github.com/shader-slang/slang/releases/download/v${SLANGC_VERSION}/slang-${SLANGC_VERSION}-linux-x86_64.tar.gz | tar -xz -C .venv

# Smoke import at build time (no GPU needed (tinycudann imports only with a GPU)): the heavy tracer JIT happens on first train run
RUN python -c "import torch, importlib.util as u; assert u.find_spec('tinycudann'); import ppisp, fused_ssim, pxr, threedgrut; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"

# slangc writes generated .cuh next to the sources at first use, so the tracer dirs must be writable by any uid
RUN chmod -R a+rwX /opt/3dgrut/threedgut_tracer /opt/3dgrut/threedgrt_tracer

# Upstream bug (e6f4552): the USD exporter reports the *active* SH degree (grows 1/1000 iters, so 2 at 2000 iters) but
# writes all coefficients of the *max* degree -> reshape ValueError for short runs. Report the max degree instead.
RUN sed -i 's/sh_degree=n_active,/sh_degree=sh_degree,/' threedgrut/export/accessor.py && grep -q 'sh_degree=sh_degree,' threedgrut/export/accessor.py

RUN mkdir -p /cache && chmod 1777 /cache
VOLUME /cache
WORKDIR /opt/3dgrut
CMD ["python", "train.py", "--help"]
