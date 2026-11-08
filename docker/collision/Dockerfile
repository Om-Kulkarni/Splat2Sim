# NB open3d is pinned to 0.19.0: 0.20.0's ScalableTSDFVolume extracted empty meshes (even for a flat synthetic plane).
# collision service: masked depth + poses -> TSDF mesh (Open3D, CPU), RANSAC table/pedestal boxes, USD collision layer.
# A separate small image (not the 3dgrut/isaac ones): Open3D needs its own numpy pins and no GPU; usd-core provides
# pxr (UsdPhysics) so the stage can author collision.usda itself. The splat2sim package is bind-mounted by compose.
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=300 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/opt/r2s \
    OMP_NUM_THREADS=8
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libgl1 libegl1 libgles2 libusb-1.0-0 libidn2-0 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install "open3d==0.19.0" "numpy>=1.26" "pillow>=10" "pyyaml>=6" "pytest>=8" "usd-core>=24.11"
WORKDIR /opt/r2s
