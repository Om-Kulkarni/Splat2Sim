#!/bin/bash
# Usage: run_lab.sh <script.py> [extra docker args]; Isaac Lab 3.0 image built FROM isaac-sim:6.1.0 (docker/Dockerfile.base, see ticket)
GPUFLAG=${GPUFLAG:---gpus all}; D=~/real2sim-data/smoke; S=$1; shift
docker run --rm --name r2s-lab-$$ $GPUFLAG --entrypoint /workspace/isaaclab/isaaclab.sh --user 0 \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e ISAACSIM_ASSET_ROOT=${ISAACSIM_ASSET_ROOT:-https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1} \
  -v $D/scripts:/scripts:ro -v $D/out:/out -v $D/data:/data \
  -v $D/cache/kit:/isaac-sim/kit/cache -v $D/cache/ov:/root/.cache/ov -v $D/cache/glcache:/root/.cache/nvidia/GLCache -v $D/cache/computecache:/root/.nv/ComputeCache \
  "$@" r2s-isaaclab3-sim61:latest -p /scripts/$S /out
