#!/bin/bash
# Usage: run_smoke.sh <script.py> [extra docker args...]  (runs in Isaac Sim 6.1 container, headless)
# Env: IMG (default nvcr.io/nvidia/isaac-sim:6.1.0), GPUFLAG (default "--gpus all"; or "--device nvidia.com/gpu=all")
IMG=${IMG:-nvcr.io/nvidia/isaac-sim:6.1.0}; GPUFLAG=${GPUFLAG:---gpus all}
D=~/real2sim-data/smoke; SCRIPT=$1; shift
docker run --rm --name r2s-smoke-$$ $GPUFLAG --entrypoint /isaac-sim/python.sh \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v $D/scripts:/scripts:ro -v $D/out:/out -v $D/data:/data \
  -v $D/cache/kit:/isaac-sim/kit/cache -v $D/cache/ov:/root/.cache/ov \
  -v $D/cache/glcache:/root/.cache/nvidia/GLCache -v $D/cache/computecache:/root/.nv/ComputeCache \
  "$@" $IMG /scripts/$SCRIPT /out
