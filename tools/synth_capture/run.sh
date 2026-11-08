#!/usr/bin/env bash
# Render the synthetic Capture in the Isaac image (workstation). Usage: tools/synth_capture/run.sh [generate.py args]
#   e.g. tools/synth_capture/run.sh --frames 600 --out /captures/synth-robot
# Output lands in $R2S_DATA/captures/ (default ~/real2sim-data). Then: ./r2s build $R2S_DATA/captures/synth-robot --lab synth
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export R2S_DATA="${R2S_DATA:-$HOME/real2sim-data}"
mkdir -p "$R2S_DATA/captures" "$R2S_DATA/cache/isaac"/{kit,ov,glcache,computecache}
cd "$REPO"
exec docker compose -f compose.yaml --project-directory . run --rm \
  -v "$R2S_DATA/captures:/captures" -v "$REPO/tools/synth_capture:/workspace/synth:ro" -v "$REPO/labs:/opt/r2s/labs:ro" \
  isaac -p /workspace/synth/generate.py "$@"
