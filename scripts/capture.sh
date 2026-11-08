#!/usr/bin/env bash
# Record a Capture on the laptop (default camera: OAK-D Pro W; R2S_CAMERA=realsense for the D455).
#   ./scripts/capture.sh <scene-name>     record (Ctrl-C to stop)
#   ./scripts/capture.sh --check          verify camera, USB 3 link, firmware; exit
#   ./scripts/capture.sh --preview        live view only, to tune exposure/WB (no recording)
# Env overrides (defaults in docker/capture/run.sh and docker/capture/oak_params.yaml, which is mounted, so editing it needs no rebuild):
#   R2S_CAMERA=oak|realsense  R2S_PREVIEW=0  R2S_PREVIEW_TOPIC  R2S_STORAGE_PRESET  R2S_LAUNCH_ARGS
#   OAK:       R2S_OAK_EXPOSURE_US R2S_OAK_ISO R2S_OAK_WB_K R2S_OAK_RGB_WIDTH/HEIGHT/FPS R2S_OAK_UNDISTORT R2S_OAK_STEREO_PRESET
#              R2S_OAK_DEPTH_MIN_MM/MAX_MM R2S_OAK_DOT R2S_OAK_FLOOD ... (list: oak_params.yaml)  R2S_OAK_PARAMS=<own yaml>
#   RealSense: R2S_EXPOSURE (x100us) R2S_GAIN R2S_WHITE_BALANCE R2S_COLOR_PROFILE R2S_DEPTH_PROFILE R2S_AUTO_EXPOSURE
#   R2S_CAPTURE_DIR (default ~/real2sim-captures)  R2S_WORKSTATION (user@host for rsync; else WORKSTATION_USER/WORKSTATION_HOST from .env)  R2S_REBUILD=1
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${R2S_CAPTURE_IMAGE:-r2s-capture}"
DIR="${R2S_CAPTURE_DIR:-$HOME/real2sim-captures}"
# Workstation for the rsync hint: R2S_WORKSTATION (user@host), or WORKSTATION_USER + WORKSTATION_HOST from the environment / repo .env
envval() { [[ -f "$HERE/.env" ]] && sed -n "s/^$1=//p" "$HERE/.env" | tail -1 | tr -d "\"'" || true; }
WORKSTATION_USER="${WORKSTATION_USER:-$(envval WORKSTATION_USER)}"
WORKSTATION_HOST="${WORKSTATION_HOST:-$(envval WORKSTATION_HOST)}"
WORKSTATION="${R2S_WORKSTATION:-${WORKSTATION_USER:+$WORKSTATION_USER@}$WORKSTATION_HOST}"
DEST="~/real2sim-data/captures/"

MODE=record; NAME=""
case "${1:-}" in
  --check) MODE=check ;;
  --preview) MODE=preview ;;
  ""|-h|--help) sed -n 2,10p "${BASH_SOURCE[0]}"; exit 64 ;;
  *) NAME="$(tr -c '\nA-Za-z0-9._-' '_' <<<"$1")-$(date +%Y%m%d-%H%M%S)" ;;
esac

if [[ "$MODE" == record && -z "$WORKSTATION_HOST" && -z "${R2S_WORKSTATION:-}" ]]; then
  echo "error: WORKSTATION_HOST is not set (put WORKSTATION_HOST=<host> and WORKSTATION_USER=<user> in .env, see .env.example, or export R2S_WORKSTATION=user@host)" >&2
  exit 78
fi

# rebuild when the image is missing or older than this checkout's Dockerfile (label r2s.capture.version)
WANT_VER="$(sed -n 's/^LABEL r2s.capture.version=//p' "$HERE/docker/capture/Dockerfile")"
HAVE_VER="$(docker image inspect -f '{{index .Config.Labels "r2s.capture.version"}}' "$IMAGE" 2>/dev/null || true)"
if [[ -n "${R2S_REBUILD:-}" ]] || [[ "$HAVE_VER" != "$WANT_VER" ]]; then
  echo "building $IMAGE (first run / Dockerfile changed, a few minutes)..."
  docker build -t "$IMAGE" "$HERE/docker/capture"
fi
mkdir -p "$DIR"

ARGS=(--rm --privileged --net=host --init -v /dev:/dev -v "$DIR:/captures"
      -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)")
[[ "$MODE" != check && -t 0 ]] && ARGS+=(-it)
while IFS= read -r v; do ARGS+=(-e "$v"); done < <(env | grep '^R2S_' | grep -v -E '^R2S_(CAPTURE_|REBUILD|OAK_PARAMS)' || true)
# OAK driver parameters: the repo's oak_params.yaml (edit it to tune), or your own copy via R2S_OAK_PARAMS=<path>
OAK_YAML="${R2S_OAK_PARAMS:-$HERE/docker/capture/oak_params.yaml}"
[[ -f "$OAK_YAML" ]] || { echo "error: R2S_OAK_PARAMS=$OAK_YAML not found" >&2; exit 66; }
ARGS+=(-v "$(readlink -f "$OAK_YAML"):/etc/r2s/oak_params.yaml:ro")
if [[ -n "${DISPLAY:-}" && "${R2S_PREVIEW:-1}" == 1 && "$MODE" != check ]]; then
  command -v xhost >/dev/null && xhost +local:root >/dev/null 2>&1 || echo "note: xhost not available; preview window may not open (R2S_PREVIEW=0 to disable)"
  ARGS+=(-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix)
fi

set +e
docker run "${ARGS[@]}" "$IMAGE" "$MODE" $NAME
rc=$?
set -e
if [[ "$MODE" == record && -d "$DIR/$NAME" ]]; then
  echo
  echo "Capture saved: $DIR/$NAME"
  echo "Copy to the workstation, then delete locally:"
  echo "  rsync -avP --partial \"$DIR/$NAME\" \"$WORKSTATION:$DEST\""
  echo "  rm -r \"$DIR/$NAME\"   # after rsync succeeds"
fi
exit $rc
