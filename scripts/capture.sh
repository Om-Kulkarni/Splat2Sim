#!/usr/bin/env bash
# Record a Capture on the laptop.
#   ./scripts/capture.sh <scene-name>     record (Ctrl-C to stop)
#   ./scripts/capture.sh --check          verify camera, USB 3 link, firmware; exit
#   ./scripts/capture.sh --preview        live view only, to tune exposure/WB (no recording)
# Env overrides (defaults in docker/capture/run.sh): R2S_EXPOSURE (x100us) R2S_GAIN R2S_WHITE_BALANCE
#   R2S_COLOR_PROFILE R2S_DEPTH_PROFILE R2S_AUTO_EXPOSURE R2S_PREVIEW=0 R2S_STORAGE_PRESET R2S_LAUNCH_ARGS
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
  ""|-h|--help) sed -n 2,9p "${BASH_SOURCE[0]}"; exit 64 ;;
  *) NAME="$(tr -c '\nA-Za-z0-9._-' '_' <<<"$1")-$(date +%Y%m%d-%H%M%S)" ;;
esac

if [[ "$MODE" == record && -z "$WORKSTATION_HOST" && -z "${R2S_WORKSTATION:-}" ]]; then
  echo "error: WORKSTATION_HOST is not set (put WORKSTATION_HOST=<host> and WORKSTATION_USER=<user> in .env, see .env.example, or export R2S_WORKSTATION=user@host)" >&2
  exit 78
fi

if [[ -n "${R2S_REBUILD:-}" ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "building $IMAGE (first run only)..."
  docker build -t "$IMAGE" "$HERE/docker/capture"
fi
mkdir -p "$DIR"

ARGS=(--rm --privileged --net=host --init -v /dev:/dev -v "$DIR:/captures"
      -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)")
[[ "$MODE" != check && -t 0 ]] && ARGS+=(-it)
while IFS= read -r v; do ARGS+=(-e "$v"); done < <(env | grep '^R2S_' | grep -v -E '^R2S_(CAPTURE_|REBUILD)' || true)
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
