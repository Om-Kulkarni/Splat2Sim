#!/usr/bin/env bash
# Container entrypoint.  Modes:  check | preview | record <outdir-name> | shell
# R2S_CAMERA=oak (default, OAK-D Pro W) | realsense (D455) selects the driver.
# All camera settings are env vars (defaults = locked capture protocol, see docs/capture-guide.md).
#   OAK:       R2S_OAK_* here + the params file docker/capture/oak_params.yaml (mounted by capture.sh; R2S_OAK_PARAMS=<host path> to use your own)
#   RealSense: R2S_COLOR_PROFILE R2S_DEPTH_PROFILE R2S_GYRO_FPS R2S_ACCEL_FPS R2S_AUTO_EXPOSURE R2S_EXPOSURE R2S_GAIN R2S_WHITE_BALANCE
#   both:      R2S_PREVIEW R2S_PREVIEW_TOPIC R2S_STORAGE_PRESET R2S_LAUNCH_ARGS
set -uo pipefail
set +u; source /opt/ros/jazzy/setup.bash; set -u

MODE="${1:-record}"; NAME="${2:-}"
: "${R2S_CAMERA:=oak}"
: "${R2S_PREVIEW:=1}"
: "${R2S_STORAGE_PRESET:=zstd_fast}"     # zstd_fast | zstd_small | fastwrite
# RealSense defaults
: "${R2S_COLOR_PROFILE:=1280,720,30}"   # W,H,fps
: "${R2S_DEPTH_PROFILE:=848,480,30}"
: "${R2S_GYRO_FPS:=400}"
: "${R2S_ACCEL_FPS:=250}"
: "${R2S_AUTO_EXPOSURE:=false}"          # manual exposure + white balance = no flicker in the reconstruction
: "${R2S_EXPOSURE:=80}"                  # RGB exposure, units of 100 us (80 = 8 ms): short => less blur
: "${R2S_GAIN:=32}"                      # RGB gain, 0-128
: "${R2S_WHITE_BALANCE:=4600}"           # Kelvin
# OAK-D Pro W defaults live in oak_params.yaml as `$(env R2S_OAK_X default)`; this list is only for the capture_info.json snapshot.
OAK_PARAMS="${R2S_OAK_PARAMS_IN_CONTAINER:-/etc/r2s/oak_params.yaml}"
export R2S_CAMERA R2S_CHECK_JSON=/tmp/camera_check.json

case "$R2S_CAMERA" in
  oak)
    CAMNAME="OAK-D Pro W"
    TOPICS=(
      /oak/rgb/image_raw /oak/rgb/camera_info
      /oak/stereo/image_raw /oak/stereo/camera_info      # depth aligned to RGB (device-side), 16UC1 mm
      /oak/imu/data /tf_static
    )
    PREVIEW_TOPIC="${R2S_PREVIEW_TOPIC:-/oak/rgb/image_raw}"
    WAIT_FOR=(/oak/imu/data /oak/stereo/image_raw /oak/rgb/image_raw)
    ;;
  realsense)
    CAMNAME="RealSense D455"
    CAM=/camera/camera
    TOPICS=(
      $CAM/color/image_raw $CAM/color/camera_info
      $CAM/aligned_depth_to_color/image_raw $CAM/aligned_depth_to_color/camera_info
      $CAM/depth/image_rect_raw $CAM/depth/camera_info
      $CAM/imu /tf_static
    )
    PREVIEW_TOPIC="${R2S_PREVIEW_TOPIC:-$CAM/color/image_raw}"
    WAIT_FOR=($CAM/imu $CAM/aligned_depth_to_color/image_raw)
    ;;
  *) echo "ERROR: R2S_CAMERA must be 'oak' or 'realsense' (got '$R2S_CAMERA')" >&2; exit 64 ;;
esac

check_realsense() {
  local out
  out="$(rs-enumerate-devices 2>&1)"
  if grep -qi "no device detected" <<<"$out" || ! grep -qi "serial number" <<<"$out"; then
    echo "ERROR: no RealSense device detected." >&2
    echo "  - plug the D455 into a USB 3 port with the short USB 3 cable; check 'lsusb | grep 8086'" >&2
    echo "  - run this script with the container --privileged (capture.sh does that)" >&2
    return 2
  fi
  grep -iE "^ *(Name|Serial Number|Firmware Version|Recommended Firmware Version|Physical Port|Usb Type)" <<<"$out" | head -8
  local usb fw rec name
  usb="$(grep -iE "usb type" <<<"$out" | head -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)"
  fw="$(grep -iE "^ *Firmware Version" <<<"$out" | head -1 | awk '{print $NF}')"
  rec="$(grep -iE "^ *Recommended Firmware Version" <<<"$out" | head -1 | awk '{print $NF}')"
  name="$(grep -iE "^ *Name" <<<"$out" | head -1 | sed 's/^ *Name *: *//')"
  printf '{"driver": "realsense2_camera", "product": "%s", "firmware": "%s", "usb": "%s"}\n' "$name" "$fw" "$usb" > "$R2S_CHECK_JSON"
  if [[ -z "$usb" ]]; then echo "WARN: could not read the USB link speed"; 
  elif [[ "$usb" == 2* ]]; then
    echo "ERROR: camera enumerated as USB $usb. 848x480 + 1280x720 + IMU needs USB 3. Use a different port / a short, quality USB 3 cable." >&2
    return 3
  else echo "USB link: $usb (ok)"; fi
  grep -q "D455" <<<"$name" || echo "WARN: camera is not a D455 (protocol and docs assume the D455; the pipeline itself reads intrinsics from the camera)"
  [[ -n "$fw" && -n "$rec" && "$fw" != "$rec" ]] && echo "WARN: firmware $fw differs from recommended $rec (update: rs-fw-update; optional)"
  echo "check OK"
}

check_device() {
  if [[ "$R2S_CAMERA" == oak ]]; then r2s-check-oak; else check_realsense; fi
}

if [[ "$MODE" == shell ]]; then exec bash; fi
if [[ "$MODE" == check ]]; then check_device; exit $?; fi

check_device || exit $?
export QT_X11_NO_MITSHM=1
if [[ "$MODE" == record ]]; then
  [[ -n "$NAME" ]] || { echo "usage: record <outdir-name>"; exit 64; }
  OUT=/captures/$NAME
  [[ -e "$OUT" ]] && { echo "ERROR: $OUT exists"; exit 65; }
fi

write_capture_info() {  # sidecar next to the .mcap: camera identity + the settings this take was made with (read by ingest -> meta.json)
  python3 - "$OUT" <<'PY'
import json, os, sys, time
out = sys.argv[1]
info = {}
try:
    info = json.load(open(os.environ.get("R2S_CHECK_JSON", "")))
except Exception:
    pass
info["camera_type"] = os.environ.get("R2S_CAMERA", "")
info["settings"] = {k: v for k, v in sorted(os.environ.items()) if k.startswith("R2S_") and k != "R2S_CHECK_JSON"}
info["recorded_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(info, open(os.path.join(out, "capture_info.json"), "w"), indent=2)
PY
}

PIDS=()
cleanup() {
  trap '' INT TERM
  [[ -n "${REC_PID:-}" ]] && { kill -TERM "$REC_PID" 2>/dev/null; echo "stopping recorder (flushing MCAP)..."; wait "$REC_PID" 2>/dev/null; }
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
  # give the driver time to close the device cleanly (a SIGKILL leaves the OAK "crashed" for the next start)
  for _ in $(seq 1 8); do alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done; [[ $alive == 0 ]] && break; sleep 1; done
  for p in "${PIDS[@]}"; do kill -KILL "$p" 2>/dev/null; done
  if [[ "$MODE" == record && -d "${OUT:-}" ]]; then
    mkdir -p "$OUT/logs"; cp /tmp/camera.log /tmp/bag.log "$OUT/logs/" 2>/dev/null   # keep the driver logs with the take
    [[ "$R2S_CAMERA" == oak ]] && cp "$OAK_PARAMS" "$OUT/logs/oak_params.yaml" 2>/dev/null; cp /tmp/oak_params_effective.yaml "$OUT/logs/" 2>/dev/null
    write_capture_info
    [[ -n "${HOST_UID:-}" ]] && chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" "$OUT"
    echo; ros2 bag info "$OUT" 2>/dev/null | grep -E "Duration|Size|Messages|Topic:" 
    du -sh "$OUT"
  fi
  exit 0
}
trap cleanup INT TERM

# setsid: children do not get the terminal Ctrl-C directly. Shut them down with SIGTERM (async bash children ignore SIGINT; rclcpp handles SIGTERM).
if [[ "$R2S_CAMERA" == oak ]]; then
  [[ -f "$OAK_PARAMS" ]] || { echo "ERROR: OAK params file $OAK_PARAMS not found"; exit 66; }
  echo "$CAMNAME: params $OAK_PARAMS"
  setsid ros2 launch depthai_ros_driver_v3 driver.launch.py \
    params_file:="$OAK_PARAMS" camera_model:=OAK-D-PRO-W \
    ${R2S_LAUNCH_ARGS:-} > /tmp/camera.log 2>&1 &
else
  setsid ros2 launch realsense2_camera rs_launch.py \
    camera_name:=camera camera_namespace:=camera initial_reset:=true \
    rgb_camera.color_profile:=$R2S_COLOR_PROFILE depth_module.depth_profile:=$R2S_DEPTH_PROFILE \
    align_depth.enable:=true enable_sync:=true \
    enable_gyro:=true enable_accel:=true gyro_fps:=$R2S_GYRO_FPS accel_fps:=$R2S_ACCEL_FPS unite_imu_method:=2 \
    rgb_camera.enable_auto_exposure:=$R2S_AUTO_EXPOSURE \
    ${R2S_LAUNCH_ARGS:-} > /tmp/camera.log 2>&1 &
fi
CAM_PID=$!; PIDS+=("$CAM_PID")

echo "waiting for camera topics..."
for i in $(seq 1 60); do
  kill -0 "$CAM_PID" 2>/dev/null || { echo "ERROR: camera node died:"; tail -20 /tmp/camera.log; exit 4; }
  tl="$(ros2 topic list 2>/dev/null)"; ok=1
  for t in "${WAIT_FOR[@]}"; do grep -qx "$t" <<<"$tl" || ok=0; done
  [[ $ok == 1 ]] && break
  sleep 1
done
sleep 2

if [[ "$R2S_CAMERA" == realsense && "$R2S_AUTO_EXPOSURE" == false ]]; then
  # set explicitly after start-up (the wrapper can ignore exposure given before auto-exposure is off)
  P="ros2 param set $CAM"
  $P rgb_camera.enable_auto_exposure false >/dev/null
  $P rgb_camera.exposure "$R2S_EXPOSURE" >/dev/null
  $P rgb_camera.gain "$R2S_GAIN" >/dev/null
  $P rgb_camera.enable_auto_white_balance false >/dev/null
  $P rgb_camera.white_balance "$R2S_WHITE_BALANCE" >/dev/null
  echo "manual colour: exposure=$R2S_EXPOSURE (x100us) gain=$R2S_GAIN WB=${R2S_WHITE_BALANCE}K"
fi
if [[ "$R2S_CAMERA" == oak ]]; then
  ros2 param dump /oak > /tmp/oak_params_effective.yaml 2>/dev/null   # every value as the driver actually runs it
  echo "applied: $(ros2 param get /oak rgb.r_exposure 2>/dev/null | sed 's/.*: //') us exposure, ISO $(ros2 param get /oak rgb.r_iso 2>/dev/null | sed 's/.*: //')," \
       "WB $(ros2 param get /oak rgb.r_whitebalance 2>/dev/null | sed 's/.*: //') K, IR dots $(ros2 param get /oak driver.r_laser_dot_intensity 2>/dev/null | sed 's/.*: //')"
fi

if [[ "$R2S_PREVIEW" == 1 && -n "${DISPLAY:-}" ]]; then
  setsid ros2 run rqt_image_view rqt_image_view "$PREVIEW_TOPIC" > /tmp/rqt.log 2>&1 &
  PIDS+=($!)
fi

if [[ "$MODE" == preview ]]; then
  echo "PREVIEW only (not recording). Ctrl-C to quit."
  while kill -0 "$CAM_PID" 2>/dev/null; do sleep 1; done; exit 0
fi

setsid ros2 bag record -s mcap --storage-preset-profile "$R2S_STORAGE_PRESET" \
  -o "$OUT" "${TOPICS[@]}" > /tmp/bag.log 2>&1 &
REC_PID=$!
echo "RECORDING -> $OUT   (Ctrl-C to stop)"
while kill -0 "$REC_PID" 2>/dev/null && kill -0 "$CAM_PID" 2>/dev/null; do
  sleep 2; printf '\r  %s  %s   ' "$(du -sh "$OUT" 2>/dev/null | cut -f1)" "$(df -h /captures | awk 'NR==2{print $4" free"}')"
done
echo; echo "recorder or camera exited unexpectedly:"; tail -5 /tmp/bag.log /tmp/camera.log
cleanup
