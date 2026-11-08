#!/usr/bin/env bash
# Container entrypoint.  Modes:  check | preview | record <outdir-name>
# All camera settings are env vars (defaults = locked capture protocol, see docs/capture-guide.md).
set -uo pipefail
set +u; source /opt/ros/jazzy/setup.bash; set -u

MODE="${1:-record}"; NAME="${2:-}"
: "${R2S_COLOR_PROFILE:=1280,720,30}"   # W,H,fps
: "${R2S_DEPTH_PROFILE:=848,480,30}"
: "${R2S_GYRO_FPS:=400}"
: "${R2S_ACCEL_FPS:=250}"
: "${R2S_AUTO_EXPOSURE:=false}"          # manual exposure + white balance = no flicker in the reconstruction
: "${R2S_EXPOSURE:=80}"                  # RGB exposure, units of 100 us (80 = 8 ms): short => less blur
: "${R2S_GAIN:=32}"                      # RGB gain, 0-128
: "${R2S_WHITE_BALANCE:=4600}"           # Kelvin
: "${R2S_PREVIEW:=1}"
: "${R2S_STORAGE_PRESET:=zstd_fast}"     # zstd_fast | zstd_small | fastwrite
CAM=/camera/camera
TOPICS=(
  $CAM/color/image_raw $CAM/color/camera_info
  $CAM/aligned_depth_to_color/image_raw $CAM/aligned_depth_to_color/camera_info
  $CAM/depth/image_rect_raw $CAM/depth/camera_info
  $CAM/imu /tf_static
)

check_device() {
  local out
  out="$(rs-enumerate-devices 2>&1)"
  if grep -qi "no device detected" <<<"$out" || ! grep -qi "serial number" <<<"$out"; then
    echo "ERROR: no RealSense device detected." >&2
    echo "  - plug the D455 into a USB 3 port with the short USB 3 cable; check 'lsusb | grep 8086'" >&2
    echo "  - run this script with the container --privileged (capture.sh does that)" >&2
    return 2
  fi
  grep -iE "^ *(Name|Serial Number|Firmware Version|Recommended Firmware Version|Physical Port|Usb Type)" <<<"$out" | head -8
  local usb fw rec
  usb="$(grep -iE "usb type" <<<"$out" | head -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)"
  fw="$(grep -iE "^ *Firmware Version" <<<"$out" | head -1 | awk '{print $NF}')"
  rec="$(grep -iE "^ *Recommended Firmware Version" <<<"$out" | head -1 | awk '{print $NF}')"
  if [[ -z "$usb" ]]; then echo "WARN: could not read the USB link speed"; 
  elif [[ "$usb" == 2* ]]; then
    echo "ERROR: camera enumerated as USB $usb. 848x480 + 1280x720 + IMU needs USB 3. Use a different port / a short, quality USB 3 cable." >&2
    return 3
  else echo "USB link: $usb (ok)"; fi
  grep -iE "^ *Name" <<<"$out" | head -1 | grep -q "D455" || echo "WARN: camera is not a D455 (protocol and docs assume the D455; the pipeline itself reads intrinsics from the camera)"
  [[ -n "$fw" && -n "$rec" && "$fw" != "$rec" ]] && echo "WARN: firmware $fw differs from recommended $rec (update: rs-fw-update; optional)"
  echo "check OK"
}

if [[ "$MODE" == check ]]; then check_device; exit $?; fi

check_device || exit $?
export QT_X11_NO_MITSHM=1
if [[ "$MODE" == record ]]; then
  [[ -n "$NAME" ]] || { echo "usage: record <outdir-name>"; exit 64; }
  OUT=/captures/$NAME
  [[ -e "$OUT" ]] && { echo "ERROR: $OUT exists"; exit 65; }
fi

PIDS=()
cleanup() {
  trap '' INT TERM
  [[ -n "${REC_PID:-}" ]] && { kill -TERM "$REC_PID" 2>/dev/null; echo "stopping recorder (flushing MCAP)..."; wait "$REC_PID" 2>/dev/null; }
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
  sleep 1; for p in "${PIDS[@]}"; do kill -KILL "$p" 2>/dev/null; done
  if [[ "$MODE" == record && -d "${OUT:-}" ]]; then
    mkdir -p "$OUT/logs"; cp /tmp/camera.log /tmp/bag.log "$OUT/logs/" 2>/dev/null   # keep the driver logs with the take
    [[ -n "${HOST_UID:-}" ]] && chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" "$OUT"
    echo; ros2 bag info "$OUT" 2>/dev/null | grep -E "Duration|Size|Messages|Topic:" 
    du -sh "$OUT"
  fi
  exit 0
}
trap cleanup INT TERM

# setsid: children do not get the terminal Ctrl-C directly. Shut them down with SIGTERM (async bash children ignore SIGINT; rclcpp handles SIGTERM).
setsid ros2 launch realsense2_camera rs_launch.py \
  camera_name:=camera camera_namespace:=camera initial_reset:=true \
  rgb_camera.color_profile:=$R2S_COLOR_PROFILE depth_module.depth_profile:=$R2S_DEPTH_PROFILE \
  align_depth.enable:=true enable_sync:=true \
  enable_gyro:=true enable_accel:=true gyro_fps:=$R2S_GYRO_FPS accel_fps:=$R2S_ACCEL_FPS unite_imu_method:=2 \
  rgb_camera.enable_auto_exposure:=$R2S_AUTO_EXPOSURE \
  ${R2S_LAUNCH_ARGS:-} > /tmp/camera.log 2>&1 &
CAM_PID=$!; PIDS+=("$CAM_PID")

echo "waiting for camera topics..."
for i in $(seq 1 60); do
  kill -0 "$CAM_PID" 2>/dev/null || { echo "ERROR: camera node died:"; tail -20 /tmp/camera.log; exit 4; }
  ros2 topic list 2>/dev/null | grep -q "$CAM/imu" && ros2 topic list 2>/dev/null | grep -q "aligned_depth_to_color/image_raw" && break
  sleep 1
done
sleep 2

if [[ "$R2S_AUTO_EXPOSURE" == false ]]; then
  # set explicitly after start-up (the wrapper can ignore exposure given before auto-exposure is off)
  P="ros2 param set $CAM"
  $P rgb_camera.enable_auto_exposure false >/dev/null
  $P rgb_camera.exposure "$R2S_EXPOSURE" >/dev/null
  $P rgb_camera.gain "$R2S_GAIN" >/dev/null
  $P rgb_camera.enable_auto_white_balance false >/dev/null
  $P rgb_camera.white_balance "$R2S_WHITE_BALANCE" >/dev/null
  echo "manual colour: exposure=$R2S_EXPOSURE (x100us) gain=$R2S_GAIN WB=${R2S_WHITE_BALANCE}K"
fi

if [[ "$R2S_PREVIEW" == 1 && -n "${DISPLAY:-}" ]]; then
  setsid ros2 run rqt_image_view rqt_image_view $CAM/color/image_raw > /tmp/rqt.log 2>&1 &
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
