# Capture guide (OAK-D Pro W, handheld, one take = one Scene)

Recording runs on the **laptop** in a Docker container (nothing installed on the host). The default camera is the **Luxonis OAK-D Pro W** (`R2S_CAMERA=oak`); the RealSense D455 still works with `R2S_CAMERA=realsense` (settings in the [D455 appendix](#appendix-realsense-d455)). Use the supplied **USB 3** cable in a USB 3.x port, **laptop on mains power**.
Optional host udev rule (the container is `--privileged`, so it normally works without): `echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules && sudo udevadm control --reload-rules && sudo udevadm trigger`.

## 0. One-time
- Print `docs/charuco-board-a3.pdf` (A4 fallback: `charuco-board-a4.pdf`) at **100% / "actual size"**, matte. Check the 100 mm bar with a ruler. Mount flat on rigid backing (Dibond/foam-core), spray adhesive.
- Measure the span of all 7 squares with a steel rule/calipers; **measured square length = span / 7** goes in `labs/<lab>.yaml` (not the nominal 55 mm). Sanity-check the print: `python tools/charuco/detect_check.py photo.jpg --square-mm <measured>` (see `tools/charuco/make_board.py` for the container one-liner; expects 24/24 corners).
- **No printer?** Show the board on a monitor or tablet instead: `docs/charuco-board-screen-{1080p,1200p,4k}.png` (match the screen's native resolution; regenerate with `tools/charuco/make_screen_board.py`), full-screen at 100 % zoom, brightness ~50 %, matte screen if possible. A 24" monitor gives ~52 mm squares, about the A3 print; a phone is too small. The recording laptop itself works with a long USB 3 cable: `docs/charuco-board-laptop-1200p.png` (~36 mm squares, readable to ~1 m only, so do the board passes at 0.7-1 m), `R2S_PREVIEW=0`, screen blanking off (`gsettings set org.gnome.desktop.session idle-delay 0`, back to 300 afterwards), mains power. Lay it **flat, face up, on the robot table** (same frame convention as the print: roll 180 deg) and keep it switched on and still for the whole take. Measure the square length **on the screen** (span of 7 squares / 7) and set `marker_length_m = square * 41/55`. Measure `board_to_base` with tape (A), not a pointed tip on the glass. If the preview shows dark bands across the screen (backlight flicker against the short exposure), raise `R2S_EXPOSURE` a bit or change the screen brightness.
- First run builds the image (a few minutes): `./scripts/capture.sh --check`. OAK: product, sensors (colour = IMX378 **rolling shutter**), USB link (**must be SUPER**), calibration model, IMU. Fails with a clear message if the camera is missing or on USB 2.

## 1. Before every take
1. **Board**: fix it rigidly to the same table/plinth the FR3 is bolted to, 0.3-0.8 m from the base, flat, outside the arm sweep, visible from your walking route, no glare. Keep it there between takes (or re-measure).
2. **Board -> base transform** (goes in the lab config; board frame: origin = ChArUco corner id 0, the first inner chessboard corner, one square in from the top-left of the pattern and marked by the margin ticks and "ORIGIN = corner id 0" on the print; x right along the printed +X arrow, y down along +Y, z into the board). Board z points INTO the table, so a flat, face-up board has roll 180 deg relative to a z-up base.
   - *Tape (A)*: level the table; measure from the joint-1 axis point to the origin mark in robot x/y with a steel rule + square, plus height; yaw from two far-apart points on the board edge (2 mm over 0.5 m = 0.23 deg).
   - *Touch-off (B, preferred)*: put a pointed tip in the gripper (calibrate tip offset by pivoting), jog/hand-guide it to >=4 non-collinear corner marks, record FK tip positions, fit a rigid transform. Residual = your real error.
3. **Robot**: move the FR3 in Desk to the saved scan pose. Do not move it during the take.
4. **Lighting**: switch on the normal lab lights, close blinds, then leave lighting alone for the whole take.
5. **Camera settings** (defaults = locked protocol, all in `docker/capture/oak_params.yaml`): colour **640x360@30** with **on-device undistortion** (`rgb.i_undistorted`), depth from the mono pair at 640x400 **aligned to colour** (same size, 16UC1 mm), preset ROBOTICS, LR-check + subpixel on, depth window 0.3-6 m, IMU united accel 480 Hz / gyro 400 Hz, **manual exposure 4 ms, ISO 600, white balance 5000 K**, IR dot projector 0.6, flood 0.
   - The colour sensor is a **rolling shutter**: keep exposure short (<= 4 ms), walk slowly, avoid whip-pans. In a dim room raise `R2S_OAK_ISO` first (max 1600), then exposure.
   - Depth: min range ~0.7 m with the mono pair at 800P (~0.35 m with `R2S_OAK_EXTENDED_DISP=true`, worse far range; less at 400P, look at the preview), good depth up to ~4-5 m, noisy beyond. The 150 deg mono lenses see far less distance than the D455 (7.5 cm baseline).
   - The IR dots do not show in the colour image (measured: projector 1.0 vs 0 gave identical colour frames), so the projector stays on.
   - Tune with `./scripts/capture.sh --preview` (shows the colour image; `R2S_PREVIEW_TOPIC=/oak/stereo/image_raw` for depth). Per-run override by env var, permanent change by editing the YAML (mounted into the container, no rebuild), or your own copy via `R2S_OAK_PARAMS=/path/my.yaml`:
     `R2S_OAK_EXPOSURE_US=3000 R2S_OAK_ISO=800 R2S_OAK_WB_K=5000 ./scripts/capture.sh <scene>`
   - Knobs: `R2S_OAK_RGB_WIDTH/HEIGHT/FPS`, `R2S_OAK_UNDISTORT` (false = record the raw lens, ingest undistorts), `R2S_OAK_MANUAL_EXPOSURE`, `R2S_OAK_EXPOSURE_US`, `R2S_OAK_ISO`, `R2S_OAK_MANUAL_WB`, `R2S_OAK_WB_K`, `R2S_OAK_MONO_WIDTH/HEIGHT`, `R2S_OAK_STEREO_FPS`, `R2S_OAK_STEREO_PRESET` (DEFAULT|FACE|HIGH_DETAIL|ROBOTICS), `R2S_OAK_LR_CHECK`, `R2S_OAK_SUBPIXEL`, `R2S_OAK_EXTENDED_DISP`, `R2S_OAK_DEPTH_MIN_MM/MAX_MM`, `R2S_OAK_DOT`, `R2S_OAK_FLOOD`, `R2S_OAK_ACCEL_HZ`, `R2S_OAK_GYRO_HZ`, `R2S_OAK_USB_SPEED`; plus `R2S_PREVIEW=0`, `R2S_STORAGE_PRESET`, `R2S_LAUNCH_ARGS`. Any other driver parameter: edit the YAML (`ros2 param list /oak` from `docker run ... r2s-capture shell`).
   - **Check the rates** in the summary printed after a take (`ros2 bag info`): colour ~30 Hz, depth ~30 Hz, IMU ~400 Hz. Measured on the development laptop: 640x360 = 30/30 Hz; 1280x720 only ~7 Hz through the ROS driver (960x540 ~12 Hz), although the same camera does 1280x720@30 with the plain DepthAI Python API. Mono 1280x800 (best depth accuracy) is capped by the camera at ~14.5 fps. Raise resolution (`R2S_OAK_RGB_WIDTH=1280 R2S_OAK_RGB_HEIGHT=720`) only if your take shows full rates.
   - The pipeline needs a pinhole image. With `rgb.i_undistorted: true` the driver sends a distortion-free image (camera_info D = 0; the field of view is cropped to ~82 deg HFOV). If you record the raw lens, ingest undistorts colour and depth (`meta.json: undistorted_in_ingest`).
   - After a take, `logs/oak_params_effective.yaml` in the take folder holds every driver value as it ran, and `capture_info.json` the camera product/serial and your `R2S_*` settings.

## 2. Record
```
./scripts/capture.sh robolab-tableA        # -> ~/real2sim-captures/robolab-tableA-<timestamp>/
```
Wait for `RECORDING`, then walk. **Ctrl-C once** to stop; wait for the summary (MCAP is flushed). Raw is ~110 MB/s at 1280x720 (~35 MB/s at 640x360); zstd chunks cut that; budget up to ~20 GB for a 3-min take.

## 3. Walking pattern (2-3 min, slow)
- 2-3 full **orbits** of the workcell at **knee, chest and head height**.
- One **close pass over the table** at ~0.7 m, camera looking down (the OAK sees no depth closer than ~0.7 m at 800P stereo: holes, not errors; keep >= 0.8 m).
- **Dwell** (~10 s, steady) near the **wrist camera** and **front camera** viewpoints (where the policy cameras see from).
- **Board in view >= 20 s in total**, from varied angles (20-50 deg oblique as well as frontal) and distances (0.7-2 m).
- Overlap: always keep ~70% of what you saw a moment ago in view; revisit the start at the end (loop closure).

**Do**: walk slowly, smooth steps, keep the camera level-ish, hold at corners, keep texture in view, check the preview now and then.
**Don't**: fast pans or rotations in place, point at windows/lamps (saturation), cover the lens, walk through featureless walls/floors only, let people or chairs move into the scene, touch the arm or board, change lighting.

## 4. After
```
# printed by capture.sh:
rsync -avP --partial ~/real2sim-captures/<name> $WORKSTATION_USER@$WORKSTATION_HOST:~/real2sim-data/captures/
rm -r ~/real2sim-captures/<name>          # once rsync succeeded
```
On the workstation: `./r2s build ~/real2sim-data/captures/<name> --lab robolab`. The quality report flags coverage gaps -> do a second take.

## 5. Troubleshooting
| Symptom | Fix |
|---|---|
| `--check`: "no OAK device" | Cable/port; `lsusb \| grep 03e7` (`03e7:2485` before boot is normal); replug; only one process can use the camera; run via `capture.sh` (privileged, `/dev` mounted, **host network**: needed for USB hotplug events). `X_LINK_DEVICE_NOT_FOUND` after a crash: unplug/replug. RealSense: `lsusb \| grep 8086`. |
| USB 2.x link reported | Different port, short quality USB 3 cable, no hub. Do **not** record on USB 2 (frames drop, IMU stalls). |
| Dropped frames / low fps | USB 2 link; slow disk (check free space; try `R2S_STORAGE_PRESET=fastwrite`); close other CPU-heavy apps; `R2S_PREVIEW=0`. Check rates with `ros2 topic hz` inside the container. |
| Colour/depth far below 30 Hz | See rates above: lower `R2S_OAK_RGB_WIDTH/HEIGHT`; mains power; mono 400P not 800P. |
| No `/oak/imu/data` (OAK) / `/camera/camera/imu` (D455) | D455 / D435i have an IMU (a plain D435 has none); firmware; `enable_gyro/accel` are on by default; check `/tmp/camera.log` (printed on startup failure). Replug. |
| No preview window | `echo $DISPLAY`, `xhost +local:root`, or use `R2S_PREVIEW=0` (recording is unaffected). |
| Image too dark / blurred | Dark: raise `R2S_OAK_ISO` then `R2S_OAK_EXPOSURE_US`; blur: shorter exposure, walk slower (rolling shutter). Flicker/colour drift: keep manual exposure + WB on (`R2S_OAK_MANUAL_EXPOSURE=true`, `R2S_OAK_MANUAL_WB=true`; D455: `R2S_AUTO_EXPOSURE=false`). |
| Firmware warning | Optional; `rs-fw-update` inside the image (D455 only). |

## Appendix: RealSense D455
`R2S_CAMERA=realsense ./scripts/capture.sh --check|--preview|<scene>`. Same container, same MCAP/ingest path.
**Camera settings** (defaults are the locked protocol): colour 1280x720@30, depth 848x480@30, `align_depth` on, `enable_sync`, gyro 400 Hz / accel 250 Hz united (linear interpolation), **manual exposure + white balance**.
Starting values (tune once per lab): `R2S_EXPOSURE=80` (units of 100 us = 8 ms; short against blur), `R2S_GAIN=32`, `R2S_WHITE_BALANCE=4600` (K). Tune with `./scripts/capture.sh --preview`: image bright but not clipped, no visible blur on a slow pan; raise gain before exposure. Then pin them:
`R2S_EXPOSURE=100 R2S_GAIN=48 R2S_WHITE_BALANCE=5000 ./scripts/capture.sh <scene>`
D455 notes: colour is a **global-shutter** sensor (no rolling-shutter skew, so pans blur less than on the D435i) with a wider ~90 deg HFOV; depth min range is ~0.5 m (keep >= 0.6-0.7 m from what you scan), and it stays accurate further out (95 mm baseline).
(Check the exposure unit on first use: if the image is black or white, the number is off by a decade.) Other overrides: `R2S_PREVIEW=0` (no window), `R2S_COLOR_PROFILE`, `R2S_DEPTH_PROFILE`, `R2S_LAUNCH_ARGS="..."`.

