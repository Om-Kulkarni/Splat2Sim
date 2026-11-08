# rtabmap service image (`r2s-rtabmap`)

Headless, CPU-only RTAB-Map **0.23.8** for offline RGB-D (+ optional IMU) camera pose estimation.

- Base: `introlab3it/rtabmap:noble` pinned by digest (`sha256:f49d2629...`, Ubuntu 24.04, ROS Jazzy base,
  OpenCV 4.6 + contrib, PCL 1.14, GTSAM 4.2, OpenGV; 6.4 GB on disk, 1.4 GB compressed pull).
  That image ships RTAB-Map **0.24.2**, so the Dockerfile rebuilds the pinned tag `0.23.8` over it
  (`make install` into `/usr/local`, ~2 min build; `rtabmap-console --version` reports 0.23.8).
- Patch `rgbd_dataset_generic.patch`: stock `rtabmap-rgbd_dataset` hard-codes TUM Kinect intrinsics, 5000/m depth and
  needs `rgb_sync/` + `depth_sync/` folders, so it cannot read a D435i folder. The patch adds
  `--fx --fy --cx --cy --width --height --depth_factor --rgb_dir --depth_dir --gt --imu --max_frames`
  (legacy TUM behaviour is unchanged when `--fx` is not given).
- No GPU needed. `QT_QPA_PLATFORM=offscreen` is set so Qt-linked tools (`rtabmap-report`, `rtabmap-export`) run headless.

Build: `docker compose build rtabmap` (or `docker build -f docker/rtabmap/Dockerfile -t r2s-rtabmap:latest docker/rtabmap`).

## Input contract

A folder with `rgb/<stamp>.png` and `depth/<stamp>.png` (same file names; depth 16-bit **millimetres** registered to the
colour image, i.e. the `ingest` layout) and optionally `imu.txt` (`t ax ay az gx gy gz`, m/s^2 and rad/s, `#` comments,
axes = camera optical frame as librealsense reports for the D435i). The tool writes `rtabmap_calib.yaml` into the input
folder, so mount it read-write (or stage a copy).

## Commands (verified)

```bash
D=~/real2sim-data/smoke/recon   # in/{rgb,depth,imu.txt}  out/
RUN="docker run --rm -u $(id -u):$(id -g) -v $D/in:/data/in -v $D/out:/data/out r2s-rtabmap"
COMMON="--fx 425 --fy 425 --cx 424 --cy 240 --width 848 --height 480 --depth_factor 1 \
  --rgb_dir rgb --depth_dir depth --Rtabmap/DetectionRate 0 --Rtabmap/CreateIntermediateNodes false \
  --RGBD/LinearUpdate 0 --RGBD/AngularUpdate 0 --output /data/out"

# 1. SLAM, RGB-D only (every frame becomes a node)
$RUN rtabmap-rgbd_dataset $COMMON --output_name noimu /data/in
# 1b. RGB-D + IMU (Madgwick orientation -> gravity-aligned world, graph gravity constraints)
$RUN rtabmap-rgbd_dataset $COMMON --imu /data/in/imu.txt --output_name imu /data/in
#   add  --gt gt.txt (TUM format: stamp tx ty tz qx qy qz qw, metres)  to store ground truth + RMSE in <name>_rmse.txt
#   add  --max_frames N  /  --skip N  to subsample

# 2. Export optimised poses (full global optimisation of the db graph)
$RUN rtabmap-export --poses --poses_camera --output_dir /data/out/export_imu /data/out/imu.db
#   *_poses.txt         world<-base  (RTAB-Map base frame: x forward, y left, z up)
#   *_camera_poses.txt  world<-camera optical frame (x right, y down, z forward)  <- use this for COLMAP/3DGRUT
#   columns: timestamp x y z qx qy qz qw id
# 3. Report vs ground truth
$RUN rtabmap-report --gt /data/out/gt_m.txt --gt_max_t 0.05 /data/out/imu.db
```

Notes
- Without `--RGBD/LinearUpdate 0 --RGBD/AngularUpdate 0` RTAB-Map only creates a node every 10 cm / ~6 deg, which yields
  very sparse poses (3 for 300 frames of slow motion). For a splat we want every (key)frame.
- `rtabmap-rgbd_dataset` also writes `<name>_poses.txt` (graph, base frame, TUM format) and `<name>.db`.
- Real intrinsics must replace the smoke values once `ingest` emits `intrinsics.json`. NOTE: fx=fy=425 (used in the 5090 smoke) is
  the *depth* sensor's focal length; SLAM&Render colour (848x480, depth registered to colour) is nominally fx=fy=609.5,
  cx=423.5, cy=239.5 (what `ingest` assumes). Use those: ATE improves and the path length matches OptiTrack (0.5 m).
- World frame: without `--imu` the world is the first camera pose; with `--imu` it is gravity-aligned (z up), still
  arbitrary in yaw/position (the ChArUco Anchor stage fixes that).
- IMU support here is a gravity/orientation prior (Madgwick filter + `Optimizer/GravitySigma`), as the research notes
  said; it is not tightly coupled VIO.

## Smoke test (2026-10-18, RTX 5090 workstation, CPU only; old machine, gone)

Input: SLAM&Render setup-1 `1-natural-tr`, 300 frames starting at frame 700 (the first ~10 s of the capture are static),
848x480, 30 Hz, approximate D435i intrinsics. Staged with copies of the PNGs (see `$D/in`).

| run | nodes | wall | ATE RMSE vs OptiTrack (rtabmap-report) |
|---|---|---|---|
| RGB-D | 300 | 36 s (+ 2 s container start) | 0.007 m over a 0.7 m path |
| RGB-D + IMU | 300 | 37 s | 0.007 m |

`rtabmap-export --poses --poses_camera` produced 300 poses each. (RMSE is position-only after alignment; angular
number from the report is meaningless because OptiTrack tracks a rigid body, not the camera frame.)

## Smoke test (2026-10-18, RTX 3090 Ti workstation, CPU only)

New workstation (16 cores). Image rebuilt from this Dockerfile (6.44 GB on disk). Same input as above:
`1-natural-tr` frames 700..999, staged from the raw TUM folder (timestamp-named PNGs), 300 frames, OptiTrack GT converted to m.

| run | intrinsics | nodes | wall | ATE RMSE (rtabmap-report/odom log) |
|---|---|---|---|---|
| RGB-D | 425 / 424 / 240 (old smoke values) | 300 | 77 s | 0.0065 m |
| RGB-D + IMU | 425 / 424 / 240 | 300 | 77 s | 0.0065 m (report: 0.007 m, path 0.7 m) |
| RGB-D | 609.5 / 423.5 / 239.5 (nominal colour) | 300 | 62 s | 0.0046 m |
| RGB-D + IMU | 609.5 / 423.5 / 239.5 | 300 | 58 s | 0.0046 m (report: 0.005 m, path 0.5 m) |

`rtabmap-export --poses --poses_camera` gave 300 poses. Wall times are slower than the 5090 box's 36 s because other image
builds were running on the same CPUs at the time (the 609.5 runs were quieter). Accuracy matches the old result.
