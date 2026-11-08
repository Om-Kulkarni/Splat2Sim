# Splat2Sim

Turn a handheld RealSense D455 **Capture** of the FR3 workcell into a **Scene** (Gaussian-splat visual layer +
collision layer) that loads in Isaac Sim 6.1 / Isaac Lab 3.0, and a **Task environment** with the FR3 + Robotiq 2F-85
and cubes. Vocabulary: `CONTEXT.md`.

Status: `ingest`, `mask`, `poses`, `anchor`, `splat`, `compose`, `report` (M1 on mock data) and `collision`, and `env` (M2, Task environment +
pick-and-lift check) are real. Without a ChArUco
Anchor (`anchor` writes `status: fallback`), `compose` uses a fallback: gravity/dominant-plane alignment (`/World/Anchor` carries
`r2s:anchor = "fallback"`; with a board it carries `"charuco"` plus the residuals as `r2s:anchor_*` attributes).
Per-run options: `r2s build ... --quick` (3k-iteration splat) and `--set poses.use_imu=false` (saved in `<run>/options.yaml`;
sections `poses`, `splat`, see `splat2sim/recon/options.py`).
SLAM runs at ~10 fps (`poses.every: auto`: stride 3 on 30 fps; full rate takes >1 h on the 98 s mock capture).
`splat/metrics.json` holds the held-out PSNR/SSIM/LPIPS from 3DGRUT's `metrics.json` (mean over the test split), gaussians,
wall time and whole-GPU VRAM peak; `splat/heldout/*.png` are [render | ground truth] pairs. 30k iterations on the 3090 Ti
take ~26 min (about 1M Gaussians); `--quick` (3k) gives ~20 dB in ~2 min. `poses/colmap/` holds the 3DGRUT dataset (`sparse/0/*.txt`, `images/`).

## Quickstart (workstation; only docker needed)

```bash
./r2s doctor                                   # docker, GPU, images, data dir, .env (secret values never printed)
./r2s build ~/real2sim-data/captures/x.mcap --lab robolab [--scene NAME] [--from STAGE] [--to STAGE] [--force STAGE]
./r2s status <scene>                           # per-stage pending / done / stale
./r2s view <scene>                             # stub
make test                                      # pytest inside the r2s-cli container
make build-images
```

`./r2s` runs the CLI in the `r2s-cli` container (python:3.12-slim + pydantic/pyyaml/numpy, built on first use) with the
docker socket, the host `docker` binary + compose plugin, the repo and the data dir mounted **at identical host paths**
(the container drives the host daemon, so bind-mount paths must match). The repo is mounted, not baked in: code edits need
no rebuild. `R2S_REBUILD=1 ./r2s ...` rebuilds the image.

Alternative without the container: `R2S_MODE=uv ./r2s ...` (or `uv run r2s ...`), needs `uv` and Python 3.12 on the host;
stages still call `docker compose`.

Environment: `R2S_DATA` (default `~/real2sim-data`, a historical name kept so existing data keeps working), secrets in `$R2S_DATA/.env` (see `.env.example`; passed with
`env_file`, never in images).

## Layout

```
r2s                  host launcher (bash)
compose.yaml         services: r2s-cli, isaac, ingest (real); sam3 rtabmap colmap 3dgrut collision (placeholders, profile "todo")
docker/<service>/    Dockerfile per service (isaac/scripts is mounted into the isaac container)
labs/<lab>.yaml      per-lab config
splat2sim/            config.py (schema+merge) runfolder.py (contract) stages.py (registry) cli.py
tests/
spikes/isaac-smoke/  Isaac Sim 6.1 smoke-test scripts
```

## Run folders and stages

```
~/real2sim-data/runs/<scene>/{frames,masks,poses,splat,collision,scene,report}   (+ env/, run.json)
```

Stages in order: `ingest -> mask -> poses -> anchor -> splat -> collision -> compose -> report -> env`
(`splat2sim/stages.py` maps each to a compose service + command and lists dependencies).

* On success a stage writes `<dir>/.done.<stage>.json` (timestamp, inputs hash, git rev, run id). `poses` and `anchor`
  share `poses/`, hence per-stage marker names.
* Inputs hash = capture fingerprint (ingest only) + the lab-config sections the stage uses + the run ids of its
  dependencies. `r2s build` skips a stage whose marker matches (`done`), reruns `stale` ones (inputs changed, or an
  upstream stage was rerun) and `pending` ones.
* `--force STAGE` (repeatable) clears that marker and reruns it; downstream stages become stale automatically.
* `--dry-run` prints the `docker compose run` commands. Stub stages exit with code 3 and "not implemented yet".
* `run.json` records capture path, lab and override so `r2s status` can compute staleness.

## Ingest stage (`./r2s build <capture> --lab robolab --to ingest`)

Service `ingest` (`docker/ingest`, python:3.12-slim + rosbags/mcap/mcap-ros2-support/opencv-headless; no ROS). The
`splat2sim/` package is bind-mounted (no rebuild on code edits); `r2s build` mounts the capture read-only at `/capture`
and runs the container as your uid. Format is autodetected (magic bytes / folder layout):

| Capture | Reader notes |
|---|---|
| ROS 2 MCAP (our capture script; file or rosbag2 dir) | topics found by suffix (`*/color/image_raw`, `*/aligned_depth_to_color/image_raw`, `*/imu` or `*/gyro/sample`+`*/accel/sample`, camera_info, `/tf_static`). Aligned depth is used; if absent, raw `*/depth/image_rect_raw` is registered to colour with camera_info + the `extrinsics/depth_to_color` topic or `/tf_static`. |
| librealsense `.bag` (ROS 1, e.g. Intel sample) | unaligned depth (1280x720) registered into the colour frame (z-buffer reprojection) with stored intrinsics/extrinsics; separate Accel/Gyro topics united by interpolation onto the faster stream. |
| `frames/` folder (synthetic, `tools/synth_capture`) | `color/ depth/ frames.csv intrinsics.json [gt_poses.csv, metres]`: copied through the ingest contract (reader `frames_reader.py`). |
| TUM folder (SLAM&Render): `rgb/ depth/ associations.txt imu.txt groundtruth.txt` | depth assumed aligned, 16-bit mm (`--depth-factor` for other units). **Intrinsics are not in the folder**: give them in `<capture>/intrinsics.yaml|json` or in `runs/<scene>/ingest.yaml` (`intrinsics: path`, relative paths resolve in the run folder). Without either, nominal D435 values are used and flagged (`intrinsics.json` `source`). The dataset's Drive index lists only the 40 zips, not `intrinsics.yaml`. |

Output `runs/<scene>/frames/`: `color/NNNNNN.png` (8-bit RGB), `depth/NNNNNN.png` (uint16 mm, aligned to colour, 0 = invalid),
`frames.csv` (index, timestamp_ns, color, depth, sharpness), `intrinsics.json` (fx fy cx cy width height distortion_model
distortion_coeffs depth_scale_m=0.001 source), `imu.csv` (timestamp_ns, gx gy gz [rad/s], ax ay az [m/s^2]; clipped to the
frame range +-0.5 s), `keyframes.txt`, `gt_poses.csv` when present (OptiTrack, **translation in mm**, not converted),
`meta.json` (format, source, serial/firmware if recorded, durations, frame/IMU rates, dropped/gap stats, depth validity),
`preview/overlay_*.png` (depth edges over colour + colourised depth: eyeball the alignment).

Keyframes: all frames are always written; `keyframes.txt` lists the indices downstream stages should use.
Options (CLI `python -m splat2sim.ingest ingest --help`, or `runs/<scene>/ingest.yaml` with the same keys, which is part of
the stage's inputs hash): `every: N`, `min_sharpness: T` (variance of the Laplacian; with `every` the sharpest frame
in each window is taken), `format`, `intrinsics`, `depth_factor`, `depth_mode: auto|aligned|register`, `max_frames`.

Tests: `make test` (r2s-cli; ingest unit tests needing cv2/mcap skip there). Full run in the ingest image:
`docker run --rm -v $PWD:/repo -w /repo -e PYTHONPATH=/repo --entrypoint python r2s-ingest:latest -m pytest -q tests`.

## Anchor stage (`./r2s build ... --to anchor`)

Service `ingest` (python:3.12-slim + opencv-contrib 4.10 + scipy; `python -m splat2sim.anchor anchor`, package
`splat2sim/anchor/`). Detects the ChArUco board (`charuco` in the lab config) in the posed frames of `poses/poses.csv`
(colour intrinsics, up to 400 evenly spaced frames), back-projects each corner with the aligned depth (5x5 median,
corners on depth edges or without valid depth rejected), fits board -> recon with Umeyama (with scale) inside RANSAC
(4 cm inlier threshold), then refines scale, R, t by minimising the soft-L1 corner reprojection error over all frames
(plus a weak depth-consistency term). The result composes with `board_to_base` into recon -> robot base.

`poses/anchor.json`: `status` (`charuco` | `fallback` | `failed`) and `reason`; for `charuco` also `scale`, `R`, `t`,
`T_base_recon` (4x4 with scale, `p_base = s R p_recon + t`; this is what `compose` and `usdutil.anchor_transform`
read), `recon_to_board`, `board_to_base`, `frames_used` / corner counts, `ransac_inliers`, `reproj_rms_px`,
`corner_3d_rms_mm` (depth-backprojected corners vs the board model, board mm), `scale_estimate`, `warnings`. Scale is
recon-units-to-metres: RTAB-Map is metric, so a deviation > 5 % warns (wrong `square_length_m`, detection or depth problem).
`poses/anchor_overlays/` shows detected vs reprojected corners. `fallback` = no board seen (mock captures); `failed` =
board seen but too few consistent frames or reprojection RMS > 5 px. Both exit 0, and `compose` then uses the gravity/plane
fallback, so board-less captures still build. Options (`options.yaml` section `anchor`, or `--set anchor.fix_scale=true`):
see `ANCHOR_DEFAULTS` in `splat2sim/anchor/stage.py`.

Where the anchor is applied: **at compose time only** (`/World/Anchor` xform; collision applies the same transform to its
mesh). `splat` trains in the recon frame, so it does not depend on `anchor`: 3DGUT is invariant to a similarity, re-measuring
`board_to_base` or fixing a board problem re-runs only the 1-minute anchor + compose instead of a 30 min training, and
the splat, its COLMAP poses and the collision mesh stay in one common frame until the final transform. Board frame
(`board_to_base`): origin at chessboard corner 0 (one square in from the board's outer corner), x along `squares_x`,
y along `squares_y` (down the printed face), z into the board.

Synthetic data with ground truth: `python tests/synth_charuco.py <run-dir> --frames 80` writes a ChArUco Capture in the frames
contract + `poses/poses.csv` + `lab.json`; run the stage with `python -m splat2sim.anchor anchor --run <run-dir>` and score it
with `python tests/synth_charuco.py <run-dir> --eval`. Unit + end-to-end tests: `tests/test_anchor.py` (need cv2/scipy: run in
the `r2s-ingest` image).

## Collision stage (`./r2s build ... --to collision`)

Service `collision` (`docker/collision`, python:3.12-slim + Open3D 0.19.0 CPU + usd-core; separate from the Isaac image because
it needs no GPU/Isaac and a small image rebuilds in a minute; Open3D is pinned because 0.20.0's `ScalableTSDFVolume` extracted
empty meshes). Masked depth (`masks/`, 255 = keep) from frames that have poses is integrated into a TSDF (1 cm voxel, depth
truncated at 2.5 m, every 8th posed frame held out) in the **recon frame**; the mesh is mapped into the anchor frame
(`poses/anchor.json` when `status == "charuco"` (`T_base_recon`), else `fallback_anchor.json`, the same choice as compose) for
fitting. Outputs in `collision/`: `tsdf_mesh.ply`, `collision_mesh.ply`/`.usd` (table top carved out, islands < 100 triangles
dropped, decimated to 60k), `primitives.json` (`table_box`, `pedestal_box`: center, size, rpy_deg/yaw_deg in the anchor frame = world,
`source` = ransac | override | absent, fit stats), `collision.usda`, `metrics.json` (held-out depth-to-collision error, plane RMS,
coverage; copied into `report/summary.json`).

* Table: dominant near-horizontal plane of the up-facing mesh vertices (RANSAC), box = largest fully observed rectangle in that
  plane (`table_extent: bounding` for the min-area rectangle), top face on the plane (follows its tilt; `table_level: true` for
  roll = pitch = 0), 4 cm thick. With the ChArUco anchor the plane is searched in z in [-0.5, 0.3] m around the robot base; with the
  fallback anchor an elevated plane (>= 25 cm above the floor-like dominant plane, below the cameras) is preferred, because the
  fallback origin sits on the dominant plane (often the floor). Override with `table_z_range`.
* Pedestal: only with a ChArUco anchor (robot base = origin): the horizontal patch near the origin, extending down to the table top.
  Fallback anchor: `absent` unless `overrides.pedestal_box` is set. `overrides.table_box/pedestal_box` (lab config, rpy in degrees) replace the fits (`source: override`).
* `collision.usda` (default prim `/Collision`, **recon frame**; compose references it at `/World/Anchor/Collision`, so one Anchor
  transform moves splat and colliders): `Mesh` with `CollisionAPI` + `MeshCollisionAPI` (approximation `none`), `TableBox` /
  `PedestalBox` as `Cube` prims with colliders; all `visibility = invisible`, no rigid bodies (static).
* **Robot clearance** (charuco anchor only; robot base = origin): mesh triangles with a vertex inside a vertical cylinder around the base axis
  (`clearance_radius_m` 0.25, from pedestal top + `clearance_z_margin_m` 0.02 up to `clearance_z_max_m` 1.2) or inside capsules around the
  arm + gripper at `robot.scan_pose_joints` (`clearance_arm`, `clearance_arm_radius_m` 0.10, `clearance_gripper_length_m` 0.22; FK in
  `splat2sim/robot_fk.py`) are removed: mask leaks, dilation borders and cables near the arm must not collide with the real robot. The boxes are kept.
  What was removed is in `metrics.json` (`collision_mesh.robot_clearance`: counts, volume, bbox). Set defaults per lab in the lab config
  (`collision:` section, same keys) or per run with `--set collision.clearance_radius_m=0.3`; `--set collision.clearance=false` disables it.
  `carve_clutter_m` (0.12) / `carve_clutter_radius_m` (0.8) also drop small things lying on the table near the base (the cubes of the Capture stay in the
  splat but must not become collision bumps under the cubes the env spawns).
* Options (`--set collision.<key>=...`, defaults in `splat2sim/collision/stage.py`): `voxel_m`, `depth_trunc_m`, `max_integrate_frames`,
  `target_triangles`, `table_extent`, `table_level`, `table_z_range`, `include_mesh` (false = boxes only), `carve_table`, ...
* Check (Isaac, on demand): `docker compose run --rm -e R2S_SCENE=<scene> isaac -p /workspace/r2s/collision_check.py` drops 4 rigid
  5 cm cubes over free spots of the table box in `scene.usda` (in memory only), writes `collision/check.json` (rest height above the
  table plane, pass if within 1 cm) and previews (`preview_cubes.png`, `preview_debug_*.png` with the colliders made visible over the splat).
* With a fallback-aligned scene there is no robot at the origin: the mesh (floor, walls) can intrude on the arm and the fitted
  table is outside its reach, so the `env` stage needs a ChArUco anchor (or `overrides.table_box` plus `--set collision.include_mesh=false`).

## Report stage (`./r2s build ... --to report`)

Three steps (`Stage.post` in `splat2sim/stages.py`): `docker/isaac/scripts/report.py` (previews + fly-through PNG frames),
`robot_silhouette.py` (Isaac Lab: the robot alone at `robot.scan_pose_joints` rendered from 24 keyframe cameras after the ChArUco anchor,
`report/silhouette/sil_*.png`), then `python -m splat2sim.report finish` in the **ingest** image (CPU; `imageio-ffmpeg` there encodes
`report/flythrough.mp4` because the Isaac image has no ffmpeg backend). Outputs: `report/summary.json`, `report/report.md`
(verdict, headline numbers, all checks with levels `ok|warn|redo|fix`, previews), `flythrough.mp4`, `coverage_table.png`,
`silhouette_overlay_*.png`. Checks and thresholds: `splat2sim/report/summary.py` (`THRESH`), explained in `docs/first-capture.md`.
`mask_vs_urdf` = SAM masks vs the URDF silhouette (fraction of the silhouette inside the mask, IoU against the dilated silhouette): it validates
`board_to_base`, the scan pose and the mask together. `table_coverage` = fraction of the fitted table top seen (depth-consistent) from >= 1/3/5 keyframes.
The `env` stage re-runs the CPU step so `report/` also carries the pick-and-lift result and the reset safety check.

## Synthetic Capture with ground truth (`tools/synth_capture/`)

`tools/synth_capture/run.sh --frames 600 --out /captures/synth-robot` renders (Isaac Sim, ~2.5 min on the 3090 Ti) a handheld-like walk around
the FR3 + 2F-85 on a pedestal, a table with the ChArUco board (true `board_to_base` in `labs/synth.yaml`), 3 cubes, clutter and a textured
room, in the `frames/` contract with `gt_poses.csv`, `gt/robot_mask/` and `synth_gt.json`. `./r2s build $R2S_DATA/captures/synth-robot --lab synth --scene synth-robot`
runs the whole pipeline on it; `tools/synth_capture/evaluate.py` scores anchor / masks / boxes against the truth (see the file header).
Dry run on the 3090 Ti (600 frames 848x480, `--quick`, ~11 min end to end): SLAM ATE 3-4 mm; anchor 1.4 px reprojection, 7 mm mean / 13 mm max placement error over the
workcell vs the best Sim(3) of the SLAM trajectory (0.33 deg, scale 0.990 vs 0.999); table box top +8 mm, footprint IoU 0.96; SAM vs the rendered robot: 85 % of its pixels
covered (IoU 0.82 vs the dilated truth); URDF silhouette inside the mask 1.00, IoU 0.93; splat 24.9 dB at 3k iterations; pick-and-lift with the fitted layer:
mesh + clearance 3/5, boxes only 3/5, **mesh without clearance 0/5** (mesh touches link 0-2 and the fingers at reset; the safety check reports it), robot left in the
mesh (no masks) 0/5 without and 3/5 with the clearance volume.

## Lab config

`labs/<lab>.yaml`, schema in `splat2sim/config.py` (units: m, kg, rad for joints, degrees for rpy/fov; a pose is
`xyz` + `rpy` with `R = Rz*Ry*Rx`, z-up, world = FR3 base). Sections: `charuco`, `board_to_base`, `robot`
(scan pose, gripper, `coupling_offset`), `policy_cameras` (wrist, front), `cubes`, optional `overrides`
(`table_box`, `pedestal_box`), optional `collision` (clearance defaults), optional `env` (`episode_length_s`, `reach_min_m`, `reach_max_m`, `spawn_margin_m`). A per-run file (`--override file.yaml`) is deep-merged on top (dicts merge, lists replace).

Add a lab: copy `labs/robolab.yaml` to `labs/<name>.yaml`, measure the board square length and `board_to_base`, then
`./r2s build <capture> --lab <name>`. Values marked TODO in `robolab.yaml` are placeholders.

## Task environment (`env` stage, `Splat2Sim-FR3-Cubes-v0`)

Code: `splat2sim/lab/` (mounted into the isaac container with the rest of `splat2sim/`, importable by training code),
`docker/isaac/scripts/check_env.py` (the stage). `geometry.py` is pure numpy (tests: `tests/test_lab_env.py`); `asset.py`,
`env_cfg.py`, `mdp.py`, `scripted.py` need Isaac Lab.

* **Robot** (`asset.py`): `fr3.usd` with its Franka hand deactivated + the bundled `Robotiq_2F_85_edit.usd` (same file and `Physx_Mimic`
  variant NVIDIA's Panda variant uses), joined by a fixed joint `fr3_link8 -> base_link` carrying `robot.coupling_offset`. Built once into
  `<run>/env/assets/fr3_robotiq_2f85-<hash>.usda` (hash = coupling offset + asset URLs; references the remote 6.1 assets). Actuators are cloned
  from Isaac Lab's `FRANKA_ROBOTIQ_GRIPPER_CFG` (`finger_joint` driven K=17 D=0.02, `.*_inner_finger_joint`, passive knuckles with zero gain,
  `right_outer_knuckle_joint` mimic from the USD); deviations: `fr3_joint*` names, FR3 velocity limits (2.62 / 5.26 rad/s), home pose = `robot.scan_pose_joints`,
  base at the world origin. The bundled 2F-85 grasped 5 cm cubes fine, so the UWLab calibrated gripper was not needed.
* **Scene**: `<run>/scene/scene.usda` is referenced at `{ENV_REGEX_NS}/Scene` (anchor at the env origin). The table the cubes rest on is a static
  collider from `read_table_box()` (`collision/primitives.json` -> `overrides.table_box` -> default 1.2 x 0.8 m at z = 0), added unless `scene.usda`
  already references the collision layer; it is invisible with a Scene (the splat is the visual), visible without one (`show_table`).
* **Cubes**: `cubes` cfg, re-sampled every reset inside the table box shrunk by `env.spawn_margin_m` and clipped to `env.reach_min_m..reach_max_m`
  from the base (`env` lab section, optional), non-overlapping, random yaw.
* **Cameras**: `wrist` (child of `fr3_link8`) and `front` (child of `fr3_link0`), RGB + `distance_to_image_plane`, poses/intrinsics from
  `policy_cameras` with Isaac Lab's `"world"` offset convention (+x forward, +z up == the lab-config frame). `check_env` projects every cube through the
  configured pose + FOV and compares with the cube's colour centroid in the render (`summary.json:camera_check`; 0.2-2.8 px on the no-scene run).
* **API**: manager-based `ManagerBasedRLEnv`, 30 Hz control (120 Hz physics). Action `(num_envs, 8)`: absolute `fr3_joint1..7` targets [rad] + `finger_joint`
  target [rad] (0 open, ~0.4 closed on a 5 cm cube, limit 0.82). Obs group `policy` (dict, not concatenated): `joint_pos` (8), `joint_vel` (8), `cube_pos` (3n),
  `cube_quat` (4n, xyzw), `wrist_rgb`/`front_rgb` (480,640,3 float 0..1), `wrist_depth`/`front_depth` (480,640,1 m). Placeholder zero-weight reward, time-out only.

```python
from splat2sim.lab.env_cfg import Splat2SimFR3CubesEnvCfg       # inside launch_simulation(...), after cfg.ensure_robot_asset()
cfg = Splat2SimFR3CubesEnvCfg.from_lab("/runs/<scene>/lab.json", "/runs/<scene>", num_envs=1)   # or gym.make(ENV_ID) with $R2S_SCENE/$R2S_LAB
env = gym.make("Splat2Sim-FR3-Cubes-v0", cfg=cfg)
```

`./r2s build <capture> --lab robolab --scene S --from env --to env` runs `check_env.py`: 5 resets, each a scripted pick (`scripted.py`: Lab's differential IK
to a top-down TCP pose above the cube, descend, close, lift 10 cm, hold 1 s). Success = cube centre >= 8 cm above its rest height for the whole hold and within
5 cm of the TCP. Writes `env/summary.json` (success rate, per-trial numbers, obs/action shapes, timings, VRAM), `env/preview/{wrist,front}_{rgb,depth}_*.png`,
`overview_*.png`, `pick_lift.gif` (no mp4: the Isaac image has no ffmpeg backend for imageio). Exit code 1 below `--min-success` (0.6). Direct run:
`docker compose run --rm -e R2S_SCENE=S isaac -p /workspace/r2s/check_env.py [--no-scene] [--trials N] [--out DIR] [--show-table]`.
The 6.1 assets come over `ISAACSIM_ASSET_ROOT` (HTTP cache in `cache/isaac/ov`).
Results on the workstation (3090 Ti): 5/5 with the m1-demo Scene (table collider at z = 0, `overrides.table_box` 1.2 x 0.8 m), 8/8 and 5/5 without; ~11 env steps/s with two
640x480 policy cameras + an overview camera, ~6.5-7 GB whole-GPU VRAM peak; a 1-env run takes ~5 min cold (shader/asset caches warm: ~25 s start + ~2.5 min trials).

## Isaac image

The `isaac` service uses the locally built `r2s-isaaclab3-sim61` (Isaac Lab 3.0 `release/3.0.0` `docker/Dockerfile.base`
with `ISAACSIM_VERSION_ARG=6.1.0`; build commands in `docker/isaac/Dockerfile`). Flags (EULA, asset root pinned to 6.1,
cache volumes, GPU reservation) are in `compose.yaml`; `./r2s` creates the world-writable cache dirs under
`$R2S_DATA/cache/isaac`. For CDI replace the `deploy` block with `devices: ["nvidia.com/gpu=all"]`
(not available on the workstation (RTX 3090 Ti): no `/etc/cdi`; use `--gpus all`/the `deploy` block, verified there).
Build tip: pin Isaac Lab to commit `5870804` (`git fetch --unshallow`, `git checkout 5870804`; the branch head has moved)
and add `ENV UV_HTTP_TIMEOUT=900` to `docker/Dockerfile.base`. Rebuilt on the workstation on 2026-10-18 (51.8 GB, about 10 min with
fast bandwidth ~25 MB/s); `spikes/isaac-smoke/run_smoke.sh` and `run_lab.sh` pass on the 3090 Ti (driver 580.178.04).

## License

MIT, see `LICENSE`.
License: MIT
