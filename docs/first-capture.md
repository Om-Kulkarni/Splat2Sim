# First real Capture: checklist (Monday 2026-10-19)

Handheld walk-around (OAK-D Pro W; D455 via `R2S_CAMERA=realsense`) of the lab: FR3 + Robotiq 2F-85 on a pedestal next to a table, printed ChArUco board on the table, 3D-printed cubes.
Read `docs/capture-guide.md` for the camera/walking details; this page is the order of work and the numbers that matter.
Everything below was dry-run on a synthetic Capture with ground truth (`tools/synth_capture/`, `labs/synth.yaml`); numbers marked (synth) come from it.

## 0. The evening before (5 min, workstation)

```bash
ssh $WORKSTATION_USER@$WORKSTATION_HOST
cd ~/real2sim-dev && git -C . log --oneline | head -1           # or rsync the branch you want to run: rsync -a --delete --exclude .git --exclude .env <repo>/ <ws>:~/real2sim-dev/
./r2s doctor                                                    # docker, GPU (nvidia-smi must show it free: shared machine), 7 images, .env has HF_TOKEN
nvidia-smi --query-gpu=memory.used --format=csv,noheader        # < 1 GB used before a GPU stage
df -h ~/real2sim-data | tail -1                                 # a 3 min take is ~10-20 GB as MCAP, ~15 GB more as PNG frames
```
Needed images: `r2s-cli r2s-ingest r2s-sam3 r2s-rtabmap r2s-3dgrut r2s-collision r2s-isaaclab3-sim61`. `r2s-ingest` must be the build with `imageio-ffmpeg`
(`docker run --rm --entrypoint python r2s-ingest:latest -c "import imageio_ffmpeg"`), else the fly-through is a GIF.
Laptop on mains power: `./scripts/capture.sh --check` (USB **3.x** link), print `docs/charuco-board-a3.pdf` at 100 %.

## 1. What to measure in the lab (before the take; 20 min)

Put the values in a copy of `labs/robolab.yaml` (e.g. `labs/robolab-1019.yaml`, run with `--lab robolab-1019`). Every number has a unit and a frame in `splat2sim/config.py`.

| What | How | Goes in | Why it matters / tolerance |
|---|---|---|---|
| **Board square length** | steel rule or calipers over all 7 squares of the printed + mounted board, divide by 7 (also the marker side: 41 mm nominal) | `charuco.square_length_m` (`marker_length_m`) | sets the metric check of the anchor: the fitted recon->board scale is reported; > 2 % off (`anchor_scale_deviation`) means a wrong length or depth problem. 0.5 mm error per 55 mm = 1 %. |
| **board_to_base** | touch-off: pointed tip in the gripper (pivot-calibrate the tip first), hand-guide to >= 4 non-collinear board corners marks, record FK tip positions in the base frame, rigid fit; residual = your real error. Fallback: tape measure + square. Board frame: origin = ChArUco corner id 0 (first inner corner, marked on the print), x along the 7-square side, y down the printed face, **z into the board**: a flat, face-up board has roll 180 deg | `board_to_base.xyz` (m), `.rpy` (deg, R = Rz Ry Rx) | everything in the Scene is placed relative to this: 5 mm here = 5 mm table offset in the env; yaw error of 1 deg = 9 mm at 0.5 m. The report's mask-vs-URDF silhouette IoU is the independent check of this number. |
| **Scan pose** | read the 7 joint angles in Desk at the pose you leave the arm in for the take (do not move it during the take) | `robot.scan_pose_joints` (rad) | the env resets to it, the URDF silhouette and the robot-clearance capsules use it. A wrong pose shows up as low silhouette IoU. |
| **Coupling offset** | flange -> 2F-85 base: plate thickness along the flange z (the default 8 mm is a placeholder), any xy offset, and the **gripper yaw about the flange z** (which way the fingers close relative to link 7's axes; typically 0 / 90 / 45 deg) | `robot.coupling_offset.xyz` (m), `.rpy` (deg) | gripper placement in the env and in the silhouette; a wrong yaw rotates the whole hand (visible in the silhouette overlay of the gripper). |
| **Cube size / mass** | calipers on 3 cubes, kitchen scale | `cubes.size_m`, `cubes.mass_kg`, `cubes.count` | grasp success (2F-85 stroke 85 mm: <= 6 cm is fine) and the finger target. |
| Pedestal / table (optional) | tape: pedestal top is the robot mounting plane (z = 0 by definition); table top height below it and table extents | `overrides.table_box`, `overrides.pedestal_box` (center, size, rpy; base frame) | only if the report flags the fitted boxes (below); overrides replace the fits. |
| Camera poses | not needed (assumed, out of scope) | `policy_cameras` | -- |

Also leave the **lights on, blinds closed**, cubes on the table where they will be during the take (they stay in the splat; they are carved out of the collision mesh).

## 2. Record (laptop), transfer

```bash
./scripts/capture.sh --check                 # OAK-D Pro W: product, sensors, USB SUPER link, calibration (R2S_CAMERA=realsense for the D455)
./scripts/capture.sh --preview               # tune exposure (bright, not clipped, no blur; short exposure, rolling shutter); pin: R2S_OAK_EXPOSURE_US=.. R2S_OAK_ISO=.. R2S_OAK_WB_K=.. or edit docker/capture/oak_params.yaml
./scripts/capture.sh robolab-tableA          # -> ~/real2sim-captures/robolab-tableA-<timestamp>/ ; wait for RECORDING, walk, Ctrl-C once
rsync -avP --partial ~/real2sim-captures/robolab-tableA-<timestamp> $WORKSTATION_USER@$WORKSTATION_HOST:~/real2sim-data/captures/
```
Walk: 2-3 slow orbits (knee / chest / head), one slow close pass over the table looking down **with the board in view for >= 30 s in total**
(20-50 deg oblique and frontal, 0.8-2 m; the OAK has no depth closer than ~0.7 m; walk slower than with a global-shutter camera; check the take summary for ~30 Hz colour/depth and ~400 Hz IMU), dwell ~10 s at the wrist- and front-camera viewpoints, revisit the start. Do not touch the arm, board or cubes.

## 3. Build (workstation)

```bash
cd ~/real2sim-dev
CAP=~/real2sim-data/captures/robolab-tableA-<timestamp>
mkdir -p ~/real2sim-data/runs/rl-a && printf 'every: 3\n' > ~/real2sim-data/runs/rl-a/ingest.yaml    # 10 fps keyframes: SLAM/mask/anchor use these; saves ~2/3 of mask time

# a) first look: ingest -> mask -> poses -> anchor, then decide before the long stages
./r2s build $CAP --lab robolab-1019 --scene rl-a --to anchor
less ~/real2sim-data/runs/rl-a/poses/anchor.json                 # status must be "charuco"; reproj_rms_px, corner_3d_rms_mm, scale_estimate
ls ~/real2sim-data/runs/rl-a/masks/preview/                      # eyeball the robot masks
# b) the rest. --quick = 3k-iteration splat (about 2 min): use it for the first pass
./r2s build $CAP --lab robolab-1019 --scene rl-a --quick
less ~/real2sim-data/runs/rl-a/report/report.md                  # verdict + numbers (previews, fly-through.mp4, coverage)
# c) when the numbers are good: the real 30k splat (this must set the option back: --quick is saved in options.yaml)
./r2s build $CAP --lab robolab-1019 --scene rl-a --set splat.iterations=30000 --force splat
```
`./r2s status rl-a` shows done / stale / pending per stage. A change in the lab yaml (`board_to_base`, scan pose, ...) makes exactly the dependent
stages stale; `--force STAGE` reruns one. `board_to_base` re-measured -> `./r2s build ... --from anchor` (anchor, collision, compose, report, env: the splat is not retrained).
Useful knobs: `--set anchor.fix_scale=true` (trust RTAB-Map's metric scale, only R, t estimated), `--set collision.clearance_radius_m=0.3`,
`--set collision.include_mesh=false` (boxes only), `--set poses.use_imu=false`, `--set mask.*` via the lab yaml `mask:` section (`dilate_px`, `prompts`).

## 4. Which report numbers mean "redo the take" (`report/report.md`, `report/summary.json: verdict`)

Capture-quality checks (level `redo` flips `verdict.redo_take`; `warn` = look, probably fine). Thresholds are in `splat2sim/report/summary.py` (`THRESH`).

| Number | OK | warn | redo the take |
|---|---|---|---|
| `anchor.status` | `charuco` | -- | `fallback` (board never detected) or `failed`: the scene is not aligned to the robot. Recheck lighting/focus/print, show the board more. |
| Board in view (`board_seen_s`, estimate) | >= 20 s | 10-20 s | < 10 s |
| `anchor reproj_rms_px` | <= 2 | 2-4 | > 4 (5 = `failed`) |
| `anchor corner_3d_rms_mm` | <= 15 | 15-30 | > 30 (depth/pose trouble at the board) |
| anchor scale deviation \|s - 1\| | <= 2 % | 2-5 % | > 5 %: wrong `square_length_m`, or depth/SLAM scale broken. |
| anchor frames used | >= 8 | 3-8 | < 3 |
| RTAB-Map registered / total | >= 90 % | 70-90 % | < 70 % (fast pans, blur, texture-less walls); 0 loop closures = warn |
| Frames where SAM found the robot | >= 60 % | 20-60 % | < 20 %: the robot is hardly in view, or the prompt fails: check `masks/preview` |
| Table top seen from >= 3 keyframes | >= 80 % | 50-80 % | < 50 %: do a slow second pass over the table |
| Held-out PSNR (30k iterations; ~24 dB at 3k on the synthetic) | >= 22 dB | 18-22 | < 18: blur, exposure changes, moving things |

Config checks (level `fix`: the Capture is fine, a measurement or option is not):

| Number | OK | `fix` | Typical cause |
|---|---|---|---|
| `mask_vs_urdf`: URDF silhouette inside the SAM mask (median) | >= 0.9 | < 0.75 warn-> fix | wrong scan pose / `board_to_base` / coupling offset, or the arm moved during the take |
| `mask_vs_urdf`: IoU vs dilated silhouette | >= 0.5 | < 0.3 | same; (synth truth: 1.00 / 0.94) |
| table plane RMS | <= 8 mm | > 15 mm | table not flat / fit on the wrong plane; set `overrides.table_box` |
| held-out depth to collision geometry, median | <= 2 cm | > 4 cm | board_to_base far off, or a large reconstruction hole |
| clearance removed share of the collision mesh | <= 25 % | > 60 % | masks leaked a lot of robot, or clearance radius too large |
| env: reset safety warning | none | any | collision mesh touches a robot link at the scan pose (fragments near the arm) |
| env: pick-and-lift success | >= 3/5 | < 3/5 | fix the above first; `check_env.py` previews show where it fails |

Also look (no automatic threshold): `poses/anchor_overlays/*.png` (green detected vs red reprojected corners should coincide),
`report/silhouette_overlay_*.png` (green SAM mask outline around the red URDF outline, ~15 px apart all around), `report/flythrough.mp4`,
`report/coverage_table.png` (red = table cells nobody saw), `splat/heldout/*.png`.

## 5. Expected timings (RTX 3090 Ti, one job at a time)

Measured on the synthetic 600-frame 848x480 take (20 s, 30 fps) unless noted; the real 3 min take at 1280x720 colour has ~9x the frames
(5400; ~1800 keyframes with `every: 3`), so scale the per-frame stages and add the splat-training difference.

| Stage | synthetic (600 fr) | real take estimate (~1800 keyframes) |
| ingest | 44 s | 10-15 min for 5400 frames (PNG encode; 1280x720 colour). `every: 3` does not shorten ingest, only mask/SLAM. |
| mask (SAM 3.1, 14 GB VRAM) | 174 s (0.29 s/frame, 300-frame chunks) | ~9 min for 1800 keyframes (~30 min if every frame is a keyframe) |
| poses (RTAB-Map, CPU) | 23 s (200 SLAM frames) | 5-10 min |
| anchor | 5 s | < 1 min |
| splat `--quick` (3k it.) / full (30k it.) | 95 s / (not run) | 2-3 min / ~26 min for 1M Gaussians (measured on the mock take) |
| collision | 11 s | 1-2 min |
| compose | 20 s | 20-30 s |
| report (previews, 144 fly-through frames, URDF silhouettes, MP4, coverage) | 97 s | 2-3 min |
| env (5 pick-and-lift trials, 4.2 GB VRAM) | 175 s | 3 min |

Total synthetic `r2s build --quick`: 11 min (15:01 -> 15:13). Plan on **about 1 h to the first report** for a real take with `--quick` (stop after `--to anchor` in ~15 min to check the anchor), and +25-30 min for the 30k splat (~26 min for 1M Gaussians measured on the SLAM&Render mock). GPU memory peaks: SAM 3.1 14 GB, 3DGRUT 2.5-8 GB, Isaac ~7 GB (check `nvidia-smi` first: the machine is shared).

## 6. If something goes wrong

* `anchor` = `fallback`/`failed`: look at `poses/anchor_overlays`, and the board frames in `frames/color`; reshoot with the board in view longer, glare-free; `--set anchor.min_corners=4` as a last resort.
* mask crashes / finds nothing: `masks/preview/`; `--set`-less options live in the lab yaml `mask:` (`prompts: ["robot arm", "robotic gripper", "<more specific>"]`). The SAM click fallback is not implemented.
* collision mesh touches the robot: raise `--set collision.clearance_radius_m=0.3 --set collision.clearance_arm_radius_m=0.12`, rerun `--from collision`.
* env `success` low but report good: run `docker compose run --rm -e R2S_SCENE=rl-a isaac -p /workspace/r2s/check_env.py --no-scene` to separate env problems from scene problems.
* A VPN/ssh drop does not kill `nohup` jobs: `nohup ./r2s build ... > ~/real2sim-data/logs/rl-a.log 2>&1 &`.
