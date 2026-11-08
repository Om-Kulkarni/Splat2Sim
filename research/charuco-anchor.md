# ChArUco Anchor: scale and Robot-base alignment

Researched 2026-10-17.

Legend: **[S]** = verified against a primary source (URL given). **[E]** = my own engineering estimate or inference, not sourced; treat as a hypothesis to validate on the first real Capture.

## TL;DR

- Use an OpenCV `CharucoBoard` with `DICT_4X4_50`, 7x5 squares, printed on A3 on rigid, flat backing. Detect with `cv2.aruco.CharucoDetector`.
- Do not use per-frame PnP poses alone to get scale. Collect the board's 2D corner detections across many frames and solve one 7-DoF Sim(3) (reconstruction frame -> board frame) by robust reprojection-error minimisation, initialised with Umeyama. Then compose with the measured board->Robot-base transform.
- The weakest link is almost certainly the physical board->base measurement, not the vision. Prefer the "touch the corners with the robot" procedure below over a tape measure.
- No surveyed real2sim paper uses a ChArUco anchor for base alignment; the ones found align to the robot via ICP against URDF geometry. A hand-rolled marker-based scale tool (`aruco-estimator`) and a 2026 NeRF/GS paper using ArUco markers for scale exist. So this is a small build, not a reuse.

## 1. Board spec

### Facts about the tooling [S]
- `CharucoBoard(size, squareLength, markerLength, dictionary, ids)`: `size` = number of squares in x and y; lengths are in any unit and the pose comes out in the same unit (use metres). Header: https://raw.githubusercontent.com/opencv/opencv/4.x/modules/objdetect/include/opencv2/objdetect/aruco_board.hpp ; tutorial: https://raw.githubusercontent.com/opencv/opencv/4.x/doc/tutorials/objdetect/charuco_detection/charuco_detection.markdown
- Why ChArUco over plain ArUco: ArUco marker corners are less accurate even with subpixel refinement; ChArUco corners are interpolated chessboard corners ("very accurate in terms of subpixel accuracy") and still allow partial views/occlusion. Same tutorial.
- Pattern-generation changed for even row counts after OpenCV 4.6.0 (`setLegacyPattern`, https://github.com/opencv/opencv/issues/23152). Pin the OpenCV version and generate the printed image and the detector's board from the same code, or use an odd/odd grid such as 7x5 to sidestep the ambiguity [E: 7x5 is odd in both so the legacy flag should not matter, but verify by checking the rendered image against the detector once].
- Predefined dictionaries and their minimum Hamming distances (4X4_50 -> 4; 5X5_50 -> 8; 6X6_50 -> 13; ARUCO_MIP_36h12 -> 12, 250 codes): https://raw.githubusercontent.com/opencv/opencv/4.x/modules/objdetect/include/opencv2/objdetect/aruco_dictionary.hpp
- `DetectorParameters` defaults: `cornerRefinementMethod = CORNER_REFINE_NONE`, `cornerRefinementWinSize = 5`, `useAruco3Detection = false`, `minMarkerPerimeterRate = 0.03`; `CharucoParameters.minMarkers = 2`, and it accepts `cameraMatrix`/`distCoeffs` (corner interpolation is then pose-based rather than homography-based). https://raw.githubusercontent.com/opencv/opencv/4.x/modules/objdetect/include/opencv2/objdetect/aruco_detector.hpp , https://raw.githubusercontent.com/opencv/opencv/4.x/modules/objdetect/include/opencv2/objdetect/charuco_detector.hpp
- `CharucoBoard.checkCharucoCornersCollinear(ids)`: PnP/calibration fail if detected corners are collinear; call it before using a frame.
- D435i colour stream: 1920x1080, FOV 69.4 x 42.5 deg (+/-3 deg, +/-5% mechanical tolerance per unit) per Intel's support article https://intel.com/content/www/us/en/support/articles/000030385/emerging-technologies/intel-realsense-technology.html . Colour shutter type: reseller pages say rolling shutter; I did not confirm in Intel's datasheet [unverified; see uncertainties].

### Recommended spec [E, derived from the facts above]
Focal length estimate at 1080p: fx ~= 960 / tan(34.7 deg) ~= 1390 px. A marker of side m at distance d spans ~ 1390*m/d px.

| | A3 (preferred) | A4 (fallback) |
|---|---|---|
| Dictionary | `DICT_4X4_50` (6x6 modules incl. border; 4 bit Hamming distance is plenty for ~17 ids, and fewer modules means more pixels per module than 5x5/6x6) | same |
| Grid (squares x, y) | 7 x 5 (odd/odd) | 7 x 5 |
| Square | 55 mm | 40 mm |
| Marker | 41 mm (ratio ~0.75) | 30 mm |
| Board extent | 385 x 275 mm, on 420 x 297 sheet (17.5 / 10 mm white margin) | 280 x 200 mm on 297 x 210 (8.5 / 5 mm margin) |
| Marker size in image at 1 m / 2 m | ~57 px / ~28 px (4.7 px per module at 2 m) | ~42 px / ~21 px (3.5 px per module at 2 m) |
| Usable range | 0.4 - 2 m, better than 1.5 m for scale | 0.4 - 1.5 m |

Notes:
- A bigger board is strictly better for the Sim(3) fit (longer baseline between corners, better pose conditioning). If A2/A1 printing is available, scale the same layout up; it costs nothing in pipeline complexity.
- Print at 100% ("actual size", no fit-to-page), matte paper/laminate (avoid gloss: specular highlights kill corners), on aluminium composite (Dibond) or glass-flat MDF/foam-core thick enough not to bow. Mount with spray adhesive, rolling out bubbles. Do not rely on A4 sheet size: measure after printing.
- Print a physical **origin mark** (corner id 0 location) and arrow on the back/edge so the measurement in section 4 is unambiguous. Check the axis convention once with `cv2.drawFrameAxes` rather than assuming it.
- Use the **measured** square length (not the nominal) in the `CharucoBoard` constructor: measure across all 7 squares with calipers/steel rule (385 mm span), divide by 7. A 0.3% print/measure error is a 0.3% metric scale error.

## 2. Detection

Use OpenCV's `cv2.aruco.CharucoDetector` (OpenCV >= 4.8; the older `cv2.aruco.interpolateCornersCharuco` is the legacy path).
- Settings to start with [E]: `cornerRefinementMethod = CORNER_REFINE_SUBPIX` (or `CONTOUR`), `cornerRefinementWinSize` ~5, `CharucoParameters.cameraMatrix/distCoeffs` set from the D435i factory intrinsics (read via pyrealsense2 `get_intrinsics()`), `minMarkers = 2`. Reject frames with fewer than ~8 charuco corners, collinear corners, or high board-to-image-plane obliquity (> ~60 deg).
- Detect on the **same, undistorted-consistent** images the reconstruction uses; the 2D observations must be in the same camera model as the reconstruction's poses.
- Blur: handheld motion blur degrades corners. Drop frames by a sharpness threshold (variance of Laplacian over the board ROI).
- Calibration quality matters: factory intrinsics for the colour stream are probably fine at 1-2 mm class but a one-off ChArUco intrinsic calibration (OpenCV `calibrateCamera` with charuco corners) is cheap and removes a systematic scale error source [E].

## 3. Solving the Sim(3)

Frames: R = reconstruction frame (arbitrary scale), B = board frame, F = Robot base frame (Franka `link0`).
Unknown: S_BR (similarity R -> B), 7 DoF. Known from physical procedure: T_FB (rigid, metric).
Result: S_FR = T_FB * S_BR; apply to the whole Scene once.

Method (recommended) [E, standard bundle-adjustment-style formulation]:
1. For each accepted frame i you have the reconstruction's camera pose T_{R<-C_i} (from the SLAM/SfM stack) and 2D charuco corner observations u_ij with ids j (known 3D board points X_j in B from `board.getChessboardCorners()`).
2. Initialise: per-frame PnP (`cv2.solvePnP`, IPPE for planar) gives camera centre c_i^B in B. Pair with the reconstruction camera centres c_i^R. Umeyama similarity fit (c_i^R -> c_i^B) gives s, R, t. Reference: Umeyama, PAMI 1991, DOI 10.1109/34.88573 (also implemented in Eigen, `Eigen::umeyama`, https://raw.githubusercontent.com/eigenteam/eigen-git-mirror/master/Eigen/src/Geometry/Umeyama.h ). Use RANSAC around it to discard frames with bad PnP.
3. Refine: minimise robust (Huber/Cauchy) reprojection error of all X_j over all frames w.r.t. the 7 parameters of S_BR with the reconstruction camera poses and intrinsics held fixed (`scipy.optimize.least_squares`). This uses all corners with proper weighting instead of trusting noisy per-frame board poses, and it automatically weights close, frontal views more.
4. Report: RMS reprojection error (px), number of frames/corners, scale s with a bootstrap/jackknife confidence interval over frames, and residual of camera-centre positions.

Notes:
- Gravity (IMU) is a useful independent check but not needed for a board that is physically known: the board pins all 6 pose DoF plus scale.
- Planar-only constraint: a single planar board gives weak constraint on rotation about axes in the board plane when seen only head-on. Capture the board from varied oblique angles (20-50 deg) and distances, not only straight-on.
- COLMAP's own `model_aligner` cannot use the board directly: it aligns to **camera-centre** priors (`--ref_images_path`, `--alignment_type custom`, RANSAC `--alignment_max_error`, `--estimate_scale`, `--transform_path` saves the Sim3). If the stack is COLMAP-based, you could feed it board-derived camera centres (from step 2) and use it as the Sim(3) solver. Source: https://raw.githubusercontent.com/colmap/colmap/main/src/colmap/exe/model.cc and https://colmap.github.io/faq.html (at least 3 images, RANSAC, 3D similarity transform). This only uses camera centres (not corner reprojection), so the step-3 refinement is more accurate.
- If the reconstruction stack is RGB-D/SLAM with D435i depth, it may already be near-metric (depth is specified within ~2% at 2 m by reseller pages; Intel datasheet not verified). Then the board is a scale **check** plus the alignment source, and `s` should come out close to 1; report |s-1| as the "scale error vs Anchor" metric the map's quality report needs.

### Expected accuracy [E, no empirical data yet]
- Vision: with 0.1-0.3 px corner noise, ~1390 px focal length, and hundreds of corners per frame over hundreds of frames, the statistical scale error is far below 0.5%. Systematic errors dominate: print scale (measure it), intrinsics/distortion, rolling shutter (if confirmed), and any scale drift inside the reconstruction itself (the board only anchors the region where it is visible; a 5 m scene can drift away from the anchor).
- Place the board **near the Robot**, because Task-relevant accuracy is needed there, and also capture frames where the board is visible from across the workspace so the Sim(3) fit is not extrapolating.
- Physical measurement (below) is typically 1-3 mm and ~0.2-0.5 deg unless using the robot touch method.
- Plan for ~0.5-1% scale and ~5 mm position / ~0.5 deg orientation as a target to **validate**, not a guarantee.

## 4. Physical placement and measurement relative to the Franka base

### What "base frame" is [S]
In `franka_description` the arm's root is `link0`, and `joint1` sits at z = 0.333 m directly above the link0 origin (x = y = 0), so the link0 origin lies on the joint-1 vertical axis at the base-mount level. Sources: https://raw.githubusercontent.com/frankaemika/franka_description/main/robots/fr3/kinematics.yaml , https://raw.githubusercontent.com/frankaemika/franka_description/main/robots/common/franka_arm.xacro . (Panda uses the same `link0` convention; I did not open its file.) Confirm in the Franka datasheet/product manual (R02210/R02212 on franka.de) where the link0 origin sits relative to the mounting surface; I could not read it from the PDF text.
Consequence: the base frame is: origin on the joint-1 axis at the base-plate plane (confirm height), z up (gravity-aligned if the table is level), x per robot's zero configuration (joint 1 = 0 points the arm along +x).

### Placement
- Fix the board **rigidly and flat to the same rigid structure the Robot is bolted to** (same table top / robot plinth), so a table or plinth shift moves both. Not on the floor or a separate cart.
- Put it within ~0.3-0.8 m of the base, in a spot the arm will not sweep, visible from the Capture walking route, with the board plane roughly horizontal (on the table) or vertical (a fixed wall/frame). Avoid the Robot mask region and glare from overhead lights.
- Keep it there for every Capture, or re-measure. Make the mounting repeatable (stops/dowels or a drawn outline) so Scene regenerations stay valid.

### Measurement procedure A (simple, tape/calipers) [E]
1. Level the table; verify the base plate plane is horizontal (spirit level).
2. Define the board origin as the physical mark for chessboard corner id 0 and the board x axis along the printed arrow.
3. Mark the joint-1 axis point on the table (plumb line or the robot's base centre).
4. Measure the vector from the joint-1 axis point to the origin mark in the robot's x and y directions (joint 1 = 0 direction), using a steel rule/calipers and a square, plus the height of the board surface above the base-mount plane (and its offset to link0 origin per datasheet). Measure the board's yaw relative to the robot x axis from two points along the board edge, not a protractor.
5. Record T_FB = (x, y, z, yaw) with the stated uncertainty; roll and pitch zero if both are flat on a level surface (verify with a level).

Error: yaw error = lateral error / lever arm, e.g. 2 mm over 0.5 m ~ 0.23 deg. So measure two far-apart points along the board edge.

### Measurement procedure B (preferred: use the robot as the measuring device) [E]
1. Fit a pointed stylus/dowel tip, or use a known fingertip, in the Robotiq gripper. Measure/calibrate the tip offset in the flange frame once (pivot calibration: pivot on a fixed dimple through several orientations, least squares gives the tip offset).
2. Hand-guide (or jog) the tip to at least 4 chessboard-corner marks on the board (non-collinear, spread across the board). Record forward-kinematics tip positions in F from the joint states.
3. Fit the rigid transform (Kabsch/Umeyama without scale) from board-frame corner coordinates X_j to measured F-frame points; residual tells you the real measurement error. Gives T_FB with ~1-2 mm and well-conditioned orientation, and incorporates the true base frame the controller uses (no ambiguity about link0).
4. Franka's quoted pose repeatability is vendor-datasheet level (not verified here); the dominant error becomes tip calibration and manual positioning.

Use A to sanity check B. Keep the home pose the Robot mask uses (Capture protocol) separate from this: the Anchor only needs the base frame, not the arm pose.

## 5. Do existing real2sim pipelines already do this?

- Real-to-Sim Robot Policy Evaluation with GS (arXiv 2511.04665, https://arxiv.org/pdf/2511.04665): aligns the robot's splat to ground-truth points sampled from the URDF with ICP+RANSAC, then reuses the transform for the background. No marker. (Per search-result summary; not read in full.)
- Splatting Physical Scenes (arXiv 2506.04120) jointly optimises robot joint angles/poses for calibration; RoboGSim (arXiv 2411.11839) has a layout-alignment module; RL-GSBridge (arXiv 2409.20291) uses COLMAP without a described base calibration; MetaFine (arXiv 2605.19986) aligns base pose by an unspecified method. None identified as using ChArUco. I only saw abstracts/snippets, not full methods.
- Scale from ArUco markers: `aruco-estimator` (PyPI https://pypi.org/project/aruco-estimator/1.1.3/; work-in-progress, needs the marker in >= 2 images, >= 5 recommended, scales a COLMAP cloud) and a 2026 NeRF/GS geometry comparison (arXiv 2604.18205) that uses four tabletop ArUco markers for origin, orientation and scale. Neither handles robot-base alignment.
- Conclusion: build it (small: ~150 lines around OpenCV + scipy). Cross-check idea for the quality report: also run ICP of the Robot's visible geometry vs the URDF as in 2511.04665 and compare the two base estimates (but note the Robot is masked out of the Scene by design, so use pre-mask frames or only for a one-off validation).

## 6. Open uncertainties

1. Exact position of `link0` origin vs the physical mounting plane (read from Franka datasheet/manual, not retrieved).
2. D435i colour shutter type and rolling-shutter effect on handheld corner accuracy (reseller claim only); a global-shutter-corrected approach is not needed unless residuals show it. Could use the infrared (global shutter) stream for board detection only if the reconstruction uses the same camera model, but that adds complexity.
3. Actual achieved accuracy numbers: all section 3 and 4 error figures are estimates; validate on the first real Capture from 2026-10-19 (and a synthetic test on mock recordings with a rendered board).
4. Scale drift of the reconstruction away from the board: depends on the chosen stack (ticket on reconstruction stack); mitigated by multiple boards or board visible throughout.
5. OpenCV version pinning for the legacy pattern issue and API (`CharucoDetector` needs >= 4.8; verify the Python binding signatures in the Docker image).
6. Couldn't fetch docs.opencv.org (HTTP 403); relied on the upstream OpenCV source headers/tutorial markdown, which are the origin of those docs.
