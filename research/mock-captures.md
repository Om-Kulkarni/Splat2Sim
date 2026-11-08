# Mock D435i captures available for download

Researched 2026-10-17. URLs were checked with HTTP HEAD/GET (status and sizes noted). Nothing multi-GB was downloaded. "Verified" means I checked it myself; "per source" means I am repeating a page or paper I read.

## Short answer

No public dataset found is a handheld D435i walk-around of a robot workcell with RGB + depth + IMU. The best usable mix is:

1. **Intel `d435i_sample_data.zip`**: the only guaranteed-live, no-login, native D435i color+depth+IMU handheld walking data. Outdoor, ~0.47 GB. Use it to test the ingest path (bag, IMU, intrinsics, depth alignment).
2. **SLAM&Render (arXiv 2504.13713)**: D435i, 848x480@30 Hz, IMU at 210 Hz, ROS2 bag, CC-BY 4.0, static tabletop scenes recorded with a camera on a KUKA arm in a robot cell. It is the closest to the lab workcell and is built for splat/NeRF evaluation. Downloads are 1.3-1.8 GB per file. The camera is robot-moved rather than handheld, and I could not confirm the IMU is inside the downloadable bags.

Fallback for a bigger handheld indoor set: **VINS-RGBD "Handheld"** (3 bags, 0.5-1.07 GB each, D435i, direct download).

## Candidates

| # | Dataset | Camera | Streams | Format | Size | Handheld indoor? | Robot/tabletop? | Licence | Access |
|---|---|---|---|---|---|---|---|---|---|
| A | Intel librealsense sample data, `d435i_sample_data.zip` | D435i (pre-production) | color, depth, IMU (zip holds `d435i_walking.bag` and `d435i_walk_around.bag`) | librealsense `.bag` (ROS1-style); `rs-convert` -> `.db3` | zip 472,413,521 B (~450 MiB); bags 723 MB and 842 MB uncompressed | Handheld walking, but outdoors | No | Not stated ("not official marketing material") | Open, HTTP 200 |
| B | SLAM&Render | D435i, 848x480@30 Hz, IMU 210 Hz | RGB, depth, IMU, robot kinematics, GT poses; intrinsics.yaml per setup; chessboard images | PNG + CSV/YAML, plus "also available as ROS2 bag files" | train 1.7 GB, test 1.3 GB, setup-2 train 1.8 GB (Drive page, 40 files total) | No: camera on a KUKA LBR iisy arm, static scenes, 5 setups x 4 lighting | Yes: robot cell, tabletop objects. Arm visibility in frames unconfirmed | CC-BY 4.0 per paper | Google Drive (virus-scan interstitial) |
| C | VINS-RGBD Handheld | D435i | color, aligned depth, IMU (`/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`, `/camera/imu`) | ROS1 bag, compressed (`rosbag decompress`) | Normal.bag 505,031,296 B; Simple.bag 978,019,039 B; With more rotation.bag 1,073,809,785 B | Yes, handheld, scenes not described | No | Code GPLv3; data licence not stated | Direct seafile download, verified HTTP 200 |
| D | OpenLORIS-Scene | D435i + T265 on a wheeled robot | color 848x480@30, depth, aligned depth, D435i accel/gyro, GT | ROS bags | not listed | No, robot-mounted at ~1 m. Office/home/cafe/market | No arm | Not named | Form required |
| E | VOID | D435i + VIO rig | RGB 640x480, depth, IMU 400 Hz (raw bags only) | release PNG; raw `raw.bag` (ROS Kinetic) | not listed | Handheld, classrooms/labs/offices/gardens | No | UC Regents, research only | OneDrive link returns 403 to curl, so unverified |
| F | DARP | unnamed RealSense on 2 Unitree arms | color, depth, IR, joint states (no IMU) | RealSense `.bag` | 49.76 GB | No, eye-in-hand | Yes, tabletop | not stated | IEEE DataPort subscription |
| G | IDLS-inlab (IEEE DataPort) | D435i | RGB + IMU only, no depth | ROS bags | large | Yes | No | not stated | DataPort subscription |
| H | `bearfouz/SLAM-dataset-BRACU-indoors` | D435i | color, depth, IR1/2, imu, gyro, accel (per `scripts/record.sh`), 640x480@30, fixed exposure | ROS2 MCAP (`room_seq01/02.tar.zst`) | not stated | Closed room, handheld | No | none | **No download link found**: repo has only scripts, no releases, `*.mcap` gitignored |
| I | LuViRA, Rosario v2, Newer College, CAVERS | D435i or other | various | rosbag | not checked | Robot, field, cave or LiDAR rig | No | not checked | Mismatch with ticket; not pursued |

## Details and evidence

### A. Intel sample data (recommended for pipeline smoke tests)
- Page: https://github.com/IntelRealSense/librealsense/blob/master/doc/sample-data.md. It lists `d435i_sample_data.zip` as "Outdoor scene with D435i pre-production sample (Depth from Stereo with IMU)". It notes `.bag` files are uncompressed, ~100 MB per second, and can be converted with `rs-convert -i file.bag -D file.db3`.
- Direct URL: https://librealsense.realsenseai.com/rs-tests/TestData/d435i_sample_data.zip. HEAD returned 200 and `content-length: 472413521`. The old host `librealsense.intel.com/rs-tests/TestData/` also answers 200.
- Contents: I read the zip central directory with HTTP range requests (~100 KB, no full download). It holds `d435i_walk_around.bag` (841,910,563 B uncompressed) and `d435i_walking.bag` (722,813,137 B).
- Intrinsics and extrinsics are embedded in librealsense bags (the device snapshot is recorded). I did not open a bag.
- Gap: the scene is outdoor, and the page does not say whether each bag contains the IMU streams, or at what rates. Check with `rs-enumerate-devices -p` or `rosbag info` after download.
- Also live, but no IMU and not D435i (not useful for the IMU requirement): `stairs.bag` (409,209,452 B, D435 pre-production), `outdoors.bag` (422,868,878 B, D415). Both are under https://librealsense.realsenseai.com/rs-tests/TestData/.

### B. SLAM&Render (recommended: closest to the lab workcell)
- Paper: https://arxiv.org/abs/2504.13713 (HTML: https://arxiv.org/html/2504.13713). Per the paper: D435i "mounted in an eye-to-hand configuration on a KUKA LBR iisy 3 R760 manipulator", RGB-D at 848x480 and 30 Hz, IMU at 210 Hz. 40 static sequences (5 setups x 4 lighting conditions, each with a train and a test trajectory). Ground-truth poses and robot kinematics are included. "All sequences are also available as ROS2 bag files." Data "under CC-BY 4.0 license". Per-setup `intrinsics.yaml` (factory plus estimated) and chessboard calibration images are released.
- The cell has non-reflective black panels and a black cloth over the support table. This is a controlled tabletop cell, much cleaner than a real lab.
- Download index: https://tinyurl.com/SLAMRender (301 to a published Google Doc: https://docs.google.com/document/d/e/2PACX-1vSvPjws3Tn0wVIYsG7m21XqXz9dwVrdAFEWMAVuizfkONvbrMZCPemRIJPFWuKPlHEeHb_eMj54kXDL/pub). It lists 40 Google Drive IDs with the pattern `https://drive.google.com/uc?export=download&id=<ID>`.
- Examples whose Drive interstitial reports sizes:
  - setup-1 natural train, `1-natural-tr.zip`, 1.7G: https://drive.google.com/uc?export=download&id=1Ki7bnv0XSJN-2woZvxEB6bdYSh3b96ga
  - setup-1 natural test, `1-natural-tt.zip`, 1.3G: https://drive.google.com/uc?export=download&id=1xi5Z6Zeu8TKS68AK6T5e6q7piW6Ufuc2
  - setup-2 natural train, `2-natural-tr.zip`, 1.8G: https://drive.google.com/uc?export=download&id=186jWlMZOGbMWKibx6geB7VP5Rtlm0tmy
  Google serves a virus-scan warning page for these, so a plain curl needs the confirm token (`gdown` handles it).
- Open questions: (1) whether the zips contain the ROS2 bags or only PNG/CSV; (2) whether the bags carry IMU; (3) whether the KUKA arm itself is visible (matters for testing Robot masking); (4) the "eye-to-hand" wording conflicts with a camera that moves with the arm.

### C. VINS-RGBD Handheld
- Repo: https://github.com/STAR-Center/VINS-RGBD (D435i, 9 bags across handheld, wheeled and tracked platforms). Dataset page: https://robotics.shanghaitech.edu.cn/datasets/VINS-RGBD.
- Handheld folder: https://robotics.shanghaitech.edu.cn/seafile/d/0ea45d1878914077ade5/ (HTTP 200; the `star-center.shanghaitech.edu.cn` host in the README gives 404).
  - Normal.bag: https://robotics.shanghaitech.edu.cn/seafile/d/0ea45d1878914077ade5/files/?p=/Normal.bag&dl=1 (505,031,296 B)
  - Simple.bag: https://robotics.shanghaitech.edu.cn/seafile/d/0ea45d1878914077ade5/files/?p=/Simple.bag&dl=1 (978,019,039 B)
  - With more rotation.bag: https://robotics.shanghaitech.edu.cn/seafile/d/0ea45d1878914077ade5/files/?p=/With%20more%20rotation.bag&dl=1 (1,073,809,785 B)
- Not stated: intrinsics, scene, depth units. Bags are ROS1; run `rosbag decompress`. From 2019, so the librealsense firmware and driver vintage is old.

### D. OpenLORIS-Scene
- https://lifelong-robotic-vision.github.io/dataset/scene.html (HTTP 200). Per page: D435i color/depth 848x480@30, accel 250 Hz, gyro 400 Hz, intrinsics and extrinsics inside the bags, OptiTrack GT for the office scene. Download is gated by a form (Google Drive / Baidu). Robot-mounted at ~1 m, so the motion is unlike handheld.

### E-G. Gated or mismatched
- VOID: https://github.com/alexklwong/void-dataset. OneDrive links as listed in the README; the `void_raw.zip` link redirected to an interactive page (403 for curl). Sizes not published. Raw bags need ROS Kinetic.
- DARP: https://doi.org/10.21227/rmv3-be47 -> https://ieee-dataport.org/documents/darp-calibrated-dual-arm-rgb-d-ir-dataset-multi-view-robotic-perception. 49.76 GB, subscription needed, no IMU. Rejected.
- IDLS-inlab: https://ieee-dataport.org/documents/idls-inlab. RGB + IMU only (no depth) and subscription-gated. Rejected.

### H. bearfouz/SLAM-dataset-BRACU-indoors
- https://github.com/bearfouz/d435i-room-dataset redirects (301) to https://github.com/bearfouz/SLAM-dataset-BRACU-indoors. Repo created 2026-09-29. The README names `room_seq01.tar.zst` and `room_seq02.tar.zst` (zstd tarballs; "recommended room_seq02"), but the repo contains only `scripts/` and the README. No releases, no licence, and `.gitignore` excludes `*.mcap`. The recording script confirms the topic set (color, depth, infra1/2, imu, gyro, accel, `/tf_static`) and MCAP storage, with sensible settings (640x480@30, manual exposure). An unverified web search snippet claimed about 39 GiB and 368 s + 280 s, which I could not confirm. Treat as not downloadable until the author is contacted or a link appears.

## Closeness to the lab workcell scan

| Property | A Intel | B SLAM&Render | C VINS-RGBD |
|---|---|---|---|
| Handheld | yes | no (arm-moved) | yes |
| Indoor | no | yes | yes (presumably) |
| Tabletop / arm | no | tabletop + robot cell | no |
| Clutter / lab realism | n/a | low (black panels) | medium |
| Bag format matches our pipeline | `.bag` | PNG/CSV + ROS2 | ROS1 `.bag` |

None of these shows a Franka or a ChArUco board. A mask-the-robot test therefore needs a synthetic stand-in or a re-render until the real camera arrives.

## Recommendation
1. Ingest/format test now: `d435i_sample_data.zip` (A). Direct and small, true D435i + IMU.
2. Workcell-like reconstruction test: SLAM&Render (B), after confirming bag contents with one 1.3 GB test zip.
3. Optional indoor handheld IMU test: VINS-RGBD `Normal.bag` (C, 505 MB).

## Uncertainties
- IMU presence and rate inside A and B bags is unverified; no bag was opened.
- B's tinyurl hosting is a Google Doc plus Drive, which could rate-limit.
- Licences: none for A and C; B's CC-BY 4.0 is as stated in the paper, not on the download page.
- The BRACU dataset might be the best match (MCAP, D435i, handheld room) if a link is published later.
