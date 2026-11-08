# Reconstruction stack options for D435i RGB-D + IMU (ticket 03)

Researched 2026-10-17. Target: one RTX 5090 (sm_120, 32 GB), driver 580, Docker, Ubuntu 24.04. Scene: ~3x3 m workcell, handheld D435i Capture.
Evidence tags: **[doc]** read in a primary source (URL given); **[code]** read in the cloned repo source; **[inferred]** my reasoning, not stated by a source; **[unverified]** could not confirm.

## TL;DR / recommended stack

| Stage | Recommendation | Fallback |
|---|---|---|
| (a) Poses | **RTAB-Map offline (RGB-D + IMU, metric)** for initial metric poses, then **COLMAP (4.2.x, masked features, optional `global_mapper`) as the photometric pose refiner** if RTAB-Map poses are too noisy for splatting; ChArUco Anchor does the final Sim(3) fix | cuVSLAM RGB-D (PyCuVSLAM); VGGT/MASt3R-SLAM only as initializers (no sensor depth/IMU, scale not guaranteed) |
| (b) Splat training | **3DGRUT (3DGUT mode)** if you want the shortest path to Isaac Sim 6 (native `ParticleField` USD export, mask support, CUDA 12.8 Docker, documented on 5090). **fVDB-Reality-Capture** if you want **metric dense sensor-depth supervision** (D435i depth) and in-toolkit TSDF/mesh and USD export. | gsplat `simple_trainer` + 3DGRUT `transcode` (PLY -> USD) |
| (c) Collision mesh | **TSDF fusion of masked sensor depth with the final poses** (Open3D, CPU is fine at this scale; nvblox if you want GPU), then attach with 3DGRUT `add_mesh_to_usdz --set_collision --set_invisible` | `tsdf_from_splats` / `mesh_from_splats` in fVDB-RC; 2DGS TSDF |
| Scale | Metric from D435i depth (RTAB-Map) -> verified/corrected against the ChArUco Anchor | |

No tool I found is *known broken* on Blackwell. Several are *stale / unverified* on it (listed under Blackwell below).

## Context that constrains the choice

- Isaac Sim 6.0 (GA June 4 2026, Kit 110) renders splats as OpenUSD `ParticleField` geometry; 3DGRUT is the recommended trainer/exporter. [doc] https://docs.isaacsim.omniverse.nvidia.com/6.0.0/assets/usd_assets_nurec.html , https://radiancefields.com/nvidia-s-isaac-sim-6.0-ships-with-nurec-gaussian-splatting
- Two formats exist: **ParticleField USD** (preferred, beta, `UsdVol ParticleField3DGaussianSplat`) and legacy **NuRec USDZ** (listed for Kit 107.3-110.1 / Isaac Sim 5.0-6.0, "being deprecated"). [doc] https://github.com/nv-tlabs/3dgrut/blob/main/threedgrut/export/README.md
- Isaac Sim 6 stated driver tested: Linux 580.95.05; RTX 5090 is not named in the requirements page (min RTX 4080, ideal RTX PRO 6000 Blackwell); GPUs without RT cores unsupported. [doc] https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html
- Splats are visual only; physics needs a mesh proxy. The NuRec doc does not describe collision for splats; the 3DGRUT export docs say the USDZ from `ply_to_usd` "contains no mesh" and provide `add_mesh_to_usdz --set_collision --set_invisible`. [doc] 3DGRUT export README above.
- Known Isaac Sim 6 gotchas: a `.usdz` opened as root stage fails to take referenced assets (use a `.usd/.usda` root that references the splat); DLSS Frame Generation can artifact on particle fields. [doc] NuRec page above. One forum user reports NuRec path = better fidelity/stability, ParticleField = better shadows/scene interaction but more flicker (single user report, not official). [doc-ish] https://forums.developer.nvidia.com/t/question-on-two-3dgs-rendering-pipelines-nurec-vs-particlefield-and-expected-quality-tradeoff-in-isaac-sim-6-0/373260 (page itself 403'd for me; relayed via search summary -> treat as **[unverified]**).
- D435i hardware: depth imagers are global shutter, RGB is **rolling shutter** 1920x1080@30; depth accuracy "<2% at 2 m" (conditional); ideal range 0.3-3 m. [doc] Intel store/compare pages (via search): https://store.intelrealsense.com/buy-intel-realsense-depth-camera-d435i.html , https://intelrealsense.com/compare-depth-cameras . Implication [inferred]: walk slowly (rolling-shutter blur on RGB), 3 m max range suits a 3x3 m cell.
- librealsense v2.59.1 supports D435i (firmware >= 5.17.3.10), Ubuntu 24.04, kernels 6.x; ROS 2 rosbag2 (db3, compressed) playback. MCAP not mentioned. [doc] https://github.com/IntelRealSense/librealsense/releases

## (a) Camera pose estimation

| Option | Uses depth / IMU | Metric? | Status / Blackwell / Docker | Verdict |
|---|---|---|---|---|
| **RTAB-Map** | RGB-D; IMU supported (Madgwick filter auto-enabled for IMU-capable sensors, gravity constraint `Optimizer/GravitySigma=0.3`) | Yes (from depth) | v0.23.8 (Jul 2026), BSD, Docker `introlab3it/rtabmap`, ROS 2 Humble..Lyrical; bundles RealSense2 2.56.5 incl. D435i; `rtabmap-export --poses_camera --images` exports optimized camera poses/images. CPU-based => no sm_120 concern. [doc] https://github.com/introlab/rtabmap , https://github.com/introlab/rtabmap/releases , export flags [code] https://github.com/introlab/rtabmap/blob/master/tools/Export/main.cpp | **Primary.** Works from a recorded database/bag, offline, metric. Caveat: loop-closure-optimized poses are good for TSDF but may need photometric refinement for sharp splats [inferred]. Mask integration: not documented; would need to blank robot pixels in RGB/depth before feeding [unverified]. |
| **COLMAP 4.2.1** (includes former GLOMAP as `global_mapper`) | RGB only (no depth/IMU) | No (arbitrary scale) | Sep 29 release; 4.1.0 fixed PatchMatch on Blackwell sm_100+; `mask_path` skips features in black mask regions; global mapper needs good focal priors (`view_graph_calibrator` otherwise); GLOMAP repo archived Mar 2026, merged into COLMAP; Docker scripts in repo (CUDA version set in Dockerfile). [doc] https://github.com/colmap/colmap/releases , https://colmap.github.io/faq.html , https://github.com/colmap/glomap , https://github.com/colmap/colmap/blob/main/docker/README.md | **Best refiner and the de-facto input format** for 3DGRUT/gsplat/fVDB. Robot mask applies natively. Scale must be fixed afterwards (depth or Anchor). Could also run `point_triangulator` with RTAB-Map poses as known poses (documented procedure in FAQ). |
| **cuVSLAM / PyCuVSLAM** | modes: Mono, **RGBD**, Multicamera, Inertial (stereo + IMU), Multisensor (experimental) | RGBD yes | NVIDIA Community License; Ubuntu 22.04/24.04 x86_64, CUDA 12 or 13; Blackwell not mentioned; RealSense examples are Mono-Depth and Stereo-Inertial (so RGB-D and IMU are not combined) [doc] https://github.com/NVlabs/PyCuVSLAM | Viable alternative; no RGB-D+IMU fusion, pose export not documented. |
| **ORB-SLAM3** | Has Mono/Stereo/RGB-D and Mono/Stereo-Inertial; **no RGB-D-inertial** documented | RGB-D yes | README dated 2021, Ubuntu 16/18-era, GPLv3 [doc] https://github.com/UZ-SLAMLab/ORB_SLAM3 | **Not recommended** (stale, no RGBD-I). (RTAB-Map 0.23.8 notes mention ORB-SLAM3 IMU fixes only as an RTAB-Map-integrated odometry option.) |
| **MASt3R-SLAM** | RGB only (video/images, optional calibration; RealSense live supported) | Not stated | PyTorch 2.5.1, CUDA 11.8/12.1/12.4 in README; Blackwell not mentioned; no Docker; Naver checkpoint license separate [doc] https://github.com/rmurai0610/MASt3R-SLAM | Not recommended as primary: ignores the depth/IMU you have; PyTorch pin needs porting to cu128. |
| **VGGT** | RGB only, feed-forward | Not stated | Outputs poses, depth, point maps; `demo_colmap.py` writes COLMAP `sparse/`; 1B checkpoint non-commercial, only VGGT-1B-Commercial licensed commercially; VGGT-Omega released May 2026 [doc] https://github.com/facebookresearch/vggt | Possible init for COLMAP on hard sequences; not a primary because of scale and license. |

Note on D435i sync [inferred]: RGB and IMU timestamps come from the same librealsense clock; RTAB-Map handles IMU as a gravity/orientation prior, not tightly coupled VIO.

## (b) Gaussian splat training

| Option | Depth supervision | Masks | Isaac Sim 6 export | Blackwell / CUDA | Docker | Runtime |
|---|---|---|---|---|---|---|
| **3DGRUT (3DGUT/3DGRT)** v2.0.0 (Jun 2026) | **None found** (no depth term in configs/losses; grep of repo) [code] | Yes: `<img>_mask.png` next to image, value 0 excluded from loss [doc] | **Native**: ParticleField USD (`standard`/lightfield), NuRec USDZ, PLY; transcoder between them; `add_mesh_to_usdz` [doc] | CUDA 11.8/12.4/12.6/12.8 (default)/13.0 experimental; README has explicit Blackwell/RTX 50 section (CUDA 12.8) [doc] https://github.com/nv-tlabs/3dgrut | Yes, `Dockerfile` with `CUDA_VERSION` arg (12.8.1 example) | Reported on RTX 5090: 3DGUT NeRF-Synthetic ~215 s avg; MipNeRF360 ~686 s avg (GS strategy). 3DGRT ~2869 s MipNeRF360, wants RT cores. [doc] README benchmarks. A 3x3 m scene ~ the MipNeRF360 class => ~10-30 min [inferred] |
| **fVDB-Reality-Capture** 0.6.0 (Sep 2026, Apache-2.0, NVIDIA/OpenVDB) | **Yes: dense depth (`DepthMapAttribute`, METRIC scale => masked L1) and sparse depth** [code] | Yes (dataset has mask/mask_path) [code] | `ParticleField` USD(Z) export "for Isaac Sim 6.0+", `--legacy` for older; can embed a mesh in the USDZ [code] | PyTorch 2.10.0, CUDA 12.8 or 13.0, Ampere+ (CC>=8.0), driver>=550; pre-built wheels [doc] https://fvdb-reality-capture.readthedocs.io/ ; source: https://github.com/openvdb/fvdb-reality-capture | Not documented [unverified] | NVIDIA claims ~50% better throughput than gsplat [doc]; no 5090 numbers found |
| **gsplat** (main, v1.6.0 unreleased; PyPI 1.5.3) | `depth_loss` is **experimental and uses COLMAP sparse points, not sensor depth** [code] | Yes in `simple_trainer` (masked L1/SSIM) [code] | None itself: PLY only; convert via 3DGRUT `transcode` (PLY -> ParticleField) [doc] | "PyTorch 2.7+ required, CUDA 12.8 and 13.2 build compat"; Blackwell not named [doc] https://github.com/nerfstudio-project/gsplat ; PyTorch 2.7 ships Blackwell + cu128 wheels [doc] https://pytorch.org/blog/pytorch-2-7/ | none in repo | not measured |
| **nerfstudio splatfacto** | Not documented on README | Not documented | `ns-export` only documents pointcloud there | README says tested CUDA 11.7/11.8, torch 2.1.2+cu118 => stale for Blackwell [doc] https://github.com/nerfstudio-project/nerfstudio | Dockerfile | - | 
| **DN-Splatter** (sensor-depth + normal splatting, `o3dtsdf` mesh) | Yes (sensor depth, EdgeAwareLogL1) | depth confidence masks | PLY via nerfstudio | Last dated update Nov 2024, nerfstudio-based => stale [doc] https://github.com/maturk/dn-splatter | - | - |

Takeaways:
- Robot masking works in all of 3DGRUT, gsplat and fVDB-RC (mask loss exclusion) plus COLMAP (`mask_path`). With a masked training region the Isaac-side splat will have holes where the robot was; the Task environment's real robot renders over it. [inferred]
- Sensor-depth supervision is only first-class in fVDB-RC. D435i depth noise (<2% @ 2 m) and holes mean a weight sweep is needed; metric depth loss is in scene units so weight must be retuned after scale changes [code docstring].
- 3DGRUT has Isaac-specific export docs and the 5090 benchmark; fVDB-RC is newer and its exporter targets the same ParticleField schema. Choice between them is a bake-off on the first real Capture [inferred]; both can consume a COLMAP-style dataset (3DGRUT: yes; fVDB-RC: `test_load_colmap_scene` present [code]).
- Both ParticleField format and 3DGRUT's NHT features are beta/evolving; pin the 3DGRUT tag and Isaac Sim version.

## (c) Collision mesh and metric scale

- **TSDF from sensor depth + final poses (recommended).** Direct metric geometry, independent of splat floaters. Open3D v0.20.0 (Sep 2026) wheels include a Linux x86_64 CUDA wheel, but CUDA toolkit version and Blackwell are not stated => run TSDF on CPU or verify GPU build [doc] https://github.com/isl-org/Open3D/releases . A 3x3 m room at ~1 cm voxels is small enough for CPU [inferred]. Blank robot-mask pixels in depth before integration.
- **nvblox** (v0.0.10, Apr 2026): GPU TSDF/ESDF, `nvblox_torch` wrapper with `output_color_mesh_ply`; Docker deps default to `nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu24.04`; builds for the native CUDA arch by default; has a RealSense example Dockerfile; license in LICENSE.md (not read) [code] https://github.com/nvidia-isaac/nvblox . Blackwell not explicitly claimed but CUDA 12.8 base suggests it builds [inferred, **unverified at runtime**].
- **Mesh from splats**: fVDB-RC ships `tsdf_from_splats`, `mesh_from_splats`, `point_cloud_from_splats` [code]; 2DGS does TSDF (Open3D) from rendered depth, README does not mention Blackwell/CUDA versions, custom CUDA rasterizer submodule needs a rebuild for sm_120 [doc] https://github.com/hbb1/2d-gaussian-splatting. PGSR not evaluated (time).
- **Attaching to Isaac:** `python -m threedgrut.export.scripts.add_mesh_to_usdz --input_usdz ... --mesh_ply ... --set_collision --set_invisible` [doc]. Caveats: USDZ per-layer 4 GiB limit (ZIP64 unsupported); use `.usd/.usda` or `--max-particles-per-field` partitions for large scenes [doc]. Which PhysX collision approximation (convex decomposition vs SDF) to use on a scanned mesh was not researched.
- **Metric scale chain:** RTAB-Map poses are metric because depth is metric; COLMAP-only poses are not. If COLMAP is the refiner, fit a Sim(3) between COLMAP and RTAB-Map camera centres (or against sensor depth), then check against ChArUco Anchor distance; the Anchor also gives the robot-base frame transform. [inferred; COLMAP `model_aligner` exists for similarity alignment per FAQ]. The Anchor residual becomes the quality-report "scale error".
- 3DGRUT export flags worth knowing: `--no-transform` / `apply_normalizing_transform` (default normalizes the scene!) and `--up-axis` (stage metadata only, does not reorient). Normalization would break metric alignment unless disabled or recorded. [doc] export README. fVDB-RC's USD export applies an Isaac scene alignment matrix [code].

## Blackwell / CUDA 12.8 summary

- Works / documented: PyTorch >= 2.7 cu128 wheels (Blackwell "prototype" support) [doc]; 3DGRUT (CUDA 12.8 Docker + 5090 benchmarks) [doc]; fVDB-RC (torch 2.10, cu128/cu130, CC>=8.0) [doc]; COLMAP >= 4.1.0 (Blackwell PatchMatch fix) [doc]; nvblox (CUDA 12.8 base image) [code]; RTAB-Map (CPU).
- Stale or unverified (not "known broken"): nerfstudio/splatfacto (docs pin torch 2.1.2+cu118), DN-Splatter, MASt3R-SLAM (torch 2.5.1, cu124 max in README), 2DGS/PGSR (no stated CUDA support), VGGT (bf16 on CC>=8.0, Blackwell unmentioned), PyCuVSLAM (CUDA 12/13, Blackwell unmentioned), Open3D CUDA wheel (arch not stated), gsplat (cu128 mentioned; sm_120 not named).
- Not found: any report of a tool explicitly failing on sm_120 beyond generic "PyTorch built without sm_120 kernels" errors, which are fixed by cu128+ wheels. A search-engine summary claimed stable PyTorch 2.7 needed a nightly; contradicted by the PyTorch 2.7 release post, so ignored.
- Isaac Sim itself: tested driver 580.95.05 matches the workstation; RTX 5090 not explicitly listed but within "RTX 5080 good / PRO 6000 ideal" tiers.

## Open uncertainties (to resolve by spike)

1. Does 3DGRUT's lack of depth supervision hurt a ~3x3 m D435i capture enough to justify fVDB-RC? (bake-off on mock + real Capture)
2. ParticleField vs NuRec visual quality/flicker in Isaac Sim 6 on the 5090 (only a single forum report).
3. RTAB-Map pose quality vs COLMAP for splat sharpness; best way to inject Robot masks into RTAB-Map.
4. Exact IMU benefit (RTAB-Map uses it as gravity/orientation prior only).
5. Open3D CUDA wheel on sm_120 (or just use CPU); nvblox runtime on 5090; fVDB-RC Docker story.
6. PhysX collision approximation settings for the scanned mesh and whether the splat's world transform composes cleanly with the Anchor transform in Isaac.
7. Reliability of D435i depth on the robot/metal/shiny workcell surfaces (active stereo; <2% @ 2 m is conditional).
8. Capture container: MCAP vs rosbag2 db3 for RTAB-Map/librealsense playback.
