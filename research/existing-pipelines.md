# Existing real2sim pipelines to reuse

Researched 2026-10-17.
Repo stats (stars, last push, licence) were read from the GitHub API / READMEs on that date.

## Question

Which existing (near-)end-to-end real2sim pipelines take handheld RGB(-D) video to a simulator scene usable for robot manipulation? For each: inputs, outputs, simulator, licence, maintenance, Isaac Sim compatibility, coverage of our Capture -> Scene -> Task environment. What to adopt wholesale vs borrow.

## Short answer

- No existing project covers our whole pipeline. The closest fit is the **NVIDIA stack (3DGRUT -> ParticleField USD -> Isaac Sim 6)** for the visual layer, which is official, Apache-2.0, current, and is the only one that targets Isaac Sim 6.
- NVIDIA's workflow gives a visual-only scene. It gives no metric scale, no collision mesh, no robot-mask handling beyond per-image loss masks, and no Isaac Lab task-environment integration. Those parts remain ours to build.
- Academic pipelines are useful as **design references** (Re3Sim for Anchor/calibration + Isaac + Franka; SplatSim for robot-masking and splat/physics decoupling; RialTo for scan -> mesh + articulations). Most are old-Isaac, research-grade, and (Re3Sim: CC BY-NC; others: no licence) not safely reusable as code.

## Findings per pipeline

### 1. NVIDIA: 3DGRUT + NuRec / ParticleField -> Isaac Sim 6 (official)
- **What:** `nv-tlabs/3dgrut` trains 3DGRT/3DGUT/3DGRUT Gaussian scenes and exports them to USD ParticleField, NuRec USDZ, PLY, and the lightfield format. Source: https://github.com/nv-tlabs/3dgrut
- **Inputs:** COLMAP-style datasets (also NeRF Synthetic, MipNeRF360, ScanNet++, NCore v4); optional per-image grayscale masks `<image>_mask.png` used only in the loss (so a Robot mask can be fed in). No depth supervision is documented in the README. Source: README above.
- **Outputs:** checkpoint, PLY, ParticleField USD / NuRec USDZ. The README says custom NuRec USDZ "is going to be deprecated and replaced by ParticleField" and recommends ParticleField for new assets; Isaac Sim 6.0 supports both. Source: README above.
- **Licence:** Apache-2.0 (GitHub API).
- **Maintenance:** v1.1.0 released 2026-06-10 (new export stack, NCore v4, MCMC, multi-camera, Blackwell/CUDA 12-13, uv installs); repo pushed 2026-09-22; ~2.4k stars. Source: https://github.com/nv-tlabs/3dgrut/releases. Secondary summary: https://radiancefields.com/nvidia-3dgrut-v1.1.0---nurec-export-pipeline-ppisp-and-training-upgrades
- **Isaac Sim:** Isaac Sim 6.0.0 GA (Kit 110.1.1) lists `omni.kit.converter.gsplat`; 6.0.1 adds the experimental `isaacsim.replicator.nurec_utils` (render + PSNR/SSIM vs ground truth, usable as a quality-report building block). Sources: https://docs.isaacsim.omniverse.nvidia.com/6.0.1/overview/release_notes.html, https://docs.isaacsim.omniverse.nvidia.com/6.0.1/assets/nurec_utils.html
- **Isaac Sim docs for loading:** scenes are USD stages using ParticleField geometry rendered by Omniverse RTX alongside polygonal content; example scenes are `stage_particle.usdz` from the HF dataset `nvidia/PhysicalAI-Robotics-NuRec`. The only physics in the examples is an optional invisible 10x10 collision ground plane. Known limitation: opening a `.usdz` as root stage blocks adding references; reference it from a `.usd/.usda` instead. Source: https://docs.isaacsim.omniverse.nvidia.com/6.0.1/assets/usd_assets_nurec.html
- **Mono-camera workflow doc** (photos -> COLMAP (PINHOLE/SIMPLE_PINHOLE) -> 3DGUT + MCMC -> USDZ -> Isaac Sim 5.0+): visual only. Collision = manually added ground plane; a proxy mesh is attached for shadows; `apply_normalizing_transform` centres/scales the scene but "does not guarantee the floor is exactly at z = 0" and metric scale is not recovered. Source: https://docs.nvidia.com/nurec/robotics/neural_reconstruction_mono.html
- **Render requirements** (from nurec_utils): single-GPU rendering (`--/renderer/multiGpu/enabled=false`); SPG/PPISP assets need `omni.rtx.spg` and extra settings. Source: nurec_utils page above.
- **Coverage:** Scene visual layer (strong). Capture protocol (photo-oriented guidance only), metric scale/Anchor, collision layer, robot, Task environment: none.
- **Caveat:** an NVIDIA forum thread (user report, plus NVIDIA staff reply per a search snippet; not read in full) says the Isaac Sim 5.1 NuRec volume path loses view-dependent colour and that ParticleField in 6.0 keeps spherical harmonics. Treat as unverified: https://forums.developer.nvidia.com/t/question-on-two-3dgs-rendering-pipelines-nurec-vs-particlefield-and-expected-quality-tradeoff-in-isaac-sim-6-0/373260
- **Unverified / not found:** how Isaac Lab (v3.0 EA released 2026-09-16, https://github.com/isaac-sim/IsaacLab/releases) loads a ParticleField background, whether policy-camera (tiled) rendering works with ParticleField, and Isaac Sim-side ParticleField docs ("Omniverse Gaussian Splats (Particle Fields)"), which I did not reach. These need a follow-up ticket / a smoke test.

### 2. Re3Sim (InternRobotics, ICRA 2026)
- **What:** Real-to-sim for manipulation: reconstruct scene with 3DGS + OpenMVS geometry, align with hand-eye calibration and an ArUco marker-to-base transform, collect demos in Isaac with a Franka, train ACT. Source: https://github.com/InternRobotics/Re3Sim, paper https://arxiv.org/abs/2502.08645
- **Inputs:** a folder of images plus an alignment image (`python reconstrct.py -i <input>`); calibration scripts (`hand_in_eye_calib.py`, `get_marker2base_aruco.py`). This is the same idea as our ChArUco Anchor.
- **Outputs/target:** Isaac Sim / **Isaac Lab v1.1.0** in a Docker image (CUDA 11.8 installed by hand); Franka USD; LMDB demonstration data.
- **Licence:** **CC BY-NC 4.0** (README) - non-commercial; no code licence file found. Poor fit for reuse as code.
- **Maintenance:** pushed 2026-03-16, ~181 stars; pinned to old Isaac Lab, so porting to Isaac Sim 6 is non-trivial.
- **Coverage:** the only one that spans scene + alignment + Franka task env in Isaac. Borrow the **Anchor/calibration flow and OpenMVS-for-collision-geometry idea**; do not adopt the code.

### 3. SplatSim (Qureshi et al., ICRA 2025)
- **What:** Replace the robot render in a static scene splat with per-link splats moved by simulated joint states; sim is PyBullet (UR5 example); not Isaac. Source: https://github.com/qureshinomaan/SplatSim, https://arxiv.org/abs/2409.10161
- **Inputs:** 1-2 min smartphone video, COLMAP, vanilla gaussian-splatting; robot isolated by a simple rectangular-bounding-box segmentation; robot in a known easy pose for calibration (README tips). The robot-at-known-pose idea matches our decision for masking.
- **Licence:** none declared (GitHub API: null) -> all rights reserved by default. Last push 2025-09-04, ~161 stars.
- **Coverage:** Capture tips and splat/physics decoupling concepts only. It shows the *inverse* of our need (robot kept in splat, articulated), whereas we mask the robot out and render the sim robot with RTX. Borrow ideas only.

### 4. RialTo (Torne et al., RSS 2024)
- **What:** Scan -> digital twin via a GUI (`RialToGUI`) -> RL fine-tuning in sim -> distill with real demos. Policy code targets **Isaac Sim 2022.2.1 + Orbit** (README). Sources: https://github.com/real-to-sim-to-real/RialToPolicyLearning, https://real-to-sim-to-real.github.io/RialTo/
- **Licence:** none declared. Last push 2024-09-17 (RialToGUI 2024-04-09). Effectively unmaintained and Isaac-version-locked.
- **Coverage:** scene + articulated-object workflow concept only; borrow nothing in code.

### 5. GaussGym (Escontrela et al., 2025)
- **What:** Gaussian-splat scenes (iPhone + Polycam -> nerfstudio splat -> mesh slices) as environments for **legged-robot** RL; uses Isaac Gym Preview 4 (setup script) and nerfstudio. Sources: https://github.com/escontra/gauss_gym, paper https://arxiv.org/pdf/2510.15352
- **Licence:** NOASSERTION (BSD-style headers copied from legged_gym). Last push 2025-10-30.
- **Coverage:** shows the "splat for visuals + mesh slices for collision" pattern, but locomotion on Isaac Gym, not Isaac Sim/manipulation. Borrow the splat+collision-mesh decoupling idea only. (I could not confirm the paper's own claims from the abstract page; I inferred scope from the README.)

### 6. DISCOVERSE / DISCOVERSE-Real2Sim (AIR, Tsinghua)
- **What:** 3DGS visual layer + MJCF physics in **MuJoCo**; Real2Sim tools: TRELLIS image-to-3D objects, laser-scan scenes, DiffusionLight relighting, Mesh2GS. Sources: https://github.com/discoverse-dev/DISCOVERSE (MIT, push 2026-03-06, ~522 stars), https://github.com/discoverse-dev/DISCOVERSE-Real2Sim (no licence, push 2025-10-29), paper https://www.arxiv.org/abs/2507.21981
- **Coverage:** MuJoCo only, so no Isaac path. Useful reference for relighting and object assets (both out of v1 scope).

### 7. Other
- **Real2Render2Real** (Berkeley/TRI, CoRL 2025): MIT, pushed 2025-11-21. Generates demonstration data by rendering without dynamics simulation; objects scanned with iPhone/Polycam, hardware: YuMi. Not an Isaac scene pipeline. https://github.com/uynitsuj/real2render2real
- **Vid2Sim** (CVPR 2025): urban navigation from video; MIT (CzzzzH fork, pushed 2026-10-01). Not manipulation. https://github.com/CzzzzH/Vid2Sim
- **RoboGSim:** only a project-page repo (`robogsim/robogsim.github.io`) found; no released code located. Treat as not reusable.
- **NVIDIA PhysicalAI-Robotics-NuRec dataset:** mobile-robot scenes with ParticleField USDZ plus meshes/occupancy maps (per a search snippet; I did not read the dataset card) - a useful reference example of "sim-ready" packaging: https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec

## Comparison

| Pipeline | Input | Output | Sim target | Licence | Maintained | Isaac Sim 6 | Covers (of Capture/Scene/Task env) |
|---|---|---|---|---|---|---|---|
| 3DGRUT + NuRec/ParticleField | COLMAP data (+ masks) | ParticleField USD / NuRec USDZ / PLY | Isaac Sim (also AlpaSim, CARLA) | Apache-2.0 | Yes (2026-06 / 2026-09) | Yes (official) | Scene visuals only |
| Re3Sim | images + align image | 3DGS + OpenMVS geometry, Isaac data | Isaac Lab 1.1.0 | CC BY-NC 4.0 | Moderate (2026-03) | No (old pin) | Scene + alignment + Franka env (concept) |
| SplatSim | phone video | per-link splats | PyBullet | none | Stale (2025-09) | No | Capture tips, masking idea |
| RialTo | scan via GUI | mesh twin | Isaac Sim 2022.2.1/Orbit | none | No (2024) | No | concept |
| GaussGym | Polycam/iPhone | splat + mesh slices | Isaac Gym | NOASSERTION | Stale (2025-10) | No | splat+collision pattern |
| DISCOVERSE | scans / images / TRELLIS | 3DGS + MJCF | MuJoCo | MIT / none | Yes (2026-03) | No | relighting, assets |

## Recommendation

1. **Adopt wholesale:** 3DGRUT (training + ParticleField export) and Isaac Sim 6's native ParticleField rendering, plus `isaacsim.replicator.nurec_utils` as the basis for the held-out render-quality metrics. Pin 3DGRUT v1.1.0 (or a later tag) in the Docker image.
2. **Build ourselves (nothing to reuse):** Capture ingestion from D435i bags, pose estimation with the Robot masked (feed masks as 3DGRUT `_mask.png`), ChArUco Anchor -> metric scale and base alignment (3DGRUT does not recover metric scale), collision-layer mesh from depth, USD composition, Isaac Lab task environment.
3. **Borrow ideas:** Re3Sim (marker-to-base alignment, OpenMVS geometry, Franka-in-Isaac data collection), SplatSim (robot at a known pose, texture-rich background tips), GaussGym (splat visuals + separate collision mesh). Do not copy code from Re3Sim (non-commercial) or unlicensed repos.

## Open uncertainties
- Isaac Lab (3.0 EA vs 2.x) integration of a ParticleField background, and whether policy-camera rendering (tiled/replicator cameras) works with splats under Isaac Lab; needs a spike on the workstation.
- ParticleField vs NuRec rendering fidelity in Isaac Sim 6 (forum report only).
- Whether 3DGRUT can ingest depth/IMU priors (README documents neither); likely needs external pose + scale handling.
- Splat/collision alignment: NVIDIA docs show only a ground plane, not a reconstructed collision mesh.
- RoboGSim code availability; GaussGym paper claims not independently read.
