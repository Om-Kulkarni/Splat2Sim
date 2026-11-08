# Isaac Sim 6 status and fit (ticket 01)

Researched 2026-10-17. Primary sources only unless marked (secondary). Facts about the workstation were read over SSH (read-only).

## Summary

- Isaac Sim 6 is GA. Latest is **6.2.0** (2026-10-14). 6.0.0 GA 2026-06-04, 6.0.1 2026-06-22, 6.1.0 2026-09-10. A 7.0.0a1 pre-release exists (2026-09-18). [GitHub releases](https://github.com/isaac-sim/IsaacSim/releases) (year inferred from the 5.1.0 -> 6.x sequence; the page shows day/month only).
- Isaac Lab for Sim 6 is **3.0, still Early Access** (v3.0.0-EA, 2026-09-16, needs Sim 6.1; GA "targeted toward the end of October 2026"). [IsaacLab releases](https://github.com/isaac-sim/IsaacLab/releases), [v3.0.0-EA notes](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA). Isaac Lab 2.3.x targets Sim 5.1; beta tags v3.0.0-beta/beta2 targeted Sim 6.0.
- Container: `nvcr.io/nvidia/isaac-sim:6.2.0` (multi-arch; tag per release). Headless livestream via `./runheadless.sh`; GUI in container is not recommended. Docs use `--gpus all`; CDI is not mentioned. [Container install](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html)
- **Biggest risk: driver 580 on RTX 5090.** Isaac Sim docs test with **595.58.03** and say nothing about the 5090. The Omniverse requirements page lists 570.169, 580.95.05 and 595.58.03 as validated for Blackwell GeForce (Omniverse-wide, not Isaac Sim 6 specific). No first-party or forum evidence found of Sim 6 on a 5090 with 580. See below.
- Splat rendering works in Sim 6 via OpenUSD `ParticleField` (preferred) or legacy NuRec USDZ. Splats have no automatic collider; attach a separate collision mesh yourself.

## 1. Release status

| Item | Status | Source |
|---|---|---|
| Isaac Sim 6.0.0 | GA, 2026-06-04 (dev builds 2025-12 and 2026-03) | [GitHub releases](https://github.com/isaac-sim/IsaacSim/releases) |
| 6.0.1 / 6.1.0 / 6.2.0 | GA; 6.2.0 is "Latest" | same |
| Kit SDK | 6.0.0: 110.1.1; 6.2.0: 110.4.0 | [6.0.0 notes](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/release_notes.html), [latest notes](https://docs.isaacsim.omniverse.nvidia.com/latest/overview/release_notes.html) |
| Isaac Lab 3.0 | Early Access, needs Sim 6.1; Python 3.12, PyTorch 2.11, Warp 1.16, Newton 1.5.2; GA ~end Oct 2026 | [v3.0.0-EA](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA) |

Python 3.12 and RT 2.0 as default renderer came with the 6.0 early developer release (search snippet of GitHub/forum notes, not fully read; treat as likely).

## 2. Container images and tags

- Image `nvcr.io/nvidia/isaac-sim`, tag `6.2.0` in current docs; companion `nvcr.io/nvidia/omniverse/hub_workstation_cache:2.2.0`. Runs as UID 1234; needs `ACCEPT_EULA=Y`; WebRTC needs `--network=host`. [Container install](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html)
- Docker Compose stack with WebRTC web viewer exists (added 6.0); Newton selectable with `ISAACSIM_PHYSICS_ENGINE=newton` (6.2). [Latest notes](https://docs.isaacsim.omniverse.nvidia.com/latest/overview/release_notes.html)
- Isaac Lab 3.0 EA documents a `uv` install (clone `release/3.0.0`, extra `isaacsim`); the release notes do not mention a Docker/NGC Isaac Lab image. `isaaclab.sh` is deprecated. So the Lab-in-Docker route for Sim 6 must be built by us (e.g. FROM the isaac-sim image or pip-install). Not verified further.
- Workstation facts (SSH): RTX 5090, compute cap 12.0, 32 GB, driver 580.178.04, Ubuntu 24.04.4, Docker 29.7.2, nvidia-ctk 1.20.0. Existing local images: `isaac-lab-base:latest` (label version 2.1.1, i.e. old Lab 2.x on an older Sim). Neither is Sim 6.

## 3. RTX 5090 / Blackwell / driver 580 / Ubuntu 24.04

- OS: Ubuntu 22.04 and 24.04 supported on x86_64 (22.04 deprecated). [Requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- GPU tiers: min RTX 4080 16 GB, "good" RTX 5080, "ideal" RTX PRO 6000 Blackwell. 5090 not named but is a faster 50-series part. Same page.
- Driver: tested on Linux **595.58.03**; for new GPUs it recommends the Latest Production Branch driver. Same page. Driver 580.159.03 appears only for DGX Spark (aarch64).
- Omniverse technical requirements validate Blackwell GeForce on 570.169, 580.95.05, 595.58.03; says newer may work; 570.144 and older had a shader-compile crash. [Technical requirements](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html). 580.178.04 is newer than 580.95.05 in the same branch, so it should be inside the validated envelope for Kit, but this is inference.
- Known issue: Franka Open Drawer example may not work on Blackwell by default (raise physics rate to 600). [Known issues](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/overview/known_issues.html) Relevant as a warning for Franka contact tuning.
- Forum evidence (user reports, not authoritative): Sim 5.1.0-rc crash on 5090 with 580 driver [thread](https://forums.developer.nvidia.com/t/isaac-sim-5-1-0-rc-19-crashes-on-startup-with-rtx-5090-and-nvidia-580-open-driver/374184); Sim 6.0.0 Docker WebRTC failure on 4090 with 580.126 [thread](https://forums.developer.nvidia.com/t/isaac-sim-6-0-0-webrtc-livestream-fails-with-streamsdkexception-800b0000-on-a-headless-rtx-4090-docker-server/372612). Workstation has monitors, so WebRTC is not needed; but GUI-in-container is unsupported by docs, so use headless Python / offscreen rendering, or a native (pip/binary) install for GUI.
- Verdict: **unconfirmed**. Must smoke test on the actual box (cheap: pull `isaac-sim:6.2.0` ~ multi-GB, run headless standalone, render one frame, check for crash). Fallback: upgrade driver to 595.58.03 (the tested one), noting 595.x also has anecdotal crashes on other hardware (DGX Spark, 4090).

## 4. Gaussian splat support and loading into a stage

- Sim 6 renders splats as OpenUSD `ParticleField` geometry (3DGS, 3DGUT) alongside mesh content with RTX. [Neural Volume Rendering](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_nurec.html), [Omniverse Gaussian Splats](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/particle-fields.html)
- Two formats: `ParticleField` USD(Z) (standard, `ParticleField3DGaussianSplat`; preferred) and legacy NuRec `Volume` USDZ (Kit 107.3 to 110.1, Sim 5.0 to 6.0; "going to be deprecated"; NuRec workflows described as beta). [3DGRUT export README](https://github.com/nv-tlabs/3dgrut/blob/main/threedgrut/export/README.md), [3DGRUT repo](https://github.com/nv-tlabs/3dgrut)
- Getting splats in: train with 3DGRUT (`export_usd.enabled=true`), or convert PLY via `threedgrut.export.scripts.transcode model.ply -o model.usdz --format lightfield` (ParticleField) or `ply_to_usd` (NuRec). Alternatives: OpenUSD's `py3dgsPlyToUsd.py`, or Kit's GSplat Converter extension (`omni.kit.converter.gsplat`, .ply/.spz). [GSplat Converter](https://docs.omniverse.nvidia.com/extensions/latest/ext_gsplat-converter.html)
- Stage composition caveat: adding references to a stage opened from a `.usdz` fails. Use a `.usd/.usda` root and reference the `.usdz`. [Neural Volume Rendering](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_nurec.html)
- Render settings: single-GPU only; recommended `/rtx/spg/enabled` + `omni.rtx.spg` for SPG/PPISP assets; disable DLSS Frame Generation if artifacts appear. Same page.
- `isaacsim.replicator.nurec_utils` (new in 6.0.1; 0.1.3 in 6.2) auto-detects particle fields/volumes, applies render settings, captures frames at TUM poses, and computes PSNR/SSIM. Marked experimental. [NuRec utils](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/nurec_utils.html) Directly useful for the held-out render quality metric in the quality report.
- Known issue (early 6.0 dev notes via search snippet): NuRec scenes with a matte object do not render correctly under RT 2.0; workaround is RT 1.0. Not confirmed fixed.
- Coordinates: standard export `--up-axis` is metadata only; NuRec export is always Z-up; a normalizing transform is applied by default (`--no-transform` to skip). 3DGRUT export README. Our Anchor-derived transform should go on a parent Xform, not baked.

## 5. Colliders for a splat scene

- No official doc found describing automatic collision for splats; splats are visual only. The documented mechanism is a **separate mesh**: 3DGRUT's `add_mesh_to_usdz --mesh_ply mesh.ply --set_collision [--set_invisible]` packages a collision mesh in the USDZ. [3DGRUT export README](https://github.com/nv-tlabs/3dgrut/blob/main/threedgrut/export/README.md)
- Alternative from Sim docs/tutorial: NuRec Volume prim has a Proxy field for a ground-plane/proxy geometry (secondary: [hyper.ai tutorial](https://hyper.ai/en/stories/8cb9fb859ce040d2fd8ca9ca4bc36efb); [Pebblous](https://blog.pebblous.ai/report/nvidia-nurec-real-to-sim-2026-07/en/)).
- Forum report: a 3DGRUT USDZ loaded in Sim 5.1 with no collision mesh, and an nvblox mesh was about 100x off in scale. [thread](https://forums.developer.nvidia.com/t/loading-usdz-files-from-nurec-pipeline-into-isaac-sim-does-not-load-mesh-file/352914) Confirms we must own mesh generation and scale alignment (both from the same metric Anchor transform).
- Practical design: Scene USD = root `.usda` referencing (a) ParticleField splat, (b) collision mesh with `UsdPhysics.CollisionAPI` (+ optionally `MeshCollisionAPI` approximation; set invisible), under a common Xform carrying the Anchor transform. Static, so no rigid body.

## 6. Breaking changes 5.x -> 6.x relevant to us

From the [6.0.0 notes](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/release_notes.html) and [6.2 notes](https://docs.isaacsim.omniverse.nvidia.com/latest/overview/release_notes.html):
- Removed `omni.isaac.*` shims; use `isaacsim.*`. Python module bindings moved into `bindings/` subdirs.
- Removed `isaacsim.asset.browser`, `isaacsim.app.selector`; RTX sensors via JSON camera prims removed (use OmniSensor / `isaacsim.sensors.experimental.rtx`).
- Deprecated `isaacsim.core.api/prims/utils` (to `isaacsim.core.experimental.*`), `isaacsim.sensors.camera`, `isaacsim.robot.manipulators`; Cortex removed in 6.2.
- URDF and MJCF importers rewritten (3.0) and now author Newton schemas; `PhysxArticulationAPI` no longer written on imported roots. Matters for any Robotiq 2F-85 URDF import.
- Multi-backend physics (PhysX default, Newton experimental). Kit 110, Python 3.12, RT 2.0 default.
- Isaac Lab 3.0: quaternion order now (x,y,z,w); `.data` returns `ProxyArray`; `--headless` removed (`--viz none`); `make_instanceable` unsupported; importers rewritten; surface-gripper tasks need CPU PhysX. [v3.0.0-EA](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA) Franka Lift is a validated PhysX/Newton transfer example.
- No Robotiq-specific known issue was found.

## Recommendation

Target **Isaac Sim 6.2.0 container** with **ParticleField USDZ** splats plus a separately generated collision mesh referenced from a `.usda` root. Pin Isaac Lab to v3.0.0-EA now and bump to 3.0 GA when released (end Oct 2026); alternatively develop Milestone 1 (Scene in Isaac Sim) on bare Isaac Sim first since it does not need Lab. Hold a fallback to Sim 5.1 / Lab 2.3.x only if the 5090 smoke test fails and driver 595 does not fix it.

## Open uncertainties

1. Sim 6.x on RTX 5090 + driver 580.178.04 + Docker/CDI: no direct evidence. Needs a smoke test (headless render of a ParticleField stage).
2. CDI vs `--gpus all`: docs only describe the latter; NVIDIA CDI should work with Docker 29 but is unverified here.
3. Whether the NuRec matte-object RT 2.0 bug is fixed in 6.2; whether ParticleField renders correctly without SPG assets (docs cover both).
4. Isaac Lab 3.0 container story is undocumented in the EA notes; GA may add it.
5. Release years on GitHub pages are inferred.
6. Exact 6.1 and 6.0 migration guide content was not read (only sidebar links seen).
