# Robot masking in handheld walk-around video (ticket 06)

Researched 2026-10-17. Target: RTX 5090 (sm_120), driver 580, Docker. Claims marked **[src]** were read from the cited primary source; **[inference]** is my reasoning, not sourced; **[unverified]** was not confirmed.

## Short answer

Use **SAM 3.1 video tracking (text prompt "robot arm" / "robotic arm with gripper", with a few click refinements on a keyframe)**, with **SAM 2.1 video propagation seeded by manual clicks/box as the fallback**. Run it once per Capture, dilate the mask, then consume it in each stage: zero the depth, pass a per-image mask to the splat trainer, and (optionally) pass it to COLMAP. Rendering the robot model at the known pose is a good *validator* and *refiner*, not the primary method.

## Candidates

### SAM 2 / SAM 2.1 (facebookresearch/sam2)
- Latest is SAM 2.1 (Sept 2024); newest update Dec 2024 (`vos_optimized` full-model compile) **[src: https://github.com/facebookresearch/sam2/blob/main/RELEASE_NOTES.md]**.
- Prompts are clicks, boxes (README lists points/boxes; the video API is `build_sam2_video_predictor`, `init_state`, `add_new_points_or_box`, `propagate_in_video`) **[src: https://github.com/facebookresearch/sam2]**. No text prompt: needs one human (or detector) prompt per Capture.
- Requires Python >=3.10, torch >=2.5.1, torchvision >=0.20.1; builds an optional CUDA extension (needs nvcc; failure is non-fatal) **[src: README]**.
- Docker: repo ships `backend.Dockerfile` and `docker-compose.yaml`, but they target the web demo, not a headless pipeline **[src: https://github.com/facebookresearch/sam2 contents listing, README]**. Expect to write our own Dockerfile.
- License: Apache-2.0 for checkpoints/code **[src: README]**.
- Blackwell: no mention in the repo. PyTorch 2.7 + CUDA 12.8 (cu128) wheels are the first with sm_120 **[src: https://github.com/pytorch/pytorch/issues/159207, maintainer comment]**; torch >=2.5.1 requirement is satisfied by that. **[unverified]** that SAM 2's custom CUDA op builds/works on sm_120; it is optional, so fallback is fine.

### SAM 3 / SAM 3.1 (facebookresearch/sam3)
- Open-vocabulary "Promptable Concept Segmentation": a noun-phrase (or exemplar) prompt returns masks for all matching instances, in images and video, with a memory-based tracker built on the SAM 2 design **[src: https://arxiv.org/abs/2511.16719; https://github.com/facebookresearch/sam3]**.
- SAM 3.1 "Object Multiplex" released 2026-03-27: faster multi-object tracking, new checkpoints `facebook/sam3.1`; "improved VOS on 6 of 7 benchmarks" **[src: https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md]**. We track one object (the robot), so speedups matter little.
- Video API: `build_sam3_video_predictor()`; `handle_request` with `start_session` (`resource_path` = JPEG folder or MP4) then `add_prompt` (`frame_index`, `text`); point refinement shown in notebooks **[src: README]**. The README does not show the propagate call; check the example notebook when implementing.
- Install: Python >=3.12, PyTorch >=2.7, CUDA >=12.6; README example uses `torch==2.10.0` from the cu128 index **[src: README]**. This matches the Blackwell requirement (cu128) so the documented path should work on a 5090, but the README never mentions Blackwell/sm_120 **[unverified on hardware]**.
- Weights are **gated on Hugging Face** (request access, `hf auth login`) **[src: README]**. Pipeline must bake the weights into a private image or mount them; needs an HF token at build time.
- License: "SAM License" (Meta, updated 2025-11-19): royalty-free, redistribution only under the same terms, must acknowledge use in published research, no reverse engineering **[src: https://github.com/facebookresearch/sam3/blob/main/LICENSE]**. Not Apache; read it before redistributing the image outside the lab.
- No Dockerfile in the repo (contents listing: `.github, assets, examples, sam3, scripts, test, ...`) **[src: GitHub contents API]**.

### Grounded-SAM-2 (IDEA-Research)
- GroundingDINO (or Florence-2 / cloud DINO-X) text detection on frame 0, then SAM 2 tracking **[src: https://github.com/IDEA-Research/Grounded-SAM-2]**.
- Ships Dockerfile + Makefile (`make build-image`, `make run`); authors used Python 3.10, torch >=2.3.1, CUDA 12.1; GroundingDINO needs a CUDA compile of its deformable-attention op; last listed update April 2025 **[src: repo README]**. Higher Blackwell/maintenance risk than SAM 3, which does text prompting natively. **Not recommended** now that SAM 3 exists.

### Render the robot model into estimated camera poses
- **[inference]** Needs camera poses in the robot base frame (from the ChArUco Anchor plus SLAM/SfM) and a URDF/mesh at the recorded home pose; then rasterise the mesh to get an exact silhouette. Pros: no learned-model failures on thin/reflective parts, fully automatic once poses exist. Cons: depends on pose error (a few px of misalignment at the thin gripper fingers leaves ghost edges), on exact mesh/joint calibration and home-pose repeatability, on the Robotiq mesh matching the real fitted gripper and cables (cables and mounts are not in the URDF), and on pose estimation running before masking. Since the robot is static in the world it is geometrically consistent for SfM/SLAM, so unmasked pose estimation is fine **[inference]**.
- Best use: render silhouette, compare IoU with the SAM mask per frame as a QA metric, and take the union (SAM mask OR dilated render) for the final mask.

## Quality on thin/reflective parts
- I found **no primary-source benchmark** for thin parts (gripper fingers, cables) or glossy/metal surfaces. SAM 2/3 papers report general VOS benchmarks only (e.g. SAM 3.1 +2.0 on MOSEv2) **[src: RELEASE_SAM3p1.md]**. Treat thin/reflective quality as **[unverified]** and plan a test on mock/real Captures: check fingers, Robotiq pads, cable, and specular flare.
- Mitigations **[inference]**: add positive clicks on the gripper and negative clicks on the background on the first and a mid-trajectory keyframe; re-prompt on frames where IoU against the rendered silhouette drops; dilate the final mask by ~10-20 px (at the working resolution) so edge bleed does not leave a halo in the splat; fill holes.

## How downstream stages consume masks

| Consumer | How masks are consumed | Source |
|---|---|---|
| COLMAP feature extraction | Per-image mask PNG named `<image filename>.png` under `mask_path`, mirroring the image tree; black (0) pixels get no features. `camera_mask_path` for one mask for all images. Check CLI reference for exact prefix (`--ImageReader.mask_path`). | https://colmap.github.io/faq.html |
| nerfstudio (splatfacto) | Per-frame `mask_path` in `transforms.json`; single-channel black/white, black = ignore; must match image resolution; if used, all images need one. Docs warn the implementation "is inefficient and will cause large memory allocations". | https://docs.nerf.studio/quickstart/data_conventions.html |
| gsplat `examples/simple_trainer.py` | Batch key `"mask"` (bool, [1,H,W], optional): rendered colour zeroed at masked pixels, L1 computed on `colors[masks]`, SSIM on masked images. Depth loss does **not** use the mask; eval metrics use full image. Note: the COLMAP parser I read builds a *per-camera* ROI mask (`mask_dict[camera_id]`), not per-image, so per-image robot masks need a small dataset patch or a different loader. | https://raw.githubusercontent.com/nerfstudio-project/gsplat/main/examples/simple_trainer.py ; .../examples/datasets/colmap.py |
| Original 3DGS (graphdeco-inria) | No documented mask feature in README, but code uses an **alpha channel**: `Camera.alpha_mask = image[3:4]` and `train.py` does `image *= alpha_mask` before the L1 loss. So supply RGBA images (alpha = 1 keep, 0 robot). | https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/scene/cameras.py ; .../train.py |
| Open3D TSDF (`ScalableTSDFVolume.integrate`) | Tutorial documents no mask argument and does not state how zero depth is treated. **[inference]** Practical route: set depth to 0 (invalid) at masked pixels before building the `RGBDImage`; verify on a synthetic test that 0-depth is skipped. | https://www.open3d.org/docs/latest/tutorial/pipelines/rgbd_integration.html |
| nvblox | I could not confirm a mask input from its README (docs site page 404). **[unverified]**; the same "zero the depth" trick would apply. | https://github.com/nvidia-isaac/nvblox |

Important caveat: masked pixels carry no supervision, so splats never form there; but splat trainers can still place floaters where the robot was seen only through the mask edge. Conversely, the collision mesh will have a hole where the robot stood. **[inference]** This is fine for the Scene (the robot is re-added in Isaac) but the base/pedestal footprint should be added to the collision layer explicitly or inpainted from the robot USD.

Also, the mask and depth need to be aligned: the D435i depth is registered to color only after `align_to(color)`; mask color-aligned frames and zero the aligned depth **[inference]**.

## Automation needs
- SAM 3: one text prompt per Capture, fully headless; optional human click refinement as a retry path. SAM 2.1: one human-supplied click/box per Capture (can be stored per cell/home pose and reused if the camera starts from a similar view, **[inference]**; a stored 2D prompt is not reusable across arbitrary starting views).
- Because the robot is held at a **known home pose**, a robust automatic seed is possible: prompt SAM 3 with text, accept only if mask IoU with a rendered silhouette (from first-frame pose via the Anchor) exceeds a threshold, otherwise fail loudly **[inference]**.

## Recommendation
1. Primary: SAM 3.1 video predictor, text prompt for the robot arm + gripper, run on the full-resolution colour frames, output per-frame binary masks.
2. Fallback: SAM 2.1 video predictor with box/click prompts (Apache-2.0, ungated, simplest Docker story).
3. QA: render robot silhouette via the URDF at home pose and report per-frame IoU in the quality report; union with the SAM mask if IoU is low near the gripper.
4. Post-process: temporal-consistency check, dilation, hole fill; write masks in the exact formats above (black = ignore for COLMAP/nerfstudio, alpha for original 3DGS, depth=0 for TSDF).
5. Docker: one CUDA 12.8-based image, torch cu128 (>=2.7; SAM 3 README uses 2.10), HF token for gated weights.

## Open uncertainties
- No on-hardware confirmation that SAM 3 / SAM 2 run on sm_120 (only PyTorch-level support is sourced).
- No benchmark of thin/reflective part quality; needs an empirical test on Captures.
- SAM 3 propagate/handle_request details for returning per-frame masks not shown in the README; check `examples/` notebooks.
- gsplat per-image mask loading and nvblox mask support not confirmed; 3DGS alpha mask behaviour taken from code, not docs.
- SAM License terms need legal reading if the image will be shared beyond the lab.
