# 3dgrut service image (`r2s-3dgrut`)

3DGRUT (3DGUT trainer, OpenUSD ParticleField export) for the RTX 3090 Ti (sm_86, current workstation) and RTX 5090 (sm_120).

- Upstream `nv-tlabs/3dgrut` pinned to commit `e6f4552` (main, 2026-09-22; pyproject 2.0.0, no 2.0.0 tag exists).
- Base `nvidia/cuda:13.0.0-base-ubuntu24.04` (already on the workstation) + CUDA **12.8** toolkit from NVIDIA's apt repo
  (nvcc + headers for the torch JIT; avoids a 3 GB `-devel` pull), PyTorch **2.8.0+cu128**, Python 3.11 (uv),
  tiny-cuda-nn, slangc **2026.5.2**. Image 23.2 GB on disk. Build: `docker compose build 3dgrut`.
- The CUDA tracer (`lib3dgut_cc`) is JIT-compiled on first run (~2.5 min) into `/cache/torch_extensions`
  (mounted from `$R2S_DATA/cache/3dgrut`); later runs skip it.
- Run as the host uid (`-u $(id -u):$(id -g) -e HOME=/tmp`): `/opt/3dgrut/threedgut_tracer` is `a+rwX` (slangc writes
  `threedgutSlang.cuh` there) and `/cache` is mode 1777 in the image. The host cache dir must be writable by that uid.
- Patch in the Dockerfile: upstream's USD exporter reports the *active* SH degree (rises every 1000 iterations) but writes
  all max-degree coefficients, so runs < 3000 iterations crash with `cannot reshape array ... (N,3,3)`.
  `sed` makes it report the max degree. Drop once upstream fixes it.
- `tools/rgbd_poses_to_colmap.py`: dev helper that builds a COLMAP text dataset from RGB-D frames + RTAB-Map poses
  (depth back-projection for the initial points, optional masks). Not the COLMAP stage.
- `tools/grut_usd_render.py`: Isaac Sim script that opens the exported `.usdz` and renders the embedded camera.

## Train + export (verified)

Input: COLMAP text/binary dataset (`images/`, `sparse/0/`), optional `<name>_mask.png` per image.

```bash
D=~/real2sim-data/smoke/recon   # colmap/  runs/ (writable)
docker run --rm --gpus all -u $(id -u):$(id -g) -e HOME=/tmp \
  -v $D/colmap:/data/colmap -v $D/runs:/runs -v ~/real2sim-data/cache/3dgrut:/cache \
  r2s-3dgrut python train.py --config-name apps/colmap_3dgut.yaml \
  path=/data/colmap out_dir=/runs experiment_name=smoke n_iterations=2000 use_wandb=false with_gui=false \
  test_last=false val_frequency=999999 dataset.normalize_world_space=false dataset.test_split_interval=0 \
  num_workers=4 export_usd.enabled=true export_usd.apply_normalizing_transform=false \
  export_usd.path=/runs/smoke/smoke.usdz checkpoint.iterations="[2000]"
```

Outputs: `runs/smoke/colmap-<date>/ckpt_last.pt`, `runs/smoke/smoke.usdz` (`default.usda` + `gaussians.usdc`;
`ParticleField3DGaussianSplat` at `/World/gaussians/Gaussians/gaussians`, Y-up, plus the first training camera
`/World/Cameras/camera_0`). With `normalize_world_space=false` the splat stays in the input (RTAB-Map world) frame.

## Isaac check

```bash
cp $D/runs/smoke/smoke.usdz ~/real2sim-data/smoke/data/
cp docker/3dgrut/tools/grut_usd_render.py ~/real2sim-data/smoke/scripts/
bash spikes/isaac-smoke/run_smoke.sh grut_usd_render.py     # -> ~/real2sim-data/smoke/out/grut_render.png
```

## Smoke test (2026-10-18, RTX 5090; old machine, gone)

Dataset: 200 SLAM&Render frames (848x480, masked) with RTAB-Map poses, 3DGUT, 2000 iterations.
- Training: **15.0 s** at 133 it/s (213k Gaussians); wall 2m52s incl. container start, dataset load, export (first run
  adds ~2.5 min of JIT). GPU memory used peaked at **1.46 GiB** (nvidia-smi, whole GPU, 2 s sampling).
- Export: `smoke.usdz` 50 MB.
- Isaac Sim 6.1.0 (`nvcr.io/nvidia/isaac-sim:6.1.0`) opens the usdz and renders the splat through `camera_0`
  (848x480, scene recognisable: objects on a table), warm run well under 1 min.

## Smoke test (2026-10-18, RTX 3090 Ti workstation)

Image built here from this Dockerfile (23.3 GB on disk, 8.2 GB content; `torch.cuda.get_arch_list()` =
`sm_70 .. sm_86, sm_90, sm_100, sm_120`, a CUDA matmul and `import tinycudann` verified on the 3090 Ti).
Dataset: 200 SLAM&Render frames (frames 700..899 of `1-natural-tr`, 848x480, **not masked**: the mask stage is not on main),
RTAB-Map RGB-D+IMU poses with fx=fy=609.5 (see `docker/rtabmap/README.md`), 3DGUT, 2000 iterations.
- Training: **31.9 s** at 62.7 it/s (about half the 5090's 133 it/s), ~149k Gaussians; wall 1m22s per run incl. container start,
  dataset load, export (the tracer JIT is cached in `$R2S_DATA/cache/3dgrut`). Whole-GPU memory peak 976 MiB with a 166 MiB
  baseline, i.e. about **0.79 GiB** for the run (old 1.46 GiB, but with 213k Gaussians).
- Export: `smoke.usdz` 35 MB. USD prints a harmless "Could not open asset @gaussians.usdc@" warning during export.
- Isaac Sim 6.1.0 opens the usdz and renders `camera_0` (848x480, 36 s warm). The splat is **blurry and streaky** (colours and
  layout recognisable, objects are not): held-out test PSNR 14.0 dB / SSIM 0.66 (`dataset.test_split_interval=10 test_last=true`)
  with either the fx=425 or fx=609.5 poses. Only 0.5 m of nearly static camera motion and 2000 iterations; this is a
  plumbing smoke, not a quality benchmark. Do not read the 5090's "scene recognisable" as a baseline for quality.
