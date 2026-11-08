# Isaac image

How the `isaac` service image (Isaac Sim 6.1 + Isaac Lab 3.0) is built and run. Used by `compose`, `report` and `env`
([Architecture](architecture.md)).

The `isaac` service uses the locally built `r2s-isaaclab3-sim61` (Isaac Lab 3.0 `release/3.0.0` `docker/Dockerfile.base`
with `ISAACSIM_VERSION_ARG=6.1.0`; build commands in `docker/isaac/Dockerfile`). Flags (EULA, asset root pinned to 6.1,
cache volumes, GPU reservation) are in `compose.yaml`; `./r2s` creates the world-writable cache dirs under
`$R2S_DATA/cache/isaac`.

| Topic | Note |
|---|---|
| CDI | replace the `deploy` block with `devices: ["nvidia.com/gpu=all"]` (not available on the workstation (RTX 3090 Ti): no `/etc/cdi`; use `--gpus all`/the `deploy` block, verified there) |
| Build tip | pin Isaac Lab to commit `5870804` (`git fetch --unshallow`, `git checkout 5870804`; the branch head has moved) and add `ENV UV_HTTP_TIMEOUT=900` to `docker/Dockerfile.base` |
| Last build | on the workstation on 2026-10-18 (51.8 GB, about 10 min with fast bandwidth ~25 MB/s) |
| Smoke tests | `spikes/isaac-smoke/run_smoke.sh` and `run_lab.sh` pass on the 3090 Ti (driver 580.178.04) |
