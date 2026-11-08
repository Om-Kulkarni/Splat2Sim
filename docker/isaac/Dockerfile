# The isaac service uses the locally built Isaac Lab 3.0 image on Isaac Sim 6.1 directly
# (compose.yaml: image r2s-isaaclab3-sim61). Build it once (see README, "Isaac image"):
#   git clone -b release/3.0.0 https://github.com/isaac-sim/IsaacLab.git
#   (add `ENV UV_HTTP_TIMEOUT=900` to docker/Dockerfile.base)
#   docker build -f docker/Dockerfile.base -t r2s-isaaclab3-sim61 \
#     --build-arg ISAACSIM_VERSION_ARG=6.1.0 --build-arg ISAACSIM_ROOT_PATH_ARG=/isaac-sim \
#     --build-arg ISAACLAB_PATH_ARG=/workspace/isaaclab --build-arg DOCKER_USER_HOME_ARG=/root .
# Pipeline scripts (compose_scene.py, report.py, ...) live in docker/isaac/scripts and are bind-mounted.
FROM r2s-isaaclab3-sim61:latest
