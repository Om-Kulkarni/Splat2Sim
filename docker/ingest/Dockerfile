# ingest service: Capture (ROS 2 MCAP / librealsense .bag / TUM folder) -> frames/. Pure python, no ROS install.
# The package is bind-mounted at run time by compose (./splat2sim -> /opt/r2s/splat2sim), so no rebuild on code edits.
# imageio-ffmpeg (bundled static ffmpeg, ~30 MB) encodes the report fly-through MP4: the Isaac image has no ffmpeg backend.
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=300 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/opt/r2s
RUN pip install "rosbags>=0.10" "mcap>=1.2" "mcap-ros2-support>=0.5" "numpy>=1.26,<2" \
    "opencv-contrib-python-headless==4.10.0.84" "scipy>=1.11" "pillow>=10" "pyyaml>=6" "pytest>=8" "imageio>=2.34" "imageio-ffmpeg>=0.5"
WORKDIR /opt/r2s
