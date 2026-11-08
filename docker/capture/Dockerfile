# Capture image: ROS 2 Jazzy + MCAP bag storage + both camera drivers (select with R2S_CAMERA=oak|realsense at run time).
#  - OAK-D Pro W (default): ros-jazzy-depthai-ros-driver-v3 (DepthAI v3, from the main ROS apt repo) + the `depthai`
#    Python wheel in /opt/r2s-venv for `--check` (device, USB speed, calibration).
#  - RealSense D455: ros-jazzy-realsense2-camera + librealsense built with the RSUSB (libusb) backend, so no
#    DKMS or kernel patches are needed.
# Neither needs host drivers: run the container --privileged with /dev mounted (scripts/capture.sh does that).
# For the OAK on a bare host a udev rule is optional but avoids a re-enumeration permission race, see docs/capture-guide.md.
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-depthai-ros-driver-v3 \
        ros-jazzy-realsense2-camera \
        ros-jazzy-librealsense2 \
        ros-jazzy-rosbag2 \
        ros-jazzy-rosbag2-storage-mcap \
        ros-jazzy-rqt-image-view \
        python3-venv python3-pip \
        usbutils \
        libx11-6 libxcb-xinerama0 libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# DepthAI Python (device probe for `--check`). Version pinned to the one the OAK-D Pro W was validated with.
RUN python3 -m venv /opt/r2s-venv \
    && /opt/r2s-venv/bin/pip install --no-cache-dir depthai==3.12.0 numpy

LABEL r2s.capture.version=2
ENV DEPTHAI_TELEMETRY=0
COPY run.sh /usr/local/bin/r2s-capture-run
COPY check_oak.py /usr/local/bin/r2s-check-oak
COPY oak_params.yaml /etc/r2s/oak_params.yaml
RUN chmod +x /usr/local/bin/r2s-capture-run /usr/local/bin/r2s-check-oak
WORKDIR /captures
ENTRYPOINT ["/usr/local/bin/r2s-capture-run"]
