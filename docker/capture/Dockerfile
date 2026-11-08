# Capture image: ROS 2 Jazzy + realsense2_camera + librealsense (RSUSB backend) + MCAP bag storage.
# ros-jazzy-librealsense2 is built with the RSUSB (libusb) backend, so no host udev rules,
# DKMS or kernel patches are needed: run the container --privileged with /dev mounted.
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-realsense2-camera \
        ros-jazzy-librealsense2 \
        ros-jazzy-rosbag2 \
        ros-jazzy-rosbag2-storage-mcap \
        ros-jazzy-rqt-image-view \
        usbutils \
        libx11-6 libxcb-xinerama0 libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

COPY run.sh /usr/local/bin/r2s-capture-run
RUN chmod +x /usr/local/bin/r2s-capture-run
WORKDIR /captures
ENTRYPOINT ["/usr/local/bin/r2s-capture-run"]
