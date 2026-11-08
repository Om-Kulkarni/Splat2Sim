# syntax=docker/dockerfile:1.7
# RTAB-Map 0.23.8 (offline RGB-D / RGB-D+IMU pose estimation), headless CLI tools only.
# The official introlab3it/rtabmap:noble image already carries the whole toolchain and deps
# (OpenCV 4.6 + contrib, PCL, GTSAM 4.2, OpenGV, ROS Jazzy base) but ships RTAB-Map 0.24.2 (2026-10-14),
# so we pin that image by digest and rebuild the pinned 0.23.x tag over it (installs into /usr/local,
# replacing the 0.24.2 binaries/libs). CPU only; no GPU needed.
FROM introlab3it/rtabmap@sha256:f49d2629bdc14a3219b6bd1988947f762d8df16619de07bee3dce7ecd20a866c

ARG RTABMAP_TAG=0.23.8
# Patch: stock rtabmap-rgbd_dataset hard-codes TUM Kinect intrinsics + 5000/m depth; the patch adds
# --fx/--fy/--cx/--cy/--width/--height/--depth_factor/--rgb_dir/--depth_dir/--gt/--imu/--max_frames.
COPY rgbd_dataset_generic.patch /tmp/rgbd_dataset_generic.patch
RUN --mount=type=cache,target=/root/.ccache \
    git clone --depth 1 --branch ${RTABMAP_TAG} https://github.com/introlab/rtabmap.git /tmp/rtabmap && \
    cd /tmp/rtabmap && git apply --whitespace=nowarn /tmp/rgbd_dataset_generic.patch && \
    source /ros_entrypoint.sh && \
    mkdir -p /tmp/rtabmap/build && cd /tmp/rtabmap/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DWITH_OPENGV=ON -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_APP=ON .. && \
    make -j"$(nproc)" && make install && ldconfig && \
    rm -rf /tmp/rtabmap /root/rtabmap

ENV QT_QPA_PLATFORM=offscreen
WORKDIR /data
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["rtabmap-console", "--version"]
