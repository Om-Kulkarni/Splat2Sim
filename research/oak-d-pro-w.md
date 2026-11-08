# OAK-D Pro W as a replacement for RealSense D455 (handheld RGB-D + IMU capture)

Researched 2026-11-08. Items marked **[UNVERIFIED]** could not be confirmed from a primary source. Several fetches returned summaries, not raw pages; treat parameter defaults as "per source file on `develop`", not guaranteed for the packaged release.

## 1. Hardware

Sources: [OAK-D Pro W](https://docs.luxonis.com/hardware/products/OAK-D%20Pro%20W.md), [OAK-D Pro W PoE](https://docs.luxonis.com/hardware/products/OAK-D%20Pro%20W%20PoE.md).

- **Two colour-sensor variants exist. Check which one you own** (`lsusb`/calibration dump/product label):
  - IMX378: 1/2.3", 4056x3040 max, **rolling shutter**, fixed focus, 120 deg DFOV / 95 HFOV / 72 VFOV.
  - OV9782: 1/4", 1280x800, **global shutter**, fixed focus, 150 DFOV / 127 HFOV / 79.5 VFOV.
  - A global-shutter RGB is better for handheld SLAM and 3DGS (no rolling-shutter skew). 1280x720@30 fits either variant.
- **Mono pair:** 2x OV9282, global shutter, 1280x800, fixed focus, 150 DFOV / 127 HFOV / 79.5 VFOV, **baseline 7.5 cm**. Both have 940 nm notch filters, so they see visible light plus the IR dot pattern and flood.
- **IR in the RGB image:** Docs say "the color camera doesn't perceive IR light". A web-search summary said the Pro W page lists a 650 nm IR-cut filter on the colour camera, but I could not reproduce that from the fetched page **[UNVERIFIED]**. Expect the dot pattern NOT to show in RGB. Verify empirically in a dark scene at full projector power.
- **IMU: BNO086** (9-axis, with on-chip fusion). Not BMI270 on this model. The driver's IMU handler treats BNO086 as having a magnetometer and rotation vector on RVC2 ([imu_param_handler.cpp](https://raw.githubusercontent.com/luxonis/depthai-ros/develop/depthai_ros_driver/src/param_handlers/imu_param_handler.cpp)). Driver defaults: accel 480 Hz, gyro 400 Hz, mag 100 Hz, rotation 400 Hz. Ceilings per the BNO086 datasheet are not checked here.
- **Depth range:** ideal ~0.7 m to 12 m. MinZ is ~20 cm (400P + extended disparity), ~40 cm (800P + extended), ~70 cm (800P, no extended). Error is <2% under 3.5 m, <4% for 3.5-6.5 m, <6% for 6.5-9 m.
- **IR emitters:** dot projector and flood LED, up to 1 W each, Class 1 laser. Power draw can reach ~15 W, so Luxonis recommends a 15 W+ supply or Y-adapter. A laptop USB port may brown out under projector plus flood **[UNVERIFIED for your laptop]**.
- **USB:** USB 2/3, up to 10 Gbps listed. The driver defaults `i_usb_speed: SUPER`. Use a USB3 cable and port. A USB2 fallback is the common cause of dropped frames.
- **Distortion model:** not confirmed for the W variant. The [manual calibration doc](https://docs.luxonis.com/hardware/platform/depth/manual-calibration.md) says wide-FOV lenses use an extended perspective model (radial + tangential + thin-prism, not all 14 coefficients active, plus tilt terms) and that Luxonis "generally avoids fisheye calibration". So expect an OpenCV-style rational/extended model, not equidistant fisheye, **[UNVERIFIED]**. Read it from the device (section 6) and check `getDistortionModel`/coefficient count. A forum thread reports edge artifacts in undistortion on Pro W PoE ([thread](https://discuss.luxonis.com/d/5801-poor-calibration-on-the-oak-d-pro-w-poe)).

## 2. ROS 2 Jazzy driver

Sources: [DepthAI ROS](https://docs.luxonis.com/software-v3/depthai/ros.md), [driver page](https://docs.luxonis.com/software-v3/depthai/ros/driver/), [depthai-ros repo](https://github.com/luxonis/depthai-ros), [Jazzy sync 2026-04-13](https://discourse.openrobotics.org/t/new-packages-for-jazzy-jalisco-2026-04-13/54004).

- **Binaries exist.**
  - Jazzy/Humble use v3 package names: `ros-jazzy-depthai-ros-v3`, `ros-jazzy-depthai-ros-driver-v3` (3.1.1-2 in the April 2026 sync). The docs say to enable the ROS *testing* repo first: `sudo apt install ros2-testing-apt-source && sudo apt update && sudo apt install ros-jazzy-depthai-ros-v3`.
  - The old v2 driver `ros-jazzy-depthai-ros` also exists (changelog up to 2.12.2, Jan 2026), based on DepthAI v2. Kilted and newer drop the `_v3` suffix.
  - Prefer v3. The package version may have moved since April 2026 **[check `apt-cache policy`]**.
- **Launch:** `ros2 launch depthai_ros_driver_v3 driver.launch.py` (RGBD plus a spatial NN, so override it). Use `rgbd_pcl.launch.py` for plain RGBD, or `rtabmap.launch.py` / `vio.launch.py`. Params file: `params_file:=/path/x.yaml`. Config examples live in `depthai_ros_driver/config/` (`rgbd.yaml`, `oak_d_pro_w.yaml`, `rtabmap.yaml`, `vio.yaml`).
- **Parameter names (v3, verified from `develop` sources).** Node namespace/name defaults to `oak`. Sub-nodes are `rgb`, `left`, `right`, `stereo`, `imu`, `driver`, `pipeline_gen`.
  - Pipeline: `pipeline_gen.i_pipeline_type: rgbd`, `pipeline_gen.i_nn_type: none`, `pipeline_gen.i_enable_imu: true`, `pipeline_gen.i_enable_rgbd`.
  - RGB: `rgb.i_width`, `rgb.i_height`, `rgb.i_fps` (float, default 30), `rgb.i_resize_mode` (CROP/LETTERBOX/STRETCH), `rgb.i_undistorted` (default true), `rgb.i_publish_full_resolution`, `rgb.i_synced`, `rgb.i_publish_topic`, `rgb.i_low_bandwidth*`.
  - RGB manual controls (runtime-changeable `r_`): `rgb.r_set_man_exposure` + `rgb.r_exposure` (us, 1-33000) + `rgb.r_iso` (100-1600); `rgb.r_set_man_whitebalance` + `rgb.r_whitebalance` (K); `rgb.r_set_man_focus`, `rgb.r_focus`; `rgb.r_set_auto_exposure_limit` + `rgb.r_auto_exposure_limit`.
  - Stereo: `stereo.i_depth_preset` (FAST_ACCURACY default in code; DEFAULT, FACE, HIGH_DETAIL, ROBOTICS), `i_lr_check` (true), `i_subpixel` (true), `i_subpixel_fractional_bits` (3), `i_extended_disp` (false), `i_stereo_conf_threshold` (15), `i_lrc_threshold` (10), `i_disparity_width` (DISPARITY_96), `i_enable_distortion_correction` (true), `i_width`/`i_height`/`i_fps`, `i_aligned` (align to board socket; default CAM_A = RGB), plus temporal/speckle/spatial/threshold/decimation filters (`i_enable_*`, `i_threshold_filter_min_range` 400, `max_range` 15000 mm; off by default).
  - IMU: `imu.i_message_type` (`IMU` | `IMU_WITH_MAG` | `IMU_WITH_MAG_SPLIT`), `imu.i_sync_method` (`COPY` | `LINEAR_INTERPOLATE_GYRO` | `LINEAR_INTERPOLATE_ACCEL`), `i_acc_freq`, `i_gyro_freq`, `i_enable_acc/gyro/mag/rotation`, `i_batch_report_threshold` (5), `i_max_batch_reports` (10). `IMU` message type = united accel+gyro in `sensor_msgs/Imu`, matching the RealSense `unite_imu_method` output.
  - IR: `driver.i_enable_ir: true`; **`r_laser_dot_intensity` (0-1, default 0.6)** and **`r_floodlight_intensity` (0-1, default 0.6)**, declared in the driver node (set as `driver.r_laser_dot_intensity` at runtime via `ros2 param set` **[exact prefix UNVERIFIED]**). The v2 names `i_laser_dot_brightness`/`i_floodlight_brightness` were not found in v3.
  - Other: `driver.i_usb_speed` (SUPER), `driver.i_device_id`, `driver.i_publish_tf_from_calibration` (true), `driver.i_rs_compat`, `driver.i_calibration_dump`, `driver.i_external_calibration_path`, `sync_threshold`/`sync_attempts` (sync node: 10 ms / 10).
- **Depth aligned to RGB:** on RVC2 (this device) the StereoDepth node's own align input is fed from the RGB socket (CAM_A) on-device (`i_aligned`; `i_run_align_on_host` only affects RVC4). Stereo is processed at `stereo.i_width/i_height`; ensure those equal the RGB size (1280x720) so aligned depth and colour match **[UNVERIFIED how the driver resizes, test it]**.
- **Undistortion:** `rgb.i_undistorted` (default true) enables device-side undistortion (`enableUndistortion` on `requestOutput`; per the [undistort example](https://docs.luxonis.com/software-v3/depthai/examples/camera/camera_undistort.md), on RVC2 `alphaScaling` only works if FOV is >90 deg and **undistortion changes the effective FOV** via crop). Intrinsics are not rewritten by the API. Check the published `camera_info`: it must describe the undistorted image (D should be ~0). **[UNVERIFIED]** whether the aligned depth follows the undistorted RGB geometry. Verify by checking depth/RGB edge overlay. For COLMAP/3DGRUT, either use the undistorted stream with a PINHOLE camera, or record distorted and undistort offline.
- **Topics (`~` = `/oak`):** by source: stereo publishes `~/stereo/image_raw` (+ `camera_info`); RGB `~/rgb/image_raw` (+ `camera_info`); mono `~/left/image_raw`, `~/right/image_raw` (rect variants `~/left/image_rect` etc., off by default); IMU `~/imu/data`; also `~/rgbd/*` for the rgbd node. These follow the code (`/image_raw`) but the full topic list was not on any page, so confirm with `ros2 topic list`. With `rs_compat:=true`: namespace/node `camera`, `color`, `depth`, `infra1/2`, frame `camera_link`, and the default depth profile becomes 1280x720@30, which is the nearest drop-in for the existing D455 layout.
- **Depth encoding:** RealSense-style stereo ROS image, expected 16UC1 mm **[UNVERIFIED; stereo.cpp only shows publisher config. Check `ros2 topic echo --once /oak/stereo/image_raw --field encoding`]**.
- **Frames / TF:** parent `oak_parent_frame`, base `oak`, per-sensor frames prefixed with the node name (e.g. `oak_rgb_camera_optical_frame`, `oak_imu_frame`), taken from the device calibration. `tf_static` is published by robot_state_publisher. Set `i_tf_imu_from_descr: true` if the IMU extrinsics are zero.
- **Sync / timestamps:** the `sync` node (`sync_threshold` 10 ms) plus per-stream `i_synced`; `imu.i_get_base_device_timestamp` and `i_update_ros_base_time_on_ros_msg` control the clock base. Device timestamps are sync'd to host monotonic time. **[UNVERIFIED: RGB/depth/IMU skew in practice; check header stamps.]**
- **Jazzy issues:** none found in sources. Packages come from ROS *testing*. The Docker images listed in docs are Kilted only (`luxonis/depthai-ros:kilted-latest`). Build your own on `ros:jazzy-ros-base`.

## 3. Docker

Sources: [Luxonis Docker discussions](https://discuss.luxonis.com/d/200-running-pipeline-on-docker-container/23), [driver docs](https://docs.luxonis.com/software-v3/depthai/ros/driver/), [X-Link device not found in ROS container](https://discuss.luxonis.com/d/5254-x-link-device-not-found-inside-depthai-ros-container/18).

- Host udev rule (on the host, not necessarily in the image): `echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules && sudo udevadm control --reload-rules && sudo udevadm trigger`.
- Run with `--privileged -v /dev/bus/usb:/dev/bus/usb` (docs also use `-v /dev/:/dev/`) and `--network host`. Without privileged you need `--device-cgroup-rule='c 189:* rmw'` plus the bus mount.
- **Re-enumeration:** on boot the VPU shows as 03e7:2485 (unbooted/bootloader-less); after the host loads firmware it re-enumerates as 03e7:f63b (booted). The USB device node changes, so a container started with a single `--device /dev/bus/usb/001/005` loses access. Mounting the whole `/dev/bus/usb` tree (privileged) handles it. Both IDs are covered by the vendor-id rule. Luxonis staff advise NOT installing udev inside the image. The 2485/f63b IDs are from the task brief and not confirmed in the pages I read **[UNVERIFIED]**.
- Restarting the container after a crash can leave the device in a booted state or on a stale node; unplug/replug if "X_LINK_DEVICE_NOT_FOUND" appears.
- Set `DEPTHAI_TELEMETRY=0` (driver README) to disable anonymous telemetry. The firmware is downloaded/embedded with the library, so no internet is needed at runtime for RVC2 (**[UNVERIFIED]**).

## 4. Recommended settings (SLAM + reconstruction)

Judgement based on the parameters above; Luxonis does not publish an official "SLAM recipe" that I found. Validate empirically.

- Stereo: 800P mono input, `i_depth_preset: ROBOTICS` or `HIGH_DETAIL` (FAST_ACCURACY is the driver default), `i_lr_check: true`, `i_subpixel: true` (better far accuracy, +cost), `i_stereo_conf_threshold` 200-240 range is **not** the driver's scale (it defaults to 15) so tune from the default **[UNVERIFIED scale]**, `i_extended_disp: false` unless you need <40 cm (it costs range and runs only at lower res), `i_enable_threshold_filter: true` with min 400 / max 6000-8000 mm for room scale, optionally `i_enable_speckle_filter` and `i_enable_temporal_filter` (temporal can smear moving handheld scenes).
- RGB: for 3DGS, lock exposure (`r_set_man_exposure: true`, short exposure ~4-8 ms, ISO 100-400) and white balance (`r_set_man_whitebalance: true`, ~4500-5500 K) to avoid brightness flicker in splats. Auto-exposure limit is a softer alternative.
- Dot projector: improves depth on textureless walls/floors, and the RGB camera does not see 940 nm, so projector ON for depth is fine. Start at 0.6 and test. Flood LED: leave at 0 unless the scene is dark, since it adds a uniform IR wash that reduces dot contrast, and mono-based features are not used here. Caveat: with the projector on, mono images carry the dot pattern, which hurts any pipeline that tracks features on the mono cameras (RTAB-Map RGB-D uses RGB, so fine).
- Valid depth for the W: the 7.5 cm baseline and 150 deg lenses give effectively less far-range accuracy than a D455 (9.5 cm, 87 deg). Plan for good depth ≲4-5 m. Per Luxonis ~2% at <3.5 m.
- Use rolling-shutter-aware reasoning if the IMX378 variant: avoid fast motion; OV9782 variant has global shutter but lower resolution (1280x800).

## 5. Native DepthAI v3 Python alternatives

Sources: [Record Video node](https://docs.luxonis.com/software-v3/depthai/depthai-components/host_nodes/record_video.md), [Holistic record/replay](https://docs.luxonis.com/software-v3/depthai/tutorials/holistic-record-replay.md), [example index](https://docs.luxonis.com/software-v3/depthai/examples/).

- PyPI shows `depthai` 3.12.0 as latest at fetch time (python >=3.9) **[date of release not confirmed]**. Install `pip install depthai`.
- **`RecordVideo` host node:** MP4 video plus MCAP (protobuf) metadata, per stream. Not ROS-compatible MCAP, and depth is not in a video file unless encoded as an ImgFrame; not suitable directly for rosbag replay.
- **Holistic record:** `DEPTHAI_RECORD=./recordings python app.py` records source nodes (Camera, IMU) plus calibration into one `recording_<date>.tar`; depth is recomputed on replay (`DEPTHAI_REPLAY=<tar>`). Replay goes host to device, can reduce FPS, timestamps can drift per loop, single-device only. Good for archival of raw RGB/mono/IMU/calibration, not for ROS bags.
- **Custom writer recommended if the ROS driver misbehaves:** a Python pipeline with `Camera` (CAM_A, 1280x720, undistort on), `StereoDepth` (`setDepthAlign(CAM_A)`, output size 1280x720), `IMU` (accel+gyro 400 Hz), a host loop that writes MCAP via the `mcap-ros2-support` Python package or raw frames plus sidecar timestamps to disk (PNG16 depth, JPG RGB, imu.csv, calibration JSON). I did not verify this code against v3 docs (the depth-align example URLs 404'd), so check `examples/python/StereoDepth` / `Camera` in depthai-core.
- A Luxonis forum thread ([MCAP recording](https://discuss.luxonis.com/d/2401-does-mcap-recording-work-any-longer-on-oak-d-with-depthai)) relates to v2-era recording. Do not rely on it.

## 6. Checking the device

- List devices: `python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"` (v3 example "Device Information" prints sensors, product/board name; page URL not fetched).
- Calibration dump ([example](https://docs.luxonis.com/software-v3/depthai/examples/calibration/calibration_dump.md)): 
  ```python
  import depthai as dai, json
  d = dai.Device(dai.UsbSpeed.HIGH)
  print(json.dumps(d.getCalibration().eepromToJson(), indent=2))      # user calibration
  print(json.dumps(d.readFactoryCalibration().eepromToJson(), indent=2))  # factory
  ```
  For intrinsics/distortion: `calib = d.readCalibration(); calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 1280, 720); calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)` (coefficient count 14 means rational model; 4/5 Brown, 8 = rational; `getDistortionModel` may exist **[UNVERIFIED]**). Also `driver.i_calibration_dump: true` in ROS.
- USB speed: `d.getUsbSpeed()` (expect `SUPER` or `SUPER_PLUS`); on host `lsusb -t` shows 5000M. "Health Check" example reports connection and bandwidth.
- Firmware/bootloader: `python -m depthai --help` / bootloader examples ("Flash bootloader") **[UNVERIFIED in v3]**; `oakctl` is for OAK 4 (RVC4) devices, not the RVC2 OAK-D Pro W **[UNVERIFIED]**. For RVC2 the firmware is loaded automatically from the library on each boot (hence the 2485 to f63b re-enumeration).

## Open items to verify on hardware

1. Which colour sensor variant (IMX378 vs OV9782).
2. IR dot visibility in RGB.
3. Depth encoding/units, topic names, `camera_info` D after undistortion, and depth/RGB alignment in the driver.
4. Distortion coefficient count of CAM_A.
5. IMU actual rates and timestamp skew vs images.

## Verified on hardware (2026-11-08, ROS 2 Jazzy, `ros-jazzy-depthai-ros-driver-v3` 3.2.1 from the main ROS repo, no testing repo needed; depthai Python 3.12.0)

- Device: OAK-D-PRO-W, colour **IMX378 (rolling shutter)**, mono 2x OV9282, BNO086 IMU, USB SUPER. CAM_A calibration: Perspective model, 14 distortion coeffs, max |k| ~12.6 (strongly distorted).
- Topics: `/oak/rgb/image_raw` (bgr8), `/oak/rgb/camera_info`, `/oak/stereo/image_raw` (**16UC1 mm, aligned to RGB, same size as RGB**), `/oak/stereo/camera_info`, `/oak/imu/data` (united `Imu`, ~400 Hz with gyro 400/accel 480), `/tf_static` (transient local, from calibration; IMU frame axes equal the colour optical axes, accel z ~ +9.87 with the lens pointing up). Frame ids `oak_rgb_camera_optical_frame`, `oak_imu_frame`.
- `rgb.i_undistorted: true` -> `camera_info` D = 14 zeros, K for the cropped, undistorted image (HFOV ~82 deg). Depth follows the same geometry (identical K on `/oak/stereo/camera_info`). Depth edges line up with the colour edges in the ingest preview overlay.
- Driver params accept `$(env VAR default)` in the params YAML (typed correctly); `r_laser_dot_intensity` / `r_floodlight_intensity` live under `driver.`; `driver.r_*` and `rgb.r_*` params applied (checked with `ros2 param get`).
- **IR dots are not visible in the colour image**: static scene, 100-frame temporal means, projector 1.0 vs 0.0 -> identical mean brightness (51.21 vs 51.23) and no extra high-frequency structure.
- Rates: plain DepthAI Python: 1280x720 colour 30 Hz alone; stereo aligned to 1280x720 = 29.5 Hz with a 640x400 mono pair, **~14.5 Hz with 1280x800 mono** (LR-check + subpixel; device compute limit). Through the ROS driver at the time of testing (laptop on battery, heavy disk I/O from other software): 640x360 colour + depth = 30/30 Hz, 960x540 ~12 Hz, 1280x720 ~7 Hz (rgb and depth equal; throughput roughly constant at ~6 MP/s; not CPU bound, not disk bound, not IR). Earlier in the session 1280x720 gave 30 Hz colour / 15 Hz depth, so the ROS path is load sensitive. Unresolved.
- Stale-device message `Device ... has crashed` appears after a container is SIGKILLed; run.sh now waits for the driver to exit. The container needs `--net=host` (libusb hotplug uses netlink uevents) or the re-enumerated device is not found.
