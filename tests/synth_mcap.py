"""Write a tiny ROS 2 MCAP with the realsense2_camera (or depthai-ros OAK) topic layout (used by tests and for manual verification).

mcap-ros2-support only *decodes*; messages are CDR-serialised with rosbags' typestore and wrapped with mcap's
writer using hand-written ros2msg schemas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HEADER = "builtin_interfaces/Time stamp\nstring frame_id"
TIME = "int32 sec\nuint32 nanosec"
SCHEMAS = {
    "sensor_msgs/msg/Image": (
        "std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\nuint32 step\nuint8[] data\n"
        f"===\nMSG: std_msgs/Header\n{HEADER}\n===\nMSG: builtin_interfaces/Time\n{TIME}\n"),
    "sensor_msgs/msg/CameraInfo": (
        "std_msgs/Header header\nuint32 height\nuint32 width\nstring distortion_model\nfloat64[] d\nfloat64[9] k\n"
        "float64[9] r\nfloat64[12] p\nuint32 binning_x\nuint32 binning_y\nsensor_msgs/RegionOfInterest roi\n"
        f"===\nMSG: std_msgs/Header\n{HEADER}\n===\nMSG: builtin_interfaces/Time\n{TIME}\n"
        "===\nMSG: sensor_msgs/RegionOfInterest\nuint32 x_offset\nuint32 y_offset\nuint32 height\nuint32 width\nbool do_rectify\n"),
    "sensor_msgs/msg/Imu": (
        "std_msgs/Header header\ngeometry_msgs/Quaternion orientation\nfloat64[9] orientation_covariance\n"
        "geometry_msgs/Vector3 angular_velocity\nfloat64[9] angular_velocity_covariance\n"
        "geometry_msgs/Vector3 linear_acceleration\nfloat64[9] linear_acceleration_covariance\n"
        f"===\nMSG: std_msgs/Header\n{HEADER}\n===\nMSG: builtin_interfaces/Time\n{TIME}\n"
        "===\nMSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w\n"
        "===\nMSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z\n"),
    "tf2_msgs/msg/TFMessage": (
        "geometry_msgs/TransformStamped[] transforms\n"
        "===\nMSG: geometry_msgs/TransformStamped\nstd_msgs/Header header\nstring child_frame_id\ngeometry_msgs/Transform transform\n"
        f"===\nMSG: std_msgs/Header\n{HEADER}\n===\nMSG: builtin_interfaces/Time\n{TIME}\n"
        "===\nMSG: geometry_msgs/Transform\ngeometry_msgs/Vector3 translation\ngeometry_msgs/Quaternion rotation\n"
        "===\nMSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w\n"
        "===\nMSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z\n"),
}

NS = "/camera/camera"


def plane_pattern(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Smooth analytic texture (uint8-range float) of metric plane coordinates; smooth so bilinear resampling error is tiny."""
    return 127.5 + 100.0 * np.sin(2 * np.pi * x / 0.35) * np.cos(2 * np.pi * y / 0.27) + 20.0 * np.sin(2 * np.pi * (x + y) / 0.9)


def render_plane(w: int, h: int, K, D=None, Z: float = 1.0) -> np.ndarray:
    """RGB uint8 image of a fronto-parallel textured plane at depth Z seen by a camera with intrinsics K and OpenCV distortion D."""
    import cv2

    K = np.asarray(K, float).reshape(3, 3)
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pts = np.stack([uu.ravel(), vv.ravel()], 1).reshape(-1, 1, 2)
    n = cv2.undistortPoints(pts, K, np.asarray(D, float) if D is not None and len(D) else None).reshape(h, w, 2)  # normalised rays
    g = plane_pattern(n[..., 0] * Z, n[..., 1] * Z)
    return np.clip(np.stack([g, 255 - g, 0.5 * g + 60], -1), 0, 255).astype(np.uint8)


def write_mcap(path: Path, n_frames: int = 12, aligned: bool = True, color_size=(160, 120), depth_size=(160, 120),
               T_color_from_depth_x: float = 0.015, t0_ns: int = 1_700_000_000_000_000_000, separate_imu: bool = False,
               layout: str = "realsense", distortion: tuple[str, list] | None = None, depth_encoding: str = "16UC1",
               color_encoding: str = "rgb8", plane: bool = False, capture_info: dict | None = None) -> dict:
    """``layout``: "realsense" (/camera/camera/...) or "oak" (/oak/rgb, /oak/stereo aligned depth, /oak/imu/data).
    ``distortion``: (camera_info model, D) for the colour camera (the image is rendered distorted when ``plane``).
    ``depth_encoding``: 16UC1 (mm) | 32FC1 (m).  ``plane``: textured fronto-parallel plane at 1 m instead of random noise.
    ``capture_info``: written as capture_info.json next to the .mcap."""
    from mcap.writer import Writer
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    T = ts.types

    def hdr(t_ns: int, frame: str):
        return T["std_msgs/msg/Header"](stamp=T["builtin_interfaces/msg/Time"](sec=t_ns // 10**9, nanosec=t_ns % 10**9), frame_id=frame)

    def ser(msg, typ):
        return bytes(ts.serialize_cdr(msg, typ))

    cw, ch = color_size
    dw, dh = depth_size
    fxc = 0.9 * cw
    Kc = [fxc, 0, cw / 2, 0, fxc, ch / 2, 0, 0, 1.0]
    fxd = 0.9 * dw
    Kd = [fxd, 0, dw / 2, 0, fxd, dh / 2, 0, 0, 1.0]

    oak = layout == "oak"
    ns = "/oak" if oak else NS
    cname, dname = ("rgb", "stereo") if oak else ("color", "depth")
    dmodel, dcoef = distortion if distortion else ("plumb_bob", [0.0] * (14 if oak else 5))
    cframe = "oak_rgb_camera_optical_frame" if oak else "camera_color_optical_frame"
    dframe = cframe if oak else "camera_depth_optical_frame"

    def info(t, frame, w, h, K, model="plumb_bob", D=None):
        return ser(T["sensor_msgs/msg/CameraInfo"](
            header=hdr(t, frame), height=h, width=w, distortion_model=model, d=np.array(D if D is not None else np.zeros(5), float), k=np.array(K), r=np.zeros(9),
            p=np.zeros(12), binning_x=0, binning_y=0,
            roi=T["sensor_msgs/msg/RegionOfInterest"](x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)), "sensor_msgs/msg/CameraInfo")

    def image(t, frame, enc, arr):
        h, w = arr.shape[:2]
        data = np.frombuffer(np.ascontiguousarray(arr).tobytes(), np.uint8)
        return ser(T["sensor_msgs/msg/Image"](header=hdr(t, frame), height=h, width=w, encoding=enc, is_bigendian=0,
                                              step=data.size // h, data=data), "sensor_msgs/msg/Image")

    def imu_msg(t, gyro, accel):
        V = T["geometry_msgs/msg/Vector3"]
        return ser(T["sensor_msgs/msg/Imu"](
            header=hdr(t, "imu"), orientation=T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
            orientation_covariance=np.zeros(9), angular_velocity=V(x=gyro[0], y=gyro[1], z=gyro[2]),
            angular_velocity_covariance=np.zeros(9), linear_acceleration=V(x=accel[0], y=accel[1], z=accel[2]),
            linear_acceleration_covariance=np.zeros(9)), "sensor_msgs/msg/Imu")

    # scene: fronto-parallel plane at 1.0 m with a nearer box (0.6 m) in the middle
    def depth_img(w, h, fx):
        d = np.full((h, w), 1000, np.uint16)
        d[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = 600
        return d

    tf = ts.serialize_cdr(T["tf2_msgs/msg/TFMessage"](transforms=[T["geometry_msgs/msg/TransformStamped"](
        header=hdr(t0_ns, "oak_rgb_camera_frame" if oak else dframe), child_frame_id=cframe,
        transform=T["geometry_msgs/msg/Transform"](translation=T["geometry_msgs/msg/Vector3"](x=T_color_from_depth_x, y=0.0, z=0.0),
                                                   rotation=T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0)))]),
        "tf2_msgs/msg/TFMessage")

    topics = {
        f"{ns}/{cname}/image_raw": "sensor_msgs/msg/Image", f"{ns}/{cname}/camera_info": "sensor_msgs/msg/CameraInfo",
        f"{ns}/{dname}/camera_info": "sensor_msgs/msg/CameraInfo", "/tf_static": "tf2_msgs/msg/TFMessage",
    }
    if oak:
        topics[f"{ns}/stereo/image_raw"] = "sensor_msgs/msg/Image"  # OAK: depth aligned to RGB on the device
    elif aligned:
        topics[f"{ns}/aligned_depth_to_color/image_raw"] = "sensor_msgs/msg/Image"
    else:
        topics[f"{ns}/depth/image_rect_raw"] = "sensor_msgs/msg/Image"
    if separate_imu:
        topics[f"{ns}/gyro/sample"] = topics[f"{ns}/accel/sample"] = "sensor_msgs/msg/Imu"
    else:
        topics[f"{ns}/imu/data" if oak else f"{ns}/imu"] = "sensor_msgs/msg/Imu"

    with open(path, "wb") as f:
        w = Writer(f, compression=__import__("mcap.writer", fromlist=["CompressionType"]).CompressionType.ZSTD)
        w.start()
        ids = {}
        for topic, typ in topics.items():
            sid = w.register_schema(name=typ, encoding="ros2msg", data=SCHEMAS[typ].encode())
            ids[topic] = w.register_channel(topic=topic, message_encoding="cdr", schema_id=sid)

        def put(topic, t, data):
            w.add_message(channel_id=ids[topic], log_time=t, publish_time=t, data=data)

        put("/tf_static", t0_ns, bytes(tf))
        rng = np.random.default_rng(0)
        base = render_plane(cw, ch, Kc, dcoef, 1.0) if plane else rng.integers(0, 255, (ch, cw, 3), dtype=np.uint8)
        dt = 33_333_333
        for i in range(n_frames):
            t = t0_ns + i * dt
            put(f"{ns}/{cname}/camera_info", t, info(t, cframe, cw, ch, Kc, dmodel, dcoef))
            put(f"{ns}/{dname}/camera_info", t, info(t, dframe, dw, dh, Kd))
            frame_img = np.roll(base, i, axis=1) if not plane else base
            if color_encoding == "bgr8":
                frame_img = frame_img[:, :, ::-1]
            put(f"{ns}/{cname}/image_raw", t, image(t, cframe, color_encoding, frame_img))
            d = depth_img(dw, dh, fxd)
            if depth_encoding == "32FC1":
                d = d.astype(np.float32) / 1000.0
            if oak:
                put(f"{ns}/stereo/image_raw", t, image(t, dframe, depth_encoding, d))
            elif aligned:
                put(f"{ns}/aligned_depth_to_color/image_raw", t, image(t, cframe, depth_encoding, d))
            else:
                put(f"{ns}/depth/image_rect_raw", t, image(t, dframe, depth_encoding, d))
        n_imu = n_frames * 12
        for k in range(n_imu):
            t = t0_ns + k * (dt // 12)
            g, a = (0.01 * k, 0.0, 0.0), (0.0, 0.0, 9.81)
            if separate_imu:
                if k % 2 == 0:
                    put(f"{ns}/gyro/sample", t, imu_msg(t, g, (0.0, 0.0, 0.0)))
                if k % 3 == 0:
                    put(f"{ns}/accel/sample", t, imu_msg(t, (0.0, 0.0, 0.0), a))
            else:
                put(f"{ns}/imu/data" if oak else f"{ns}/imu", t, imu_msg(t, g, a))
        w.finish()
    if capture_info is not None:
        import json

        (Path(path).parent / "capture_info.json").write_text(json.dumps(capture_info))
    return {"n_frames": n_frames, "Kc": Kc, "Kd": Kd, "t0_ns": t0_ns}


if __name__ == "__main__":
    import sys

    write_mcap(Path(sys.argv[1]), n_frames=int(sys.argv[2]) if len(sys.argv) > 2 else 30,
               aligned=(sys.argv[3] != "raw") if len(sys.argv) > 3 else True,
               color_size=(640, 360), depth_size=(424, 240) if len(sys.argv) > 3 and sys.argv[3] == "raw" else (640, 360))
