"""Decode sensor_msgs/Image-like messages (ROS 1 via rosbags, ROS 2 via mcap-ros2-support) into numpy."""

from __future__ import annotations

import numpy as np


def stamp_ns(header, fallback_ns: int) -> int:
    st = header.stamp
    sec = getattr(st, "sec", 0)
    nsec = getattr(st, "nanosec", getattr(st, "nsec", 0))
    ns = int(sec) * 1_000_000_000 + int(nsec)
    return ns if ns > 0 else int(fallback_ns)


def _raw(msg, dtype, channels: int) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    buf = np.frombuffer(bytes(msg.data) if not isinstance(msg.data, (bytes, bytearray, memoryview, np.ndarray)) else msg.data, np.uint8)
    itemsize = np.dtype(dtype).itemsize
    step = int(msg.step) or w * channels * itemsize
    rows = buf[: h * step].reshape(h, step)[:, : w * channels * itemsize]
    a = np.ascontiguousarray(rows).view(dtype)
    return a.reshape(h, w, channels) if channels > 1 else a.reshape(h, w)


def to_rgb(msg) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc == "rgb8":
        return _raw(msg, np.uint8, 3)
    if enc == "bgr8":
        return _raw(msg, np.uint8, 3)[:, :, ::-1].copy()
    if enc == "rgba8":
        return _raw(msg, np.uint8, 4)[:, :, :3].copy()
    if enc == "bgra8":
        return _raw(msg, np.uint8, 4)[:, :, 2::-1].copy()
    if enc in ("mono8", "8uc1"):
        import cv2

        return cv2.cvtColor(_raw(msg, np.uint8, 1), cv2.COLOR_GRAY2RGB)
    raise ValueError(f"unsupported colour encoding {msg.encoding!r}")


def to_depth_units(msg) -> np.ndarray:
    """Raw uint16 depth in sensor units (16UC1/mono16). 32FC1 is metres -> returned as mm."""
    enc = msg.encoding.lower()
    if enc in ("16uc1", "mono16"):
        return _raw(msg, np.dtype("<u2"), 1)
    if enc == "32fc1":
        m = _raw(msg, np.dtype("<f4"), 1)
        return np.clip(np.nan_to_num(m * 1000.0), 0, 65535).astype(np.uint16)
    raise ValueError(f"unsupported depth encoding {msg.encoding!r}")


def camera_info_to_dict(msg) -> dict:
    g = lambda *names: next((getattr(msg, n) for n in names if hasattr(msg, n)), None)  # noqa: E731
    return {
        "width": int(msg.width), "height": int(msg.height),
        "model": str(g("distortion_model") or "plumb_bob"),
        "D": [float(x) for x in (g("d", "D") if g("d", "D") is not None else [])],
        "K": [float(x) for x in g("k", "K")],
        "frame_id": getattr(msg.header, "frame_id", ""),
    }
