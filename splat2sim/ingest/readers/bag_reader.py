"""librealsense ``.bag`` (ROS 1 bag) reader, e.g. Intel's d435i_sample_data.

Streams are unaligned (depth 1280x720 vs colour 640x480): depth is registered into the colour frame with the
stored intrinsics and the depth->colour extrinsic (``<sensor>/tf/0`` of each stream is reference->stream;
the reference here is the depth sensor, whose own transform is the identity). IMU comes as separate
Accel_0 / Gyro_0 topics at different rates and is united by interpolation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np

from ..geometry import Intrinsics, invert_T, make_T, quat_to_R, register_depth_to_color
from ..imu import ImuData, unite
from ..pairing import StreamPairer
from . import rosimg
from .base import Frame, Source

_PAT = {
    "color": re.compile(r"/Color_\d+/image/data$"),
    "depth": re.compile(r"/Depth_\d+/image/data$"),
    "color_info": re.compile(r"/Color_\d+/info/camera_info$"),
    "depth_info": re.compile(r"/Depth_\d+/info/camera_info$"),
    "color_tf": re.compile(r"/Color_\d+/tf/\d+$"),
    "depth_tf": re.compile(r"/Depth_\d+/tf/\d+$"),
    "accel": re.compile(r"/Accel_\d+/imu/data$"),
    "gyro": re.compile(r"/Gyro_\d+/imu/data$"),
    "depth_units": re.compile(r"/sensor_\d+/option/Depth Units/value$"),
}


def _tf_to_T(tr) -> np.ndarray:
    return make_T(quat_to_R(tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w),
                  [tr.translation.x, tr.translation.y, tr.translation.z])


def _strings(raw: bytes) -> list[str]:
    return [s.decode("ascii", "ignore") for s in re.findall(rb"[\x20-\x7e]{4,}", raw)]


class BagSource(Source):
    fmt = "rosbag1-librealsense"

    def __init__(self, path: Path):
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_typestore

        self.path = Path(path)
        self._Reader = Reader
        self.ts = get_typestore(Stores.ROS1_NOETIC)
        with Reader(self.path) as r:
            self.conns = {c.topic: c.msgtype for c in r.connections}
            self.duration_s = r.duration / 1e9
        self.topic = {role: next((t for t in self.conns if pat.search(t)), None) for role, pat in _PAT.items()}
        for need in ("color", "depth", "color_info", "depth_info"):
            if not self.topic[need]:
                raise ValueError(f"{path}: librealsense bag lacks the '{need}' topic; found {sorted(self.conns)[:12]}")
        self._intr: Intrinsics | None = None
        self._K_d: np.ndarray | None = None
        self._T_c_d = np.eye(4)
        self._depth_units = 0.001
        self._stats: dict = {}
        self._info: dict = {}
        self._scan()

    def _msgs(self, topics: list[str]):
        with self._Reader(self.path) as r:
            conns = [c for c in r.connections if c.topic in topics]
            for c, t, raw in r.messages(connections=conns):
                yield c, t, raw

    def _scan(self) -> None:
        small = [self.topic[k] for k in ("color_info", "depth_info", "color_tf", "depth_tf", "depth_units") if self.topic[k]]
        T_color = T_depth = np.eye(4)
        seen: set[str] = set()
        for c, _t, raw in self._msgs(small):
            if c.topic in seen:
                continue
            seen.add(c.topic)
            m = self.ts.deserialize_ros1(raw, c.msgtype)
            if c.topic == self.topic["color_info"]:
                d = rosimg.camera_info_to_dict(m)
                self._intr = Intrinsics.from_K(d["K"], d["width"], d["height"], d["model"], d["D"], "bag camera_info")
            elif c.topic == self.topic["depth_info"]:
                self._K_d = np.array(rosimg.camera_info_to_dict(m)["K"]).reshape(3, 3)
            elif c.topic == self.topic["color_tf"]:
                T_color = _tf_to_T(m)
            elif c.topic == self.topic["depth_tf"]:
                T_depth = _tf_to_T(m)
            elif c.topic == self.topic["depth_units"]:
                self._depth_units = float(m.data)
        # tf/0 is the extrinsic reference(depth) -> stream: p_stream = T_stream @ p_depth (D435: colour t_x = +15 mm;
        # verified against edge alignment on the Intel sample, the opposite sign scores clearly worse).
        self._T_c_d = T_color @ invert_T(T_depth)
        # Device info records have a custom (realsense_msgs) type; pull key/value strings out of the raw bytes.
        kv: dict[str, str] = {}
        with self._Reader(self.path) as r:
            for c in r.connections:
                if re.search(r"/info$", c.topic) and "camera_info" not in c.topic:
                    for _c, _t, raw in r.messages(connections=[c]):
                        st = _strings(bytes(raw))
                        if len(st) == 2:
                            kv[f"{c.topic.split('/')[-2]}:{st[0]}"] = st[1]
        self._info = {"device_info": kv}
        self.camera_name = next((v for k, v in kv.items() if k.endswith(":Name") and "RealSense" in v), None)
        self.camera_serial = next((v for k, v in kv.items() if "serial" in k.lower()), None)
        self.camera_firmware = next((v for k, v in kv.items() if k.endswith(":Firmware Version")), None)

    def intrinsics(self) -> Intrinsics:
        return self._intr  # type: ignore[return-value]

    def frames(self) -> Iterator[Frame]:
        pairer = StreamPairer(tol_ns=int(0.5e9 / 30), max_pending=8)
        n_c = n_d = 0
        intr = self._intr
        assert intr is not None and self._K_d is not None
        mm_per_unit = self._depth_units / 0.001
        for c, t, raw in self._msgs([self.topic["color"], self.topic["depth"]]):
            m = self.ts.deserialize_ros1(raw, c.msgtype)
            ts = rosimg.stamp_ns(m.header, t)
            if c.topic == self.topic["color"]:
                n_c += 1
                kind, payload = "color", rosimg.to_rgb(m)
            else:
                n_d += 1
                kind, payload = "depth", rosimg.to_depth_units(m)
            for ts_ns, color, depth in pairer.push(kind, ts, payload):
                h, w = color.shape[:2]
                reg = register_depth_to_color(depth, self._K_d, intr.K, self._T_c_d, (w, h), self._depth_units)
                if abs(mm_per_unit - 1) > 1e-6:  # register_depth keeps sensor units; convert to mm
                    reg = np.clip(np.rint(reg * mm_per_unit), 0, 65535).astype(np.uint16)
                yield Frame(ts_ns, color, reg)
        pairer.flush()
        self._stats = {"color_msgs": n_c, "depth_msgs": n_d, "unpaired_dropped": dict(pairer.dropped)}

    def imu(self) -> ImuData | None:
        if not (self.topic["accel"] and self.topic["gyro"]):
            return None
        gt, g, at, a = [], [], [], []
        for c, t, raw in self._msgs([self.topic["accel"], self.topic["gyro"]]):
            m = self.ts.deserialize_ros1(raw, c.msgtype)
            ts = rosimg.stamp_ns(m.header, t)
            if c.topic == self.topic["gyro"]:
                gt.append(ts)
                g.append([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
            else:
                at.append(ts)
                a.append([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        if len(gt) < 2 or len(at) < 2:
            return None
        o1, o2 = np.argsort(gt, kind="stable"), np.argsort(at, kind="stable")
        return unite(np.array(gt)[o1], np.array(g)[o1], np.array(at)[o2], np.array(a)[o2])

    def meta(self) -> dict:
        return {"topics": self.topic, "depth_units_m": self._depth_units,
                "T_color_from_depth": self._T_c_d.tolist(), "bag_duration_s": self.duration_s,
                **self._info, **self._stats,
                "camera_name": self.camera_name}
