"""ROS 2 MCAP reader (our capture script: realsense2_camera + `ros2 bag record -s mcap`).

Topics are discovered by suffix so the camera namespace (``/camera/camera`` or ``/camera``) does not matter:
  colour           ``*/color/image_raw``                       sensor_msgs/Image (rgb8)
  aligned depth    ``*/aligned_depth_to_color/image_raw``     16UC1, mm, colour geometry   (preferred)
  raw depth        ``*/depth/image_rect_raw``                  fallback: registered to colour here using
                                                               camera_info + extrinsics (``*/extrinsics/depth_to_color``
                                                               topic, else /tf_static)
  camera_info      ``*/color/camera_info`` (+ depth variants)
  IMU              ``*/imu`` (united) or ``*/gyro/sample`` + ``*/accel/sample`` (united here)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..geometry import Intrinsics, invert_T, make_T, register_depth_to_color
from ..imu import ImuData, sort_dedupe, unite
from ..pairing import StreamPairer
from . import rosimg
from .base import Frame, Source
from .tf import TfGraph

IMAGE_T = "sensor_msgs/msg/Image"
INFO_T = "sensor_msgs/msg/CameraInfo"
IMU_T = "sensor_msgs/msg/Imu"


def _pick(topics: dict[str, str], suffixes: tuple[str, ...], msgtype: str | None) -> str | None:
    cands = [t for t, ty in topics.items() if any(t.endswith(s) for s in suffixes) and (msgtype is None or ty == msgtype)]
    return sorted(cands, key=lambda t: (len(t), t))[0] if cands else None


def discover_topics(topics: dict[str, str]) -> dict[str, str | None]:
    """``topics``: {topic: schema name}. Returns role -> topic (None if absent)."""
    return {
        "color": _pick(topics, ("/color/image_raw",), IMAGE_T),
        "depth_aligned": _pick(topics, ("/aligned_depth_to_color/image_raw",), IMAGE_T),
        "depth_raw": _pick(topics, ("/depth/image_rect_raw", "/depth/image_raw"), IMAGE_T),
        "color_info": _pick(topics, ("/color/camera_info",), INFO_T),
        "depth_info": _pick(topics, ("/depth/camera_info",), INFO_T),
        "aligned_info": _pick(topics, ("/aligned_depth_to_color/camera_info",), INFO_T),
        "imu": _pick(topics, ("/imu",), IMU_T),
        "gyro": _pick(topics, ("/gyro/sample",), IMU_T),
        "accel": _pick(topics, ("/accel/sample",), IMU_T),
        "tf_static": "/tf_static" if topics.get("/tf_static", "").endswith("TFMessage") else None,
        "extrinsics": _pick(topics, ("/extrinsics/depth_to_color",), None),
    }


class McapSource(Source):
    fmt = "ros2-mcap"

    def __init__(self, path: Path, depth_mode: str = "auto"):
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory

        self.path = Path(path)
        self._make_reader, self._DecoderFactory = make_reader, DecoderFactory
        with open(self.path, "rb") as f:
            r = make_reader(f)
            s = r.get_summary()
            if s is None or not s.channels:
                raise ValueError(f"{path}: MCAP has no summary/channels (truncated recording? run `mcap recover`)")
            schemas = {sid: sc.name for sid, sc in s.schemas.items()}
            self.topics = {ch.topic: schemas.get(ch.schema_id, "") for ch in s.channels.values()}
            self.counts = {ch.topic: (s.statistics.channel_message_counts.get(cid, 0) if s.statistics else 0)
                           for cid, ch in s.channels.items()}
            self.mcap_metadata = {m.name: dict(m.metadata) for m in r.iter_metadata()}
        self.roles = discover_topics(self.topics)
        if not self.roles["color"]:
            raise ValueError(f"{path}: no */color/image_raw topic; found: {sorted(self.topics)}")
        if depth_mode == "auto":
            depth_mode = "aligned" if self.roles["depth_aligned"] else "register"
        if depth_mode == "aligned" and not self.roles["depth_aligned"]:
            raise ValueError("no aligned_depth_to_color topic in capture")
        if depth_mode == "register" and not self.roles["depth_raw"]:
            raise ValueError("no depth topic (aligned or raw) in capture")
        self.depth_mode = depth_mode
        self._stats: dict = {}
        self._intr: Intrinsics | None = None
        self._depth_info: dict | None = None
        self._T_c_d: np.ndarray | None = None
        self._info_scanned = False

    # -- helpers --
    def _iter(self, topics: list[str]):
        with open(self.path, "rb") as f:
            r = self._make_reader(f, decoder_factories=[self._DecoderFactory()])
            for _schema, channel, message, msg in r.iter_decoded_messages(topics=topics):
                yield channel.topic, message.log_time, msg

    def _scan_info(self) -> None:
        if self._info_scanned:
            return
        want = [self.roles[k] for k in ("color_info", "depth_info", "tf_static", "extrinsics") if self.roles[k]]
        ci = di = None
        graph = TfGraph()
        ext = None
        for topic, _t, msg in self._iter(want):
            if topic == self.roles["color_info"] and ci is None:
                ci = rosimg.camera_info_to_dict(msg)
            elif topic == self.roles["depth_info"] and di is None:
                di = rosimg.camera_info_to_dict(msg)
            elif topic == self.roles["tf_static"]:
                for tr in msg.transforms:
                    graph.add_msg(tr)
            elif topic == self.roles["extrinsics"] and ext is None:
                R = np.array(list(msg.rotation), float).reshape(3, 3).T  # librealsense: column-major
                ext = make_T(R, list(msg.translation))
        if ci is None:
            raise ValueError("no colour camera_info in capture; cannot build intrinsics")
        self._intr = Intrinsics.from_K(ci["K"], ci["width"], ci["height"], ci["model"], ci["D"],
                                       f"camera_info {self.roles['color_info']}")
        self._depth_info = di
        if self.depth_mode == "register":
            if di is None:
                raise ValueError("raw-depth fallback needs depth camera_info")
            T = ext
            if T is None and di["frame_id"] and ci["frame_id"]:
                T = graph.lookup(ci["frame_id"], di["frame_id"])
            if T is None:
                raise ValueError("raw-depth fallback needs depth->colour extrinsics (extrinsics topic or /tf_static)")
            self._T_c_d = T
        self._info_scanned = True

    # -- Source API --
    def intrinsics(self) -> Intrinsics:
        self._scan_info()
        return self._intr  # type: ignore[return-value]

    def frames(self) -> Iterator[Frame]:
        self._scan_info()
        depth_topic = self.roles["depth_aligned"] if self.depth_mode == "aligned" else self.roles["depth_raw"]
        pairer = StreamPairer(tol_ns=int(0.4e9 / 30), max_pending=8)
        intr = self._intr
        assert intr is not None
        n_color = n_depth = 0
        K_d = None
        if self.depth_mode == "register":
            K_d = np.array(self._depth_info["K"]).reshape(3, 3)  # type: ignore[index]
        for topic, log_t, msg in self._iter([self.roles["color"], depth_topic]):
            ts = rosimg.stamp_ns(msg.header, log_t)
            if topic == self.roles["color"]:
                n_color += 1
                kind, payload = "color", rosimg.to_rgb(msg)
            else:
                n_depth += 1
                kind, payload = "depth", rosimg.to_depth_units(msg)
            for ts_ns, color, depth in pairer.push(kind, ts, payload):
                h, w = color.shape[:2]
                if self.depth_mode == "register":
                    depth = register_depth_to_color(depth, K_d, intr.K, self._T_c_d, (w, h), 0.001)  # type: ignore[arg-type]
                elif depth.shape[:2] != (h, w):
                    import cv2

                    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
                yield Frame(ts_ns, color, depth)
        pairer.flush()
        self._stats = {"color_msgs": n_color, "depth_msgs": n_depth, "unpaired_dropped": dict(pairer.dropped),
                       "depth_mode": self.depth_mode, "depth_topic": depth_topic}

    def imu(self) -> ImuData | None:
        r = self.roles
        if r["imu"]:
            ts, g, a = [], [], []
            for _topic, log_t, m in self._iter([r["imu"]]):
                ts.append(rosimg.stamp_ns(m.header, log_t))
                g.append([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
                a.append([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
            if ts:
                return sort_dedupe(ImuData(np.array(ts, np.int64), np.array(g), np.array(a)))
        if r["gyro"] and r["accel"]:
            gt, g, at, a = [], [], [], []
            for topic, log_t, m in self._iter([r["gyro"], r["accel"]]):
                t = rosimg.stamp_ns(m.header, log_t)
                if topic == r["gyro"]:
                    gt.append(t)
                    g.append([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
                else:
                    at.append(t)
                    a.append([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
            if len(gt) > 1 and len(at) > 1:
                return unite(np.array(gt), np.array(g), np.array(at), np.array(a))
        return None

    def meta(self) -> dict:
        d = {"topics": self.roles, "mcap_metadata": self.mcap_metadata,
             "message_counts": {t: c for t, c in self.counts.items() if t in set(v for v in self.roles.values() if v)}}
        d.update(self._stats)
        if self._T_c_d is not None:
            d["T_color_from_depth"] = self._T_c_d.tolist()
        return d
