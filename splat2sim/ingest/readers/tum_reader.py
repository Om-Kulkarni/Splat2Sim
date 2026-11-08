"""TUM-RGBD-style folder reader (SLAM&Render): rgb/, depth/, associations.txt, imu.txt, groundtruth.txt.

The folder carries no intrinsics: provide them via ``--intrinsics FILE`` or an ``intrinsics.yaml|json`` sidecar
inside the capture folder (see intrinsics_io.py); otherwise nominal D435 values are used and flagged in meta.json.
Depth is assumed to be aligned to colour already, 16-bit PNG with ``depth_factor`` units per metre (default 1000 = mm).
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Iterator

import numpy as np

from ..geometry import Intrinsics, nominal_d435_color
from ..imu import ImuData, sort_dedupe
from ..intrinsics_io import find_sidecar, load_intrinsics
from .base import Frame, Source


def parse_ts_ns(s: str) -> int:
    """'1739373727.450895309' -> int ns, exactly (no float rounding)."""
    return int((Decimal(s) * 1_000_000_000).to_integral_value())


def _read_table(path: Path) -> list[list[str]]:
    rows = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            rows.append(ln.replace(",", " ").split())
    return rows


def associate(rgb: list[tuple[int, str]], depth: list[tuple[int, str]], max_dt_ns: int = 20_000_000):
    """Nearest-timestamp association of two sorted lists, one-to-one. Returns [(ts, rgb_name, depth_name)]."""
    out, j = [], 0
    for t, name in rgb:
        while j + 1 < len(depth) and abs(depth[j + 1][0] - t) <= abs(depth[j][0] - t):
            j += 1
        if depth and abs(depth[j][0] - t) <= max_dt_ns:
            out.append((t, name, depth[j][1]))
    return out


class TumSource(Source):
    fmt = "tum-folder"

    def __init__(self, path: Path, intrinsics_file: Path | None = None, depth_factor: float = 1000.0):
        self.root = Path(path)
        self.depth_factor = depth_factor
        self.pairs = self._pairs()
        if not self.pairs:
            raise ValueError(f"{path}: no rgb/depth pairs found")
        import cv2

        first = cv2.imread(str(self.root / self.pairs[0][1]), cv2.IMREAD_COLOR)
        if first is None:
            raise ValueError(f"cannot read {self.root / self.pairs[0][1]}")
        h, w = first.shape[:2]
        side = intrinsics_file or find_sidecar(self.root)
        if side:
            self._intr = load_intrinsics(Path(side), (w, h))
            if (self._intr.width, self._intr.height) != (w, h):
                self._intr = self._intr.scaled(w, h)
        else:
            self._intr = nominal_d435_color(w, h)
        self._stats: dict = {}

    def _pairs(self):
        assoc = self.root / "associations.txt"
        if assoc.is_file():
            rows = _read_table(assoc)
            return [(parse_ts_ns(r[0]), r[1], r[3]) for r in rows if len(r) >= 4]
        def listing(d):
            return [(parse_ts_ns(p.stem), f"{d}/{p.name}") for p in sorted((self.root / d).glob("*.png"))]
        return associate(listing("rgb"), listing("depth"))

    def intrinsics(self) -> Intrinsics:
        return self._intr

    def frames(self) -> Iterator[Frame]:
        import cv2

        for ts, rgb, dep in self.pairs:
            c = cv2.imread(str(self.root / rgb), cv2.IMREAD_COLOR)
            d = cv2.imread(str(self.root / dep), cv2.IMREAD_UNCHANGED)
            if c is None or d is None or d.dtype != np.uint16:
                raise ValueError(f"bad frame {rgb} / {dep} (depth must be 16-bit PNG)")
            if d.ndim == 3:
                d = d[:, :, 0]
            if d.shape != c.shape[:2]:
                d = cv2.resize(d, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_NEAREST)
            if abs(self.depth_factor - 1000.0) > 1e-9:
                d = np.clip(np.rint(d.astype(np.float64) * (1000.0 / self.depth_factor)), 0, 65535).astype(np.uint16)
            yield Frame(ts, cv2.cvtColor(c, cv2.COLOR_BGR2RGB), d)
        self._stats = {"associations": len(self.pairs)}

    def imu(self) -> ImuData | None:
        p = self.root / "imu.txt"
        if not p.is_file():
            return None
        rows = _read_table(p)  # t ax ay az gx gy gz
        t = np.array([parse_ts_ns(r[0]) for r in rows], np.int64)
        v = np.array([[float(x) for x in r[1:7]] for r in rows])
        return sort_dedupe(ImuData(t, v[:, 3:6], v[:, 0:3]))

    gt_units = "mm"

    def gt_poses(self):
        p = self.root / "groundtruth.txt"
        if not p.is_file():
            return None
        rows = _read_table(p)  # t qx qy qz qw tx ty tz
        t = np.array([parse_ts_ns(r[0]) for r in rows], np.int64)
        vals = np.array([[float(x) for x in r[5:8]] + [float(x) for x in r[1:5]] for r in rows])
        return t, vals

    def meta(self) -> dict:
        return {"depth_factor_per_m": self.depth_factor, "intrinsics_source": self._intr.source,
                "has_robot_data": (self.root / "robot_data").is_dir(), **self._stats}
