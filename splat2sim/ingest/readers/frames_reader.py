"""Pre-ingested ``frames/`` folder reader (synthetic captures from ``tools/synth_capture``): ``color/``, ``depth/``,
``frames.csv`` (index, timestamp_ns, color, depth), ``intrinsics.json`` and optionally ``gt_poses.csv`` (camera poses in
the world frame, **metres**, ``timestamp_ns,tx,ty,tz,qx,qy,qz,qw``)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

import numpy as np

from ..intrinsics_io import load_intrinsics
from .base import Frame, Source


class FramesFolderSource(Source):
    fmt = "frames-folder"
    gt_units = "m"

    def __init__(self, path: Path):
        self.root = Path(path)
        with open(self.root / "frames.csv", newline="") as f:
            self.rows = [(int(r["timestamp_ns"]), r["color"], r["depth"]) for r in csv.DictReader(f)]
        if not self.rows:
            raise ValueError(f"{path}: frames.csv is empty")
        self._intr = load_intrinsics(self.root / "intrinsics.json")

    def intrinsics(self):
        return self._intr

    def frames(self) -> Iterator[Frame]:
        import cv2

        for ts, c, d in self.rows:
            col = cv2.imread(str(self.root / c), cv2.IMREAD_COLOR)
            dep = cv2.imread(str(self.root / d), cv2.IMREAD_UNCHANGED)
            if col is None or dep is None or dep.dtype != np.uint16:
                raise ValueError(f"bad frame {c} / {d} (depth must be 16-bit PNG)")
            yield Frame(ts, cv2.cvtColor(col, cv2.COLOR_BGR2RGB), dep)

    def gt_poses(self):
        p = self.root / "gt_poses.csv"
        if not p.is_file():
            return None
        with open(p, newline="") as f:
            rows = list(csv.reader(f))[1:]
        return np.array([int(r[0]) for r in rows], np.int64), np.array([[float(x) for x in r[1:8]] for r in rows])
