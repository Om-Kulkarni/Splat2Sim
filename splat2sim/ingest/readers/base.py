from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..geometry import Intrinsics
from ..imu import ImuData


@dataclass
class Frame:
    ts_ns: int
    color: np.ndarray  # (H,W,3) uint8 RGB
    depth: np.ndarray  # (H,W) uint16 millimetres, aligned to colour, 0 = invalid


class Source(ABC):
    """A Capture reader. Output depth is always uint16 mm aligned to colour."""

    fmt: str = ""

    @abstractmethod
    def intrinsics(self) -> Intrinsics: ...

    @abstractmethod
    def frames(self) -> Iterator[Frame]: ...

    def imu(self) -> ImuData | None:
        return None

    def gt_poses(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(t_ns int64 (N,), values (N,7) = tx,ty,tz,qx,qy,qz,qw) in the source's own units (mm for SLAM&Render)."""
        return None

    gt_units: str = ""
    camera_serial: str | None = None
    camera_firmware: str | None = None
    undistorted_in_ingest: bool = False  # True when the reader remapped colour/depth to a pinhole K (see ingest/undistort.py)
    original_intrinsics: dict | None = None  # camera_info as recorded (model, distortion_coeffs, K) when undistorted_in_ingest

    def meta(self) -> dict:
        """Source-specific facts, valid after ``frames()`` is exhausted (dropped counts etc.)."""
        return {}
