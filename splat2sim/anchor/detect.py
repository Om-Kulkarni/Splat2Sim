"""ChArUco detection + depth sampling (needs opencv-contrib >= 4.8; the ``ingest`` image has 4.10).

The board definition follows ``tools/charuco/common.py`` (same OpenCV call, so the printed board and the detector agree).
Board frame (``board_to_base`` in the lab config): origin at the first inner corner (chessboard corner id 0), axes
as returned by ``CharucoBoard.getChessboardCorners()`` (x along squares_x, y along squares_y).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def make_board(spec: dict):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.get("dictionary", "DICT_4X4_50")))
    return cv2.aruco.CharucoBoard((int(spec["squares_x"]), int(spec["squares_y"])), float(spec["square_length_m"]),
                                  float(spec["marker_length_m"]), d)


def board_corners(board) -> np.ndarray:
    """(N,3) inner-corner positions in the board frame (origin = corner 0)."""
    c = np.asarray(board.getChessboardCorners(), float).reshape(-1, 3)
    return c - c[0]


def make_detector(board):
    dp = cv2.aruco.DetectorParameters()
    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), dp)


@dataclass
class Detection:
    ids: np.ndarray       # (n,) charuco corner ids
    uv_raw: np.ndarray    # (n,2) detected pixels (distorted image)
    uv: np.ndarray        # (n,2) undistorted pixels (pinhole K)
    z: np.ndarray         # (n,) depth m (nan = invalid / depth edge)


def sample_depth(depth_m: np.ndarray, uv: np.ndarray, win: int = 5, zmin: float = 0.15, zmax: float = 4.0,
                 min_valid_frac: float = 0.5) -> np.ndarray:
    """Median depth in a (win x win) window per pixel; nan when too few valid samples or the window straddles a depth edge
    (spread > 3 cm + 3% of range, which also tolerates a tilted plane across 5 px)."""
    h, w = depth_m.shape
    r = win // 2
    out = np.full(len(uv), np.nan)
    for k, (u, v) in enumerate(uv):
        ui, vi = int(round(u)), int(round(v))
        if ui < r or vi < r or ui >= w - r or vi >= h - r:
            continue
        patch = depth_m[vi - r : vi + r + 1, ui - r : ui + r + 1].ravel()
        good = patch[(patch > zmin) & (patch < zmax)]
        if len(good) < min_valid_frac * patch.size:
            continue
        med = float(np.median(good))
        if good.max() - good.min() > 0.03 + 0.03 * med:
            continue
        out[k] = med
    return out


def undistort(uv: np.ndarray, K: np.ndarray, dist: np.ndarray | None) -> np.ndarray:
    if dist is None or not np.any(dist):
        return uv.copy()
    return cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), K, dist, P=K).reshape(-1, 2)


def detect_frame(det, gray: np.ndarray, depth_m: np.ndarray | None, K: np.ndarray, dist: np.ndarray | None,
                 min_corners: int = 6) -> Detection | None:
    corners, ids, _, _ = det.detectBoard(gray)
    if ids is None or len(ids) < min_corners:
        return None
    uv_raw = corners.reshape(-1, 2).astype(float)
    ids = ids.reshape(-1)
    z = sample_depth(depth_m, uv_raw) if depth_m is not None else np.full(len(ids), np.nan)
    return Detection(ids=ids, uv_raw=uv_raw, uv=undistort(uv_raw, K, dist), z=z)


def read_depth_m(path: Path, scale: float) -> np.ndarray | None:
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return None if d is None else d.astype(np.float32) * scale
