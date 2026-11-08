"""Lens-distortion removal at ingest (pinhole invariant).

Everything downstream of ingest (RTAB-Map, the COLMAP PINHOLE writer, TSDF, silhouettes, the anchor detector) assumes an
ideal pinhole camera with ``distortion_coeffs`` ~ 0. RealSense D400 streams are already rectified; a wide-angle camera
such as the OAK-D Pro W colour sensor is not unless the driver undistorts on the device (``rgb.i_undistorted``).
If the ``camera_info`` still carries non-negligible coefficients, colour (bilinear) and depth (nearest, so no depth
values are invented across edges) are remapped here to a new pinhole K with zero distortion.

Supported ``camera_info.distortion_model`` values: ``plumb_bob`` (4/5 coeffs) and ``rational_polynomial`` (8 coeffs,
or the 12/14 OpenCV extension with thin-prism and tilt terms, which is what DepthAI reports for its wide lenses).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import Intrinsics

# max |coefficient| below which a camera counts as undistorted (pass-through)
NEGLIGIBLE = 1e-4
_VALID_LEN = (4, 5, 8, 12, 14)


def opencv_coeffs(model: str, D) -> np.ndarray:
    """camera_info (model, D) -> OpenCV distortion vector (k1 k2 p1 p2 [k3 [k4 k5 k6 [s1..s4 [taux tauy]]]])."""
    d = np.asarray(D, float).ravel()
    if d.size == 0:
        return np.zeros(5)
    if model in ("equidistant", "fisheye", "kannala_brandt"):
        raise ValueError(f"distortion model {model!r} (fisheye) is not supported at ingest; undistort on the device "
                         "or record with a pinhole-like profile")
    if model not in ("plumb_bob", "rational_polynomial", ""):
        raise ValueError(f"unsupported distortion model {model!r}")
    if d.size not in _VALID_LEN:
        raise ValueError(f"unsupported number of distortion coefficients for {model!r}: {d.size} (expected one of {_VALID_LEN})")
    return d


def is_distorted(model: str, D, tol: float = NEGLIGIBLE) -> bool:
    d = np.asarray(D, float).ravel()
    return bool(d.size and np.max(np.abs(d)) > tol)


@dataclass
class Undistorter:
    """Remap tables from a distorted camera (K, D) to a distortion-free pinhole with ``new_K`` (same image size)."""

    K: np.ndarray
    dist: np.ndarray
    size: tuple[int, int]  # (w, h)
    new_K: np.ndarray
    map1: np.ndarray
    map2: np.ndarray
    model: str
    valid_fraction: float  # share of output pixels that sample inside the source image

    @classmethod
    def build(cls, intr: Intrinsics) -> "Undistorter":
        w, h = intr.width, intr.height
        K = intr.K
        dist = opencv_coeffs(intr.distortion_model, intr.distortion)
        newK, _roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0.0, (w, h))  # alpha=0: only valid pixels (no black border)
        m1, m2, frac = None, None, 0.0
        for _ in range(30):  # alpha=0 can still leave a few edge pixels outside on strong distortion: widen until clean
            if not (np.isfinite(newK).all() and newK[0, 0] > 1 and newK[1, 1] > 1):
                newK = K.copy()
            m1, m2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_32FC1)
            inside = (m1 >= 0) & (m1 <= w - 1) & (m2 >= 0) & (m2 <= h - 1)
            frac = float(inside.mean())
            if frac >= 0.9995:
                break
            newK = newK.copy()
            newK[0, 0] *= 1.04
            newK[1, 1] *= 1.04
        return cls(K, dist, (w, h), newK, m1, m2, intr.distortion_model, frac)

    def intrinsics(self, source: str) -> Intrinsics:
        n = self.new_K
        return Intrinsics(float(n[0, 0]), float(n[1, 1]), float(n[0, 2]), float(n[1, 2]), self.size[0], self.size[1],
                          "plumb_bob", [0.0] * 5, source)

    def color(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def depth(self, d: np.ndarray) -> np.ndarray:
        # nearest: interpolating depth would invent surfaces between foreground and background. Out-of-image -> 0 (invalid).
        return cv2.remap(d, self.map1, self.map2, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def maybe_build(intr: Intrinsics, tol: float = NEGLIGIBLE) -> Undistorter | None:
    """Undistorter for ``intr`` if its coefficients are non-negligible, else None (pass-through)."""
    return Undistorter.build(intr) if is_distorted(intr.distortion_model, intr.distortion, tol) else None
