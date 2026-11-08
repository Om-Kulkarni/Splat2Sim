"""Pure-numpy transform helpers for composing the Scene USD (no pxr import, so unit-testable anywhere).

Frame conventions
-----------------
* **recon frame**: the frame of ``poses/poses.csv`` and of the 3DGRUT training data (RTAB-Map world).
  With IMU it is gravity-aligned z-up; without it, it is the first camera frame (y down).
* **3DGRUT ParticleField export**: Gaussian data and camera poses are authored *as-is in the recon frame*
  (``apply_coordinate_transform=false`` and ``apply_normalizing_transform=false``), but the stage claims
  ``upAxis = "Y"``. USD references do not reorient on upAxis, so that metadata is ignored when the splat is
  referenced into our z-up Scene. :func:`splat_local_transform` makes that choice explicit.
* **Anchor frame** (``/World/Anchor``'s parent space = Scene world): z-up, metres. ``T_anchor_recon`` maps recon
  points into it (recon -> robot base once the ChArUco Anchor exists; gravity/plane fallback otherwise).
* USD matrices are row-vector (translation in the last row); numpy 4x4 here are column-vector.
* USD cameras look down -Z with +Y up; OpenCV cameras look down +Z with +Y down: ``T_usdcam = T_cv @ CV_TO_USD_CAM``.
"""

from __future__ import annotations

import numpy as np

CV_TO_USD_CAM = np.diag([1.0, -1.0, -1.0, 1.0])

# Rotation taking a y-up frame to a z-up frame: (x, y, z) -> (x, -z, y), i.e. Rx(+90 deg).
YUP_TO_ZUP = np.array([[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])


def np_to_usd_matrix(T: np.ndarray) -> tuple[tuple[float, ...], ...]:
    """Column-vector 4x4 -> nested row-major tuples in USD's row-vector layout (``Gf.Matrix4d(*rows)``)."""
    M = np.asarray(T, float).T
    return tuple(tuple(float(v) for v in row) for row in M)


def usd_to_np_matrix(rows) -> np.ndarray:
    return np.array([[float(v) for v in r] for r in rows]).T


def splat_local_transform(content: str = "recon") -> np.ndarray:
    """Transform from the splat prim's authored space to the recon frame.

    ``recon``: identity (what our exporter settings produce). ``yup``: for an exporter that really applied a
    y-up convention (e.g. ``apply_coordinate_transform=true``), the inverse of YUP_TO_ZUP.
    """
    if content == "recon":
        return np.eye(4)
    if content == "yup":
        return np.linalg.inv(YUP_TO_ZUP)
    raise ValueError(f"unknown splat content frame {content!r}")


def anchor_transform(anchor: dict) -> tuple[np.ndarray, str]:
    """Pick the /World/Anchor matrix from anchor json dicts: ``{"T_base_recon": 4x4}`` (ChArUco Anchor stage) or the
    fallback ``{"T_anchor_recon": 4x4}``. Returns (T, kind)."""
    if "T_base_recon" in anchor:
        return np.array(anchor["T_base_recon"], float).reshape(4, 4), "charuco"
    if "T_anchor_recon" in anchor:
        return np.array(anchor["T_anchor_recon"], float).reshape(4, 4), "fallback"
    raise ValueError("anchor json needs T_base_recon (charuco) or T_anchor_recon (fallback)")


def cv_camera_to_usd(T_world_cv: np.ndarray) -> np.ndarray:
    return np.asarray(T_world_cv, float) @ CV_TO_USD_CAM


def look_at_usd_camera(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera-to-world for a USD camera (looks down -Z, +Y up) at ``eye`` looking at ``target``."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, [1.0, 0.0, 0.0])
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, u, -f, eye
    return T


def bbox_transform(bbox_min, bbox_max, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AABB of an AABB after a rigid/similarity transform."""
    lo, hi = np.asarray(bbox_min, float), np.asarray(bbox_max, float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    w = corners @ T[:3, :3].T + T[:3, 3]
    return w.min(0), w.max(0)


def overview_eyes(bbox_min, bbox_max, n: int = 2, elevation_deg: float = 35.0, azimuth0_deg: float = 215.0,
                  distance_scale: float = 1.6) -> list[tuple[np.ndarray, np.ndarray]]:
    """``n`` (eye, target) pairs orbiting the box centre, evenly spaced in azimuth."""
    lo, hi = np.asarray(bbox_min, float), np.asarray(bbox_max, float)
    c = 0.5 * (lo + hi)
    r = max(0.5 * np.linalg.norm(hi - lo), 0.3) * distance_scale
    out = []
    for k in range(n):
        az, el = np.radians(azimuth0_deg + 360.0 * k / n), np.radians(elevation_deg)
        out.append((c + r * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]), c))
    return out
