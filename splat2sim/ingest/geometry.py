"""Pure-numpy geometry: intrinsics, rigid transforms, depth-to-color registration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_model: str = "plumb_bob"
    distortion: list[float] = field(default_factory=lambda: [0.0] * 5)
    source: str = ""

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @classmethod
    def from_K(cls, K, width: int, height: int, model: str = "plumb_bob", D=None, source: str = "") -> "Intrinsics":
        K = np.asarray(K, dtype=float).reshape(3, 3)
        return cls(float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), int(width), int(height),
                   model or "plumb_bob", [float(x) for x in (D if D is not None and len(D) else [0.0] * 5)], source)

    def scaled(self, width: int, height: int) -> "Intrinsics":
        sx, sy = width / self.width, height / self.height
        return Intrinsics(self.fx * sx, self.fy * sy, (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5,
                          width, height, self.distortion_model, list(self.distortion), self.source + " (rescaled)")


def nominal_d435_color(width: int, height: int) -> Intrinsics:
    """Approximate D435/D435i colour intrinsics (factory fx ~1380 at 1920x1080, no distortion). Last resort only."""
    f = 1380.0 * width / 1920.0
    return Intrinsics(f, f, (width - 1) / 2, (height - 1) / 2, width, height, "plumb_bob", [0.0] * 5,
                      "ASSUMED nominal D435 colour intrinsics (no calibration found)")


def quat_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def make_T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def register_depth_to_color(
    depth: np.ndarray,
    K_depth: np.ndarray,
    K_color: np.ndarray,
    T_color_from_depth: np.ndarray,
    color_size: tuple[int, int],
    depth_scale_m: float = 0.001,
    max_depth_m: float = 0.0,
) -> np.ndarray:
    """Reproject a uint16 depth image into the colour camera frame.

    ``depth`` is HxW in sensor units (``depth_scale_m`` metres each, 0 = invalid). Returns uint16 (h, w) with
    the same units where ``(w, h) = color_size``; the z-value is the depth along the colour optical axis.
    Nearest surface wins when several depth pixels land on one colour pixel (z-buffer). When one depth pixel
    spans several colour pixels (colour focal length larger) each point is splatted over its footprint so no
    holes appear.
    Lens distortion is ignored (D400 streams are rectified and the stored coefficients are zero).
    """
    cw, ch = color_size
    out = np.zeros((ch, cw), dtype=np.uint16)
    v, u = np.nonzero(depth)
    if v.size == 0:
        return out
    d = depth[v, u].astype(np.float64)
    z = d * depth_scale_m
    if max_depth_m > 0:
        keep = z <= max_depth_m
        u, v, z = u[keep], v[keep], z[keep]
    x = (u - K_depth[0, 2]) / K_depth[0, 0] * z
    y = (v - K_depth[1, 2]) / K_depth[1, 1] * z
    R, t = T_color_from_depth[:3, :3], T_color_from_depth[:3, 3]
    xc = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z + t[0]
    yc = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z + t[1]
    zc = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z + t[2]
    ok = zc > 1e-4
    xc, yc, zc = xc[ok], yc[ok], zc[ok]
    uf = K_color[0, 0] * xc / zc + K_color[0, 2]
    vf = K_color[1, 1] * yc / zc + K_color[1, 2]
    dval = np.clip(np.rint(zc / depth_scale_m), 1, 65535).astype(np.uint16)

    # Colour pixels are smaller than depth pixels when the colour camera has the larger focal length: splat each
    # point over the colour pixels whose centres fall inside its footprint (eps keeps shared edges hole-free).
    ru, rv = K_color[0, 0] / K_depth[0, 0], K_color[1, 1] / K_depth[1, 1]
    eps = 1e-6
    idx_all, val_all = [], []
    if ru > 1 + eps or rv > 1 + eps:
        ru, rv = max(ru, 1.0), max(rv, 1.0)
        lo_u, hi_u = np.ceil(uf - ru / 2 - eps).astype(np.int64), np.ceil(uf + ru / 2 + eps).astype(np.int64) - 1
        lo_v, hi_v = np.ceil(vf - rv / 2 - eps).astype(np.int64), np.ceil(vf + rv / 2 + eps).astype(np.int64) - 1
        pairs = [(du, dv) for du in range(int(np.ceil(ru)) + 2) for dv in range(int(np.ceil(rv)) + 2)]
        for du, dv in pairs:
            uu, vv = lo_u + du, lo_v + dv
            m = (uu <= hi_u) & (vv <= hi_v) & (uu >= 0) & (uu < cw) & (vv >= 0) & (vv < ch)
            idx_all.append(vv[m] * cw + uu[m])
            val_all.append(dval[m])
    else:
        uu, vv = np.rint(uf).astype(np.int64), np.rint(vf).astype(np.int64)
        m = (uu >= 0) & (uu < cw) & (vv >= 0) & (vv < ch)
        idx_all.append(vv[m] * cw + uu[m])
        val_all.append(dval[m])
    idx = np.concatenate(idx_all)
    val = np.concatenate(val_all)
    if idx.size == 0:
        return out
    order = np.lexsort((val, idx))  # by pixel, then nearest first
    idx, val = idx[order], val[order]
    first = np.ones(idx.size, dtype=bool)
    first[1:] = idx[1:] != idx[:-1]
    out.reshape(-1)[idx[first]] = val[first]
    return out
