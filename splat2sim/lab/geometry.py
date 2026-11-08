"""Pure geometry helpers for the Task environment (numpy + stdlib + the lab config models; no Isaac, no torch).

Conventions are those of ``splat2sim.config``: metres, degrees for rpy/fov, poses are xyz + rpy with R = Rz*Ry*Rx,
camera frame +x forward, +y left, +z up. Quaternions here are **xyzw** (Isaac Lab 3.0 order).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Box, CameraSpec, Intrinsics, LabConfig, Pose, SpawnRegion

# Isaac Lab ``CameraCfg.OffsetCfg(convention="world")``: forward +X, up +Z (== our lab-config camera frame).
CAMERA_CONVENTION = "world"

# Lab-frame parent -> prim name below the robot root (the fr3.usd link names).
PARENT_LINKS = {"flange": "fr3_link8", "robot_base": "fr3_link0"}

# Used when neither primitives.json nor overrides.table_box exist: a 1.2 x 0.8 m table top at z = 0 in front of the robot.
DEFAULT_TABLE_BOX = Box(center=(0.5, 0.0, -0.02), size=(1.2, 0.8, 0.04))


def camera_offset(spec: CameraSpec) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """(pos, quat xyzw) of a policy camera in its parent frame, for ``CameraCfg.OffsetCfg(convention="world")``."""
    return tuple(float(v) for v in spec.pose.xyz), spec.pose.quat_xyzw()  # type: ignore[return-value]


def camera_parent_link(parent_frame: str) -> str:
    try:
        return PARENT_LINKS[parent_frame]
    except KeyError:
        raise ValueError(f"unknown camera parent_frame {parent_frame!r}; known: {sorted(PARENT_LINKS)}") from None


def intrinsic_matrix(intr: Intrinsics) -> np.ndarray:
    """Pinhole K (pixels) from width/height and the *horizontal* fov in degrees; square pixels, centred principal point."""
    f = (intr.width / 2.0) / math.tan(math.radians(intr.fov) / 2.0)
    return np.array([[f, 0.0, intr.width / 2.0], [0.0, f, intr.height / 2.0], [0.0, 0.0, 1.0]])


def vertical_fov_deg(intr: Intrinsics) -> float:
    f = intrinsic_matrix(intr)[0, 0]
    return math.degrees(2.0 * math.atan((intr.height / 2.0) / f))


def pinhole_aperture(intr: Intrinsics, horizontal_aperture_mm: float = 20.955) -> tuple[float, float, float]:
    """(focal_length_mm, horizontal_aperture_mm, vertical_aperture_mm) giving the requested hfov with square pixels."""
    f_px = intrinsic_matrix(intr)[0, 0]
    focal = f_px * horizontal_aperture_mm / intr.width
    return focal, horizontal_aperture_mm, horizontal_aperture_mm * intr.height / intr.width


def camera_forward_in_parent(pose: Pose) -> np.ndarray:
    """Optical axis (camera +x) expressed in the parent frame."""
    return pose.rotation_matrix() @ np.array([1.0, 0.0, 0.0])


# --------------------------------------------------------------------------------------------------------------------
# table box + cube spawning
# --------------------------------------------------------------------------------------------------------------------
@dataclass  # not frozen: Isaac Lab's configclass post-init walks dataclass attributes of a cfg and re-assigns them
class TableBox:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw_deg: float = 0.0
    source: str = "default"

    @property
    def top_z(self) -> float:
        return self.center[2] + 0.5 * self.size[2]

    def as_box(self) -> Box:
        return Box(center=self.center, size=self.size, rpy=(0.0, 0.0, self.yaw_deg))


def read_table_box(lab: LabConfig, run_dir: Path | None = None) -> TableBox:
    """The one place that decides where the table is (robot-base frame = world).

    Order: ``<run>/collision/primitives.json`` (fitted by the collision stage; it already folds in the lab override),
    then ``lab.overrides.table_box``, then :data:`DEFAULT_TABLE_BOX`.
    """
    if run_dir is not None:
        p = Path(run_dir) / "collision" / "primitives.json"
        if p.is_file():
            tb = (json.loads(p.read_text()) or {}).get("table_box")
            if tb and tb.get("center") and tb.get("size"):
                yaw = tb.get("yaw_deg")
                if yaw is None:
                    yaw = (tb.get("rpy_deg") or [0, 0, 0])[2]
                return TableBox(tuple(tb["center"]), tuple(tb["size"]), float(yaw), f"primitives.json:{tb.get('source', 'fit')}")
    ov = lab.overrides.table_box
    if ov is not None:
        return TableBox(tuple(ov.center), tuple(ov.size), float(ov.rpy[2]), "lab.overrides.table_box")
    b = DEFAULT_TABLE_BOX
    return TableBox(tuple(b.center), tuple(b.size), 0.0, "default")


def spawn_region(lab: LabConfig, table: TableBox, reach_min: float, reach_max: float, margin: float) -> SpawnRegion:
    """Axis-aligned cube spawn rectangle (robot-base frame) with ``z`` = table-top height.

    ``cubes.spawn_region`` wins. Otherwise: the table box shrunk by ``margin + half a cube`` and clipped to the
    robot's comfortable reach (``x`` in [reach_min, reach_max], ``|y| <= reach_max``).
    """
    if lab.cubes.spawn_region is not None:
        return lab.cubes.spawn_region
    m = margin + 0.5 * lab.cubes.size_m
    cx, cy, _ = table.center
    sx, sy, _ = table.size
    if abs(table.yaw_deg) > 1e-6:  # rotated table: conservative axis-aligned square inscribed in the rotated rectangle
        c, s_ = abs(math.cos(math.radians(table.yaw_deg))), abs(math.sin(math.radians(table.yaw_deg)))
        sx = sy = min(table.size[0], table.size[1]) / (c + s_)
    x0, x1 = max(cx - sx / 2 + m, reach_min), min(cx + sx / 2 - m, reach_max)
    y0, y1 = max(cy - sy / 2 + m, -reach_max), min(cy + sy / 2 - m, reach_max)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"table box {table} leaves no cube spawn region within reach [{reach_min}, {reach_max}] m "
            f"(got x [{x0:.2f}, {x1:.2f}], y [{y0:.2f}, {y1:.2f}]); set cubes.spawn_region in the lab config"
        )
    return SpawnRegion(x_range=(x0, x1), y_range=(y0, y1), z=table.top_z)


def sample_cube_poses(
    rng: np.random.Generator, count: int, region: SpawnRegion, size: float, min_gap: float = 0.04, max_tries: int = 200
) -> np.ndarray:
    """(count, 4) array of [x, y, z_centre, yaw_rad] with cubes non-overlapping (centre distance >= size*sqrt2 + min_gap)."""
    min_d = size * math.sqrt(2.0) + min_gap
    out: list[tuple[float, float]] = []
    for _ in range(count):
        for _try in range(max_tries):
            p = (rng.uniform(*region.x_range), rng.uniform(*region.y_range))
            if all(math.dist(p, q) >= min_d for q in out):
                out.append(p)
                break
        else:
            raise ValueError(f"could not place {count} cubes of {size} m in region {region}; enlarge it or reduce cubes.count")
    yaw = rng.uniform(-math.pi, math.pi, size=count)
    return np.array([[x, y, region.z + 0.5 * size, w] for (x, y), w in zip(out, yaw)])


def yaw_to_quat_xyzw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def grasp_yaw(cube_yaw: float, around: float = 0.0) -> float:
    """Gripper yaw aligned with the cube faces (a cube is 4-fold symmetric), wrapped to [around - pi/4, around + pi/4)."""
    return (cube_yaw - around + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0 + around


# --------------------------------------------------------------------------------------------------------------------
# camera projection (used to verify the camera conventions against rendered images)
# --------------------------------------------------------------------------------------------------------------------
def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def project_to_pixels(p_world, cam_pos_w, cam_quat_xyzw_w, intr: Intrinsics) -> tuple[float, float, float]:
    """World point -> (u, v, depth) for a camera in the lab convention (+x forward, +y left, +z up); u right, v down, pixels.

    ``depth`` is the distance along the optical axis (camera +x); a point behind the camera has depth <= 0.
    """
    R = quat_xyzw_to_matrix(cam_quat_xyzw_w)
    p = R.T @ (np.asarray(p_world, float) - np.asarray(cam_pos_w, float))
    k = intrinsic_matrix(intr)
    if p[0] <= 1e-9:
        return float("nan"), float("nan"), float(p[0])
    return float(k[0, 2] - k[0, 0] * p[1] / p[0]), float(k[1, 2] - k[1, 1] * p[2] / p[0]), float(p[0])


def compose_pose(parent_pos, parent_quat_xyzw, child_pos, child_quat_xyzw) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """pose of a child given in the parent frame -> world pose (pos, quat xyzw)."""
    Rp, Rc = quat_xyzw_to_matrix(parent_quat_xyzw), quat_xyzw_to_matrix(child_quat_xyzw)
    pos = np.asarray(parent_pos, float) + Rp @ np.asarray(child_pos, float)
    return pos, matrix_to_quat_xyzw(Rp @ Rc)


def matrix_to_quat_xyzw(m: np.ndarray) -> tuple[float, float, float, float]:
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(m[i, i] - m[j, j] - m[k, k] + 1.0) * 2
        v = [0.0] * 3
        v[i] = 0.25 * s
        v[j] = (m[j, i] + m[i, j]) / s
        v[k] = (m[k, i] + m[i, k]) / s
        q = (v[0], v[1], v[2], (m[k, j] - m[j, k]) / s)
    return tuple(float(x) for x in q)  # type: ignore[return-value]


def _hue_sat_val(a: np.ndarray):
    mx, mn = a.max(-1), a.min(-1)
    d = mx - mn
    safe = np.where(d > 1e-9, d, 1.0)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    h = np.where(mx == r, ((g - b) / safe) % 6, np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4)) * 60.0
    h = np.where(d > 1e-9, h, 0.0)
    sat = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0.0)
    return h, sat, mx


def color_centroid(img: np.ndarray, rgb: tuple[float, float, float], hue_tol_deg: float = 25.0, min_sat: float = 0.25, min_pixels: int = 12):
    """Pixel centroid (u, v) and count of pixels with the hue of ``rgb`` (and some saturation); None if fewer than ``min_pixels``.

    ``img`` is float rgb in [0, 1] (H, W, 3). Hue matching survives the tone mapping / lighting of the renderer, which changes
    brightness and saturation of a cube but not its hue.
    """
    a = img[..., :3].astype(np.float64)
    h, sat, val = _hue_sat_val(a)
    th, ts, _ = _hue_sat_val(np.asarray(rgb, float)[None, :])
    dh = np.abs((h - th[0] + 180.0) % 360.0 - 180.0)
    m = (dh < hue_tol_deg) & (sat > min_sat) & (val > 0.25)
    n = int(m.sum())
    if n < min_pixels:
        return None
    vs, us = np.nonzero(m)
    return float(us.mean()), float(vs.mean()), n
