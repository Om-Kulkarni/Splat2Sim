"""Primitive fitting for the collision layer (numpy only; unit-testable anywhere).

All fitting happens in the **anchor frame** (z-up, metres; robot base at the origin once a ChArUco Anchor exists,
otherwise the gravity/plane fallback frame). A box is described by ``center`` (x y z), ``size`` (x y z extents in its own
frame) and ``rpy_deg`` (R = Rz*Ry*Rx, like the lab config); fitted boxes are level (roll = pitch = 0) with a yaw.
"""

from __future__ import annotations

import numpy as np

from ..recon.geometry import fit_plane_ransac


# ---------------------------------------------------------------- 2D helpers
def _cross2(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def convex_hull(xy: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain; returns hull vertices in CCW order (Mx2)."""
    pts = np.unique(np.round(np.asarray(xy, float), 9), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(points):
        h: list[np.ndarray] = []
        for p in points:
            while len(h) >= 2 and _cross2(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h

    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def min_area_rect(xy: np.ndarray) -> dict:
    """Minimum-area oriented rectangle of 2D points: {center (2,), size (2,) (len along u, along v), yaw (rad of u)}.
    ``yaw`` is wrapped to (-pi/2, pi/2] and size[0] >= size[1] is NOT enforced (u is the hull-edge direction)."""
    hull = convex_hull(xy)
    if len(hull) < 3:
        c = hull.mean(0)
        return {"center": c, "size": np.zeros(2), "yaw": 0.0}
    edges = np.roll(hull, -1, axis=0) - hull
    best = None
    for e in edges:
        ln = np.linalg.norm(e)
        if ln < 1e-12:
            continue
        u = e / ln
        v = np.array([-u[1], u[0]])
        a, b = hull @ u, hull @ v
        size = np.array([a.max() - a.min(), b.max() - b.min()])
        area = size[0] * size[1]
        if best is None or area < best[0] - 1e-12:
            ctr = u * 0.5 * (a.max() + a.min()) + v * 0.5 * (b.max() + b.min())
            best = (area, ctr, size, float(np.arctan2(u[1], u[0])))
    _, ctr, size, yaw = best
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2  # u and -u describe the same rectangle
    if abs(yaw - (-np.pi / 2)) < 1e-9:
        yaw = np.pi / 2
    return {"center": ctr, "size": size, "yaw": yaw}


def largest_component(xy: np.ndarray, cell: float = 0.04, min_points_per_cell: int = 2) -> np.ndarray:
    """Boolean mask of the points in the largest 8-connected occupied-grid component (cells with too few points are
    ignored). Removes isolated outliers before the extent fit."""
    ij = np.floor(np.asarray(xy) / cell).astype(np.int64)
    keys, inv, counts = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).reshape(-1)
    solid = counts >= min_points_per_cell
    if not solid.any():
        return np.ones(len(xy), bool)
    index = {tuple(k): n for n, k in enumerate(keys) if solid[n]}
    label = {}
    sizes: list[int] = []
    for k in index:
        if k in label:
            continue
        lid = len(sizes)
        label[k] = lid
        stack, total = [k], 0
        while stack:
            cur = stack.pop()
            total += int(counts[index[cur]])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cur[0] + dx, cur[1] + dy)
                    if nb in index and nb not in label:
                        label[nb] = lid
                        stack.append(nb)
        sizes.append(total)
    best = int(np.argmax(sizes))
    cell_ok = np.array([solid[n] and label.get(tuple(keys[n])) == best for n in range(len(keys))])
    return cell_ok[inv]


def _max_rect_in_grid(occ: np.ndarray) -> tuple[int, int, int, int, int]:
    """Largest all-True axis-aligned rectangle (histogram stack). Returns (area, row0, col0, height, width) in cells."""
    H, W = occ.shape
    heights = np.zeros(W, int)
    best = (0, 0, 0, 0, 0)
    for r in range(H):
        heights = np.where(occ[r], heights + 1, 0)
        stack: list[tuple[int, int]] = []  # (start col, height)
        for c in range(W + 1):
            h = int(heights[c]) if c < W else 0
            start = c
            while stack and stack[-1][1] >= h:
                sc, sh = stack.pop()
                if sh * (c - sc) > best[0]:
                    best = (sh * (c - sc), r - sh + 1, sc, sh, c - sc)
                start = sc
            stack.append((start, h))
    return best


def _close(occ: np.ndarray, it: int = 2) -> np.ndarray:
    """Binary closing (3x3) to fill small holes left by objects standing on the table."""
    def shift_or(a, fn):
        out = a.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                sh = np.roll(np.roll(a, dr, 0), dc, 1)
                out = fn(out, sh)
        return out
    pad = np.pad(occ, it + 1)
    for _ in range(it):
        pad = shift_or(pad, np.logical_or)
    for _ in range(it):
        pad = shift_or(pad, np.logical_and)
    return pad[it + 1:-(it + 1), it + 1:-(it + 1)]


def inscribed_rect(xy: np.ndarray, cell: float = 0.03, yaw_step_deg: float = 3.0, yaws_extra=()) -> dict:
    """Largest rectangle fully covered by the (hole-filled) occupancy of ``xy`` over a sweep of yaws. Unlike the
    bounding rectangle it never covers empty space (an L-shaped or partially scanned top still gives a box the cubes
    can rest on). Returns {center (2,), size (2,), yaw (rad), area}."""
    xy = np.asarray(xy, float)
    yaws = list(np.radians(np.arange(0.0, 90.0, yaw_step_deg))) + [float(y) for y in yaws_extra]
    best = None
    for yaw in yaws:
        c, s_ = np.cos(yaw), np.sin(yaw)
        q = xy @ np.array([[c, -s_], [s_, c]])  # coordinates along u=(c,s), v=(-s,c)
        lo = q.min(0) - 2 * cell
        ij = np.floor((q - lo) / cell).astype(int)
        occ = np.zeros(ij.max(0)[::-1] + 3, bool)
        occ[ij[:, 1], ij[:, 0]] = True
        occ = _close(occ)
        area, r0, c0, h, w = _max_rect_in_grid(occ)
        if best is None or area > best[0]:
            best = (area, yaw, lo, r0, c0, h, w)
    area, yaw, lo, r0, c0, h, w = best
    u0, v0 = lo[0] + c0 * cell, lo[1] + r0 * cell
    cq = np.array([u0 + 0.5 * w * cell, v0 + 0.5 * h * cell])
    c, s_ = np.cos(yaw), np.sin(yaw)
    center = np.array([cq[0] * c - cq[1] * s_, cq[0] * s_ + cq[1] * c])
    return {"center": center, "size": np.array([w * cell, h * cell]), "yaw": float(yaw), "area": float(area * cell * cell)}


# ---------------------------------------------------------------- box helpers
def rpy_to_R(rpy_deg) -> np.ndarray:
    r, p, y = np.radians(rpy_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_rpy_deg(R: np.ndarray) -> list[float]:
    """Inverse of :func:`rpy_to_R` (R = Rz*Ry*Rx), degrees."""
    b = -np.arcsin(np.clip(R[2, 0], -1, 1))
    g = np.arctan2(R[2, 1], R[2, 2])
    a = np.arctan2(R[1, 0], R[0, 0])
    return [float(np.degrees(g)), float(np.degrees(b)), float(np.degrees(a))]


def tilted_frame(normal: np.ndarray, yaw: float) -> np.ndarray:
    """Rotation whose z axis is ``normal`` and whose x axis is the heading ``yaw`` projected onto the plane."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    ux = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    x = ux - float(ux @ n) * n
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(n, x), n], axis=1)


def box_dict(center, size, rpy_deg, source: str, **extra) -> dict:
    rpy = [float(v) for v in rpy_deg]
    return {"center": [float(v) for v in center], "size": [float(v) for v in size], "rpy_deg": rpy,
            "yaw_deg": rpy[2], "source": source, **extra}


def box_surface_distance(points: np.ndarray, center, size, rpy_deg) -> np.ndarray:
    """Unsigned distance from points to the surface of an oriented box (0 on the surface, inside included)."""
    R = rpy_to_R(rpy_deg)
    q = (np.asarray(points, float) - np.asarray(center)) @ R  # box frame
    half = 0.5 * np.asarray(size, float)
    d = np.abs(q) - half
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(d.max(axis=1), 0.0)  # <= 0
    return np.abs(outside + inside)


def box_to_frame(box: dict, T: np.ndarray) -> dict:
    """Express an (anchor-frame) box in another frame given ``T`` mapping *that* frame into the box's frame, i.e. for
    recon: ``T = T_anchor_recon`` (box_recon = T^-1 box_anchor). Handles a uniform scale in ``T``. Returns
    {"matrix": 4x4 (columns = scaled box axes, t = center), "center", "size", "R", "scale"} of the box in the source frame,
    where ``matrix`` already includes the box size (a unit cube maps onto the box)."""
    T = np.asarray(T, float)
    s = float(np.cbrt(abs(np.linalg.det(T[:3, :3]))))
    R_T = T[:3, :3] / s
    Tinv = np.eye(4)
    Tinv[:3, :3] = R_T.T / s
    Tinv[:3, 3] = -(R_T.T / s) @ T[:3, 3]
    c = Tinv[:3, :3] @ np.asarray(box["center"]) + Tinv[:3, 3]
    R = R_T.T @ rpy_to_R(box["rpy_deg"])
    size = np.asarray(box["size"], float) / s
    M = np.eye(4)
    M[:3, :3] = R * size[None, :]
    M[:3, 3] = c
    return {"matrix": M, "center": c, "size": size, "R": R, "scale": s}


# ---------------------------------------------------------------- table
def _subsample(p: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(p) <= n:
        return p
    return p[rng.choice(len(p), n, replace=False)]


def find_horizontal_planes(points: np.ndarray, k: int = 3, thresh: float = 0.015, max_tilt_deg: float = 8.0,
                           z_range: tuple[float, float] | None = None, min_inliers: int = 200,
                           iters: int = 1500, rng: np.random.Generator | None = None) -> list[dict]:
    """Up to ``k`` near-horizontal planes (sequential RANSAC, z-up frame), largest first. If ``z_range`` is given only
    planes whose mean height lies inside it are kept. Each: {normal (z>0), d, inlier_idx, count, z, rms, tilt_deg}."""
    rng = rng or np.random.default_rng(0)
    up = np.array([0.0, 0.0, 1.0])
    remaining = np.arange(len(points))
    out: list[dict] = []
    for _ in range(k + 3):  # a few extra attempts when z_range rejects planes
        if len(out) >= k or len(remaining) < min_inliers:
            break
        sub_idx = remaining if len(remaining) <= 40000 else rng.choice(remaining, 40000, replace=False)
        fit = fit_plane_ransac(points[sub_idx], iters=iters, thresh=thresh, rng=rng, up_hint=up, max_tilt_deg=max_tilt_deg)
        if fit is None:
            break
        n, d = fit["normal"], fit["d"]
        if n[2] < 0:
            n, d = -n, -d
        dist = np.abs(points[remaining] @ n + d)
        inl = remaining[dist < thresh]
        remaining = remaining[dist >= thresh]
        if len(inl) < min_inliers:
            break
        q = points[inl]
        # refine with the full inlier set
        c0 = q.mean(0)
        n2 = np.linalg.svd(q - c0, full_matrices=False)[2][-1]
        if n2[2] < 0:
            n2 = -n2
        d2 = -float(n2 @ c0)
        res = q @ n2 + d2
        z = float(q[:, 2].mean())
        plane = {"normal": n2, "d": d2, "inlier_idx": inl, "count": int(len(inl)), "z": z,
                 "rms": float(np.sqrt(np.mean(res**2))), "tilt_deg": float(np.degrees(np.arccos(np.clip(n2[2], -1, 1))))}
        if z_range is not None and not (z_range[0] <= z <= z_range[1]):
            continue
        out.append(plane)
    return out


def fit_table_box(points: np.ndarray, normals: np.ndarray | None = None, thickness: float = 0.04,
                  thresh: float = 0.015, z_range: tuple[float, float] | None = None, cell: float = 0.04,
                  min_inliers: int = 200, rng: np.random.Generator | None = None,
                  extent: str = "inscribed", level: bool = False, prefer_elevated: bool = False,
                  z_max: float | None = None) -> dict | None:
    """Table-top box from anchor-frame surface points (mesh vertices): dominant horizontal plane (RANSAC, up-facing
    points only when ``normals`` given), extent = minimum-area rectangle of the largest connected inlier patch.
    ``prefer_elevated``: pick the largest plane >= 25 cm above the dominant one (floor) when it holds >= 15 % of its
    inliers and lies below ``z_max`` (fallback anchor only; with a ChArUco anchor use ``z_range`` around the base).
    ``extent``: "inscribed" (default; largest fully-observed rectangle) or "bounding" (minimum-area rectangle).
    ``level=False`` (default): the box follows the fitted plane's tilt (a handheld Scene's gravity estimate is rarely
    exact, and the box must sit on the splat's table); ``level=True`` keeps roll = pitch = 0 with the top at the mean
    inlier height. ``thickness`` extends downwards.
    Returns a box dict (source "ransac") with ``stats`` or None when no plane was found."""
    pts = np.asarray(points, float)
    if normals is not None:
        pts = pts[np.asarray(normals)[:, 2] > 0.8]
    planes = find_horizontal_planes(pts, k=5, thresh=thresh, z_range=z_range, min_inliers=min_inliers, rng=rng)
    if not planes:
        return None
    pl = planes[0]
    if prefer_elevated:
        # without a robot/board reference the dominant plane is often the floor: a table is a sizeable horizontal plane
        # clearly above it (and below the cameras, which look down on it)
        up = [p for p in planes[1:] if p["z"] > pl["z"] + 0.25 and p["count"] >= 0.15 * pl["count"]
              and (z_max is None or p["z"] < z_max)]
        if up:
            pl = max(up, key=lambda p: p["count"])
    inl = pts[pl["inlier_idx"]]
    keep = largest_component(inl[:, :2], cell=cell)
    inl = inl[keep]
    bound = min_area_rect(inl[:, :2])
    rect = inscribed_rect(inl[:, :2], cell=min(cell, 0.03), yaws_extra=[bound["yaw"]]) if extent == "inscribed" else bound
    top = float(inl[:, 2].mean())
    size = np.array([rect["size"][0], rect["size"][1], thickness])
    n = pl["normal"]
    if level:
        rpy = [0.0, 0.0, float(np.degrees(rect["yaw"]))]
        center = np.array([rect["center"][0], rect["center"][1], top - 0.5 * thickness])
    else:
        rpy = R_to_rpy_deg(tilted_frame(n, rect["yaw"]))
        x0, y0 = rect["center"]
        z_plane = -(n[0] * x0 + n[1] * y0 + pl["d"]) / n[2]  # n.x + d = 0
        center = np.array([x0, y0, z_plane]) - n * 0.5 * thickness
        top = float(z_plane)
    stats = {"extent": extent, "bounding_size": [float(v) for v in bound["size"]],
        "plane_rms_m": pl["rms"], "level_rms_m": float(inl[:, 2].std()), "level": level, "inliers": int(len(inl)),
        "inlier_fraction": float(pl["count"] / max(len(pts), 1)), "tilt_deg": pl["tilt_deg"], "top_z": top,
        "patch_area_m2": float(len(np.unique(np.floor(inl[:, :2] / cell).astype(np.int64), axis=0)) * cell * cell),
        "rect_area_m2": float(size[0] * size[1]), "bounding_area_m2": float(bound["size"][0] * bound["size"][1]),
        "thresh_m": thresh,
        "other_planes_z": [float(p["z"]) for p in planes if p is not pl], "chosen_plane_rank": next(i for i, p in enumerate(planes) if p is pl),
    }
    stats["fill_ratio"] = stats["patch_area_m2"] / max(stats["bounding_area_m2"], 1e-9)  # patch vs bounding rect
    return box_dict(center, size, rpy, "ransac", stats=stats)


# ---------------------------------------------------------------- pedestal
def fit_pedestal_box(points: np.ndarray, normals: np.ndarray | None, table_top_z: float | None,
                     base_xy=(0.0, 0.0), search_radius: float = 0.45, z_window: tuple[float, float] = (-0.08, 0.08),
                     default_height: float = 0.5, thresh: float = 0.015, min_inliers: int = 150,
                     rng: np.random.Generator | None = None) -> dict | None:
    """Box under the robot base: the horizontal patch within ``search_radius`` of ``base_xy`` and ``z_window`` (anchor
    frame, base origin z = 0, which is the robot's mounting plane = the pedestal top) is the pedestal top; the box extends
    down to the table top (when that is clearly below) else ``default_height``. Points on the table plane (+-3 cm) are
    excluded: a table top close below the base would otherwise be taken for the pedestal. None when no patch is seen."""
    pts = np.asarray(points, float)
    m = (np.linalg.norm(pts[:, :2] - np.asarray(base_xy), axis=1) < search_radius) & (pts[:, 2] > z_window[0]) & (pts[:, 2] < z_window[1])
    if normals is not None:
        m &= np.asarray(normals)[:, 2] > 0.8
    if table_top_z is not None:
        m &= np.abs(pts[:, 2] - table_top_z) > 0.03
    sub = pts[m]
    planes = find_horizontal_planes(sub, k=1, thresh=thresh, min_inliers=min_inliers, rng=rng)
    if not planes:
        return None
    inl = sub[planes[0]["inlier_idx"]]
    inl = inl[largest_component(inl[:, :2], cell=0.03)]
    rect = min_area_rect(inl[:, :2])
    top = float(inl[:, 2].mean())
    if table_top_z is not None and table_top_z < top - 0.05:
        bottom = table_top_z
    else:
        bottom = top - default_height
    size = np.array([rect["size"][0], rect["size"][1], top - bottom])
    center = np.array([rect["center"][0], rect["center"][1], 0.5 * (top + bottom)])
    stats = {"plane_rms_m": planes[0]["rms"], "inliers": int(len(inl)), "top_z": top, "bottom_z": float(bottom)}
    return box_dict(center, size, [0.0, 0.0, np.degrees(rect["yaw"])], "ransac", stats=stats)


def override_box(spec: dict) -> dict:
    """Lab-config ``Box`` (center, size, rpy in degrees) -> box dict with source "override"."""
    return box_dict(spec["center"], spec["size"], spec.get("rpy", [0.0, 0.0, 0.0]), "override")
