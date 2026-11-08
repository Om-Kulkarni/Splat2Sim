"""Rigid-alignment, trajectory error and plane-fitting helpers (numpy only)."""

from __future__ import annotations

import numpy as np

from .poseio import quat_to_R


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares ``dst ~ s * R @ src + t`` (Umeyama 1991). Returns (s, R, t)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale and var_s > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def ate_rmse(est: np.ndarray, gt: np.ndarray, with_scale: bool = False) -> dict:
    """Absolute trajectory error after Umeyama alignment of est onto gt (both Nx3, metres)."""
    s, R, t = umeyama(est, gt, with_scale)
    err = np.linalg.norm(gt - (s * est @ R.T + t), axis=1)
    return {"rmse": float(np.sqrt((err ** 2).mean())), "mean": float(err.mean()), "median": float(np.median(err)),
            "max": float(err.max()), "scale": s, "R": R.tolist(), "t": t.tolist()}


def ate_with_lever_arm(est: np.ndarray, gt: np.ndarray, gt_R: np.ndarray, with_scale: bool = False, iters: int = 60) -> dict:
    """ATE when GT tracks a rigid *body* that is offset from the camera.

    gt(t) = s R_a est(t) + t_a + R_gt(t) d, with a constant lever arm d in the GT body frame. Alternate
    Umeyama and a linear least-squares solve for d. Returns the ATE dict plus ``lever_arm_m``.
    """
    d = np.zeros(3)
    A = gt_R.reshape(-1, 3)  # stacked R_gt(t), shape (3N, 3)
    for _ in range(iters):
        out = ate_rmse(est, gt - np.einsum("nij,j->ni", gt_R, d), with_scale)
        aligned = out["scale"] * est @ np.array(out["R"]).T + np.array(out["t"])
        # residual of the *uncorrected* GT against the aligned estimate is R_gt(t) d (+ noise)
        d_new = np.linalg.lstsq(A, (gt - aligned).reshape(-1), rcond=None)[0]
        converged = np.linalg.norm(d_new - d) < 1e-9
        d = d_new
        if converged:
            break
    out = ate_rmse(est, gt - np.einsum("nij,j->ni", gt_R, d), with_scale)
    out["lever_arm_m"] = d.tolist()
    return out


def resample_positions(t_src: np.ndarray, p_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Linear interpolation of positions (Nx3) at query times; returns Mx3."""
    return np.stack([np.interp(t_query, t_src, p_src[:, k]) for k in range(3)], axis=1)


def nearest_rotations(t_src: np.ndarray, quat_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(t_src, t_query), 1, len(t_src) - 1)
    idx = np.where(np.abs(t_src[idx - 1] - t_query) < np.abs(t_src[idx] - t_query), idx - 1, idx)
    return np.array([quat_to_R(quat_src[i]) for i in idx])


def backproject(depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float, stride: int = 4,
                zmin: float = 0.2, zmax: float = 4.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth (HxW metres) -> camera-frame points (OpenCV) at a pixel stride. Returns (xyz Nx3, u, v)."""
    vs, us = np.mgrid[0 : depth_m.shape[0] : stride, 0 : depth_m.shape[1] : stride]
    z = depth_m[vs, us]
    ok = (z > zmin) & (z < zmax)
    u, v, z = us[ok], vs[ok], z[ok]
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1), u, v


def transform_points(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


def voxel_downsample(p: np.ndarray, voxel: float) -> np.ndarray:
    key = np.floor(p / voxel).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return p[np.sort(first)]


def fit_plane_ransac(p: np.ndarray, iters: int = 400, thresh: float = 0.02, rng: np.random.Generator | None = None,
                     up_hint: np.ndarray | None = None, max_tilt_deg: float = 12.0) -> dict | None:
    """Dominant plane by RANSAC + least-squares refit. With ``up_hint`` only near-horizontal planes
    (normal within ``max_tilt_deg`` of the hint) are considered. Returns {normal, d, inliers, count} with
    ``normal . x + d = 0`` (unit normal, sign unspecified) or None."""
    rng = rng or np.random.default_rng(0)
    n = len(p)
    if n < 50:
        return None
    cos_lim = np.cos(np.radians(max_tilt_deg))
    best, best_cnt = None, 0
    for _ in range(iters):
        a, b, c = p[rng.choice(n, 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        if up_hint is not None and abs(float(nrm @ up_hint)) < cos_lim:
            continue
        d = -float(nrm @ a)
        cnt = int((np.abs(p @ nrm + d) < thresh).sum())
        if cnt > best_cnt:
            best, best_cnt = (nrm, d), cnt
    if best is None:
        return None
    nrm, d = best
    inl = np.abs(p @ nrm + d) < thresh
    q = p[inl]
    c0 = q.mean(0)
    nrm = np.linalg.svd(q - c0, full_matrices=False)[2][-1]  # refit
    if up_hint is not None and abs(float(nrm @ up_hint)) < cos_lim:
        nrm = best[0]
    d = -float(nrm @ c0)
    inl = np.abs(p @ nrm + d) < thresh
    return {"normal": nrm, "d": d, "inliers": inl, "count": int(inl.sum())}


def gravity_plane_anchor(points: np.ndarray, cam_centers: np.ndarray, cam_forward0: np.ndarray,
                         up_hint: np.ndarray | None = None, thresh: float = 0.02,
                         rng: np.random.Generator | None = None) -> dict:
    """Fallback anchor: T_anchor_recon (4x4) mapping recon points into a z-up frame whose z=0 plane is the dominant
    horizontal plane and whose origin is the plane-inlier centroid. x points along the first camera's viewing direction
    projected onto the plane. Cameras end up above the plane (z>0).

    ``up_hint``: gravity-up in the recon frame when trustworthy (IMU); otherwise the largest plane is taken
    and ``up_source`` is "plane" (low confidence: it may be a wall).
    """
    fit = fit_plane_ransac(points, thresh=thresh, rng=rng, up_hint=up_hint)
    if fit is None and up_hint is not None:  # no horizontal plane found -> trust gravity alone
        up = up_hint / np.linalg.norm(up_hint)
        origin = np.median(points, axis=0)
        info = {"plane_inliers": 0, "plane_inlier_fraction": 0.0, "up_source": "imu_only"}
    elif fit is None:
        raise ValueError("no plane found in the back-projected depth (too few points)")
    else:
        up = fit["normal"]
        if float(np.median(cam_centers @ up + fit["d"])) < 0:  # cameras must be on the +z side
            up = -up
        d = -float(up @ (points[fit["inliers"]].mean(0)))
        origin = points[fit["inliers"]].mean(0)
        origin = origin - (float(up @ origin) + d) * up  # on the plane
        info = {"plane_inliers": fit["count"], "plane_inlier_fraction": fit["count"] / len(points),
                "up_source": "imu+plane" if up_hint is not None else "plane"}
    fwd = cam_forward0 - float(cam_forward0 @ up) * up
    if np.linalg.norm(fwd) < 1e-3:
        fwd = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 1.0, 0.0])
    x = fwd / np.linalg.norm(fwd)
    y = np.cross(up, x)
    R = np.stack([x, y, up])  # rows = anchor axes expressed in recon
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ origin
    return {"T_anchor_recon": T, "origin_recon": origin, "up_recon": up, **info}
