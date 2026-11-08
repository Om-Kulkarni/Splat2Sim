"""Sim(3) estimation (numpy + scipy): Umeyama in RANSAC, then robust reprojection refinement.

Convention: a Sim3 is ``(s, R, t)`` with ``p_dst = s * R @ p_src + t``. The board->recon Sim3 is what is optimised
(``p_recon = s R q_board + t``); the stage inverts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from splat2sim.recon.geometry import umeyama

Sim3 = tuple[float, np.ndarray, np.ndarray]


def apply(sim: Sim3, p: np.ndarray) -> np.ndarray:
    s, R, t = sim
    return s * np.asarray(p) @ R.T + t


def invert(sim: Sim3) -> Sim3:
    s, R, t = sim
    return 1.0 / s, R.T, -(R.T @ t) / s


def compose(a: Sim3, b: Sim3) -> Sim3:
    """a after b."""
    sa, Ra, ta = a
    sb, Rb, tb = b
    return sa * sb, Ra @ Rb, sa * Ra @ tb + ta


def to_matrix(sim: Sim3) -> np.ndarray:
    s, R, t = sim
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def rigid_matrix(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rpy_deg_matrix(xyz, rpy_deg) -> np.ndarray:
    """Lab-config pose (``R = Rz*Ry*Rx``, degrees) -> 4x4."""
    r, p, y = np.radians(rpy_deg)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rigid_matrix(Rz @ Ry @ Rx, xyz)


def sim_error(est: Sim3, gt: Sim3, probe: np.ndarray | None = None) -> dict:
    """Translation (mm, at the origin), rotation (deg) and scale (%) error of ``est`` vs ``gt``; optionally also the mean
    distance (mm) of ``probe`` points mapped by both."""
    se, Re, te = est
    sg, Rg, tg = gt
    ang = np.degrees(np.arccos(np.clip((np.trace(Rg.T @ Re) - 1) / 2, -1, 1)))
    out = {"translation_mm": float(np.linalg.norm(te - tg) * 1000), "rotation_deg": float(ang),
           "scale_pct": float((se / sg - 1) * 100)}
    if probe is not None:
        out["probe_mm"] = float(np.linalg.norm(apply(est, probe) - apply(gt, probe), axis=1).mean() * 1000)
    return out


def ransac_sim3(q: np.ndarray, p: np.ndarray, thresh: float, iters: int = 3000, rng: np.random.Generator | None = None,
                with_scale: bool = True, scale_range: tuple[float, float] = (0.5, 2.0), sample: int = 4) -> tuple[Sim3, np.ndarray] | None:
    """Fit ``p ~ s R q + t`` robustly. Returns (sim refit on the inliers, inlier mask) or None.

    Minimal samples (``sample`` points, degenerate/collinear ones skipped) are fitted with Umeyama; the model with the
    most inliers (residual < ``thresh``, in dst units) wins; the winner is refit on its inliers until the set is stable.
    """
    rng = rng or np.random.default_rng(0)
    n = len(q)
    if n < max(sample, 4):
        return None
    best_cnt, best_cost, best = 0, np.inf, None
    for _ in range(iters):
        idx = rng.choice(n, sample, replace=False)
        qs = q[idx]
        sv = np.linalg.svd(qs - qs.mean(0), compute_uv=False)
        if sv[1] < 1e-3 * max(sv[0], 1e-9) or sv[1] < 1e-3:  # near collinear sample
            continue
        s, R, t = umeyama(qs, p[idx], with_scale)
        if with_scale and not (scale_range[0] < s < scale_range[1]):
            continue
        r = np.linalg.norm(p - (s * q @ R.T + t), axis=1)
        inl = r < thresh
        cnt = int(inl.sum())
        cost = float(np.minimum(r, thresh).sum())
        if cnt > best_cnt or (cnt == best_cnt and cost < best_cost):
            best_cnt, best_cost, best = cnt, cost, (s, R, t)
    if best is None or best_cnt < 4:
        return None
    sim = best
    mask = np.linalg.norm(p - apply(sim, q), axis=1) < thresh
    for _ in range(10):
        s, R, t = umeyama(q[mask], p[mask], with_scale)
        sim = (s, R, t)
        new = np.linalg.norm(p - apply(sim, q), axis=1) < thresh
        if (new == mask).all():
            break
        mask = new
    return sim, mask


@dataclass
class FrameObs:
    """Detections of one frame. ``uv``: undistorted pixels (n,2); ``q``: board-frame points (n,3); ``z``: measured depth
    along the optical axis in metres (nan = none); ``T_cw``: recon -> camera (4x4)."""

    T_cw: np.ndarray
    uv: np.ndarray
    q: np.ndarray
    z: np.ndarray
    frame: int = 0
    extra: dict = field(default_factory=dict)


def _params_to_sim(x: np.ndarray, with_scale: bool, s0: float) -> Sim3:
    s = float(np.exp(x[0])) if with_scale else s0
    return s, Rotation.from_rotvec(x[1:4]).as_matrix(), x[4:7]


def reproj_residuals(sim: Sim3, obs: list[FrameObs], K: np.ndarray) -> list[np.ndarray]:
    """Per frame pixel residual vectors (n,2): projected - detected."""
    out = []
    for o in obs:
        pc = apply(sim, o.q) @ o.T_cw[:3, :3].T + o.T_cw[:3, 3]
        z = np.maximum(pc[:, 2], 1e-6)
        uv = np.stack([K[0, 0] * pc[:, 0] / z + K[0, 2], K[1, 1] * pc[:, 1] / z + K[1, 2]], axis=1)
        out.append(uv - o.uv)
    return out


def refine_sim3(sim0: Sim3, obs: list[FrameObs], K: np.ndarray, with_scale: bool = True, depth_weight: float = 1.0,
                px_sigma: float = 1.0, f_scale: float = 2.0) -> tuple[Sim3, dict]:
    """Minimise robust (soft-L1) corner reprojection error over all frames, plus an optional depth-consistency term.

    Free parameters: scale (log), rotation (rotvec), translation of the board in the recon frame. The depth term
    (weight ``depth_weight``, 0 = pure reprojection) compares the predicted camera-frame z of each corner with the
    measured depth, sigma 1 cm + 0.5% of range.
    """
    s0, R0, t0 = sim0
    x0 = np.concatenate([[np.log(s0)], Rotation.from_matrix(R0).as_rotvec(), t0])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Rcw = [o.T_cw[:3, :3] for o in obs]
    tcw = [o.T_cw[:3, 3] for o in obs]

    def fun(x):
        sim = _params_to_sim(x, with_scale, s0)
        res = []
        for o, R, t in zip(obs, Rcw, tcw):
            pc = apply(sim, o.q) @ R.T + t
            z = np.maximum(pc[:, 2], 1e-6)
            res.append((fx * pc[:, 0] / z + cx - o.uv[:, 0]) / px_sigma)
            res.append((fy * pc[:, 1] / z + cy - o.uv[:, 1]) / px_sigma)
            if depth_weight > 0:
                ok = np.isfinite(o.z)
                dz = np.zeros(len(z))
                dz[ok] = (z[ok] - o.z[ok]) / (0.01 + 0.005 * o.z[ok])
                res.append(depth_weight * dz)
        return np.concatenate(res)

    sol = least_squares(fun, x0, loss="soft_l1", f_scale=f_scale, x_scale="jac", max_nfev=200)
    sim = _params_to_sim(sol.x, with_scale, s0)
    return sim, {"cost": float(sol.cost), "nfev": int(sol.nfev), "success": bool(sol.success)}
