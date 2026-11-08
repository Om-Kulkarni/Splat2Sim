"""``anchor`` stage: detect the ChArUco board in the keyframes, fit the recon -> board Sim(3), compose with the lab's
board -> base transform and write ``poses/anchor.json``.

Inputs: ``frames/{frames.csv,intrinsics.json,color,depth}``, ``poses/poses.csv`` (T_world_cam, OpenCV camera),
``lab.json`` (``charuco``, ``board_to_base``). Without a usable board the status is ``fallback`` (no board seen) or
``failed`` (board seen but the fit is unreliable); both leave downstream stages on the gravity/plane fallback.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from splat2sim.recon import poseio
from splat2sim.recon.options import stage_options

from . import detect, solve

ANCHOR_DEFAULTS = {
    "max_frames": 400,        # frames (evenly subsampled from the posed ones) to run the detector on
    "min_corners": 6,         # per-frame corners needed to use a detection
    "min_frames": 3,          # frames with RANSAC inliers needed for status "charuco"
    "ransac_thresh_m": 0.04,  # 3D residual of a corner (recon frame) to count as inlier
    "ransac_iters": 3000,
    "fix_scale": False,       # true: recon is trusted as metric (scale = 1), only R, t are estimated
    "depth_weight": 1.0,      # weight of the depth-consistency term in the refinement (0 = reprojection only)
    "inlier_px": 4.0,         # reprojection error below which a corner counts as inlier in the final statistics
    "max_reproj_rms_px": 5.0,
    "scale_warn": 0.05,       # warn when |scale - 1| exceeds this
    "overlays": 3,
}


def _load_lab(root: Path) -> dict:
    p = root / "lab.json"
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing (written by `r2s build`)")
    lab = json.loads(p.read_text())
    for k in ("charuco", "board_to_base"):
        if k not in lab:
            raise ValueError(f"{p}: no '{k}' section")
    return lab


def _intrinsics(frames: Path) -> tuple[np.ndarray, np.ndarray, float]:
    d = json.loads((frames / "intrinsics.json").read_text())
    K = np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1.0]])
    dist = np.array(d.get("distortion_coeffs") or [], float)
    return K, (dist if dist.size else None), float(d.get("depth_scale_m", 0.001))


def collect(frames: Path, poses: Path, lab: dict, o: dict) -> dict:
    """Run the detector on posed frames. Returns {obs: [(frame, T_wc, Detection)], counts...}."""
    K, dist, dscale = _intrinsics(frames)
    fidx, _, color, depth = poseio.read_frames_csv(frames / "frames.csv")
    row = {int(i): k for k, i in enumerate(fidx)}
    pidx, _, Ts = poseio.read_poses_csv(poses / "poses.csv")
    sel = [(int(i), T) for i, T in zip(pidx, Ts) if int(i) in row]
    if o["max_frames"] and len(sel) > o["max_frames"]:
        keep = np.unique(np.linspace(0, len(sel) - 1, o["max_frames"]).round().astype(int))
        sel = [sel[k] for k in keep]
    board = detect.make_board(lab["charuco"])
    det = detect.make_detector(board)
    out = []
    for i, T in sel:
        gray = cv2.imread(str(frames / color[row[i]]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        dm = detect.read_depth_m(frames / depth[row[i]], dscale) if depth[row[i]] else None
        d = detect.detect_frame(det, gray, dm, K, dist, o["min_corners"])
        if d is not None:
            out.append((i, T, d))
    return {"K": K, "dist": dist, "board": board, "considered": len(sel), "dets": out,
            "board_pts": detect.board_corners(board)}


def solve_anchor(K: np.ndarray, board_pts: np.ndarray, dets: list, board_to_base: np.ndarray, o: dict,
                 rng: np.random.Generator | None = None) -> dict:
    """Pure solver on collected detections. Returns the anchor dict (status "charuco" or "failed") plus the Sim3s."""
    rng = rng or np.random.default_rng(0)
    # correspondences with depth: board point q <-> recon point p
    Q, P, FR = [], [], []
    for k, (i, T_wc, d) in enumerate(dets):
        ok = np.isfinite(d.z)
        ray = np.linalg.inv(K) @ np.vstack([d.uv[ok].T, np.ones(ok.sum())])
        pc = (ray * d.z[ok]).T
        P.append(pc @ T_wc[:3, :3].T + T_wc[:3, 3])
        Q.append(board_pts[d.ids[ok]])
        FR.append(np.full(ok.sum(), k))
    if not Q or sum(len(x) for x in Q) < 4:
        return {"status": "failed", "reason": "board detected but fewer than 4 corners have valid depth"}
    Q, P, FR = np.concatenate(Q), np.concatenate(P), np.concatenate(FR)
    fit = solve.ransac_sim3(Q, P, o["ransac_thresh_m"], o["ransac_iters"], rng, with_scale=not o["fix_scale"])
    if fit is None:
        return {"status": "failed", "reason": "RANSAC found no consistent board->recon Sim(3) (depth/pose trouble or a misdetection)",
                "n_correspondences": int(len(Q))}
    sim0, mask = fit
    # frames that keep >= min_corners inliers take part in the refinement (all their corners, incl. those without depth)
    per_frame = np.bincount(FR[mask], minlength=len(dets))
    used = [k for k in range(len(dets)) if per_frame[k] >= o["min_corners"]]
    if len(used) < o["min_frames"]:
        return {"status": "failed", "reason": f"only {len(used)} frames with >= {o['min_corners']} consistent corners "
                f"(need {o['min_frames']})", "n_correspondences": int(len(Q)), "ransac_inliers": int(mask.sum())}
    obs = []
    for k in used:
        i, T_wc, d = dets[k]
        obs.append(solve.FrameObs(T_cw=np.linalg.inv(T_wc), uv=d.uv, q=board_pts[d.ids], z=d.z, frame=i))
    sim, info = solve.refine_sim3(sim0, obs, K, with_scale=not o["fix_scale"], depth_weight=o["depth_weight"])

    # statistics
    res = solve.reproj_residuals(sim, obs, K)
    err = np.concatenate([np.linalg.norm(r, axis=1) for r in res])
    inl = err < o["inlier_px"]
    # 3D residual of depth-backprojected corners vs the board model, in board metres
    d3 = []
    for k in used:
        i, T_wc, d = dets[k]
        ok = np.isfinite(d.z)
        ray = np.linalg.inv(K) @ np.vstack([d.uv[ok].T, np.ones(ok.sum())])
        p = ((ray * d.z[ok]).T) @ T_wc[:3, :3].T + T_wc[:3, 3]
        d3.append(np.linalg.norm(p - solve.apply(sim, board_pts[d.ids[ok]]), axis=1) / sim[0])
    d3 = np.concatenate(d3) if d3 else np.zeros(0)
    d3_in = d3[d3 < o["ransac_thresh_m"] / max(sim[0], 1e-9)] if len(d3) else d3
    r_init = np.linalg.norm(P[mask] - solve.apply(sim0, Q[mask]), axis=1) / sim0[0]
    out = {
        "sim_board_recon": sim,           # board -> recon (s, R, t)
        "reproj_rms_px": float(np.sqrt((err[inl] ** 2).mean())) if inl.any() else float("nan"),
        "reproj_median_px": float(np.median(err)),
        "reproj_all_rms_px": float(np.sqrt((err ** 2).mean())),
        "corner_3d_rms_mm": float(np.sqrt((d3_in ** 2).mean()) * 1000) if len(d3_in) else float("nan"),
        "corner_3d_median_mm": float(np.median(d3) * 1000) if len(d3) else float("nan"),
        "init_corner_3d_rms_mm": float(np.sqrt((r_init ** 2).mean()) * 1000),
        "n_correspondences": int(len(Q)), "ransac_inliers": int(mask.sum()),
        "frames_used": [dets[k][0] for k in used], "corners_used": int(sum(len(r) for r in res)),
        "corners_inlier_px": int(inl.sum()), "refine": info, "status": "charuco",
    }
    return out


def _overlay(frames_dir: Path, color_rel: str, sim, T_wc, K, board_pts, d, path: Path) -> None:
    img = cv2.imread(str(frames_dir / color_rel))
    if img is None:
        return
    pc = solve.apply(sim, board_pts) @ np.linalg.inv(T_wc)[:3, :3].T + np.linalg.inv(T_wc)[:3, 3]
    for p in pc[pc[:, 2] > 0.05]:
        cv2.circle(img, (int(K[0, 0] * p[0] / p[2] + K[0, 2]), int(K[1, 1] * p[1] / p[2] + K[1, 2])), 5, (0, 0, 255), 1)
    for u, v in d.uv:
        cv2.circle(img, (int(round(u)), int(round(v))), 2, (0, 255, 0), -1)
    cv2.putText(img, "green = detected, red = reprojected from fitted board pose", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(str(path), img)


def run(root: Path) -> dict:
    t0 = time.time()
    root = Path(root)
    frames, poses = root / "frames", root / "poses"
    o = stage_options(root, "anchor", ANCHOR_DEFAULTS)
    lab = _load_lab(root)
    for f in ("anchor.json",):
        (poses / f).unlink(missing_ok=True)
    if not (poses / "poses.csv").is_file():
        raise FileNotFoundError(f"{poses / 'poses.csv'} missing; run the poses stage first")
    b2b = lab["board_to_base"]
    T_base_board = solve.rpy_deg_matrix(b2b["xyz"], b2b["rpy"])
    col = collect(frames, poses, lab, o)
    res: dict = {"charuco": lab["charuco"], "board_to_base": {**b2b, "T_base_board": T_base_board.tolist()},
                 "frames_considered": col["considered"], "frames_with_board": len(col["dets"]),
                 "corners_detected": int(sum(len(d.ids) for _, _, d in col["dets"])),
                 "corners_with_depth": int(sum(np.isfinite(d.z).sum() for _, _, d in col["dets"])), "options": o}
    warnings: list[str] = []
    if len(col["dets"]) == 0:
        res.update(status="fallback", reason=f"no ChArUco board detected in {col['considered']} posed frames")
    else:
        sol = solve_anchor(col["K"], col["board_pts"], col["dets"], T_base_board, o)
        if sol["status"] != "charuco":
            res.update(sol)
        else:
            sim_bo = sol.pop("sim_board_recon")
            sim_recon_board = solve.invert(sim_bo)
            sim_base_board = (1.0, T_base_board[:3, :3], T_base_board[:3, 3])
            sim_base_recon = solve.compose(sim_base_board, sim_recon_board)
            scale = sim_recon_board[0]
            res.update(sol)
            if not np.isfinite(res["reproj_rms_px"]) or res["reproj_rms_px"] > o["max_reproj_rms_px"]:
                res.update(status="failed", reason=f"reprojection RMS {res['reproj_rms_px']:.2f} px > {o['max_reproj_rms_px']} px")
            if abs(scale - 1) > o["scale_warn"]:
                warnings.append(f"recon->board scale {scale:.4f} deviates from 1 by {abs(scale - 1) * 100:.1f}% "
                                "(RTAB-Map is metric: check the measured square_length_m, board detection and depth)")
            if res["frames_used"] and len(res["frames_used"]) < 8:
                warnings.append(f"only {len(res['frames_used'])} frames used; view the board from more angles/distances")
            if res["status"] == "charuco":  # the transform is only published for a trusted fit
                res.update(
                    scale=float(sim_base_recon[0]), R=sim_base_recon[1].tolist(), t=sim_base_recon[2].tolist(),
                    T_base_recon=solve.to_matrix(sim_base_recon).tolist(),
                    recon_to_board={"scale": float(scale), "R": sim_recon_board[1].tolist(), "t": sim_recon_board[2].tolist(),
                                    "T": solve.to_matrix(sim_recon_board).tolist()},
                    scale_estimate=float(scale))
            else:
                res["scale_estimate"] = float(scale)
            if res["status"] == "charuco" and o["overlays"]:
                od = poses / "anchor_overlays"
                od.mkdir(exist_ok=True)
                for f in od.glob("*.png"):
                    f.unlink()
                _, _, color, _ = poseio.read_frames_csv(frames / "frames.csv")
                fidx = {int(i): k for k, i in enumerate(poseio.read_frames_csv(frames / "frames.csv")[0])}
                used = set(res["frames_used"])
                cands = [x for x in col["dets"] if x[0] in used]
                for j in np.unique(np.linspace(0, len(cands) - 1, o["overlays"]).round().astype(int)):
                    i, T_wc, d = cands[j]
                    _overlay(frames, color[fidx[i]], sim_bo, T_wc, col["K"], col["board_pts"], d, od / f"overlay_{i:06d}.png")
    res["warnings"] = warnings
    res["seconds"] = round(time.time() - t0, 1)
    (poses / "anchor.json").write_text(json.dumps(res, indent=2))
    for w in warnings:
        print(f"[anchor] WARNING: {w}")
    if res["status"] == "charuco":
        print(f"[anchor] charuco: {len(res['frames_used'])}/{res['frames_with_board']} frames, reproj {res['reproj_rms_px']:.2f} px, "
              f"3D {res['corner_3d_rms_mm']:.1f} mm, scale {res['scale_estimate']:.4f}")
    return res
