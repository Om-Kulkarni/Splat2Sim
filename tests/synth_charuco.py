"""Synthetic ChArUco Capture with known ground truth, in the ingest/poses file contract.

Writes ``<run>/frames/{color,depth,frames.csv,intrinsics.json}``, ``<run>/poses/poses.csv`` (same format as the poses
stage), ``<run>/lab.json`` (charuco + board_to_base) and ``<run>/synth_gt.json``.

World: the board (frame origin = first inner corner) lies flat on a grey table plane (board z = 0; the board frame has z pointing INTO the board, y down in the
printed face, see splat2sim/anchor/detect.py). A camera orbits it, looks at it from 0.7-1.5 m with roll, and sometimes
looks away (no board). Rendering: homography warp of ``CharucoBoard.generateImage`` (2x supersampled), analytic plane
depth (uint16 mm) with range-dependent noise and holes, image noise, random occluders, SLAM-like pose error
(slow drift + per-frame jitter), a few frames without a pose. ``scale_a`` enlarges the physical board
(the printed board is ``scale_a`` times larger than the lab config says; poses and depth stay metric) to exercise the scale estimate.

``python tests/synth_charuco.py OUT_RUN_DIR [--frames N] [--seed S]`` writes a set by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

LAB = {"lab": "synth",
       "charuco": {"dictionary": "DICT_4X4_50", "squares_x": 7, "squares_y": 5, "square_length_m": 0.055, "marker_length_m": 0.041},
       "board_to_base": {"xyz": [0.50, 0.10, 0.02], "rpy": [0.0, 0.0, 15.0]}}


def look_at(c: np.ndarray, target: np.ndarray, roll_deg: float = 0.0, down_hint=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Camera-to-world (OpenCV: x right, y down, z forward) at ``c`` looking at ``target``."""
    f = target - c
    f = f / np.linalg.norm(f)
    x = np.cross(down_hint, f)
    x /= np.linalg.norm(x)
    y = np.cross(f, x)
    R = np.stack([x, y, f], axis=1) @ Rotation.from_euler("z", roll_deg, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, c
    return T


def _T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def make_synth(out: Path, n_frames: int = 60, seed: int = 0, size=(848, 480), fx: float = 609.5, scale_a: float = 1.0,
               pose_drift_cm: float = 1.0, pose_drift_deg: float = 1.0, pose_jitter_mm: float = 3.0, pose_jitter_deg: float = 0.2,
               px_noise: float = 2.0, frac_away: float = 0.15, frac_occluded: float = 0.25, frac_nopose: float = 0.05,
               board_visible: bool = True, lab: dict | None = None) -> dict:
    rng = np.random.default_rng(seed)
    lab = lab or LAB
    out = Path(out)
    for d in ("frames/color", "frames/depth", "poses"):
        (out / d).mkdir(parents=True, exist_ok=True)
    W, H = size
    K = np.array([[fx, 0, W / 2 - 0.5], [0, fx, H / 2 - 0.5], [0, 0, 1.0]])
    cs = lab["charuco"]
    sq = cs["square_length_m"]
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cs["dictionary"]))
    board = cv2.aruco.CharucoBoard((cs["squares_x"], cs["squares_y"]), sq, cs["marker_length_m"], d)
    bw, bh = cs["squares_x"] * sq, cs["squares_y"] * sq
    ppm, margin = 1500, 90  # source texture px/m; white paper margin (px)
    tex = board.generateImage((int(bw * ppm) + 2 * margin, int(bh * ppm) + 2 * margin), marginSize=margin, borderBits=1)
    # texture px -> board-frame metres (origin = first inner corner = outer corner + one square)
    M = np.diag([scale_a, scale_a, 1.0]) @ np.array([[1 / ppm, 0, -margin / ppm - sq], [0, 1 / ppm, -margin / ppm - sq], [0, 0, 1.0]])
    ss = 2
    Kss = K.copy()
    Kss[:2, :2] *= ss
    Kss[:2, 2] = ss * (K[:2, 2] + 0.5) - 0.5
    # ground-truth recon <- board
    R_rb = Rotation.from_euler("zyx", [35.0, 4.0, -3.0], degrees=True).as_matrix()
    t_rb = np.array([0.35, -0.20, 0.08])
    gt_sim = {"scale": scale_a, "R": R_rb.tolist(), "t": t_rb.tolist()}  # p_recon = a * R p_board_nominal + t_rb
    centre = scale_a * np.array([bw / 2 - sq, bh / 2 - sq, 0.0])  # physical board metres
    # trajectory in the board frame
    az = np.linspace(-60, 60, n_frames) + rng.normal(0, 4, n_frames)
    el = rng.uniform(25, 60, n_frames)
    dist = rng.uniform(0.7, 1.5, n_frames)
    away = rng.random(n_frames) < frac_away if board_visible else np.ones(n_frames, bool)
    occl = rng.random(n_frames) < frac_occluded
    nopose = rng.random(n_frames) < frac_nopose
    drift_t, drift_r = np.zeros(3), np.zeros(3)
    step_t = pose_drift_cm / 100 / np.sqrt(n_frames)
    step_r = np.radians(pose_drift_deg) / np.sqrt(n_frames)
    rows, pose_rows, T_true_list = [], [], []
    for k in range(n_frames):
        a, e = np.radians(az[k]), np.radians(el[k])
        # camera in front of the printed face: board z points into the board, so the camera sits at z < 0
        c = centre + dist[k] * np.array([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), -np.sin(e)])
        tgt = centre + rng.normal(0, 0.02, 3) * (1, 1, 0)
        if away[k]:
            tgt = c + (c - centre) / np.linalg.norm(c - centre) + rng.normal(0, 0.3, 3)  # look away from the board
        T_bc = look_at(c, tgt, rng.uniform(-15, 15))
        T_cb = np.linalg.inv(T_bc)
        R, t = T_cb[:3, :3], T_cb[:3, 3]
        # --- render at 2x ---
        H_ = Kss @ np.stack([R[:, 0], R[:, 1], t], axis=1) @ M
        img = cv2.warpPerspective(tex, H_, (W * ss, H * ss), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=110)
        vv, uu = np.mgrid[0:H, 0:W]
        rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones(uu.shape)], -1)
        n = R[:, 2]  # board plane normal in camera coords; plane: n . (p - t) = 0 -> depth = n.t / n.ray
        den = rays @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(np.abs(den) > 1e-9, (n @ t) / den, 0.0)
        depth = np.where((depth > 0.15) & (depth < 4.0), depth, 0.0)
        gray = img.astype(np.float32)
        if occl[k] and not away[k]:  # a rectangular occluder (hand/tool) in front of the board
            cx, cy = rng.uniform(0.25, 0.75) * W, rng.uniform(0.25, 0.75) * H
            w2, h2 = rng.uniform(0.06, 0.18) * W, rng.uniform(0.1, 0.3) * H
            x0, x1, y0, y1 = int((cx - w2) * ss), int((cx + w2) * ss), int((cy - h2) * ss), int((cy + h2) * ss)
            gray[max(y0, 0) : y1, max(x0, 0) : x1] = 35
            depth[max(y0 // ss, 0) : y1 // ss, max(x0 // ss, 0) : x1 // ss] = rng.uniform(0.4, 0.6)
        small = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 0.6) + rng.normal(0, px_noise, small.shape)
        small = np.clip(small, 0, 255).astype(np.uint8)
        sig = 0.0005 + 0.002 * depth ** 2
        dn = depth + rng.normal(0, 1, depth.shape) * sig
        dn[rng.random(depth.shape) < 0.02] = 0
        dmm = np.clip(np.round(dn * 1000), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(out / f"frames/color/{k:06d}.png"), cv2.cvtColor(small, cv2.COLOR_GRAY2BGR))
        cv2.imwrite(str(out / f"frames/depth/{k:06d}.png"), dmm)
        ts = 1_700_000_000_000_000_000 + k * 33_333_333
        rows.append((k, ts, f"color/{k:06d}.png", f"depth/{k:06d}.png", 100.0))
        # --- pose in recon (true + SLAM-like error) ---
        T_rc_true = np.eye(4)
        T_rc_true[:3, :3] = R_rb @ T_bc[:3, :3]
        T_rc_true[:3, 3] = R_rb @ T_bc[:3, 3] + t_rb
        drift_t += rng.normal(0, step_t, 3)
        drift_r += rng.normal(0, step_r, 3)
        dT = _T(Rotation.from_rotvec(drift_r + rng.normal(0, np.radians(pose_jitter_deg), 3)).as_matrix(),
                drift_t + rng.normal(0, pose_jitter_mm / 1000, 3))
        T_est = dT @ T_rc_true
        T_true_list.append(T_rc_true)
        if not nopose[k]:
            pose_rows.append((k, ts, T_est))
    with open(out / "frames/frames.csv", "w") as f:
        f.write("index,timestamp_ns,color,depth,sharpness\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    (out / "frames/intrinsics.json").write_text(json.dumps({
        "fx": fx, "fy": fx, "cx": K[0, 2], "cy": K[1, 2], "width": W, "height": H, "distortion_model": "plumb_bob",
        "distortion_coeffs": [0.0] * 5, "depth_scale_m": 0.001, "source": "synthetic"}, indent=2))
    from splat2sim.recon import poseio

    poseio.write_poses_csv(out / "poses/poses.csv", [r[0] for r in pose_rows], [r[1] for r in pose_rows], [r[2] for r in pose_rows])
    (out / "lab.json").write_text(json.dumps(lab, indent=2))
    gt = {"sim_recon_from_board": gt_sim, "n_frames": n_frames, "frames_board_visible": int((~away).sum()),
          "frames_without_pose": int(nopose.sum()), "frames_occluded": int((occl & ~away).sum()), "seed": seed}
    (out / "synth_gt.json").write_text(json.dumps(gt, indent=2))
    return gt


def gt_sim_base_recon(gt: dict, lab: dict):
    """(s, R, t) recon -> base for the ground truth, in the solver's Sim3 form."""
    from splat2sim.anchor import solve

    g = gt["sim_recon_from_board"]
    sim_rb = (1.0, np.array(g["R"]), np.array(g["t"]))  # board -> recon, metres -> recon units (scale a)
    sim_rb = (g["scale"], sim_rb[1], sim_rb[2])
    b = lab["board_to_base"]
    Tb = solve.rpy_deg_matrix(b["xyz"], b["rpy"])
    return solve.compose((1.0, Tb[:3, :3], Tb[:3, 3]), solve.invert(sim_rb))


def evaluate(run: Path) -> dict:
    """Error of ``poses/anchor.json`` vs the ground truth: at the board centre (translation), rotation, scale, and the mean
    error over a 1 m x 1 m x 0.5 m cloud of points around the board (what a Scene actually sees)."""
    from splat2sim.anchor import solve

    run = Path(run)
    gt = json.loads((run / "synth_gt.json").read_text())
    lab = json.loads((run / "lab.json").read_text())
    a = json.loads((run / "poses/anchor.json").read_text())
    if a["status"] != "charuco":
        return {"status": a["status"], "reason": a.get("reason")}
    g = gt_sim_base_recon(gt, lab)
    est = (a["scale"], np.array(a["R"]), np.array(a["t"]))
    cs = lab["charuco"]
    centre_b = np.array([(cs["squares_x"] / 2 - 1) * cs["square_length_m"], (cs["squares_y"] / 2 - 1) * cs["square_length_m"], 0.0])
    gs = gt["sim_recon_from_board"]
    centre_recon = gs["scale"] * np.array(gs["R"]) @ centre_b + np.array(gs["t"])
    rng = np.random.default_rng(1)
    probe = centre_recon + rng.uniform(-0.5, 0.5, (500, 3)) * [1, 1, 0.5] * gs["scale"]
    err = solve.sim_error(est, g, probe)
    err["centre_mm"] = float(np.linalg.norm(solve.apply(est, centre_recon) - solve.apply(g, centre_recon)) * 1000)
    return err


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale-a", type=float, default=1.0)
    ap.add_argument("--no-board", action="store_true")
    ap.add_argument("--eval", action="store_true", help="only evaluate poses/anchor.json of OUT against synth_gt.json")
    a = ap.parse_args()
    if a.eval:
        print(json.dumps(evaluate(Path(a.out)), indent=1))
        raise SystemExit(0)
    print(json.dumps(make_synth(Path(a.out), a.frames, a.seed, scale_a=a.scale_a, board_visible=not a.no_board)))
