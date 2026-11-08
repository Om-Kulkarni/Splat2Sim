"""``poses`` stage (rtabmap container): frames/ -> poses/{poses.csv, colmap/, eval.json, stats.json, ...}.

Steps: stage a RTAB-Map dataset folder of symlinks -> ``rtabmap-rgbd_dataset`` (RGB-D, optional IMU) ->
``rtabmap-export`` optimised camera poses -> poses.csv -> COLMAP text model for 3DGRUT -> scene stats +
fallback gravity/plane anchor -> optional ATE against ``frames/gt_poses.csv``.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

import numpy as np

from . import colmap, geometry as geo, poseio
from .options import POSES_DEFAULTS, stage_options

LOOP_LINK_TYPES = (1, 2)  # rtabmap Link::kGlobalClosure, kLocalSpaceClosure


def imu_trustworthy(imu_csv: Path) -> tuple[bool, dict]:
    """Accelerometer norm must look like gravity (9.81 +- ~8%) and be stable; the Intel mock IMU fails this."""
    if not imu_csv.is_file():
        return False, {"reason": "no imu.csv"}
    a = np.loadtxt(imu_csv, delimiter=",", skiprows=1, ndmin=2)
    if len(a) < 100:
        return False, {"reason": f"only {len(a)} imu samples"}
    norm = np.linalg.norm(a[:, 4:7], axis=1)
    med, iqr = float(np.median(norm)), float(np.subtract(*np.percentile(norm, [75, 25])))
    ok = 9.0 <= med <= 10.6 and iqr < 2.0
    return ok, {"accel_norm_median": med, "accel_norm_iqr": iqr, "samples": int(len(a)),
                "reason": "ok" if ok else "accel norm not ~9.81 m/s^2 (broken/uncalibrated IMU)"}


def auto_stride(meta: dict, idx: np.ndarray, ts_ns: np.ndarray, kf: list[int], target_hz: float = 10.0) -> int:
    """SLAM stride so the processed rate is ~``target_hz`` (RTAB-Map time grows with every node; 30 fps adds nothing)."""
    k = np.array(kf)
    if len(k) < 2:
        return 1
    dt = np.median(np.diff(ts_ns[np.searchsorted(idx, k)])) / 1e9
    return max(1, int(round(1.0 / (dt * target_hz)))) if dt > 0 else 1


def _imread(path: Path, flags=None):
    import cv2

    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED if flags is None else flags)


def stage_dataset(frames: Path, work: Path, sel: np.ndarray, ts_ns: np.ndarray, color: list[str], depth: list[str],
                  imu_csv: Path | None, t0_ns: int) -> dict[int, int]:
    """Create work/{rgb,depth}/<t>.png symlinks (+ imu.txt). Returns {stamp_us: frame_index}."""
    shutil.rmtree(work, ignore_errors=True)
    (work / "rgb").mkdir(parents=True)
    (work / "depth").mkdir()
    stamp_to_idx: dict[int, int] = {}
    for i in sel:
        us = int(round((int(ts_ns[i]) - t0_ns) / 1e3))
        name = f"{us / 1e6:.6f}.png"
        stamp_to_idx[us] = int(i)
        (work / "rgb" / name).symlink_to(os.path.relpath(frames / color[i], work / "rgb"))
        (work / "depth" / name).symlink_to(os.path.relpath(frames / depth[i], work / "depth"))
    if imu_csv is not None:
        a = np.loadtxt(imu_csv, delimiter=",", skiprows=1, ndmin=2)  # t_ns gx gy gz ax ay az
        with open(work / "imu.txt", "w") as f:
            f.write("# t ax ay az gx gy gz  (camera optical frame)\n")
            for r in a:
                f.write(f"{(int(r[0]) - t0_ns) / 1e9:.6f} {r[4]:.6f} {r[5]:.6f} {r[6]:.6f} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f}\n")
    return stamp_to_idx


def run_rtabmap(work: Path, intr: dict, use_imu: bool, log: Path) -> Path:
    out = work / "out"
    out.mkdir(exist_ok=True)
    cmd = ["rtabmap-rgbd_dataset",
           "--fx", str(intr["fx"]), "--fy", str(intr["fy"]), "--cx", str(intr["cx"]), "--cy", str(intr["cy"]),
           "--width", str(intr["width"]), "--height", str(intr["height"]), "--depth_factor", "1",
           "--rgb_dir", "rgb", "--depth_dir", "depth",
           "--Rtabmap/DetectionRate", "0", "--Rtabmap/CreateIntermediateNodes", "false",
           "--RGBD/LinearUpdate", "0", "--RGBD/AngularUpdate", "0",
           "--output", str(out), "--output_name", "slam"]
    if use_imu:
        cmd += ["--imu", str(work / "imu.txt")]
    cmd.append(str(work))
    with open(log, "w") as lf:
        subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT)
    return out / "slam.db"


def export_poses(db: Path) -> Path:
    exp = db.parent / "export"
    exp.mkdir(exist_ok=True)
    subprocess.run(["rtabmap-export", "--poses", "--poses_camera", "--output_dir", str(exp), str(db)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    cands = sorted(exp.glob("*camera_poses.txt"))
    if not cands:
        raise RuntimeError(f"rtabmap-export produced no *camera_poses.txt in {exp}")
    return cands[0]


def loop_closures(db: Path) -> dict:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        by_type = {int(t): int(n) for t, n in con.execute("SELECT type, COUNT(*) FROM Link GROUP BY type")}
    finally:
        con.close()
    return {"loop_closures": sum(by_type.get(t, 0) for t in LOOP_LINK_TYPES), "links_by_type": by_type}


def evaluate_gt(poses_ts_ns: np.ndarray, T: np.ndarray, gt_csv: Path, gt_unit: str) -> dict:
    scale = {"mm": 1e-3, "m": 1.0, "cm": 1e-2}.get(gt_unit, None)
    if scale is None:
        return {"error": f"unknown GT translation unit {gt_unit!r}"}
    gt_t, gt_p, gt_q = poseio.read_gt_csv(gt_csv, scale)
    est = T[:, :3, 3]
    ts = poses_ts_ns
    ok = (ts >= gt_t[0]) & (ts <= gt_t[-1])

    def sub(off_ns: int):
        m = (ts + off_ns >= gt_t[0]) & (ts + off_ns <= gt_t[-1])
        return m, geo.resample_positions(gt_t, gt_p, ts[m] + off_ns)

    best = (None, 1e9)
    for off_ms in range(-300, 301, 20):  # camera/mocap clock offset search
        m, g = sub(off_ms * 1_000_000)
        if m.sum() < 20:
            continue
        r = geo.ate_rmse(est[m], g)["rmse"]
        if r < best[1]:
            best = (off_ms, r)
    if best[0] is None or ok.sum() < 20:
        return {"error": "GT does not overlap the poses"}
    off = best[0] * 1_000_000
    m, g = sub(off)
    gt_R = geo.nearest_rotations(gt_t, gt_q, ts[m] + off)
    se3 = geo.ate_rmse(est[m], g)
    sim3 = geo.ate_rmse(est[m], g, with_scale=True)
    lever = geo.ate_with_lever_arm(est[m], g, gt_R)
    path = float(np.linalg.norm(np.diff(est[m], axis=0), axis=1).sum())
    return {
        "n_matched": int(m.sum()), "time_offset_ms": best[0], "path_length_m": path,
        "ate_rmse_se3_m": se3["rmse"], "ate_max_se3_m": se3["max"],
        "ate_rmse_sim3_m": sim3["rmse"], "sim3_scale": sim3["scale"],
        "ate_rmse_se3_leverarm_m": lever["rmse"], "lever_arm_body_m": lever["lever_arm_m"],
        "note": "GT = OptiTrack rigid body (mm in the mocap frame), not the camera: positions aligned with Umeyama; "
                "'leverarm' additionally fits a constant body->camera offset. sim3_scale ~ 1 means the metric scale is right.",
    }


def build_colmap(run: Path, frames: Path, masks: Path | None, sel_pose_idx: np.ndarray, T: np.ndarray, color: list[str],
                 depth: list[str], intr: dict, o: dict) -> dict:
    root = run / "poses" / "colmap"
    shutil.rmtree(root, ignore_errors=True)
    img_dir = root / "images"
    img_dir.mkdir(parents=True)
    n = len(sel_pose_idx)
    keep = np.unique(np.linspace(0, n - 1, min(n, int(o["train_max_images"]))).round().astype(int))
    names, Ts, n_mask = [], [], 0
    pts, cols = [], []
    for k, j in enumerate(keep):
        fi = int(sel_pose_idx[j])
        name = Path(color[fi]).name
        (img_dir / name).symlink_to(os.path.relpath(frames / color[fi], img_dir))
        names.append(name)
        Ts.append(T[j])
        mk = None
        if masks is not None and (masks / name).is_file():
            m = _imread(masks / name, 0)  # grayscale
            if m is not None:
                robot = m > 127 if o["mask_robot_is_white"] else m <= 127
                mk = (~robot)
                import cv2

                cv2.imwrite(str(img_dir / (Path(name).stem + "_mask.png")), (mk.astype(np.uint8) * 255))
                n_mask += 1
        if k % int(o["points_every"]) == 0:
            d = _imread(frames / depth[fi]).astype(np.float32) / 1000.0
            p, u, v = geo.backproject(d, intr["fx"], intr["fy"], intr["cx"], intr["cy"], stride=4, zmax=float(o["max_depth_m"]))
            if mk is not None:
                good = mk[v, u]
                p, u, v = p[good], u[good], v[good]
            import cv2

            c = cv2.cvtColor(_imread(frames / color[fi]), cv2.COLOR_BGR2RGB)
            pts.append(geo.transform_points(T[j], p))
            cols.append(c[v, u])
    P, C = np.concatenate(pts), np.concatenate(cols)
    if len(P) > int(o["max_points"]):
        sel = np.random.default_rng(0).choice(len(P), int(o["max_points"]), replace=False)
        P, C = P[sel], C[sel]
    colmap.write_model(root, (intr["width"], intr["height"]), (intr["fx"], intr["fy"], intr["cx"], intr["cy"]),
                       names, np.array(Ts), P, C)
    return {"images": len(names), "masked_images": n_mask, "points": int(len(P)), "dir": str(root)}


def scene_stats_and_anchor(frames: Path, poses_dir: Path, sel_pose_idx, T, depth: list[str], intr: dict, o: dict, up_hint) -> None:
    sub = np.unique(np.linspace(0, len(sel_pose_idx) - 1, min(len(sel_pose_idx), 60)).round().astype(int))
    pts = []
    for j in sub:
        d = _imread(frames / depth[int(sel_pose_idx[j])]).astype(np.float32) / 1000.0
        p, _, _ = geo.backproject(d, intr["fx"], intr["fy"], intr["cx"], intr["cy"], stride=6, zmax=float(o["ransac_max_depth_m"]))
        pts.append(geo.transform_points(T[j], p))
    P = geo.voxel_downsample(np.concatenate(pts), 0.02)
    if len(P) > 200000:
        P = P[np.random.default_rng(0).choice(len(P), 200000, replace=False)]
    centers = T[:, :3, 3]
    lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
    stats = {"points_bbox_min": lo.tolist(), "points_bbox_max": hi.tolist(),
             "camera_bbox_min": centers.min(0).tolist(), "camera_bbox_max": centers.max(0).tolist(), "n_points": int(len(P))}
    (poses_dir / "scene_stats.json").write_text(json.dumps(stats, indent=2))
    anc = geo.gravity_plane_anchor(P, centers, T[0, :3, 2], up_hint=up_hint)
    out = {"kind": "fallback", "method": "gravity_plane",
           "T_anchor_recon": anc["T_anchor_recon"].tolist(), "origin_recon": anc["origin_recon"].tolist(),
           "up_recon": anc["up_recon"].tolist(), "up_source": anc["up_source"],
           "plane_inliers": anc["plane_inliers"], "plane_inlier_fraction": anc["plane_inlier_fraction"],
           "note": "No ChArUco Anchor available: z-up from the dominant horizontal plane, origin at its centroid. "
                   "Not aligned to the robot base."}
    (poses_dir / "fallback_anchor.json").write_text(json.dumps(out, indent=2))


def run(scene_root: Path) -> dict:
    t_start = time.time()
    o = stage_options(scene_root, "poses", POSES_DEFAULTS)
    frames = scene_root / "frames"
    poses_dir = scene_root / "poses"
    masks = scene_root / "masks" if any((scene_root / "masks").glob("*.png")) else None
    shutil.rmtree(poses_dir / "work", ignore_errors=True)
    poses_dir.mkdir(exist_ok=True)
    for f in ("poses.csv", "eval.json", "stats.json", "fallback_anchor.json", "scene_stats.json"):
        (poses_dir / f).unlink(missing_ok=True)
    intr = json.loads((frames / "intrinsics.json").read_text())
    meta = json.loads((frames / "meta.json").read_text())
    idx, ts_ns, color, depth = poseio.read_frames_csv(frames / "frames.csv")
    kf = [int(x) for x in (frames / "keyframes.txt").read_text().split()] if (frames / "keyframes.txt").is_file() else list(idx)
    every = auto_stride(meta, idx, ts_ns, kf) if o["every"] == "auto" else int(o["every"])
    sel = np.array(kf[::every])
    if o["max_frames"]:
        sel = sel[: int(o["max_frames"])]
    trusted, imu_info = imu_trustworthy(frames / "imu.csv")
    use_imu = trusted if o["use_imu"] == "auto" else bool(o["use_imu"])
    if use_imu and not (frames / "imu.csv").is_file():
        raise ValueError("poses.use_imu=true but frames/imu.csv does not exist")
    print(f"[poses] stride {every}: {len(sel)} frames to SLAM; imu: use={use_imu} (setting={o['use_imu']}, {imu_info})", flush=True)

    t0 = int(ts_ns[sel[0]])
    if use_imu:
        t0 = min(t0, int(np.loadtxt(frames / "imu.csv", delimiter=",", skiprows=1, usecols=0, max_rows=1)))
    t0 -= 1_000_000_000  # RTAB-Map's image-folder reader rejects a file named 0.000000.png ("Conversion filename to timestamp failed"): first stamp = 1 s
    work = poses_dir / "work"
    stamp_to_idx = stage_dataset(frames, work, sel, ts_ns, color, depth, frames / "imu.csv" if use_imu else None, t0)
    t_slam = time.time()
    db = run_rtabmap(work, intr, use_imu, poses_dir / "rtabmap.log")
    slam_s = time.time() - t_slam
    cam_file = export_poses(db)
    ts_s, T_all, _ = poseio.read_tum_poses(cam_file)
    fidx, Ts = [], []
    for t, T in zip(ts_s, T_all):
        us = int(round(t * 1e6))
        if us in stamp_to_idx:
            fidx.append(stamp_to_idx[us])
            Ts.append(T)
    if not fidx:
        raise RuntimeError("no exported pose could be matched to a frame (timestamp mismatch)")
    order = np.argsort(fidx)
    fidx = np.array(fidx)[order]
    Ts = np.array(Ts)[order]
    poseio.write_poses_csv(poses_dir / "poses.csv", fidx, ts_ns[fidx], Ts)

    lc = loop_closures(db)
    stats = {"frames_total": int(len(sel)), "frames_registered": int(len(fidx)),
             "registered_fraction": float(len(fidx) / len(sel)), **lc,
             "imu": {"used": use_imu, "setting": o["use_imu"], **imu_info}, "slam_seconds": round(slam_s, 1), "stride": every,
             "intrinsics_source": intr.get("source"), "frame": "recon = RTAB-Map world (gravity-aligned z-up if imu used, else first camera)"}
    warnings = []
    if len(fidx) < 0.95 * len(sel):
        lost = np.setdiff1d(sel, fidx)
        warnings.append(f"{len(lost)} of {len(sel)} SLAM frames got no pose (tracking lost); first lost frame index "
                        f"{int(lost[0])}, last {int(lost[-1])}")
    stats["warnings"] = warnings
    if len(fidx) < 30:
        raise RuntimeError(f"only {len(fidx)} of {len(sel)} frames registered; SLAM failed (see poses/rtabmap.log)")

    stats["colmap"] = build_colmap(scene_root, frames, masks, fidx, Ts, color, depth, intr, o)
    scene_stats_and_anchor(frames, poses_dir, fidx, Ts, depth, intr, o, np.array([0.0, 0.0, 1.0]) if use_imu else None)
    stats["fallback_anchor"] = json.loads((poses_dir / "fallback_anchor.json").read_text())["up_source"]

    gt_csv = frames / "gt_poses.csv"
    if gt_csv.is_file():
        ev = evaluate_gt(ts_ns[fidx], Ts, gt_csv, (meta.get("gt_poses") or {}).get("translation_units", "m"))
        (poses_dir / "eval.json").write_text(json.dumps(ev, indent=2))
        stats["ate"] = {k: ev.get(k) for k in ("ate_rmse_se3_m", "ate_rmse_se3_leverarm_m", "sim3_scale", "time_offset_ms", "error")}
    stats["seconds"] = round(time.time() - t_start, 1)
    (poses_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    shutil.rmtree(work / "rgb", ignore_errors=True)
    shutil.rmtree(work / "depth", ignore_errors=True)
    print(f"[poses] done: {stats['frames_registered']}/{stats['frames_total']} registered, "
          f"{lc['loop_closures']} loop closures, ate={stats.get('ate')}, {stats['seconds']} s", flush=True)
    return stats
