"""Score a pipeline run on the synthetic Capture against its ground truth (CPU; run in the r2s-ingest image).

    docker run --rm -v ~/real2sim-data:/data -v $PWD:/repo -w /repo -e PYTHONPATH=/repo --entrypoint python r2s-ingest:latest \
        tools/synth_capture/evaluate.py --run /data/runs/synth-robot --capture /data/captures/synth-robot

Writes ``<run>/synth_eval.json`` and prints it. Reports: anchor error (anchored SLAM trajectory vs GT camera poses in the base frame;
anchor vs the Sim(3) fit of the SLAM trajectory to GT), mask IoU vs the rendered robot, table / pedestal boxes vs truth.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from splat2sim.anchor import solve
from splat2sim.recon.poseio import read_poses_csv


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def anchor_eval(run: Path, cap: Path) -> dict:
    a = json.loads((run / "poses/anchor.json").read_text())
    if a.get("status") != "charuco":
        return {"status": a.get("status"), "reason": a.get("reason")}
    idx, ts, Ts = read_poses_csv(run / "poses/poses.csv")
    import csv

    with open(cap / "gt_poses.csv", newline="") as f:
        rows = list(csv.reader(f))[1:]
    gt_ts = np.array([int(r[0]) for r in rows], np.int64)
    gt_v = np.array([[float(x) for x in r[1:8]] for r in rows])
    P_est, P_gt, R_est, R_gt = [], [], [], []
    from splat2sim.recon.poseio import quat_to_R

    for t, T in zip(ts, Ts):
        j = int(np.argmin(np.abs(gt_ts - int(t))))
        if abs(int(gt_ts[j]) - int(t)) > 1_000_000:
            continue
        row = gt_v[j]
        P_est.append(T[:3, 3]); P_gt.append(row[:3]); R_est.append(T[:3, :3]); R_gt.append(quat_to_R(row[3:7]))
    P_est, P_gt = np.array(P_est), np.array(P_gt)
    T_ba = np.array(a["T_base_recon"], float)
    s = float(a["scale"])
    R_a = T_ba[:3, :3] / s
    mapped = s * P_est @ R_a.T + T_ba[:3, 3]
    err = np.linalg.norm(mapped - P_gt, axis=1)
    rerr = [rot_deg((R_a @ Re).T @ Rg) for Re, Rg in zip(R_est, R_gt)]
    # best Sim(3) of the SLAM trajectory onto GT (what a perfect anchor would be, up to SLAM drift)
    best = solve.umeyama(P_est, P_gt, with_scale=True)
    s_b, R_b, t_b = best
    mapped_b = s_b * P_est @ R_b.T + t_b
    # anchor vs best fit, evaluated on a cloud around the board (the workcell the scene is used for)
    gtj = json.loads((cap / "synth_gt.json").read_text())
    probe = np.random.default_rng(1).uniform([0.0, -0.6, -0.5], [1.2, 0.6, 0.8], (500, 3))  # base-frame points
    probe_recon = (probe - T_ba[:3, 3]) @ R_a / s
    d_probe = np.linalg.norm((s * probe_recon @ R_a.T + T_ba[:3, 3]) - (s_b * probe_recon @ R_b.T + t_b), axis=1)
    bc = np.array(gtj["board_to_base"]["T"], float)
    cs = gtj["charuco"]
    centre_b = np.array([(cs["squares_x"] / 2 - 1) * cs["square_length_m"], (cs["squares_y"] / 2 - 1) * cs["square_length_m"], 0.0])
    centre_base = bc[:3, :3] @ centre_b + bc[:3, 3]
    centre_recon = (centre_base - T_ba[:3, 3]) @ R_a / s
    return {"status": "charuco", "n_poses": int(len(P_est)),
            "anchored_trajectory_vs_gt": {"pos_rmse_mm": float(np.sqrt((err ** 2).mean()) * 1000), "pos_median_mm": float(np.median(err) * 1000),
                                          "rot_median_deg": float(np.median(rerr)), "rot_p90_deg": float(np.percentile(rerr, 90))},
            "slam_best_sim3_vs_gt": {"scale": float(s_b), "pos_rmse_mm": float(np.sqrt(((mapped_b - P_gt) ** 2).sum(1).mean()) * 1000)},
            "anchor_scale": s, "anchor_scale_error_pct": float((s / s_b - 1) * 100),
            "anchor_vs_best_fit_mm_over_workcell": {"mean": float(d_probe.mean() * 1000), "max": float(d_probe.max() * 1000)},
            "anchor_rot_vs_best_fit_deg": rot_deg(R_a.T @ R_b),
            "board_centre_base_truth": centre_base.tolist(),
            "reproj_rms_px": a.get("reproj_rms_px"), "corner_3d_rms_mm": a.get("corner_3d_rms_mm")}


def mask_eval(run: Path, cap: Path) -> dict:
    mdir, gdir = run / "masks", cap / "gt/robot_mask"
    meta = json.loads((mdir / "meta.json").read_text()) if (mdir / "meta.json").is_file() else {}
    dil = int(meta.get("dilate_px", 15))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1))
    rows = []
    for f in sorted(mdir.glob("0*.png")):
        g = cv2.imread(str(gdir / f.name), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) < 128
        gb = g > 127
        if gb.sum() < 300:
            continue
        gd = cv2.dilate(gb.astype(np.uint8), k).astype(bool)
        rows.append({"frame": int(f.stem), "iou_raw": float((m & gb).sum() / (m | gb).sum()), "iou_vs_dilated_gt": float((m & gd).sum() / (m | gd).sum()),
                     "gt_covered": float((m & gb).sum() / gb.sum()), "mask_px_outside_dilated_gt": float((m & ~gd).sum())})
    if not rows:
        return {"status": "no frames with a visible robot"}
    r = lambda key: float(np.mean([x[key] for x in rows]))  # noqa: E731
    return {"frames": len(rows), "dilate_px": dil, "mean_iou_raw": r("iou_raw"), "mean_iou_vs_dilated_gt": r("iou_vs_dilated_gt"),
            "mean_gt_covered": r("gt_covered"), "min_gt_covered": float(np.min([x["gt_covered"] for x in rows])),
            "frames_with_gt_covered_lt_0.95": int(sum(x["gt_covered"] < 0.95 for x in rows))}


def boxes_eval(run: Path, cap: Path) -> dict:
    gt = json.loads((cap / "synth_gt.json").read_text())
    pr = json.loads((run / "collision/primitives.json").read_text())
    out = {}
    for key in ("table_box", "pedestal_box"):
        b, t = pr.get(key), gt[key]
        if not b or "center" not in b:
            out[key] = {"status": "absent"}
            continue
        tc, ts_ = np.array(t["center"]), np.array(t["size"])
        c, s = np.array(b["center"]), np.array(b["size"])
        top_est, top_gt = c[2] + s[2] / 2, tc[2] + ts_[2] / 2
        # XY overlap (IoU of axis-aligned footprints; yaw ignored for the table which is axis-aligned here)
        lo, hi = np.maximum(c[:2] - s[:2] / 2, tc[:2] - ts_[:2] / 2), np.minimum(c[:2] + s[:2] / 2, tc[:2] + ts_[:2] / 2)
        inter = float(np.prod(np.clip(hi - lo, 0, None)))
        iou = inter / (float(np.prod(s[:2])) + float(np.prod(ts_[:2])) - inter)
        out[key] = {"source": b["source"], "center_est": c.tolist(), "center_gt": tc.tolist(), "size_est": s.tolist(), "size_gt": ts_.tolist(),
                    "top_z_error_mm": float((top_est - top_gt) * 1000), "center_xy_error_mm": float(np.linalg.norm(c[:2] - tc[:2]) * 1000),
                    "footprint_iou": iou, "yaw_deg": b.get("yaw_deg")}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--capture", required=True)
    a = ap.parse_args()
    run, cap = Path(a.run), Path(a.capture)
    res = {}
    for name, fn in (("anchor", anchor_eval), ("mask", mask_eval), ("boxes", boxes_eval)):
        try:
            res[name] = fn(run, cap)
        except Exception as e:
            res[name] = {"error": f"{type(e).__name__}: {e}"}
    (run / "synth_eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
