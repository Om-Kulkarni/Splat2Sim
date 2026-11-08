"""Mask-vs-URDF silhouette comparison. ``report/silhouette/sil_<frame>.png`` (255 = rendered robot, from robot_silhouette.py)
against ``masks/<frame>.png`` (0 = masked robot)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def compare(mask_png: Path, sil_png: Path, dilate_px: int) -> dict | None:
    m = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE)
    s = cv2.imread(str(sil_png), cv2.IMREAD_GRAYSCALE)
    if m is None or s is None:
        return None
    if m.shape != s.shape:
        s = cv2.resize(s, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
    robot_mask = m < 128
    sil = s > 127
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)) if dilate_px > 0 else None
    sil_d = cv2.dilate(sil.astype(np.uint8), k).astype(bool) if k is not None else sil
    inter = (robot_mask & sil).sum()
    union = (robot_mask | sil).sum()
    inter_d = (robot_mask & sil_d).sum()
    union_d = (robot_mask | sil_d).sum()
    return {"silhouette_px": int(sil.sum()), "mask_px": int(robot_mask.sum()),
            "iou_raw": float(inter / union) if union else None,
            "iou_vs_dilated_silhouette": float(inter_d / union_d) if union_d else None,
            "silhouette_covered_by_mask": float(inter / sil.sum()) if sil.sum() else None}


def overlay(color_png: Path, mask_png: Path, sil_png: Path, out: Path) -> None:
    img = cv2.imread(str(color_png))
    m = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE) < 128
    s = cv2.imread(str(sil_png), cv2.IMREAD_GRAYSCALE) > 127
    if s.shape != m.shape:
        s = cv2.resize(s.astype(np.uint8), (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    for mk, col in ((m, (0, 255, 0)), (s, (0, 0, 255))):
        cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, col, 2)
    cv2.putText(img, "green = SAM mask (dilated), red = URDF silhouette at the scan pose", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(str(out), img)


def evaluate(run: Path) -> dict:
    sd = run / "report" / "silhouette"
    meta_p = sd / "meta.json"
    if not meta_p.is_file():
        return {"status": "skipped: no report/silhouette (needs a ChArUco anchor and masks)"}
    meta = json.loads(meta_p.read_text())
    if meta.get("status") != "ok":
        return {"status": meta.get("status", "skipped")}
    mm = run / "masks" / "meta.json"
    dil = int(json.loads(mm.read_text()).get("dilate_px", 15)) if mm.is_file() else 15
    rows = []
    for f in meta["frames"]:
        r = compare(run / "masks" / f"{f:06d}.png", sd / f"sil_{f:06d}.png", dil)
        if r:
            rows.append({"frame": f, **r})
    vis = [r for r in rows if r["silhouette_px"] > 200]  # frames where the robot is actually in view
    out = {"status": "ok", "frames_rendered": len(rows), "frames_robot_visible": len(vis), "dilate_px": dil,
           "scan_pose_joints": meta.get("scan_pose_joints"), "per_frame": rows}
    for key in ("iou_raw", "iou_vs_dilated_silhouette", "silhouette_covered_by_mask"):
        v = [r[key] for r in vis if r[key] is not None]
        out[f"median_{key}"] = float(np.median(v)) if v else None
        out[f"min_{key}"] = float(np.min(v)) if v else None
    if vis:
        cands = sorted(vis, key=lambda r: -r["silhouette_px"])[:1] + sorted(vis, key=lambda r: r["silhouette_covered_by_mask"] or 1)[:1]
        ovs = []
        for r in cands:
            o = run / "report" / f"silhouette_overlay_{r['frame']:06d}.png"
            col = run / "frames" / "color" / f"{r['frame']:06d}.png"
            overlay(col, run / "masks" / f"{r['frame']:06d}.png", sd / f"sil_{r['frame']:06d}.png", o)
            ovs.append(o.name)
        out["overlays"] = ovs
    return out
