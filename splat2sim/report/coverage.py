"""Coverage of the fitted table top: fraction of its surface seen (depth-consistent) from >= N keyframes. numpy + cv2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def table_grid(table: dict, cell: float = 0.03) -> np.ndarray:
    """Cell centres (N,3) on the top face of an (anchor-frame) box dict {center,size,rpy_deg}."""
    from ..collision.fitting import rpy_to_R

    sx, sy, sz = table["size"]
    nx, ny = int(sx / cell + 1e-6), int(sy / cell + 1e-6)
    us = (np.arange(nx) + 0.5) * cell - sx / 2
    vs = (np.arange(ny) + 0.5) * cell - sy / 2
    g = np.stack(np.meshgrid(us, vs, indexing="ij"), -1).reshape(-1, 2)
    local = np.c_[g, np.full(len(g), sz / 2)]
    return local @ rpy_to_R(table["rpy_deg"]).T + np.asarray(table["center"])


def view_counts(points_recon: np.ndarray, Ts: np.ndarray, depth_loader, intr: dict, depth_tol: float = 0.05,
                z_range=(0.2, 3.5)) -> np.ndarray:
    """Number of frames from which each point is seen: in the image, in range, and the measured depth agrees with its distance
    (so occluded points do not count). ``depth_loader(k)`` -> (H,W) float metres (0 = invalid) for pose index k."""
    cnt = np.zeros(len(points_recon), int)
    W, H = int(intr["width"]), int(intr["height"])
    for k, T in enumerate(Ts):
        d = depth_loader(k)
        if d is None:
            continue
        pc = (points_recon - T[:3, 3]) @ T[:3, :3]
        z = pc[:, 2]
        ok = (z > z_range[0]) & (z < z_range[1])
        u = np.round(intr["fx"] * pc[:, 0] / np.where(ok, z, 1) + intr["cx"]).astype(int)
        v = np.round(intr["fy"] * pc[:, 1] / np.where(ok, z, 1) + intr["cy"]).astype(int)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.where(ok)[0]
        dm = d[v[idx], u[idx]]
        good = (dm > 0) & (np.abs(dm - z[idx]) < depth_tol + 0.01 * z[idx])
        cnt[idx[good]] += 1
    return cnt


def table_coverage(run: Path, min_views=(1, 3, 5), max_frames: int = 150) -> dict:
    from PIL import Image

    from ..collision.stage import load_anchor
    from ..recon.poseio import read_frames_csv, read_poses_csv

    prim = run / "collision" / "primitives.json"
    if not prim.is_file():
        return {"status": "skipped: no collision/primitives.json"}
    tb = json.loads(prim.read_text()).get("table_box")
    if not tb or "center" not in tb:
        return {"status": "skipped: no table box"}
    T_a, kind, _ = load_anchor(run)
    pts_a = table_grid(tb)
    Ti = np.linalg.inv(T_a)
    pts = pts_a @ Ti[:3, :3].T + Ti[:3, 3]  # anchor -> recon
    intr = json.loads((run / "frames" / "intrinsics.json").read_text())
    idx, _, Ts = read_poses_csv(run / "poses" / "poses.csv")
    fidx, _, col, dep = read_frames_csv(run / "frames" / "frames.csv")
    row = {int(i): k for k, i in enumerate(fidx)}
    sel = [k for k in np.unique(np.linspace(0, len(idx) - 1, min(max_frames, len(idx))).round().astype(int)) if int(idx[k]) in row]
    scale = float(intr.get("depth_scale_m", 0.001))

    def load(j):
        k = sel[j]
        p = run / "frames" / dep[row[int(idx[k])]]
        a = np.array(Image.open(p))
        return a.astype(np.float32) * scale

    cnt = view_counts(pts, Ts[sel], load, intr)
    out = {"anchor": kind, "cells": int(len(cnt)), "cell_m": 0.03, "frames_used": len(sel),
           "table_size_m": tb["size"][:2], "source": tb.get("source")}
    for n in min_views:
        out[f"fraction_ge_{n}_views"] = float((cnt >= n).mean())
    out["median_views"] = float(np.median(cnt))
    # heat map
    import cv2

    sx, sy = tb["size"][:2]
    nx, ny = int(sx / 0.03 + 1e-6), int(sy / 0.03 + 1e-6)
    grid = cnt.reshape(nx, ny).T  # rows = y
    img = np.clip(grid / 8.0, 0, 1)
    hm = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    hm[grid == 0] = (40, 40, 200)  # unseen = red
    hm = cv2.resize(hm, (nx * 10, ny * 10), interpolation=cv2.INTER_NEAREST)
    cv2.putText(hm, "table top views (red = unseen, yellow >= 8)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    out["heatmap"] = "coverage_table.png"
    cv2.imwrite(str(run / "report" / "coverage_table.png"), hm)
    return out
