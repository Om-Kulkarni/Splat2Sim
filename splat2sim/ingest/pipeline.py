"""Ingest driver: Source -> frames/ contract."""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import imu as imu_mod
from . import keyframes, preview
from .detect import detect_format
from .readers import open_source

OWNED = ("color", "depth", "preview", "frames.csv", "intrinsics.json", "imu.csv", "meta.json", "keyframes.txt", "gt_poses.csv")
DEPTH_SCALE_M = 0.001


@dataclass
class Options:
    fmt: str | None = None
    every: int = 1
    min_sharpness: float = 0.0
    max_frames: int = 0
    intrinsics_file: Path | None = None
    depth_factor: float = 1000.0  # TUM only: depth units per metre
    depth_mode: str = "auto"  # mcap: auto|aligned|register
    no_preview: bool = False


def _clean(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for n in OWNED:
        p = out / n
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


def frame_stats(ts_ns: np.ndarray) -> dict:
    if len(ts_ns) < 2:
        return {}
    dt = np.diff(ts_ns) / 1e9
    med = float(np.median(dt))
    gaps = dt > 1.5 * med
    return {
        "rate_hz_mean": float((len(ts_ns) - 1) / (ts_ns[-1] - ts_ns[0]) * 1e9),
        "dt_median_s": med, "dt_max_s": float(dt.max()),
        "gaps_over_1.5x_median": int(gaps.sum()),
        "estimated_missing_frames": int(np.maximum(np.rint(dt[gaps] / med) - 1, 0).sum()),
    }


def run(capture: Path, out: Path, opt: Options) -> dict:
    t_start = time.time()
    capture = Path(capture)
    fmt = opt.fmt or detect_format(capture)
    src = open_source(capture, fmt, opt.intrinsics_file, opt.depth_factor, opt.depth_mode)
    intr = src.intrinsics()
    _clean(out)
    (out / "color").mkdir()
    (out / "depth").mkdir()

    ts_list: list[int] = []
    sharp: list[float] = []
    valid_frac: list[float] = []
    rows = []
    mid_idx = None
    for i, fr in enumerate(src.frames()):
        if opt.max_frames and i >= opt.max_frames:
            break
        name = f"{i:06d}.png"
        h, w = fr.color.shape[:2]
        if fr.depth.shape != (h, w) or fr.depth.dtype != np.uint16:
            raise ValueError(f"frame {i}: depth {fr.depth.shape}/{fr.depth.dtype} does not match colour {(h, w)}/uint16")
        cv2.imwrite(str(out / "color" / name), cv2.cvtColor(fr.color, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(out / "depth" / name), fr.depth, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        ts_list.append(fr.ts_ns)
        sharp.append(keyframes.sharpness(fr.color))
        valid_frac.append(float((fr.depth > 0).mean()))
        rows.append([i, fr.ts_ns, f"color/{name}", f"depth/{name}", f"{sharp[-1]:.2f}"])
        if i == 30 or (mid_idx is None and i == 0):  # keep a preview frame early enough to be cheap
            mid_idx = i
            keep = fr
        if i % 100 == 0:
            print(f"[ingest] {i} frames", flush=True)
    if not rows:
        raise ValueError(f"{capture}: no colour/depth frame pairs found")
    if (intr.width, intr.height) != (w, h):
        intr = intr.scaled(w, h)

    with open(out / "frames.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["index", "timestamp_ns", "color", "depth", "sharpness"])
        wr.writerows(rows)

    ts = np.array(ts_list, np.int64)
    kf = keyframes.select(sharp, opt.every, opt.min_sharpness)
    (out / "keyframes.txt").write_text("".join(f"{i}\n" for i in kf))

    (out / "intrinsics.json").write_text(json.dumps({
        "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy, "width": intr.width, "height": intr.height,
        "distortion_model": intr.distortion_model, "distortion_coeffs": intr.distortion,
        "depth_scale_m": DEPTH_SCALE_M, "source": intr.source,
    }, indent=2))

    # IMU
    imu = src.imu()
    imu_info: dict = {"present": False}
    if imu is not None and len(imu):
        imu = imu_mod.clip(imu, int(ts[0]), int(ts[-1]))
    if imu is not None and len(imu):
        with open(out / "imu.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["timestamp_ns", "gx", "gy", "gz", "ax", "ay", "az"])
            for t, g, a in zip(imu.t_ns, imu.gyro, imu.accel):
                wr.writerow([int(t), *(f"{x:.9g}" for x in g), *(f"{x:.9g}" for x in a)])
        am = np.linalg.norm(imu.accel, axis=1)
        imu_info = {"present": True, "samples": len(imu), "rate_hz": imu_mod.rate_hz(imu.t_ns),
                    "accel_norm_median": float(np.median(am)), "gyro_norm_median": float(np.median(np.linalg.norm(imu.gyro, axis=1))),
                    "first_ns": int(imu.t_ns[0]), "last_ns": int(imu.t_ns[-1])}
    else:
        imu_info = {"present": False, "note": "no IMU found in capture or none overlapping the frames"}

    # ground truth
    gt = src.gt_poses()
    gt_info: dict = {"present": False}
    if gt is not None:
        gt_t, gt_v = gt
        m = (gt_t >= ts[0] - 500_000_000) & (gt_t <= ts[-1] + 500_000_000)
        with open(out / "gt_poses.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
            for t, v in zip(gt_t[m], gt_v[m]):
                wr.writerow([int(t), *(f"{x:.9g}" for x in v)])
        gt_info = {"present": True, "poses": int(m.sum()), "translation_units": src.gt_units or "unknown",
                   "note": "OptiTrack pose of the tracked body in the mocap frame; translation in "
                           f"{src.gt_units or 'source units'}, quaternion qx qy qz qw. Not converted."}

    if not opt.no_preview and mid_idx is not None:
        (out / "preview").mkdir(exist_ok=True)
        ov = preview.overlay(keep.color, keep.depth)
        cv2.imwrite(str(out / "preview" / f"overlay_{mid_idx:06d}.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

    meta = {
        "source_format": src.fmt, "source_path": os.environ.get("R2S_CAPTURE_SRC") or str(capture),
        "frames": len(rows), "keyframes": len(kf), "keyframe_every": opt.every, "keyframe_min_sharpness": opt.min_sharpness,
        "image_size": [intr.width, intr.height],
        "duration_s": float((ts[-1] - ts[0]) / 1e9), "first_timestamp_ns": int(ts[0]), "last_timestamp_ns": int(ts[-1]),
        "frame_timing": frame_stats(ts),
        "depth_valid_fraction_mean": float(np.mean(valid_frac)),
        "sharpness_median": float(np.median(sharp)),
        "imu": imu_info, "gt_poses": gt_info,
        "camera_serial": src.camera_serial, "camera_firmware": src.camera_firmware,
        "source": src.meta(),
        "ingest_seconds": round(time.time() - t_start, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[ingest] done: {len(rows)} frames, {len(kf)} keyframes, imu={imu_info.get('samples', 0)} -> {out}")
    return meta
