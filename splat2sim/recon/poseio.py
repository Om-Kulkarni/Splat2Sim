"""Pose / frame file IO and rotation helpers (numpy only).

``poses/poses.csv``: ``index,timestamp_ns,tx,ty,tz,qx,qy,qz,qw``. Each row is a camera-to-world pose
(T_world_cam) with the OpenCV camera convention (x right, y down, z forward), metres, quaternion xyzw.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

POSES_HEADER = ["index", "timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]


def quat_to_R(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (x, y, z, w), w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s])
    q = q / np.linalg.norm(q)
    return -q if q[3] < 0 else q


def pose_matrix(t, q) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_R(q)
    T[:3, 3] = t
    return T


def write_poses_csv(path: Path, indices, ts_ns, Ts) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(POSES_HEADER)
        for i, t, T in zip(indices, ts_ns, Ts):
            q = R_to_quat(T[:3, :3])
            w.writerow([int(i), int(t), *(f"{v:.9g}" for v in T[:3, 3]), *(f"{v:.9g}" for v in q)])


def read_poses_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (indices[N] int, ts_ns[N] int64, T[N,4,4])."""
    idx, ts, Ts = [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            idx.append(int(r["index"]))
            ts.append(int(r["timestamp_ns"]))
            Ts.append(pose_matrix([float(r[k]) for k in ("tx", "ty", "tz")], [float(r[k]) for k in ("qx", "qy", "qz", "qw")]))
    return np.array(idx, int), np.array(ts, np.int64), np.array(Ts).reshape(-1, 4, 4)


def read_frames_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """ingest ``frames.csv`` -> (index, timestamp_ns, color relpaths, depth relpaths)."""
    idx, ts, col, dep = [], [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            idx.append(int(r["index"]))
            ts.append(int(r["timestamp_ns"]))
            col.append(r["color"])
            dep.append(r["depth"])
    return np.array(idx, int), np.array(ts, np.int64), col, dep


def read_gt_csv(path: Path, unit_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ingest ``gt_poses.csv`` (timestamp_ns, tx ty tz [source units], qx qy qz qw) -> (ts_ns, pos_m, quat_xyzw)."""
    rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return rows[:, 0].astype(np.int64), rows[:, 1:4] * unit_scale, rows[:, 4:8]


def read_tum_poses(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TUM-style ``t x y z qx qy qz qw [id]`` (``#`` comments) -> (t_s[N], T[N,4,4], ids)."""
    ts, Ts, ids = [], [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        v = line.split()
        ts.append(float(v[0]))
        Ts.append(pose_matrix([float(x) for x in v[1:4]], [float(x) for x in v[4:8]]))
        ids.append(int(v[8]) if len(v) > 8 else len(ids))
    return np.array(ts), np.array(Ts).reshape(-1, 4, 4), np.array(ids)
