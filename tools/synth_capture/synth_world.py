"""World constants, procedural textures and the handheld camera path of the synthetic Capture (numpy/cv2/scipy only)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

FLOOR_Z = -0.9
PEDESTAL = {"center": (0.0, 0.0, -0.45), "size": (0.30, 0.30, 0.90)}  # top at z = 0 (robot mounting plane)
TABLE = {"center": (0.75, 0.0, -0.22), "size": (1.0, 0.8, 0.04), "top_z": -0.20}
ROOM = {"x": (-3.0, 3.4), "y": (-3.0, 3.2), "height_z": 2.0}
BOARD_SHEET_MARGIN_M = 0.011  # white paper margin around the 7x5 pattern
BOARD_PX_PER_M = 5000


def clutter_specs() -> list[dict]:
    tt = TABLE["top_z"]
    return [
        {"name": "book", "center": (1.10, -0.28, tt + 0.02), "size": (0.22, 0.16, 0.04), "tex": "poster", "on_table": True},
        {"name": "box", "center": (1.12, 0.25, tt + 0.07), "size": (0.12, 0.12, 0.14), "tex": "poster", "on_table": True},
        {"name": "tool", "center": (0.95, 0.02, tt + 0.025), "size": (0.20, 0.06, 0.05), "tex": "metal", "on_table": True},
        {"name": "cabinet", "center": (-1.6, -1.9, FLOOR_Z + 0.85), "size": (0.6, 0.5, 1.7), "tex": "wall", "on_table": False},
        {"name": "shelf", "center": (2.3, 1.6, FLOOR_Z + 0.5), "size": (0.8, 0.4, 1.0), "tex": "poster", "on_table": False},
        {"name": "crate", "center": (1.9, -1.7, FLOOR_Z + 0.25), "size": (0.5, 0.5, 0.5), "tex": "wood", "on_table": False},
        {"name": "rack", "center": (-1.3, 1.8, FLOOR_Z + 0.6), "size": (0.5, 0.7, 1.2), "tex": "metal", "on_table": False},
    ]


def _noise(rng, n, octaves=(4, 8, 16, 32, 64)):
    acc = np.zeros((n, n), np.float32)
    for k, o in enumerate(octaves):
        g = cv2.resize(rng.random((o, o)).astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
        acc += g / (1.6 ** k)
    acc -= acc.min()
    return acc / acc.max()


def make_texture(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """RGB uint8 (n,n). Strongly textured on purpose (feature-rich for RGB-D SLAM)."""
    base = {"floor": (0.45, 0.43, 0.40), "wall": (0.75, 0.72, 0.66), "wood": (0.55, 0.38, 0.22), "metal": (0.55, 0.57, 0.60),
            "poster": (0.8, 0.8, 0.8)}.get(kind, (0.6, 0.6, 0.6))
    nz = _noise(rng, n)
    img = np.stack([nz * 0.7 + 0.3] * 3, -1) * np.array(base, np.float32)
    img *= (0.8 + 0.4 * _noise(rng, n, (2, 3)))[..., None]
    if kind == "floor":  # tiles with grout lines + stains
        for i in range(0, n, n // 4):
            img[i : i + 3] *= 0.4
            img[:, i : i + 3] *= 0.4
        for _ in range(25):
            c = (int(rng.integers(0, n)), int(rng.integers(0, n)))
            cv2.circle(img, c, int(rng.integers(8, 50)), tuple(float(x) for x in rng.random(3) * 0.5), -1)
    elif kind in ("wall", "poster"):
        for _ in range(40 if kind == "poster" else 14):
            x0, y0 = int(rng.integers(0, n - 60)), int(rng.integers(0, n - 60))
            w, h = int(rng.integers(40, n // 4)), int(rng.integers(40, n // 4))
            col = tuple(float(x) for x in rng.random(3))
            cv2.rectangle(img, (x0, y0), (min(x0 + w, n - 1), min(y0 + h, n - 1)), col, -1)
            cv2.rectangle(img, (x0 + 6, y0 + 6), (min(x0 + w - 6, n - 1), min(y0 + h - 6, n - 1)), tuple(float(x) for x in rng.random(3)), 3)
            cv2.putText(img, "".join(rng.choice(list("ABCDEFGHKLMNPRSTXZ0123456789"), 4)), (x0 + 10, y0 + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0.1, 0.1, 0.1), 2)
    elif kind == "wood":
        stripes = 0.85 + 0.15 * np.sin(np.arange(n)[None, :] / 6.0 + 10 * gaussian_filter(rng.random((n, n)).astype(np.float32), 20))
        img *= stripes[..., None]
    elif kind == "metal":
        img += rng.normal(0, 0.03, img.shape).astype(np.float32)
        for _ in range(60):
            cv2.circle(img, (int(rng.integers(0, n)), int(rng.integers(0, n))), int(rng.integers(3, 12)), (0.25, 0.25, 0.28), -1)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def board_texture(spec):
    """(BGR/gray image, (sheet_w_m, sheet_h_m)) of the printed sheet; pattern origin (outer corner) at sheet offset = margin."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    board = cv2.aruco.CharucoBoard((spec.squares_x, spec.squares_y), spec.square_length_m, spec.marker_length_m, d)
    pw = int(round(spec.squares_x * spec.square_length_m * BOARD_PX_PER_M))
    ph = int(round(spec.squares_y * spec.square_length_m * BOARD_PX_PER_M))
    m = int(round(BOARD_SHEET_MARGIN_M * BOARD_PX_PER_M))
    img = board.generateImage((pw + 2 * m, ph + 2 * m), marginSize=m, borderBits=1)
    img = (img.astype(np.float32) * 0.92 + 8).astype(np.uint8)  # matt paper
    return img, ((pw + 2 * m) / BOARD_PX_PER_M, (ph + 2 * m) / BOARD_PX_PER_M)


def _smooth(x, sigma):
    return gaussian_filter1d(x, sigma, axis=0, mode="nearest")


def _lookat(eye, tgt, down_hint):
    f = tgt - eye
    f = f / np.linalg.norm(f)
    x = np.cross(down_hint, f)
    x /= np.linalg.norm(x)
    y = np.cross(f, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, f], 1)
    T[:3, 3] = eye
    return T


def camera_path(n: int, rng: np.random.Generator, board_center_world, fps: float = 30.0) -> dict:
    """Handheld-like walk. Phases (fractions of n): A spiral orbit knee->head, B close pass around the board, C dwell at the
    front-camera viewpoint, D arc back to the start (loop closure)."""
    c0 = np.array([0.45, 0.0, -0.15])
    b = np.asarray(board_center_world, float)
    nA, nB, nC = int(0.38 * n), int(0.34 * n), int(0.08 * n)
    nD = n - nA - nB - nC
    eyes, tgts, names = [], [], []
    # A: az0 -> az0 - 300 deg, radius 1.7 +- 0.25, height -0.4 -> 0.85
    az0 = np.radians(200.0)
    s = np.linspace(0, 1, nA)
    az = az0 - np.radians(300.0) * s
    r = 1.7 + 0.25 * np.sin(2 * np.pi * s * 1.5)
    z = -0.4 + 1.25 * s
    eyeA = np.stack([c0[0] + r * np.cos(az), c0[1] + r * np.sin(az), z], 1)
    tgA = c0 + np.stack([0.12 * np.sin(6 * s), 0.12 * np.cos(5 * s), 0.1 * np.sin(4 * s)], 1)
    # B: around the board, az from -70 to 100 deg (x > board side), distance 0.85 -> 0.6, elevation 40 -> 65
    s = np.linspace(0, 1, nB)
    azb = np.radians(-70 + 170 * s)
    d = 0.85 - 0.25 * s
    el = np.radians(40 + 25 * np.sin(np.pi * s))
    eyeB = b + np.stack([d * np.cos(el) * np.cos(azb), d * np.cos(el) * np.sin(azb), d * np.sin(el)], 1)
    tgB = b + np.stack([0.05 * np.sin(8 * s), 0.05 * np.cos(7 * s), 0 * s], 1)
    # C: dwell at the front policy-camera viewpoint, looking at the base
    eyeC = np.tile([1.0, 0.0, 0.8], (nC, 1)) + 0.0
    tgC = np.tile([0.0, 0.0, 0.1], (nC, 1))
    # D: arc about c0 from az 0 (dwell) to az0 - 360 (= start), radius 0.55 -> 1.7, height 0.8 -> -0.4
    s = np.linspace(0, 1, nD)
    azd = 0.0 - np.radians(160.0) * s
    rd = 0.55 + 1.15 * s
    eyeD = np.stack([c0[0] + rd * np.cos(azd), c0[1] + rd * np.sin(azd), 0.8 - 1.2 * s], 1)
    tgD = c0 + np.stack([0 * s, 0 * s, 0.05 * np.sin(3 * s)], 1)
    eye = np.vstack([eyeA, eyeB, eyeC, eyeD])
    tgt = np.vstack([tgA, tgB, tgC, tgD])
    # phase-boundary smoothing + handheld sway
    eye, tgt = _smooth(eye, 10), _smooth(tgt, 10)
    sway = lambda sig, amp: _smooth(rng.normal(0, 1, (n, 3)), sig) * amp * np.sqrt(sig)  # noqa: E731
    eye = eye + sway(12, 0.006)
    dist = np.linalg.norm(tgt - eye, axis=1, keepdims=True)
    tgt = tgt + sway(8, 0.004) * dist
    roll = np.radians(_smooth(rng.normal(0, 1, n), 15) * 2.5 * np.sqrt(15))
    Ts = np.zeros((n, 4, 4))
    for k in range(n):
        hint = np.array([np.sin(roll[k]), 0.0, -np.cos(roll[k])])
        f = tgt[k] - eye[k]
        if abs(f[2]) / np.linalg.norm(f) > 0.995:
            hint = np.array([1.0, 0.0, 0.0])
        Ts[k] = _lookat(eye[k], tgt[k], hint)
    speed = np.linalg.norm(np.diff(eye, axis=0), axis=1) * fps
    phases = [{"name": nme, "first": a, "last": z_ - 1} for nme, a, z_ in
              (("orbit", 0, nA), ("board_pass", nA, nA + nB), ("dwell_front", nA + nB, nA + nB + nC), ("return", nA + nB + nC, n))]
    return {"T_world_cv": Ts, "phases": phases, "speed_mean_mps": float(speed.mean()), "speed_max_mps": float(speed.max())}
