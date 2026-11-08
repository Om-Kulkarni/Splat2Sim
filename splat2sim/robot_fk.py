"""Franka FR3 forward kinematics (modified DH, numpy only) and link capsules for clearance volumes.

Frame: robot base (``fr3_link0``) = world origin of the Scene, z up. The FR3 shares the Panda's DH parameters.
"""

from __future__ import annotations

import numpy as np

# (a, d, alpha) of joints 1..7 (Craig convention: T = RotX(alpha) TransX(a) RotZ(theta) TransZ(d))
DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
      (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]
FLANGE_D = 0.107


def _dh(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1.0]])


def fk_frames(q) -> list[np.ndarray]:
    """4x4 poses of frames 1..7 and the flange (8 matrices) in the base frame."""
    T = np.eye(4)
    out = []
    for (a, d, al), th in zip(DH, q):
        T = T @ _dh(a, d, al, float(th))
        out.append(T.copy())
    F = T.copy()
    F[:3, 3] += F[:3, 2] * FLANGE_D
    out.append(F)
    return out


def arm_capsules(q, arm_radius: float = 0.10, gripper_radius: float = 0.10, gripper_length: float = 0.22,
                 coupling_z: float = 0.0) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Link capsules (p0, p1, radius) of the arm + gripper at joint vector ``q`` (radius includes the safety margin)."""
    fr = fk_frames(q)
    p = [np.zeros(3)] + [f[:3, 3] for f in fr]  # base, frame1..7, flange
    caps = [(p[0], p[1], arm_radius), (p[2], p[3], arm_radius), (p[3], p[4], arm_radius), (p[4], p[5], arm_radius),
            (p[5], p[7], arm_radius), (p[7], p[8], arm_radius)]
    z = fr[-1][:3, 2]
    caps.append((p[8], p[8] + z * (gripper_length + coupling_z), gripper_radius))
    return [(a, b, r) for a, b, r in caps if np.linalg.norm(b - a) > 1e-6]


def points_in_capsules(pts: np.ndarray, caps) -> np.ndarray:
    """Boolean mask of ``pts`` (N,3) inside any capsule."""
    inside = np.zeros(len(pts), bool)
    for a, b, r in caps:
        ab = b - a
        t = np.clip(((pts - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        inside |= np.linalg.norm(pts - (a + t[:, None] * ab), axis=1) < r
    return inside
