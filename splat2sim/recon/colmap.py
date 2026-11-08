"""COLMAP *text* model writer (cameras.txt / images.txt / points3D.txt) for 3DGRUT.

Layout written by :func:`write_model`::

    <out>/images/<name>[, <stem>_mask.png]   (symlinks, supplied by the caller)
    <out>/sparse/0/{cameras,images,points3D}.txt

Poses are given camera-to-world in the OpenCV convention; COLMAP stores world-to-camera
(quaternion w x y z + translation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .poseio import R_to_quat, quat_to_R


def write_cameras_txt(path: Path, width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> None:
    Path(path).write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n"
    )


def write_images_txt(path: Path, names: list[str], T_wc: np.ndarray) -> None:
    lines = ["# Image list with two lines of data per image:\n",
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
             "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"]
    for i, (name, T) in enumerate(zip(names, T_wc)):
        Tcw = np.linalg.inv(T)
        qx, qy, qz, qw = R_to_quat(Tcw[:3, :3])
        t = Tcw[:3, 3]
        lines.append(f"{i + 1} {qw:.12g} {qx:.12g} {qy:.12g} {qz:.12g} {t[0]:.9g} {t[1]:.9g} {t[2]:.9g} 1 {name}\n\n")
    Path(path).write_text("".join(lines))


def write_points3d_txt(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i, (p, c) in enumerate(zip(xyz, rgb)):
            f.write(f"{i + 1} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[0])} {int(c[1])} {int(c[2])} 0.5\n")


def write_model(out: Path, size: tuple[int, int], K: tuple[float, float, float, float],
                names: list[str], T_wc: np.ndarray, xyz: np.ndarray, rgb: np.ndarray) -> Path:
    sp = Path(out) / "sparse" / "0"
    sp.mkdir(parents=True, exist_ok=True)
    write_cameras_txt(sp / "cameras.txt", size[0], size[1], *K)
    write_images_txt(sp / "images.txt", names, T_wc)
    write_points3d_txt(sp / "points3D.txt", xyz, rgb)
    return sp


def read_images_txt(path: Path) -> tuple[list[str], np.ndarray]:
    """Parse images.txt back into (names, T_wc[N,4,4]); used by tests and sanity checks."""
    names, Ts = [], []
    lines = [ln for ln in Path(path).read_text().splitlines() if not ln.startswith("#")]
    for ln in lines[::2]:  # two lines per image; the second (2D points) is blank here
        v = ln.split()
        qw, qx, qy, qz = (float(x) for x in v[1:5])
        Tcw = np.eye(4)
        Tcw[:3, :3] = quat_to_R([qx, qy, qz, qw])
        Tcw[:3, 3] = [float(x) for x in v[5:8]]
        Ts.append(np.linalg.inv(Tcw))
        names.append(v[9])
    return names, np.array(Ts).reshape(-1, 4, 4)
