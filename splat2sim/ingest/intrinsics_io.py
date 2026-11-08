"""Load colour-camera intrinsics from a sidecar yaml/json (used for TUM folders, which carry none)."""

from __future__ import annotations

import json
from pathlib import Path

from .geometry import Intrinsics

SIDECAR_NAMES = ("intrinsics.yaml", "intrinsics.yml", "intrinsics.json")


def _find(d: dict, *keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _data(x):
    if isinstance(x, dict):
        return x.get("data", x.get("values"))
    return x


def parse_intrinsics(d: dict, default_size: tuple[int, int] | None = None, source: str = "") -> Intrinsics:
    """Accepts flat {fx,fy,cx,cy,width,height,[distortion|D|dist_coeffs]}, ROS camera_info style {K,D,width,height},
    OpenCV {camera_matrix:{data},distortion_coefficients:{data},image_width,image_height}, or a one-level nesting
    ({color:{...}} / {camera:{...}} / {rgb:{...}})."""
    for nest in ("color", "rgb", "camera", "intrinsics"):
        if nest in d and isinstance(d[nest], dict) and not {"fx", "K", "camera_matrix"} & d.keys():
            return parse_intrinsics(d[nest], default_size, source)
    w = _find(d, "width", "image_width", "w")
    h = _find(d, "height", "image_height", "h")
    if w is None or h is None:
        if default_size is None:
            raise ValueError("intrinsics file has no width/height and the image size is unknown")
        w, h = default_size
    D = _data(_find(d, "distortion", "D", "d", "dist_coeffs", "distortion_coefficients", "distCoeffs")) or []
    model = _find(d, "distortion_model", "model") or "plumb_bob"
    if all(k in d for k in ("fx", "fy", "cx", "cy")):
        return Intrinsics(float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]), int(w), int(h),
                          str(model), [float(x) for x in D] or [0.0] * 5, source)
    K = _data(_find(d, "K", "k", "camera_matrix", "camera_intrinsics"))
    if K is not None and len(K) == 9:
        return Intrinsics.from_K(K, w, h, str(model), D, source)
    raise ValueError("intrinsics file: expected fx/fy/cx/cy or a 3x3 K / camera_matrix")


def load_intrinsics(path: Path, default_size: tuple[int, int] | None = None) -> Intrinsics:
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".json":
        d = json.loads(text)
    else:
        import yaml

        # OpenCV FileStorage yaml starts with "%YAML:1.0" and uses !!opencv-matrix; strip those.
        lines = [ln for ln in text.splitlines() if not ln.startswith("%YAML") and ln.strip() != "---"]
        d = yaml.safe_load("\n".join(lines).replace("!!opencv-matrix", ""))
    return parse_intrinsics(d, default_size, source=f"sidecar {path.name}")


def find_sidecar(capture_dir: Path) -> Path | None:
    for n in SIDECAR_NAMES:
        p = capture_dir / n
        if p.is_file():
            return p
    return None
