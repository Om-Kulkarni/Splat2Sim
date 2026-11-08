"""Per-run stage options: ``/runs/<scene>/options.yaml`` (sections ``poses:``, ``splat:``, ...)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

POSES_DEFAULTS: dict[str, Any] = {
    "use_imu": "auto",         # auto | true | false. auto = use imu.csv only if its accel norm looks like gravity
    "every": "auto",           # stride over frames/keyframes.txt for SLAM; auto = target ~10 fps (30 fps -> 3)
    "max_frames": 0,           # 0 = all
    "train_max_images": 600,   # images exported to poses/colmap (evenly subsampled from the registered frames)
    "points_every": 6,         # use depth of every Nth exported image for the initial point cloud
    "max_points": 200000,
    "max_depth_m": 4.0,
    "mask_robot_is_white": False,  # masks/<frame>.png follow the mask stage: 255 = keep, 0 = robot (True = inverted)
    "ransac_max_depth_m": 3.0,
}
SPLAT_DEFAULTS: dict[str, Any] = {
    "iterations": 30000,
    "test_split_interval": 8,
    "num_workers": 4,
}
QUICK_ITERATIONS = 3000


def load_run_options(run_root: Path) -> dict[str, Any]:
    p = Path(run_root) / "options.yaml"
    if not p.is_file():
        return {}
    import yaml

    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: top level must be a mapping")
    return data


def stage_options(run_root: Path, section: str, defaults: dict[str, Any]) -> dict[str, Any]:
    opts = dict(defaults)
    opts.update((load_run_options(run_root).get(section) or {}))
    unknown = set(opts) - set(defaults)
    if unknown:
        raise ValueError(f"options.yaml [{section}]: unknown keys {sorted(unknown)}; valid: {sorted(defaults)}")
    return opts


def set_option(run_root: Path, dotted: str, value: Any) -> None:
    """``poses.use_imu`` -> options.yaml {poses: {use_imu: value}} (merged)."""
    import yaml

    section, _, key = dotted.partition(".")
    if not key:
        raise ValueError(f"--set expects section.key=value, got {dotted!r}")
    data = load_run_options(run_root)
    data.setdefault(section, {})[key] = value
    Path(run_root).mkdir(parents=True, exist_ok=True)
    (Path(run_root) / "options.yaml").write_text(yaml.safe_dump(data, sort_keys=True))
