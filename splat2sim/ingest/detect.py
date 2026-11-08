"""Capture format autodetection."""

from __future__ import annotations

from pathlib import Path

MCAP = "ros2-mcap"
BAG = "rosbag1"
TUM = "tum"
FRAMES = "frames-folder"


class UnknownFormat(ValueError):
    pass


def detect_format(path: Path) -> str:
    path = Path(path)
    if path.is_file():
        with open(path, "rb") as f:
            head = f.read(16)
        if head.startswith(b"\x89MCAP"):
            return MCAP
        if head.startswith(b"#ROSBAG V"):
            return BAG
        if path.suffix == ".mcap":
            return MCAP
        if path.suffix == ".bag":
            return BAG
        if path.suffix == ".db3":
            raise UnknownFormat(f"{path}: rosbag2 sqlite3 (.db3) is not supported; record with `-s mcap`")
        raise UnknownFormat(f"{path}: not an MCAP, ROS1/librealsense bag, or TUM folder")
    if path.is_dir():
        if (path / "frames.csv").is_file() and (path / "intrinsics.json").is_file() and (path / "color").is_dir():
            return FRAMES
        if (path / "associations.txt").is_file() or ((path / "rgb").is_dir() and (path / "depth").is_dir()):
            return TUM
        mcaps = sorted(path.glob("*.mcap"))
        if mcaps:  # ros2 bag directory (metadata.yaml + *.mcap)
            return MCAP
        bags = sorted(path.glob("*.bag"))
        if len(bags) == 1:
            return BAG
        raise UnknownFormat(f"{path}: directory has no rgb/ + depth/ (TUM), *.mcap or *.bag")
    raise UnknownFormat(f"{path}: not found")


def resolve_file(path: Path, fmt: str) -> Path:
    """For directory captures of file formats, return the single file."""
    path = Path(path)
    if path.is_file():
        return path
    pat = "*.mcap" if fmt == MCAP else "*.bag"
    files = sorted(path.glob(pat))
    if len(files) != 1:
        raise UnknownFormat(f"{path}: expected exactly one {pat}, found {len(files)}")
    return files[0]
