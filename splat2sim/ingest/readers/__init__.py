from __future__ import annotations

from pathlib import Path

from ..detect import BAG, FRAMES, MCAP, TUM, detect_format, resolve_file
from .base import Frame, Source


def open_source(path: Path, fmt: str | None = None, intrinsics_file: Path | None = None,
                depth_factor: float = 1000.0, depth_mode: str = "auto") -> Source:
    fmt = fmt or detect_format(path)
    if fmt == MCAP:
        from .mcap_reader import McapSource

        return McapSource(resolve_file(path, fmt), depth_mode=depth_mode)
    if fmt == BAG:
        from .bag_reader import BagSource

        return BagSource(resolve_file(path, fmt))
    if fmt == TUM:
        from .tum_reader import TumSource

        return TumSource(path, intrinsics_file, depth_factor)
    if fmt == FRAMES:
        from .frames_reader import FramesFolderSource

        return FramesFolderSource(path)
    raise ValueError(f"unknown format {fmt!r}")


__all__ = ["open_source", "Frame", "Source"]
