"""``python -m splat2sim.ingest ingest [options]`` (container entrypoint; paths default from the compose env)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .pipeline import Options, run

try:  # per-run options file: /runs/<scene>/ingest.yaml (same keys as the long options, underscores)
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m splat2sim.ingest")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest", help="Capture -> frames/")
    i.add_argument("--capture", default=os.environ.get("R2S_CAPTURE", "/capture"), help="capture file/folder (default $R2S_CAPTURE)")
    i.add_argument("--out", help="output dir (default /runs/$R2S_SCENE/frames)")
    i.add_argument("--format", dest="fmt", choices=["ros2-mcap", "rosbag1", "tum", "frames-folder"], help="override autodetection")
    i.add_argument("--every", type=int, default=None, help="keyframe stride (default 1 = all)")
    i.add_argument("--min-sharpness", type=float, default=None, help="drop keyframes with Laplacian variance below this (default 0 = off)")
    i.add_argument("--max-frames", type=int, default=None)
    i.add_argument("--intrinsics", help="TUM only: yaml/json with fx fy cx cy [width height distortion] (else <capture>/intrinsics.yaml|json)")
    i.add_argument("--depth-factor", type=float, default=None, help="TUM only: depth PNG units per metre (default 1000 = mm)")
    i.add_argument("--depth-mode", choices=["auto", "aligned", "register"], default=None, help="MCAP: use aligned depth topic or register raw depth")
    i.add_argument("--no-preview", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    scene = os.environ.get("R2S_SCENE", "")
    out = Path(a.out) if a.out else Path("/runs") / scene / "frames"
    cfg: dict = {}
    run_cfg = out.parent / "ingest.yaml"
    if yaml and run_cfg.is_file():
        cfg = yaml.safe_load(run_cfg.read_text()) or {}
        print(f"[ingest] options from {run_cfg}: {cfg}")
    pick = lambda k, cli, default: cli if cli is not None else cfg.get(k, default)  # noqa: E731
    intr = pick("intrinsics", a.intrinsics, None)
    if intr and not Path(intr).is_absolute() and not Path(intr).exists():
        intr = str(out.parent / intr)  # relative paths resolve inside the run folder
    opt = Options(
        fmt=pick("format", a.fmt, None),
        every=int(pick("every", a.every, 1)),
        min_sharpness=float(pick("min_sharpness", a.min_sharpness, 0.0)),
        max_frames=int(pick("max_frames", a.max_frames, 0)),
        intrinsics_file=Path(intr) if intr else None,
        depth_factor=float(pick("depth_factor", a.depth_factor, 1000.0)),
        depth_mode=pick("depth_mode", a.depth_mode, "auto"),
        no_preview=a.no_preview,
    )
    try:
        run(Path(a.capture), out, opt)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0
