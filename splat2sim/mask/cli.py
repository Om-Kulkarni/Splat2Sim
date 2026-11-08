"""``python -m splat2sim.mask mask [options]`` (container entrypoint of the sam3 service).

Options come from the CLI, else the lab config's ``mask`` section (``/runs/$R2S_SCENE/lab.json`` when present),
else the defaults below. Run folder: ``/runs/$R2S_SCENE`` (override with ``--run``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_PROMPTS = ["robot arm", "robotic gripper"]  # keep in sync with splat2sim.config.MaskCfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m splat2sim.mask")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mask", help="frames/ -> masks/ with SAM 3.1")
    m.add_argument("--run", help="run folder (default /runs/$R2S_SCENE)")
    m.add_argument("--out", help="output dir (default <run>/masks)")
    m.add_argument("--prompts", help="comma-separated text prompts (default: lab config mask.prompts)")
    m.add_argument("--dilate-px", type=int, default=None, help="dilation radius in px (default lab cfg, 15)")
    m.add_argument("--no-fill-holes", action="store_true")
    m.add_argument("--max-frames", type=int, default=0, help="only process the first N selected frames (0 = all)")
    m.add_argument("--start-frame", type=int, default=0, help="skip the first N selected frames (for testing)")
    m.add_argument("--all-frames", action="store_true", help="ignore keyframes.txt, mask every frame")
    m.add_argument("--chunk-size", type=int, default=300, help="frames per SAM session; tracking restarts (text re-prompt) at each chunk")
    m.add_argument("--threshold", type=float, default=0.5, help="mask probability threshold")
    m.add_argument("--model", choices=["sam3.1", "sam3"], default="sam3.1", help="sam3.1 -> falls back to sam3 if its weights cannot be fetched")
    m.add_argument("--offload-video-to-cpu", action="store_true", help="keep frame tensors on CPU (less VRAM, slower)")
    m.add_argument("--probe-step", type=int, default=10, help="robot not found on a chunk's first frame: probe every Nth frame until it is")
    m.add_argument("--grounding-batch", type=int, default=4, help="frames per detector batch (lower = less VRAM)")
    m.add_argument("--n-preview", type=int, default=8)
    return p


def lab_mask_cfg(run: Path) -> dict:
    p = run / "lab.json"
    if p.is_file():
        return json.loads(p.read_text()).get("mask", {}) or {}
    return {}


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    run = Path(a.run) if a.run else Path("/runs") / os.environ.get("R2S_SCENE", "")
    cfg = lab_mask_cfg(run)
    prompts = [s.strip() for s in a.prompts.split(",") if s.strip()] if a.prompts else cfg.get("prompts", DEFAULT_PROMPTS)
    from .pipeline import Options, run_mask  # lazy: pulls torch

    opt = Options(
        prompts=list(prompts),
        dilate_px=a.dilate_px if a.dilate_px is not None else int(cfg.get("dilate_px", 15)),
        fill_holes=False if a.no_fill_holes else bool(cfg.get("fill_holes", True)),
        max_frames=a.max_frames, start_frame=a.start_frame, all_frames=a.all_frames, chunk_size=a.chunk_size, threshold=a.threshold,
        model=a.model, offload_video_to_cpu=a.offload_video_to_cpu, n_preview=a.n_preview, grounding_batch=a.grounding_batch, probe_step=a.probe_step,
    )
    try:
        run_mask(run, Path(a.out) if a.out else run / "masks", opt)
    except (ValueError, FileNotFoundError) as e:
        print(f"[mask] error: {e}", file=sys.stderr)
        return 2
    return 0
