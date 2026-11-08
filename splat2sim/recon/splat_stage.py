"""``splat`` stage (3dgrut container): poses/colmap -> splat/splat.usdz + splat/metrics.json.

Stdlib only (the 3dgrut venv has no extra deps for us). Trains 3DGUT with scene normalisation off and exports a
ParticleField USDZ without the normalising transform, so the splat stays in the recon frame.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .options import SPLAT_DEFAULTS, stage_options

GRUT = Path("/opt/3dgrut")


def train_command(colmap: Path, out_dir: Path, usdz: Path, o: dict) -> list[str]:
    it = int(o["iterations"])
    return [
        "python", "train.py", "--config-name", "apps/colmap_3dgut.yaml",
        f"path={colmap}", f"out_dir={out_dir}", "experiment_name=train", f"n_iterations={it}",
        "use_wandb=false", "with_gui=false", "test_last=true", "val_frequency=999999",
        "dataset.normalize_world_space=false", f"dataset.test_split_interval={int(o['test_split_interval'])}",
        f"num_workers={int(o['num_workers'])}",
        "export_usd.enabled=true", "export_usd.apply_normalizing_transform=false", f"export_usd.path={usdz}",
        f"checkpoint.iterations=[{it}]",
    ]


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def parse_metrics(log_text: str) -> dict:
    """Pull final test metrics out of the training log (3DGRUT prints a table / lines with PSNR, SSIM, LPIPS)."""
    text = _ANSI.sub("", log_text)
    out: dict = {}
    for key, pat in (("psnr", r"psnr"), ("ssim", r"ssim"), ("lpips", r"lpips")):
        vals = re.findall(rf"(?i)\b(?:mean\s+)?{pat}\b[^0-9\-\n]{{0,12}}(-?\d+\.\d+)", text)
        if vals:
            out[key] = float(vals[-1])
    m = re.findall(r"(?i)number of (?:gaussians|particles)\D{0,10}(\d+)", text)
    if not m:  # densification log lines: "Density-pruned 293 / 925829 (0.03%) gaussians"
        m = re.findall(r"(?:pruned|Cloned|Splitted)\s+\d+\s*/\s*(\d+)\s*\(", text)
    if m:
        out["num_gaussians"] = int(m[-1])
    return out


def read_test_metrics(train_dir: Path) -> dict:
    """Held-out metrics from 3DGRUT's own ``metrics.json`` (mean over the test split). Falls back to log parsing."""
    cands = sorted(train_dir.glob("**/metrics.json"))
    if not cands:
        return {}
    m = json.loads(cands[-1].read_text())
    return {k: float(m[f"mean_{k}"]) for k in ("psnr", "ssim", "lpips") if f"mean_{k}" in m}


class VramPoller(threading.Thread):
    """Whole-GPU memory.used sampled via nvidia-smi (shared GPU: also reports the baseline before training)."""

    def __init__(self, period: float = 2.0):
        super().__init__(daemon=True)
        self.period, self.samples, self._stop_evt = period, [], threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout.split()
                self.samples.append(int(out[0]))
            except Exception:
                pass
            self._stop_evt.wait(self.period)

    def stop(self) -> dict:
        self._stop_evt.set()
        self.join(timeout=15)
        if not self.samples:
            return {}
        return {"gpu_mem_peak_mib": max(self.samples), "gpu_mem_baseline_mib": self.samples[0]}


def save_heldout_previews(colmap: Path, train_dir: Path, out: Path, interval: int, n: int = 4) -> list[dict]:
    """Side-by-side [render | ground truth] PNGs (``splat/heldout/``) for evenly spread held-out views.

    3DGRUT renders the test split in order; test view k is image 8k of ``images.txt`` (index % interval == 0).
    """
    import numpy as np
    from PIL import Image

    if interval <= 0:
        return []
    names = [ln.split()[-1] for ln in (colmap / "sparse/0/images.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#") and ln.split()[-1].lower().endswith((".png", ".jpg"))]
    renders = sorted(train_dir.glob("**/ours_*/renders/*.png"))
    if not renders:
        return []
    out.mkdir(exist_ok=True)
    rows = []
    for k in sorted({int(round(x)) for x in np.linspace(0, len(renders) - 1, n)}):
        gt_name = names[k * interval]
        r = np.asarray(Image.open(renders[k]).convert("RGB"))
        g = np.asarray(Image.open(colmap / "images" / gt_name).convert("RGB"))
        mse = float(((r.astype(np.float64) - g) ** 2).mean()) if r.shape == g.shape else float("nan")
        psnr = float(10 * np.log10(255.0 ** 2 / mse)) if mse > 0 else float("inf")
        Image.fromarray(np.concatenate([r, g], axis=1)).save(out / f"heldout_{k:03d}.png")
        rows.append({"file": f"heldout_{k:03d}.png", "frame": gt_name, "psnr": round(psnr, 2), "layout": "render | ground truth"})
    return rows


def run(scene_root: Path) -> dict:
    o = stage_options(scene_root, "splat", SPLAT_DEFAULTS)
    colmap = scene_root / "poses" / "colmap"
    splat = scene_root / "splat"
    if not (colmap / "sparse" / "0" / "images.txt").is_file():
        raise FileNotFoundError(f"{colmap} missing; run the poses stage first")
    for p in splat.glob("*"):
        if p.name.startswith(".done"):
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    splat.mkdir(exist_ok=True)
    out_dir = splat / "train"
    usdz = splat / "splat.usdz"
    cmd = train_command(colmap, out_dir, usdz, o)
    print("[splat]", " ".join(cmd), flush=True)
    t0 = time.time()
    poller = VramPoller()
    poller.start()
    log = splat / "train.log"
    with open(log, "w") as lf:
        rc = subprocess.run(cmd, cwd=GRUT, stdout=lf, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    vram = poller.stop()
    text = log.read_text(errors="replace")
    if rc != 0 or not usdz.is_file():
        print(text[-3000:])
        raise RuntimeError(f"3dgrut training failed (exit {rc}); see {log}")
    metrics = {"iterations": int(o["iterations"]), "test_split_interval": int(o["test_split_interval"]),
               "wall_seconds": round(wall, 1), "usdz_bytes": usdz.stat().st_size, **{k: v for k, v in parse_metrics(text).items() if k == 'num_gaussians'}, **read_test_metrics(out_dir),
               **vram}
    try:
        metrics["heldout_previews"] = save_heldout_previews(colmap, out_dir, splat / "heldout", int(o["test_split_interval"]))
    except Exception as e:  # previews are nice-to-have
        metrics["heldout_previews_error"] = str(e)
    (splat / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[splat] done: {metrics}", flush=True)
    return metrics
