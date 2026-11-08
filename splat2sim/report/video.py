"""Encode report/flythrough/*.png into report/flythrough.mp4 (imageio-ffmpeg; the Isaac image has no ffmpeg backend)."""

from __future__ import annotations

from pathlib import Path


def encode(frames_dir: Path, out: Path, fps: int = 20) -> dict:
    files = sorted(frames_dir.glob("*.png"))
    if not files:
        return {"status": "skipped: no frames in report/flythrough"}
    try:
        import imageio.v2 as iio
    except ImportError as e:
        return {"status": f"skipped: imageio missing ({e})", "frames": len(files)}
    try:
        w = iio.get_writer(str(out), fps=fps, codec="libx264", quality=7, macro_block_size=2, pixelformat="yuv420p")
        for f in files:
            w.append_data(iio.imread(str(f)))
        w.close()
        return {"status": "ok", "file": out.name, "frames": len(files), "fps": fps, "bytes": out.stat().st_size}
    except Exception as e:  # no ffmpeg binary: leave a GIF
        from PIL import Image

        ims = [Image.open(f).convert("RGB").resize((480, 270)) for f in files[::3]]
        g = out.with_suffix(".gif")
        ims[0].save(g, save_all=True, append_images=ims[1:], duration=150, loop=0)
        return {"status": f"mp4 failed ({type(e).__name__}: {str(e)[:80]}); wrote GIF", "file": g.name, "frames": len(files)}
