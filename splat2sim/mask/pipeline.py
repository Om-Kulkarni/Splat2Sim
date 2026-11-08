"""frames/ -> masks/ with the SAM 3.1 (Object Multiplex) video predictor and text prompts.

Frames are fed to SAM as a temp folder of symlinks (``000000.png`` ... in selected order, so frame ids are
positions in the selection). The selection is ``frames/keyframes.txt`` when present (else every frame), optionally
cut to ``--max-frames``. The selection is processed in chunks (``--chunk-size``): SAM keeps per-frame features in
memory (1008x1008 fp16 per frame on the GPU), so chunking bounds VRAM; each chunk starts a fresh session and
re-prompts on its first frame (text prompts re-detect the robot, so continuity is not lost).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import postprocess as pp

# TODO(click fallback): SAM 2.1-style point/box prompt path (Sam3 `add_prompt` already accepts points/boxes) for
# Captures where the text prompt misses the robot; out of scope for now.

MODEL_REPOS = {"sam3.1": ("facebook/sam3.1", "sam3.1_multiplex.pt"), "sam3": ("facebook/sam3", "sam3.pt")}


@dataclass
class Options:
    prompts: list[str] = field(default_factory=lambda: ["robot arm", "robotic gripper"])
    dilate_px: int = 15
    fill_holes: bool = True
    max_frames: int = 0
    start_frame: int = 0  # skip the first N selected frames (testing)
    all_frames: bool = False
    chunk_size: int = 300
    threshold: float = 0.5
    model: str = "sam3.1"
    offload_video_to_cpu: bool = False
    n_preview: int = 8
    grounding_batch: int = 4  # frames per detector batch (VRAM vs speed)
    probe_step: int = 10  # when the robot is not detected on a chunk's first frame, probe every Nth frame until it is


def select_frames(frames_dir: Path, opt: Options) -> list[Path]:
    colors = sorted((frames_dir / "color").glob("*.png"))
    if not colors:
        raise FileNotFoundError(f"no frames in {frames_dir / 'color'} (run ingest first)")
    by_idx = {int(p.stem): p for p in colors}
    kf = frames_dir / "keyframes.txt"
    if kf.is_file() and not opt.all_frames:
        idx = [int(t) for t in kf.read_text().split()]
        sel = [by_idx[i] for i in idx if i in by_idx]
        if sel:
            colors = sel
    colors = colors[opt.start_frame :]
    if opt.max_frames:
        colors = colors[: opt.max_frames]
    return colors


def _load_predictor(opt: Options):
    import torch
    from huggingface_hub import hf_hub_download
    from sam3.model_builder import build_sam3_predictor

    t0 = time.time()
    tried: list[str] = []
    for ver in ([opt.model] + (["sam3"] if opt.model == "sam3.1" else [])):
        repo, fname = MODEL_REPOS[ver]
        try:
            ckpt = hf_hub_download(repo_id=repo, filename=fname)
        except Exception as e:  # gated / no token / offline
            tried.append(f"{repo}: {type(e).__name__}")
            print(f"[mask] cannot fetch {repo}/{fname} ({type(e).__name__}); is HF_TOKEN set and access granted?", flush=True)
            continue
        # FlashAttention-3 is not installed (and not built for sm_120): use the SDPA path.
        pred = build_sam3_predictor(checkpoint_path=ckpt, version=ver, use_fa3=False)
        _accept_only_known_init_kwargs(pred.model)
        if ver == "sam3.1":  # builder hardcodes 16 / 16, which needs >20 GiB of transient activations at 848x480
            pred.model.batched_grounding_batch_size = opt.grounding_batch
            pred.model.postprocess_batch_size = opt.grounding_batch
        return pred, ver, ckpt, time.time() - t0
    raise RuntimeError("could not download any SAM 3 checkpoint: " + "; ".join(tried))


def _accept_only_known_init_kwargs(model) -> None:
    """sam3 @7f473cf: the base predictor's start_session passes ``offload_state_to_cpu`` (and maybe other kwargs) to
    ``model.init_state``, which the SAM 3.1 multiplex model does not accept (TypeError). Drop unknown kwargs."""
    import inspect

    orig = model.init_state
    known = set(inspect.signature(orig).parameters)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(orig).parameters.values()):
        return

    def init_state(*a, **kw):
        return orig(*a, **{k: v for k, v in kw.items() if k in known})

    model.init_state = init_state


def _any_mask(outputs) -> bool:
    try:
        m = np.asarray(outputs["out_binary_masks"])
    except (KeyError, TypeError):
        return False
    return bool(m.size and m.reshape(-1, *m.shape[-2:]).any())


def _propagate(pred, sid, thr, start: int = 0) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for r in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid, output_prob_thresh=thr,
                                             start_frame_index=start)):
        o = r["outputs"]
        m = np.asarray(o["out_binary_masks"])
        if m.size:
            out[int(r["frame_index"])] = m.reshape(-1, *m.shape[-2:]).any(axis=0)
    return out


def prompt_and_propagate(pred, sid, text: str, n_frames: int, thr: float, step: int) -> tuple[dict[int, np.ndarray], int | None]:
    """Text-prompt the first frame; when nothing is detected there (the robot is out of view at the start of a chunk) probe
    every ``step``-th frame until it is, then propagate both ways from that frame. SAM raises ("No points are provided")
    when propagating without a detection, so a chunk without any detection returns ({}, None)."""
    for f0 in range(0, max(n_frames, 1), max(step, 1)):
        r = pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=f0, text=text, output_prob_thresh=thr))
        if _any_mask(r.get("outputs")):
            try:
                return _propagate(pred, sid, thr, start=f0), f0
            except RuntimeError as e:  # tracker found nothing to follow
                print(f"[mask] propagate from frame {f0} failed ({e}); continuing", flush=True)
        pred.handle_request(dict(type="reset_session", session_id=sid))
    return {}, None


def run_mask(run: Path, out: Path, opt: Options) -> dict:
    import torch

    frames_dir = run / "frames"
    sel = select_frames(frames_dir, opt)
    h, w = cv2.imread(str(sel[0]), cv2.IMREAD_COLOR).shape[:2]
    print(f"[mask] {len(sel)} frames ({w}x{h}), prompts={opt.prompts}, dilate={opt.dilate_px}px, chunk={opt.chunk_size}", flush=True)

    for sub in (out, out / "preview"):
        sub.mkdir(parents=True, exist_ok=True)
    for f in list(out.glob("*.png")) + list((out / "preview").glob("*.png")) + [out / "meta.json", out / "summary.json"]:
        f.unlink(missing_ok=True)  # we own masks/*.png, preview/ and meta.json

    pred, version, ckpt, t_load = _load_predictor(opt)
    print(f"[mask] {version} loaded in {t_load:.1f}s ({Path(ckpt).name})", flush=True)

    prev_ids = set(np.linspace(0, len(sel) - 1, min(opt.n_preview, len(sel))).round().astype(int).tolist())
    fractions: dict[str, float] = {}
    raw_fractions: dict[str, float] = {}
    chunk_times: list[dict] = []
    tmp = Path(tempfile.mkdtemp(prefix="r2s_mask_"))
    try:
        for c0 in range(0, len(sel), opt.chunk_size):
            chunk = sel[c0 : c0 + opt.chunk_size]
            cdir = tmp / f"c{c0}"
            cdir.mkdir()
            for i, p in enumerate(chunk):
                os.symlink(p.resolve(), cdir / f"{i:06d}.png")
            t0 = time.time()
            sid = pred.handle_request(dict(type="start_session", resource_path=str(cdir),
                                           offload_video_to_cpu=opt.offload_video_to_cpu))["session_id"]
            t_start = time.time() - t0
            union: dict[int, np.ndarray] = {}
            per_prompt = {}
            t1 = time.time()
            for k, text in enumerate(opt.prompts):
                if k:
                    pred.handle_request(dict(type="reset_session", session_id=sid))
                res, first = prompt_and_propagate(pred, sid, text, len(chunk), opt.threshold, opt.probe_step)
                per_prompt[text] = sum(int(m.any()) for m in res.values())
                if first:
                    per_prompt[f"{text} (first detection at chunk frame)"] = first
                for fi, m in res.items():
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    union[fi] = union[fi] | m if fi in union else m
            pred.handle_request(dict(type="close_session", session_id=sid))
            torch.cuda.synchronize()
            t_prop = time.time() - t1
            for i, p in enumerate(chunk):
                raw = union.get(i, np.zeros((h, w), bool))
                fin = pp.finalize(raw, opt.dilate_px, opt.fill_holes)
                cv2.imwrite(str(out / f"{p.stem}.png"), pp.to_png_mask(fin))
                fractions[p.stem] = round(float(fin.mean()), 6)
                raw_fractions[p.stem] = round(float(raw.mean()), 6)
                if c0 + i in prev_ids:
                    cv2.imwrite(str(out / "preview" / f"{p.stem}.png"), pp.overlay(cv2.imread(str(p)), fin))
            chunk_times.append({"first_frame": chunk[0].stem, "frames": len(chunk), "session_start_s": round(t_start, 2),
                                "propagate_s": round(t_prop, 2), "s_per_frame_per_prompt": round(t_prop / len(chunk) / len(opt.prompts), 4),
                                "frames_with_detection_per_prompt": per_prompt})
            print(f"[mask] chunk @{chunk[0].stem}: {len(chunk)} frames, {t_prop:.1f}s, detections/prompt {per_prompt}", flush=True)
            del union
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    total = sum(c["propagate_s"] + c["session_start_s"] for c in chunk_times)
    meta = {
        "convention": "masks/NNNNNN.png: uint8, 255 = keep (background), 0 = robot (ignore)",
        "model": version, "checkpoint": Path(ckpt).name, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "prompts": opt.prompts, "dilate_px": opt.dilate_px, "fill_holes": opt.fill_holes, "threshold": opt.threshold,
        "chunk_size": opt.chunk_size, "frames": len(sel), "image_size": [w, h],
        "used_keyframes": not opt.all_frames and (frames_dir / "keyframes.txt").is_file(),
        "masked_fraction": fractions, "masked_fraction_before_postprocess": raw_fractions,
        "masked_fraction_mean": round(float(np.mean(list(fractions.values()))), 6),
        "frames_with_any_mask": sum(1 for v in fractions.values() if v > 0),
        "timings": {"model_load_s": round(t_load, 2), "total_inference_s": round(total, 2),
                    "s_per_frame": round(total / len(sel), 4), "chunks": chunk_times},
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    fr = np.array(list(fractions.values()))
    summary = {
        "frames": len(sel), "image_size": [w, h], "prompts": opt.prompts, "model": version, "dilate_px": opt.dilate_px,
        "fill_holes": opt.fill_holes,
        "masked_fraction": {"min": round(float(fr.min()), 6), "mean": round(float(fr.mean()), 6),
                            "median": round(float(np.median(fr)), 6), "p95": round(float(np.percentile(fr, 95)), 6),
                            "max": round(float(fr.max()), 6)},
        "frames_with_any_mask": meta["frames_with_any_mask"],
        "timing": {"model_load_s": meta["timings"]["model_load_s"], "inference_s": meta["timings"]["total_inference_s"],
                   "frames_per_s": round(len(sel) / max(total, 1e-9), 2)},
        "peak_vram_gib": meta["peak_vram_gib"], "details": "meta.json",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[mask] done: {len(sel)} frames, mean masked {meta['masked_fraction_mean']:.4f}, "
          f"{meta['timings']['s_per_frame']:.3f}s/frame, peak VRAM {meta['peak_vram_gib']} GiB -> {out}", flush=True)
    return meta
