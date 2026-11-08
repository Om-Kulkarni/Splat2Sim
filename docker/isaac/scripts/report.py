"""report stage (Isaac, minimal M1 version): headless renders of scene.usda + summary.json.

Writes /runs/<scene>/report/preview_train.png (a training camera pose), preview_overview_1/2.png and summary.json.
Fly-through frames go to report/flythrough/; summary.json, report.md, coverage, mask-vs-URDF IoU and the MP4 are produced by
the CPU `finish` step (splat2sim/report/).
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/opt/r2s")
from _common import RUN, SCENE, open_permissions, read_json, start_app  # noqa: E402

from splat2sim.recon import usdutil  # noqa: E402
from splat2sim.recon.poseio import read_poses_csv  # noqa: E402

t0 = time.time()
out = RUN / "report"
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("preview_*.png"):
    f.unlink()
app = start_app()
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux  # noqa: E402

scene_usd = RUN / "scene" / "scene.usda"
omni.usd.get_context().open_stage(str(scene_usd))
for _ in range(60):
    app.update()
stage = omni.usd.get_context().get_stage()
meta = json.loads((stage.GetRootLayer().customLayerData or {}).get("r2s_json", "{}"))
intr = read_json(RUN / "frames" / "intrinsics.json", {"fx": 600.0, "width": 848, "height": 480})


def mat(T):
    return Gf.Matrix4d(*[v for row in usdutil.np_to_usd_matrix(T) for v in row])


def add_camera(path, T_world_or_parent, fx, size, parent_is_anchor=False):
    cam = UsdGeom.Camera.Define(stage, path)
    ha = 20.955
    cam.GetHorizontalApertureAttr().Set(ha)
    cam.GetVerticalApertureAttr().Set(ha * size[1] / size[0])
    cam.GetFocalLengthAttr().Set(fx * ha / size[0])
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, 100.0))
    cam.AddTransformOp().Set(mat(T_world_or_parent))
    return path, size


views = []
with Usd.EditContext(stage, stage.GetSessionLayer()):  # previews never touch scene.usda
    UsdLux.DomeLight.Define(stage, "/World/ReportLight").GetIntensityAttr().Set(500)
    UsdGeom.Xform.Define(stage, "/World/Report")
    idx, _, Ts = read_poses_csv(RUN / "poses" / "poses.csv")
    mid = len(Ts) // 2  # training camera pose, expressed in the recon frame under /World/Anchor
    views.append(("train",) + add_camera("/World/Anchor/Cameras/preview_train", usdutil.cv_camera_to_usd(Ts[mid]),
                                         intr["fx"], (intr["width"], intr["height"])))
    if "bbox_world_min" in meta:
        lo, hi = np.array(meta["bbox_world_min"]), np.array(meta["bbox_world_max"])
    else:
        lo, hi = np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, 1.0])
    for k, (eye, tgt) in enumerate(usdutil.overview_eyes(lo, hi, n=2), start=1):
        views.append((f"overview_{k}",) + add_camera(f"/World/Report/overview_{k}", usdutil.look_at_usd_camera(eye, tgt),
                                                     900.0, (1280, 720)))

omni.timeline.get_timeline_interface().play()
for _ in range(120):
    app.update()

written = []
for name, path, size in views:
    rp = rep.create.render_product(path, size)
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([rp])
    for _ in range(30):
        app.update()
    data = ann.get_data()
    for _ in range(12):
        if np.asarray(data).ndim == 3 and np.asarray(data)[:, :, :3].mean() > 0:
            break
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
        data = ann.get_data()
    img = np.asarray(data)[:, :, :3]
    Image.fromarray(img).save(out / f"preview_{name}.png")
    written.append({"name": f"preview_{name}.png", "size": list(size), "mean_pixel": float(img.mean())})
    print(f"[report] wrote preview_{name}.png mean={img.mean():.1f}", flush=True)
    ann.detach([rp])
    rp.destroy()

# --- fly-through frames (PNG; the MP4 is encoded in the ingest image by `python -m splat2sim.report finish`: no ffmpeg here) ---
ft_dir = out / "flythrough"
ft_dir.mkdir(exist_ok=True)
for f in ft_dir.glob("*.png"):
    f.unlink()
n_orbit, n_train = 48, 96
FW, FH = 960, 540
with Usd.EditContext(stage, stage.GetSessionLayer()):
    cam = UsdGeom.Camera.Define(stage, "/World/Anchor/Cameras/flythrough")
    cam.GetHorizontalApertureAttr().Set(20.955)
    cam.GetVerticalApertureAttr().Set(20.955 * FH / FW)
    cam.GetFocalLengthAttr().Set(intr["fx"] * 20.955 / intr["width"])  # same field of view as the training camera
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, 100.0))
    fop = cam.AddTransformOp()
    # world-frame orbit around the scene bbox (camera lives under /World/Anchor: map the world eye back into recon with T_anchor^-1)
    T_anchor = np.array(meta["T_anchor_recon"], float).reshape(4, 4)
    Ti = np.linalg.inv(T_anchor)
    orbit = []
    for eye, tgt in usdutil.overview_eyes(lo, hi, n=n_orbit, elevation_deg=30.0, distance_scale=1.3):
        orbit.append(Ti @ usdutil.look_at_usd_camera(eye, tgt))
    sel = np.unique(np.linspace(0, len(Ts) - 1, n_train).round().astype(int))
    smooth = [usdutil.cv_camera_to_usd(Ts[i]) for i in sel]
frp = rep.create.render_product("/World/Anchor/Cameras/flythrough", (FW, FH))
fann = rep.AnnotatorRegistry.get_annotator("rgb")
fann.attach([frp])
n_ft = 0
for k, M in enumerate(orbit + smooth):
    fop.Set(mat(M))
    for _ in range(14 if k == 0 else 4):
        app.update()
    d = np.asarray(fann.get_data())
    if d.ndim != 3 or d[:, :, :3].mean() == 0:
        for _ in range(8):
            app.update()
        d = np.asarray(fann.get_data())
    if d.ndim == 3:
        Image.fromarray(d[:, :, :3]).save(ft_dir / f"{k:04d}.png")
        n_ft += 1
print(f"[report] fly-through: {n_ft} frames", flush=True)
fann.detach([frp])
frp.destroy()

(out / "isaac.json").write_text(json.dumps({"scene": SCENE, "previews": written, "scene_meta": meta, "flythrough_frames": n_ft,
                                            "seconds": round(time.time() - t0, 1)}, indent=2))
print("[report] isaac.json written (summary.json / report.md / MP4 come from the finish step)")
open_permissions(out)
app.close()
