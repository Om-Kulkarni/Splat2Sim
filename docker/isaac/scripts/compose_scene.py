"""compose stage (Isaac): splat + anchor -> /runs/<scene>/scene/scene.usda.

    /World                         Xform  (default prim; z-up, metres)
      /World/Anchor                Xform  xformOp:transform = recon -> robot-base (anchor.json) or the fallback
        /World/Anchor/Splat        Xform  references ../splat/splat.usdz </World/gaussians> (ParticleField, recon frame)
        /World/Anchor/Collision    Xform  references ../collision/collision.usda (invisible static colliders; empty without it)
        /World/Anchor/Cameras      Xform  (empty; policy cameras)

The root is a .usda because a .usdz cannot take references/overrides. Pure-python math is in splat2sim.recon.usdutil.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/opt/r2s")
from _common import RUN, SCENE, open_permissions, read_json, start_app  # noqa: E402

from splat2sim.recon import usdutil  # noqa: E402
from splat2sim.recon.poseio import read_poses_csv  # noqa: E402

t0 = time.time()
app = start_app()
from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

out_dir = RUN / "scene"
out_dir.mkdir(parents=True, exist_ok=True)
for f in out_dir.glob("scene.usd*"):
    f.unlink()
splat_rel = "../splat/splat.usdz"
splat_abs = (out_dir / splat_rel).resolve()
if not splat_abs.is_file():
    raise SystemExit(f"error: {splat_abs} missing; run the splat stage first")

# --- anchor: the real ChArUco anchor if present, else the gravity/plane fallback from the poses stage ---
anchor_json = read_json(RUN / "poses" / "anchor.json")
if anchor_json is not None and anchor_json.get("status") == "charuco":
    T_anchor, kind = usdutil.anchor_transform(anchor_json)
    anchor_info = {"anchor": kind, "scale": anchor_json["scale"], "frames_used": len(anchor_json["frames_used"]),
                   "reproj_rms_px": anchor_json["reproj_rms_px"], "corner_3d_rms_mm": anchor_json["corner_3d_rms_mm"]}
else:  # status fallback / failed / no anchor.json: keep the gravity/plane fallback from the poses stage
    fb = read_json(RUN / "poses" / "fallback_anchor.json")
    if fb is None:
        raise SystemExit("error: neither poses/anchor.json nor poses/fallback_anchor.json exists")
    T_anchor, kind = usdutil.anchor_transform(fb)
    anchor_info = {"anchor": "fallback", "method": fb["method"], "up_source": fb["up_source"],
                   "plane_inlier_fraction": fb["plane_inlier_fraction"], "note": fb["note"]}
T_splat = usdutil.splat_local_transform("recon")

# --- stage ---
stage = Usd.Stage.CreateNew(str(out_dir / "scene.usda"))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())


def mat(T):
    return Gf.Matrix4d(*[v for row in usdutil.np_to_usd_matrix(T) for v in row])


anchor = UsdGeom.Xform.Define(stage, "/World/Anchor")
anchor.AddTransformOp().Set(mat(T_anchor))
anchor.GetPrim().CreateAttribute("r2s:anchor", Sdf.ValueTypeNames.String).Set(anchor_info["anchor"])
if anchor_info["anchor"] == "charuco":  # residuals as custom attrs
    for k, ty in (("scale", Sdf.ValueTypeNames.Double), ("reproj_rms_px", Sdf.ValueTypeNames.Double),
                  ("corner_3d_rms_mm", Sdf.ValueTypeNames.Double), ("frames_used", Sdf.ValueTypeNames.Int)):
        anchor.GetPrim().CreateAttribute(f"r2s:anchor_{k}", ty).Set(anchor_info[k])
splat = UsdGeom.Xform.Define(stage, "/World/Anchor/Splat")
splat.AddTransformOp().Set(mat(T_splat))
splat.GetPrim().GetReferences().AddReference(splat_rel, "/World/gaussians")
coll = UsdGeom.Xform.Define(stage, "/World/Anchor/Collision")
coll_rel = "../collision/collision.usda"  # invisible static colliders in the recon frame (collision stage), optional
if (out_dir / coll_rel).resolve().is_file():
    coll.GetPrim().GetReferences().AddReference(coll_rel)
UsdGeom.Xform.Define(stage, "/World/Anchor/Cameras")

# --- checks against the exported usdz: frame sanity + render settings to carry over ---
check = {}
src = Usd.Stage.Open(str(splat_abs))
rs = (src.GetRootLayer().customLayerData or {}).get("renderSettings")
layer_data = {"renderSettings": dict(rs)} if rs else {}
cam = src.GetPrimAtPath("/World/Cameras/camera_0")
if cam and (RUN / "poses" / "poses.csv").is_file():
    _, _, Ts = read_poses_csv(RUN / "poses" / "poses.csv")
    xf = UsdGeom.Xformable(cam).GetLocalTransformation(Usd.TimeCode(0))
    c0 = usdutil.usd_to_np_matrix([list(r) for r in xf])[:3, 3]
    d = float(np.min(np.linalg.norm(Ts[:, :3, 3] - c0, axis=1)))
    check = {"camera0_to_nearest_pose_m": d, "splat_frame_is_recon": d < 5e-3}
n_part = [p for p in src.Traverse() if p.GetTypeName().startswith("ParticleField")]
check["particle_fields"] = len(n_part)
check["collision_layer"] = coll_rel if coll.GetPrim().HasAuthoredReferences() else None
if n_part:
    pos = n_part[0].GetAttribute("positions").Get()
    check["num_gaussians"] = int(len(pos)) if pos is not None else None
stats = read_json(RUN / "poses" / "scene_stats.json", {})
meta = {"scene": SCENE, **anchor_info, "T_anchor_recon": T_anchor.tolist(), "splat": splat_rel,
        "splat_content_frame": "recon (exporter upAxis=Y metadata ignored)", "check": check}
if stats:
    lo, hi = usdutil.bbox_transform(stats["points_bbox_min"], stats["points_bbox_max"], T_anchor)
    meta["bbox_world_min"], meta["bbox_world_max"] = lo.tolist(), hi.tolist()
meta["compose_seconds"] = round(time.time() - t0, 1)
layer_data.update({"r2s_anchor": anchor_info["anchor"], "r2s_json": json.dumps(meta)})
stage.GetRootLayer().customLayerData = layer_data
stage.GetRootLayer().Save()

# --- verify by reloading ---
chk = Usd.Stage.Open(str(out_dir / "scene.usda"))
pf = [p.GetPath().pathString for p in chk.Traverse() if p.GetTypeName().startswith("ParticleField")]
print("[compose] ParticleField prims:", pf)
print("[compose] meta:", json.dumps(meta, indent=1))
if not pf:
    open_permissions(out_dir)
    raise SystemExit("error: composed scene has no ParticleField prim (reference did not resolve)")
open_permissions(out_dir)
app.close()
