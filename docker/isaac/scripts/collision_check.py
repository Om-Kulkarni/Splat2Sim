"""collision check (Isaac, run on demand after compose): drop 5 cm rigid cubes over the fitted table box in scene.usda,
verify they come to rest on the table top (not through it), and render previews.

Nothing is written to scene.usda (edits stay in memory). Outputs in /runs/<scene>/collision/:
  check.json              per-cube rest height vs table top, pass/fail
  preview_cubes.png       splat + resting cubes (colliders invisible)
  preview_debug_*.png     same with the collision mesh/boxes made visible (alignment check against the splat)

Run: docker compose run --rm -e R2S_SCENE=<scene> isaac -p /workspace/r2s/collision_check.py
"""
import json
import sys
from contextlib import nullcontext

import numpy as np

sys.path.insert(0, "/opt/r2s")
from _common import RUN, open_permissions, read_json, start_app  # noqa: E402

from splat2sim.recon import usdutil  # noqa: E402
from splat2sim.recon.poseio import read_poses_csv  # noqa: E402

N_CUBES = 4
SIZE = 0.05
out = RUN / "collision"
prim_json = read_json(out / "primitives.json")
if not prim_json or not prim_json.get("table_box"):
    raise SystemExit("error: collision/primitives.json has no table_box")
tb = prim_json["table_box"]

app = start_app()
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, Vt  # noqa: E402

omni.usd.get_context().open_stage(str(RUN / "scene" / "scene.usda"))
for _ in range(60):
    app.update()
stage = omni.usd.get_context().get_stage()
meta = json.loads((stage.GetRootLayer().customLayerData or {}).get("r2s_json", "{}"))


def mat(T):
    return Gf.Matrix4d(*[v for row in usdutil.np_to_usd_matrix(T) for v in row])


c, size, yaw = np.array(tb["center"]), np.array(tb["size"]), np.radians(tb["yaw_deg"])
Rbox = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
_r, _p, _y = np.radians(tb["rpy_deg"])
Rbox = (np.array([[np.cos(_y), -np.sin(_y), 0], [np.sin(_y), np.cos(_y), 0], [0, 0, 1]])
        @ np.array([[np.cos(_p), 0, np.sin(_p)], [0, 1, 0], [-np.sin(_p), 0, np.cos(_p)]])
        @ np.array([[1, 0, 0], [0, np.cos(_r), -np.sin(_r)], [0, np.sin(_r), np.cos(_r)]]))
nrm = Rbox[:, 2]  # table-top normal (the box follows the fitted plane tilt)
c_top = c + nrm * 0.5 * size[2]
top = float(c_top[2])


def plane_z(xy):
    return float(c_top[2] - (nrm[0] * (xy[0] - c_top[0]) + nrm[1] * (xy[1] - c_top[1])) / nrm[2])


Rz = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
rng = np.random.default_rng(0)
margin = SIZE
# --- preview camera: a training pose (recon frame, under /World/Anchor like the splat, so the splat is sharp) whose optical
# axis hits the table box 0.25-2.2 m away from above; cubes are dropped on free spots around that hit point ---
_, _, Ts = read_poses_csv(RUN / "poses" / "poses.csv")
intr = read_json(RUN / "frames" / "intrinsics.json", {"fx": 600.0, "width": 848, "height": 480})
T_ar = np.array(meta["T_anchor_recon"])
Tinv = np.linalg.inv(T_ar)
Rb = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])  # world xy -> box frame
best, hit, best_score = None, c[:2], -1.0
for T in Ts:
    Tw = T_ar @ T  # camera in the anchor frame
    f, o_ = Tw[:3, 2], Tw[:3, 3]
    if f[2] > -0.05 or not (top + 0.05 < o_[2] < top + 2.0):
        continue
    t_hit = (top - o_[2]) / f[2]
    h_ = o_ + t_hit * f
    if not (0.25 < t_hit < 2.2) or np.any(np.abs(Rb @ (h_[:2] - c[:2])) > 0.5 * size[:2]):
        continue
    score = -f[2] - 0.1 * t_hit  # steeper, closer
    if score > best_score:
        best, hit, best_score = T, h_[:2], score
if best is None:  # no pose looks onto the box: fall back to the nearest pose looking at its centre
    tr = (Tinv @ np.append([c[0], c[1], top], 1.0))[:3]
    best = max(Ts, key=lambda T: float(T[:3, 2] @ (tr - T[:3, 3])) / np.linalg.norm(tr - T[:3, 3]) - 0.2 * np.linalg.norm(tr - T[:3, 3]))
    hit = c[:2]
print(f"[collision_check] preview camera hits the table box at {np.round(hit, 3)} (score {best_score:.2f})", flush=True)

# free spots: box cells with no collision-mesh geometry standing above the table top (mock scenes put objects on it)
T_ar0 = np.array(meta["T_anchor_recon"])
msrc = Usd.Stage.Open(str(out / "collision_mesh.usd"))
mpts = np.array(UsdGeom.Mesh(msrc.GetPrimAtPath("/Mesh")).GetPointsAttr().Get(), float) if msrc.GetPrimAtPath("/Mesh") else np.zeros((0, 3))
mpts = mpts @ T_ar0[:3, :3].T + T_ar0[:3, 3]
hrel = (mpts - c_top) @ nrm
above = mpts[(hrel > 0.015) & (hrel < 0.6)][:, :2]
cand = c[:2] + (rng.uniform(-1, 1, (4000, 2)) * np.maximum(0.5 * size[:2] - margin, 0.0)) @ Rz.T
if len(above):
    dmin = np.min(np.linalg.norm(cand[:, None, :] - above[None, ::max(1, len(above) // 4000), :], axis=2), axis=1)
    cand = cand[dmin > 0.07]
cand = cand[np.argsort(np.linalg.norm(cand - hit, axis=1))]  # nearest the camera's hit point first
spots = []
for q in cand:
    if all(np.linalg.norm(q - o2) > 0.12 for o2 in spots):
        spots.append(q)
    if len(spots) == N_CUBES:
        break
print(f"[collision_check] {len(cand)} free candidate spots, using {len(spots)}", flush=True)
while len(spots) < N_CUBES:  # no free space found: random spots (cubes may land on objects)
    spots.append(c[:2] + Rz @ (rng.uniform(-1, 1, 2) * np.maximum(0.5 * size[:2] - margin, 0.0)))
cubes = []
with nullcontext():  # in-memory root-layer edits, never saved (PhysX ignored session-layer prims)
    stage.SetTimeCodesPerSecond(60.0)  # a usable time range so the timeline actually advances (and PhysX steps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(100000)
    sc = UsdPhysics.Scene.Define(stage, "/World/CheckPhysics")
    sc.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    sc.CreateGravityMagnitudeAttr(9.81)
    UsdLux.DomeLight.Define(stage, "/World/CheckLight").GetIntensityAttr().Set(600)
    UsdGeom.Xform.Define(stage, "/World/CheckCubes")
    colors = [(0.9, 0.1, 0.1), (0.1, 0.7, 0.1), (0.1, 0.2, 0.9), (0.95, 0.8, 0.1)]
    for i in range(N_CUBES):
        xy = spots[i]
        p = np.array([xy[0], xy[1], plane_z(xy) + 0.12 + 0.09 * i])
        cube = UsdGeom.Cube.Define(stage, f"/World/CheckCubes/cube_{i}")
        cube.CreateSizeAttr(SIZE)
        cube.AddTranslateOp().Set(Gf.Vec3d(*p))
        cube.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*colors[i % 4])]))
        pr = cube.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(pr)
        UsdPhysics.CollisionAPI.Apply(pr)
        UsdPhysics.MassAPI.Apply(pr).CreateMassAttr(0.05)
        cubes.append((pr, p))

import carb.settings  # noqa: E402

_cs = carb.settings.get_settings()
for k, v in (("/physics/updateToUsd", True), ("/physics/updateVelocitiesToUsd", True), ("/physics/fabricEnabled", False),
             ("/physics/suppressReadback", False)):
    _cs.set(k, v)  # write simulated poses back to USD so they can be read with UsdGeom
tl = omni.timeline.get_timeline_interface()
tl.play()
cache = UsdGeom.XformCache()


def cube_z():
    cache.Clear()
    return [float(cache.GetLocalToWorldTransform(p).ExtractTranslation()[2]) for p, _ in cubes]


print("[collision_check] attr translate", [p.GetAttribute("xformOp:translate").Get() for p, _ in cubes[:1]],
      "physics scenes:", [x.GetPath().pathString for x in stage.Traverse() if x.IsA(UsdPhysics.Scene)], flush=True)


trace = []
for step in range(420):
    app.update()
    if step % 30 == 0:
        trace.append(cube_z())
print("[collision_check] timeline time", tl.get_current_time(), "attr translate",
      cubes[0][0].GetAttribute("xformOp:translate").Get(), flush=True)
final = cube_z()
final_xy = []
cache.Clear()
for p, _ in cubes:
    t = cache.GetLocalToWorldTransform(p).ExtractTranslation()
    final_xy.append([float(t[0]), float(t[1])])
res = []
for i, (z, (pr, p0)) in enumerate(zip(final, cubes)):
    pc = np.array([final_xy[i][0], final_xy[i][1], z])
    h_above = float((pc - c_top) @ nrm) - 0.5 * SIZE  # cube underside above the table-top plane, along its normal
    # allow resting on a mesh object standing on the table: reject only below-table or hovering results
    res.append({"cube": i, "start_z": float(p0[2]), "final_center_z": z, "table_top_z_at_center": top,
                "bottom_minus_top_m": h_above, "final_xy": final_xy[i],
                "height_above_table_m": h_above, "on_table": abs(h_above) <= 0.01, "fell_through": h_above < -0.03})
settled = float(np.max(np.abs(np.array(trace[-1]) - np.array(trace[-2])))) < 1e-3 if len(trace) > 1 else False
check = {"table_box": tb, "cubes": res, "settled": settled, "z_trace": trace,
         "passed": bool(all(r["on_table"] for r in res) and not any(r["fell_through"] for r in res))}
print("[collision_check]", json.dumps({k: check[k] for k in ("passed", "settled")}),
      [round(r["bottom_minus_top_m"], 4) for r in res], flush=True)

# --- previews: pause physics, render from a camera looking at the table ---

target = np.mean([[x, y, top + 0.03] for x, y in final_xy], axis=0)
views = {}
SZ = (intr["width"], intr["height"])
with nullcontext():
    for k, back in enumerate((0.0, 0.5)):
        T = best.copy()
        T[:3, 3] = T[:3, 3] - back * T[:3, 2]  # k=1: same view direction pulled back along the optical axis
        cam = UsdGeom.Camera.Define(stage, f"/World/Anchor/Cameras/check_{k}")
        ha = 20.955
        cam.GetHorizontalApertureAttr().Set(ha)
        cam.GetVerticalApertureAttr().Set(ha * SZ[1] / SZ[0])
        cam.GetFocalLengthAttr().Set(intr["fx"] * ha / SZ[0])
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, 100.0))
        cam.AddTransformOp().Set(mat(usdutil.cv_camera_to_usd(T)))
        views[k] = f"/World/Anchor/Cameras/check_{k}"
    cam = UsdGeom.Camera.Define(stage, "/World/CheckTop")  # world-frame view from above (splat is blurry off-trajectory)
    cam.GetHorizontalApertureAttr().Set(20.955)
    cam.GetVerticalApertureAttr().Set(20.955 * SZ[1] / SZ[0])
    cam.GetFocalLengthAttr().Set(intr["fx"] * 20.955 / SZ[0])
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, 100.0))
    cam.AddTransformOp().Set(mat(usdutil.look_at_usd_camera(target + np.array([0.35, -0.35, 0.8]), target)))
    views[2] = "/World/CheckTop"
check["preview_camera_hit_xy"] = [float(v) for v in hit]


def render(path, fname):
    for _ in range(30):
        app.update()
    rp = rep.create.render_product(path, SZ)
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
    Image.fromarray(img).save(out / fname)
    print(f"[collision_check] wrote {fname} mean={img.mean():.1f}", flush=True)
    ann.detach([rp])
    rp.destroy()


# The simulated rigid cubes are not picked up by the RTX renderer after the physics run, so the previews show static
# visual copies at the settled poses (session layer; the rigid cubes are hidden).
tl.stop()
with Usd.EditContext(stage, stage.GetSessionLayer()):
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/CheckCubes")).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdGeom.Xform.Define(stage, "/World/CheckVisual")
    for i, (xy, z) in enumerate(zip(final_xy, final)):
        v = UsdGeom.Cube.Define(stage, f"/World/CheckVisual/cube_{i}")
        v.CreateSizeAttr(SIZE)
        v.AddTranslateOp().Set(Gf.Vec3d(xy[0], xy[1], z))
        v.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*colors[i % 4])]))
render(views[0], "preview_cubes.png")
render(views[2], "preview_cubes_top.png")
# debug: make the invisible colliders visible, tinted and translucent
with nullcontext():  # in-memory root-layer edits, never saved (PhysX ignored session-layer prims)
    for path, col in (("/World/Anchor/Collision/Mesh", (0.1, 0.8, 1.0)), ("/World/Anchor/Collision/TableBox", (1.0, 0.4, 0.0)),
                      ("/World/Anchor/Collision/PedestalBox", (0.7, 0.2, 0.9))):
        pr = stage.GetPrimAtPath(path)
        if not pr:
            continue
        g = UsdGeom.Gprim(pr)
        g.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        g.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*col)]))
        g.CreateDisplayOpacityAttr(Vt.FloatArray([0.45]))
render(views[0], "preview_debug_close.png")
render(views[1], "preview_debug_wide.png")
render(views[2], "preview_debug_top.png")
UsdGeom.Imageable(stage.GetPrimAtPath("/World/Anchor/Splat")).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
render(views[1], "preview_collision_only.png")  # splat hidden: colliders + cubes only
UsdGeom.Imageable(stage.GetPrimAtPath("/World/Anchor/Collision/Mesh")).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
render(views[1], "preview_boxes_only.png")  # splat + mesh hidden: boxes + cubes
(out / "check.json").write_text(json.dumps(check, indent=1))
open_permissions(out)
app.close()
