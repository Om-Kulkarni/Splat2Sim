"""Splat smoke: PLY -> ParticleField USD (gsplat converter), root .usda referencing splat + collider mesh,
physics ball drops onto collider, camera sensor renders splat to PNG.
Run: scripts/run_smoke.sh splat_scene.py  (expects /data/synth.ply, made by make_splat.py)"""
import sys, os
out = sys.argv[1] if len(sys.argv) > 1 else "/out"
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 640, "height": 480})
import numpy as np, time
import omni.usd, omni.timeline, omni.kit.app
from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.kit.converter.gsplat"); app.update()
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf, Sdf
from usd_convert_gsplat import convertPlyUSD, UP_AXIS_Z
t = time.time()
convertPlyUSD("/data/synth.ply", "/data/synth_splat.usd", prim_name="splat", up_axis=UP_AXIS_Z)
print("CONVERT_S", round(time.time() - t, 2))
# collider mesh in its own layer (a flat quad, as would come from reconstruction mesh)
cs = Usd.Stage.CreateNew("/data/synth_collider.usda"); UsdGeom.SetStageUpAxis(cs, UsdGeom.Tokens.z)
m = UsdGeom.Mesh.Define(cs, "/collider"); m.CreatePointsAttr([(-1.5,-1.5,0),(1.5,-1.5,0),(1.5,1.5,0),(-1.5,1.5,0)])
m.CreateFaceVertexCountsAttr([4]); m.CreateFaceVertexIndicesAttr([0,1,2,3])
UsdPhysics.CollisionAPI.Apply(m.GetPrim()); UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("none")
UsdGeom.Imageable(m).CreateVisibilityAttr("invisible"); cs.SetDefaultPrim(m.GetPrim()); cs.Save()
ss = Usd.Stage.Open("/data/synth_splat.usd"); print("SPLAT_DEFAULT_PRIM", ss.GetDefaultPrim().GetPath() if ss.GetDefaultPrim() else None,
      [ (p.GetPath().pathString, p.GetTypeName()) for p in ss.Traverse()][:5])
dp = ss.GetDefaultPrim().GetPath().pathString if ss.GetDefaultPrim() else "/splat"

root = Usd.Stage.CreateNew("/data/synth_scene.usda"); UsdGeom.SetStageUpAxis(root, UsdGeom.Tokens.z)
root.SetDefaultPrim(UsdGeom.Xform.Define(root, "/World").GetPrim())
sc = UsdGeom.Xform.Define(root, "/World/Scene")  # Anchor transform would go here
sp = root.DefinePrim("/World/Scene/splat"); sp.GetReferences().AddReference("./synth_splat.usd", dp)
cp = root.DefinePrim("/World/Scene/collider"); cp.GetReferences().AddReference("./synth_collider.usda")
UsdPhysics.Scene.Define(root, "/World/physicsScene")
b = UsdGeom.Sphere.Define(root, "/World/ball"); b.GetRadiusAttr().Set(0.1); UsdGeom.XformCommonAPI(b).SetTranslate((-0.8, -0.8, 1.0))
UsdPhysics.RigidBodyAPI.Apply(b.GetPrim()); UsdPhysics.CollisionAPI.Apply(b.GetPrim()); b.GetDisplayColorAttr().Set([(1,0,1)])
UsdLux.DomeLight.Define(root, "/World/dome").GetIntensityAttr().Set(500)
cam = UsdGeom.Camera.Define(root, "/World/cam"); UsdGeom.XformCommonAPI(cam).SetTranslate((1.6, -1.6, 1.2)); UsdGeom.XformCommonAPI(cam).SetRotate((65, 0, 45))
root.GetRootLayer().Save()

omni.usd.get_context().open_stage("/data/synth_scene.usda")
for _ in range(30): app.update()
import omni.replicator.core as rep
stage = omni.usd.get_context().get_stage()
print("SPLAT_PRIM_TYPE", stage.GetPrimAtPath("/World/Scene/splat").GetTypeName(),
      [c.GetTypeName() for c in stage.GetPrimAtPath("/World/Scene/splat").GetChildren()])
rp = rep.create.render_product("/World/cam", (640, 480)); rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach([rp])
omni.timeline.get_timeline_interface().play()
for _ in range(120): app.update()
zs = stage.GetPrimAtPath("/World/ball").GetAttribute("xformOp:translate").Get()
print("BALL_POS_AFTER", zs, "(z~0.1 => stopped on collider mesh)")
for _ in range(60): app.update()
data = rgb.get_data()
for _ in range(10):
    if np.asarray(data).ndim == 3 and np.asarray(data).mean() > 0: break
    rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False); data = rgb.get_data()
from PIL import Image
Image.fromarray(np.asarray(data)[:, :, :3]).save(f"{out}/splat_render.png"); print("WROTE splat_render.png", np.asarray(data).shape)
app.close()
