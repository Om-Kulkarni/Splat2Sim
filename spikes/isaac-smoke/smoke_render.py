"""Standalone Isaac Sim smoke: stage + physics step + camera render to PNG.
Run inside the container: /isaac-sim/python.sh smoke_render.py /out
"""
import sys, os
out = sys.argv[1] if len(sys.argv) > 1 else "/out"
os.makedirs(out, exist_ok=True)
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RealTimePathTracing", "width": 640, "height": 480})
import numpy as np
import omni.usd, omni.timeline
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, Gf, Sdf, PhysxSchema
import omni.kit.app

ctx = omni.usd.get_context(); ctx.new_stage(); stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdPhysics.Scene.Define(stage, "/World/physicsScene")
g = UsdGeom.Cube.Define(stage, "/World/ground"); g.GetSizeAttr().Set(1.0)
UsdGeom.XformCommonAPI(g).SetScale((10, 10, 0.1)); UsdGeom.XformCommonAPI(g).SetTranslate((0, 0, -0.05))
UsdPhysics.CollisionAPI.Apply(g.GetPrim())
s = UsdGeom.Sphere.Define(stage, "/World/ball"); s.GetRadiusAttr().Set(0.1)
UsdGeom.XformCommonAPI(s).SetTranslate((0, 0, 1.0))
UsdPhysics.RigidBodyAPI.Apply(s.GetPrim()); UsdPhysics.CollisionAPI.Apply(s.GetPrim())
s.GetDisplayColorAttr().Set([(1, 0, 0)])
UsdLux.DistantLight.Define(stage, "/World/sun").GetIntensityAttr().Set(3000)
cam = UsdGeom.Camera.Define(stage, "/World/cam")
UsdGeom.XformCommonAPI(cam).SetTranslate((1.5, -1.5, 1.0)); UsdGeom.XformCommonAPI(cam).SetRotate((70, 0, 45))

import omni.replicator.core as rep
rp = rep.create.render_product("/World/cam", (640, 480))
rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach([rp])
omni.timeline.get_timeline_interface().play()
for i in range(120):
    app.update()
z = s.GetPrim().GetAttribute("xformOp:translate").Get()
print("BALL_Z_AFTER_STEPS", z)  # ~0.1 if physics ran (fell from 1.0)
for i in range(60):  # let RTX warm up / compile shaders
    app.update()
data = rgb.get_data()
for _ in range(10):
    if np.asarray(data).ndim == 3 and np.asarray(data).mean() > 0:
        break
    rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
    data = rgb.get_data()
from PIL import Image
Image.fromarray(np.asarray(data)[:, :, :3]).save(f"{out}/smoke_render.png")
print("WROTE", f"{out}/smoke_render.png", np.asarray(data).shape, "mean", float(np.asarray(data).mean()))
app.close()
