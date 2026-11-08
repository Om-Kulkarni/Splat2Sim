"""Load a 3DGRUT-exported ParticleField .usdz in Isaac Sim 6.1 and render it with the camera embedded by the exporter.
Run: run_smoke.sh grut_usd_render.py   (expects /data/smoke.usdz; writes /out/grut_render.png)"""
import sys
out = sys.argv[1] if len(sys.argv) > 1 else "/out"
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 848, "height": 480})
import numpy as np, omni.usd, omni.timeline
from pxr import UsdGeom
omni.usd.get_context().open_stage("/data/smoke.usdz")
for _ in range(60): app.update()
stage = omni.usd.get_context().get_stage()
print("PRIMS", [(p.GetPath().pathString, p.GetTypeName()) for p in stage.Traverse()][:8])
import omni.replicator.core as rep
cam = "/World/Cameras/camera_0"
rp = rep.create.render_product(cam, (848, 480)); rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach([rp])
omni.timeline.get_timeline_interface().play()
for _ in range(120): app.update()
data = rgb.get_data()
for _ in range(10):
    if np.asarray(data).ndim == 3 and np.asarray(data).mean() > 0: break
    rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False); data = rgb.get_data()
from PIL import Image
Image.fromarray(np.asarray(data)[:, :, :3]).save(f"{out}/grut_render.png")
print("WROTE grut_render.png", np.asarray(data).shape, "mean", float(np.asarray(data).mean()))
app.close()
