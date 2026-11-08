"""report stage, part 2 (Isaac Lab): render the robot ALONE at ``robot.scan_pose_joints`` from sampled keyframe cameras (after the
ChArUco anchor) -> /runs/<scene>/report/silhouette/sil_<frame>.png (255 = robot) + meta.json. ``python -m splat2sim.report finish``
compares these with ``masks/``.

Camera pose in the robot-base frame: T_base_cam = T_base_recon (scale s) applied to the recon-frame pose of the frame.
Status ``skipped: ...`` (no Kit start) when the anchor is not ``charuco`` or there are no masks.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/r2s")
T0 = time.time()
RUN = Path("/runs") / os.environ.get("R2S_SCENE", "")
OUT = RUN / "report" / "silhouette"


def skip(reason):
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*"):
        f.unlink()
    (OUT / "meta.json").write_text(json.dumps({"status": f"skipped: {reason}"}))
    os.system(f"chmod -R a+rwX {OUT}")
    print(f"[silhouette] skipped: {reason}", flush=True)
    sys.exit(0)


anchor = json.loads((RUN / "poses" / "anchor.json").read_text()) if (RUN / "poses" / "anchor.json").is_file() else {}
if anchor.get("status") != "charuco":
    skip(f"anchor is {anchor.get('status', 'missing')}, robot base unknown")
if not any((RUN / "masks").glob("0*.png")):
    skip("no masks/ (mask stage not run)")

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=24)
add_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
args.headless = True

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from splat2sim.lab import ENV_ID  # noqa: E402
from splat2sim.lab.env_cfg import Splat2SimFR3CubesEnvCfg  # noqa: E402
from splat2sim.recon import usdutil  # noqa: E402
from splat2sim.recon.poseio import read_poses_csv  # noqa: E402


def main():
    intr = json.loads((RUN / "frames" / "intrinsics.json").read_text())
    W, H = int(intr["width"]), int(intr["height"])
    idx, _, Ts = read_poses_csv(RUN / "poses" / "poses.csv")
    have = [k for k, i in enumerate(idx) if (RUN / "masks" / f"{int(i):06d}.png").is_file()]
    if not have:
        skip("no pose has a mask")
    sel = [have[j] for j in np.unique(np.linspace(0, len(have) - 1, min(args.frames, len(have))).round().astype(int))]
    T_br = np.array(anchor["T_base_recon"], float).reshape(4, 4)
    s = float(np.cbrt(abs(np.linalg.det(T_br[:3, :3]))))
    R_br = T_br[:3, :3] / s

    cfg = Splat2SimFR3CubesEnvCfg.from_lab(RUN / "lab.json", RUN, scene_usd=None, seed=0)
    for d in (OUT,):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*"):
            f.unlink()
    with launch_simulation(cfg, args):
        import gymnasium as gym
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom

        cfg.ensure_robot_asset()
        env = gym.make(ENV_ID, cfg=cfg)
        env.reset(seed=0)
        u = env.unwrapped
        robot = u.scene["robot"]
        scan = torch.tensor([list(cfg.lab.robot.scan_pose_joints) + [0.0]], dtype=torch.float32, device=u.device)
        for _ in range(60):
            env.step(scan)
        origin = u.scene.env_origins[0].cpu().numpy()
        bpos = robot.data.body_pos_w.torch[0].cpu().numpy() - origin
        bq = robot.data.body_quat_w.torch[0].cpu().numpy()
        stage = omni.usd.get_context().get_stage()
        from splat2sim.lab.visual import add_robot_visual_copy

        placed = add_robot_visual_copy(stage, cfg.robot_usd, "/World/sil/robot", list(robot.body_names), bpos, bq)
        app = omni.kit.app.get_app()
        cam = UsdGeom.Camera.Define(stage, "/World/sil/cam")
        ha = 20.955
        cam.GetHorizontalApertureAttr().Set(ha)
        cam.GetVerticalApertureAttr().Set(ha * H / W)
        cam.GetFocalLengthAttr().Set(intr["fx"] * ha / W)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 50.0))
        op = cam.AddTransformOp()
        rp = rep.create.render_product("/World/sil/cam", (W, H))
        seg = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
        seg.attach([rp])
        for _ in range(20):
            app.update()
        frames = []
        for n, k in enumerate(sel):
            T_c = np.eye(4)
            T_c[:3, :3] = R_br @ Ts[k][:3, :3]
            T_c[:3, 3] = s * R_br @ Ts[k][:3, 3] + T_br[:3, 3]
            op.Set(Gf.Matrix4d(*[v for r in usdutil.np_to_usd_matrix(usdutil.cv_camera_to_usd(T_c)) for v in r]))
            for _ in range(10 if n == 0 else 4):
                app.update()
            d = seg.get_data()
            ids = [int(i) for i, v in d["info"]["idToLabels"].items() if (v.get("class") if isinstance(v, dict) else v) == "robot"]
            m = np.isin(np.asarray(d["data"]).reshape(H, W), ids)
            f = int(idx[k])
            Image.fromarray((m * 255).astype(np.uint8)).save(OUT / f"sil_{f:06d}.png")
            frames.append(f)
        (OUT / "meta.json").write_text(json.dumps({
            "status": "ok", "frames": frames, "links_placed": placed, "scan_pose_joints": list(cfg.lab.robot.scan_pose_joints),
            "coupling_offset": cfg.lab.robot.coupling_offset.model_dump(), "seconds": round(time.time() - T0, 1)}, indent=2))
        os.system(f"chmod -R a+rwX {OUT}")
        print(f"[silhouette] {len(frames)} frames -> {OUT} in {time.time() - T0:.0f}s", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
