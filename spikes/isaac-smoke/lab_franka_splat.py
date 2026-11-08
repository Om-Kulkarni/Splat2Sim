"""Isaac Lab 3.0 smoke: splat+collider background, FRANKA_ROBOTIQ_GRIPPER_CFG, sweep finger_joint, camera PNG.
Run inside r2s-isaaclab3-sim61 image: isaaclab.sh -p /scripts/lab_franka_splat.py /out   (see run_lab.sh)"""
import argparse, sys, os
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); parser.add_argument("out", nargs="?", default="/out")
add_launcher_args(parser); args_cli = parser.parse_args(); args_cli.enable_cameras = True
import torch, numpy as np
from isaaclab_physx.renderers import IsaacRtxRendererCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import clone, instantiate
from isaaclab_assets.robots.franka import FRANKA_ROBOTIQ_GRIPPER_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        from pxr import Usd, UsdGeom
        print("ASSET_ROOT", ISAAC_NUCLEUS_DIR)
        # background layer: only splat + collider (no physics scene / ball / camera of the other smoke)
        bg = Usd.Stage.CreateNew("/data/lab_bg.usda"); UsdGeom.SetStageUpAxis(bg, UsdGeom.Tokens.z)
        bg.SetDefaultPrim(UsdGeom.Xform.Define(bg, "/Scene").GetPrim())
        bg.DefinePrim("/Scene/splat").GetReferences().AddReference("./synth_splat.usd", "/splat")
        bg.DefinePrim("/Scene/collider").GetReferences().AddReference("./synth_collider.usda")
        bg.GetRootLayer().Save()
        sim_utils.UsdFileCfg(usd_path="/data/lab_bg.usda").func("/World/Scene", sim_utils.UsdFileCfg(usd_path="/data/lab_bg.usda"))
        sim_utils.DomeLightCfg(intensity=1500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1500.0))
        cfg = clone(FRANKA_ROBOTIQ_GRIPPER_CFG); cfg.prim_path = "/World/Robot"
        cfg.init_state.pos = (-0.6, 0.0, 0.0)
        robot = Articulation(cfg=cfg)
        cam = instantiate(CameraCfg(prim_path="/World/CameraSensor", update_period=0, height=480, width=640, data_types=["rgb"],
            renderer_cfg=IsaacRtxRendererCfg(),
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.05, 100.0))))
        sim.reset()
        cam.set_world_poses_from_view(torch.tensor([[1.2, -1.6, 1.3]], device=sim.device), torch.tensor([[-0.3, 0.0, 0.4]], device=sim.device))
        print("JOINTS", robot.joint_names)
        i = robot.joint_names.index("finger_joint")
        lim = robot.data.joint_pos_limits.torch[0, i].tolist(); print("FINGER_LIMITS", lim)
        dt = sim.get_physics_dt(); hi = lim[1]; rows = []
        for step in range(600):
            tgt = robot.data.default_joint_pos.torch.clone()
            tgt[:, i] = hi * min(1.0, step / 300.0)
            robot.set_joint_position_target_index(target=tgt, full_data=True)
            robot.write_data_to_sim(); sim.step(); robot.update(dt); cam.update(dt)
            if step % 100 == 0 or step == 599:
                jp = robot.data.joint_pos.torch[0]
                rows.append((step, float(tgt[0, i]), {n: round(float(jp[k]), 3) for k, n in enumerate(robot.joint_names) if "finger" in n or "knuckle" in n}))
        for r in rows: print("SWEEP", r)
        img = cam.data.output["rgb"].torch[0].cpu().numpy()
        from PIL import Image
        Image.fromarray(img[:, :, :3].astype(np.uint8)).save(f"{args_cli.out}/lab_franka_splat.png"); print("WROTE lab_franka_splat.png", img.shape, "mean", float(img.mean()))
if __name__ == "__main__":
    main()
