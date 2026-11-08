"""env stage (Isaac): scripted pick-and-lift in Splat2Sim-FR3-Cubes-v0 -> /runs/<scene>/env/{summary.json,preview/,*.gif|mp4}.

Run:  isaaclab.sh -p /workspace/r2s/check_env.py [--trials 5] [--no-scene] [--out DIR] [--show-table] [--seed N]
The lab config comes from /runs/<scene>/lab.json (written by `r2s build`), the Scene from /runs/<scene>/scene/scene.usda when present.
Exit code 1 when the success rate is below --min-success (the summary is written either way).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/r2s")
T_START = time.time()

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", default=os.environ.get("R2S_SCENE", ""))
parser.add_argument("--lab", default=None, help="lab.json / lab yaml (default: /runs/<scene>/lab.json, else labs/robolab.yaml)")
parser.add_argument("--trials", type=int, default=5)
parser.add_argument("--no-scene", action="store_true", help="ground plane + table only (separates env bugs from scene bugs)")
parser.add_argument("--show-table", action="store_true", help="render the table collider even with a Scene")
parser.add_argument("--out", default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--min-success", type=float, default=0.6)
parser.add_argument("--no-video", action="store_true")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True

import numpy as np  # noqa: E402
import torch  # noqa: E402

from splat2sim.lab import ENV_ID, asset, geometry  # noqa: E402
from splat2sim.lab.env_cfg import Splat2SimFR3CubesEnvCfg  # noqa: E402
from splat2sim.lab.scripted import PickCfg, PickLift  # noqa: E402

RUN = Path("/runs") / args_cli.scene if args_cli.scene else Path("/tmp/r2s_run")
OUT = Path(args_cli.out) if args_cli.out else RUN / "env"
PREV = OUT / "preview"


def gpu_mem_mib() -> int:
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return int(o.stdout.strip().splitlines()[0])
    except Exception:
        free, total = torch.cuda.mem_get_info()
        return int((total - free) / 2**20)


def to_png(arr: np.ndarray, path: Path, depth: bool = False):
    from PIL import Image

    a = arr
    if depth:
        d = np.where(np.isfinite(a), a, 0.0)[..., 0]
        far = 2.0
        g = (255 * (1 - np.clip(d / far, 0, 1))).astype(np.uint8)
        g[d <= 0] = 0
        a = np.stack([g] * 3, -1)
    else:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8) if a.dtype != np.uint8 else a
    Image.fromarray(a[..., :3]).save(path)


def camera_check(u, cfg, o):
    """Predicted pixel of each cube centre (lab-config camera pose + intrinsics, via splat2sim.lab.geometry) vs the centroid of that
    cube's colour in the rendered RGB: verifies the pose/FOV conventions end to end."""
    from splat2sim.lab.env_cfg import CUBE_COLORS

    robot = u.scene["robot"]
    origin = u.scene.env_origins[0].cpu().numpy()
    cubes = o["policy"]["cube_pos"][0].cpu().numpy().reshape(-1, 3)
    out = {}
    for name in ("wrist", "front"):
        spec = getattr(cfg.lab.policy_cameras, name)
        bi = robot.find_bodies(geometry.camera_parent_link(spec.parent_frame))[0][0]
        ppos = robot.data.body_pos_w.torch[0, bi].cpu().numpy() - origin
        pquat = robot.data.body_quat_w.torch[0, bi].cpu().numpy()
        cpos, cquat = geometry.compose_pose(ppos, pquat, *geometry.camera_offset(spec))
        img = o["policy"][f"{name}_rgb"][0].cpu().numpy()
        rows = []
        for i, c in enumerate(cubes):
            pu, pv, depth = geometry.project_to_pixels(c, cpos, cquat, spec.intrinsics)
            m = geometry.color_centroid(img, CUBE_COLORS[i % len(CUBE_COLORS)])
            in_view = bool(depth > 0 and 0 <= pu < spec.intrinsics.width and 0 <= pv < spec.intrinsics.height)
            row = {"cube": i, "predicted_px": [round(pu, 1), round(pv, 1)], "depth_m": round(depth, 3), "in_view": in_view,
                   "measured_px": None if m is None else [round(m[0], 1), round(m[1], 1)], "pixels": None if m is None else m[2]}
            if m is not None and in_view:
                row["error_px"] = round(float(np.hypot(pu - m[0], pv - m[1])), 1)
            rows.append(row)
        out[name] = rows
    return out


def scene_safety(env, cfg, robot, picker):
    """At reset (home = scan pose): does the Scene's collision layer touch the robot? Settles 15 steps first and also reports how far
    the joints sit from their targets (PhysX depenetration pushes them away when a collider overlaps a link)."""
    from splat2sim.lab import safety

    try:
        for _ in range(15):
            picker._hold(0.0, "safety", None)
        u = env.unwrapped
        origin = u.scene.env_origins[0].cpu().numpy()
        pos = robot.data.body_pos_w.torch[0].cpu().numpy() - origin
        quat = robot.data.body_quat_w.torch[0].cpu().numpy()
        import omni.usd

        res = safety.check_robot_vs_collision(RUN, omni.usd.get_context().get_stage(), "/World/envs/env_0/Robot",
                                              list(robot.body_names), pos, quat)
        q = robot.data.joint_pos.torch[0, picker.arm_ids].cpu().numpy()
        home = np.array(list(cfg.lab.robot.scan_pose_joints))
        res["joint_error_from_scan_pose_rad_max"] = float(np.abs(q - home).max())
        for w in res.get("warnings", []):
            print(f"[env-check] SAFETY WARNING: {w}", flush=True)
        print(f"[env-check] scene safety: {res.get('status')} (mesh min distance {res.get('mesh_min_distance_m')})", flush=True)
        return res
    except Exception as e:  # the check must never break the pick-and-lift run
        import traceback

        traceback.print_exc()
        return {"status": f"error: {type(e).__name__}: {e}"}


def main():
    lab_path = args_cli.lab or (str(RUN / "lab.json") if (RUN / "lab.json").is_file() else str(Path("/opt/r2s/labs/robolab.yaml")))
    scene_usd = None if args_cli.no_scene else "auto"
    cfg = Splat2SimFR3CubesEnvCfg.from_lab(lab_path, RUN, scene_usd=scene_usd, overview_camera=True, seed=args_cli.seed,
                                          show_table=True if args_cli.show_table else None)
    PREV.mkdir(parents=True, exist_ok=True)
    for old in PREV.glob("*.png"):
        old.unlink()
    summary = {"env_id": ENV_ID, "scene": args_cli.scene, "lab": lab_path, "with_scene": cfg.scene_usd is not None,
               "scene_usd": cfg.scene_usd, "trials_requested": args_cli.trials, "seed": args_cli.seed,
               "table": {"center": cfg.table.center, "size": cfg.table.size, "yaw_deg": cfg.table.yaw_deg, "source": cfg.table.source,
                         "collider_added_by_env": cfg.add_table},
               "spawn_region": cfg.spawn_region.model_dump(), "cubes": cfg.lab.cubes.model_dump(),
               "coupling_offset": cfg.lab.robot.coupling_offset.model_dump(), "policy_cameras": cfg.lab.policy_cameras.model_dump()}
    timings = {}
    vram = [gpu_mem_mib()]
    summary["vram_idle_mib"] = vram[0]
    frames = []

    with launch_simulation(cfg, args_cli):
        import gymnasium as gym

        timings["app_start_s"] = round(time.time() - T_START, 1)
        t = time.time()
        usd = cfg.ensure_robot_asset()
        timings["asset_build_s"] = round(time.time() - t, 1)
        summary["robot_usd"] = str(usd)
        t = time.time()
        env = gym.make(ENV_ID, cfg=cfg)
        obs, _ = env.reset(seed=args_cli.seed)
        timings["env_create_and_first_reset_s"] = round(time.time() - t, 1)
        vram.append(gpu_mem_mib())
        u = env.unwrapped
        robot = u.scene["robot"]
        pol = obs["policy"]
        summary["obs_shapes"] = {k: list(v.shape) for k, v in pol.items()}
        summary["action_shape"] = list(env.action_space.shape)
        summary["joint_names"] = list(robot.joint_names)
        summary["body_count"] = len(robot.body_names)
        summary["step_dt_s"] = u.step_dt
        picker = PickLift(env, PickCfg())
        summary["scene_safety"] = scene_safety(env, cfg, robot, picker)
        summary["warnings"] = list(summary["scene_safety"].get("warnings", []))  # collision mesh vs robot at reset

        def snap(tag, o, include_overview=True):
            p = o["policy"]
            for cam in ("wrist", "front"):
                to_png(p[f"{cam}_rgb"][0].cpu().numpy(), PREV / f"{cam}_rgb_{tag}.png")
                to_png(p[f"{cam}_depth"][0].cpu().numpy(), PREV / f"{cam}_depth_{tag}.png", depth=True)
            ov = u.scene["overview_cam"].data.output["rgb"]
            ov = (ov.torch if hasattr(ov, "torch") else ov)[0, ..., :3].cpu().numpy()
            to_png(ov, PREV / f"overview_{tag}.png")

        results = []
        step_count = 0
        t_loop = time.time()
        for k in range(args_cli.trials):
            if k > 0:
                obs, _ = env.reset()
            cube = k % max(1, len(cfg.cube_names))
            if k == 0:
                for _ in range(10):  # let the first frames render (RTX warm-up) before the reset snapshot
                    obs = picker._hold(0.0, "warmup", None)
                snap("reset", obs)
                summary["camera_check"] = camera_check(u, cfg, obs)

            def on_step(phase, o):
                nonlocal step_count
                step_count += 1
                if step_count % 30 == 0:
                    vram.append(gpu_mem_mib())
                if k == 0 and not args_cli.no_video and step_count % 3 == 0:
                    p = o["policy"]
                    ov = u.scene["overview_cam"].data.output["rgb"]
                    ov = (ov.torch if hasattr(ov, "torch") else ov)[0, ..., :3].cpu().numpy()
                    frames.append((p["front_rgb"][0].cpu().numpy(), p["wrist_rgb"][0].cpu().numpy(), ov))

            # snapshot hooks keyed on phase transitions, trial 0 only
            last = {"phase": None, "o": None}

            def hook(phase, o):
                if k == 0 and last["phase"] is not None and phase != last["phase"]:
                    snap(last["phase"] + "_end", last["o"])
                last["phase"], last["o"] = phase, o
                on_step(phase, o)

            res = picker.run(obs, cube=cube, on_step=hook)
            if k == 0 and last["o"] is not None:
                snap(last["phase"] + "_end", last["o"])
            d = {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in vars(res).items() if kk != "trace"}
            results.append(d)
            print(f"[env-check] trial {k} cube {cube}: success={res.success} rise_in_hold_min={res.min_rise_in_hold_m:.3f} "
                  f"max={res.max_rise_m:.3f} tcp_dist_max={res.max_tcp_dist_in_hold_m:.3f} finger={res.finger_at_grasp_rad:.3f}", flush=True)
        timings["trials_wall_s"] = round(time.time() - t_loop, 1)
        timings["env_steps"] = step_count
        timings["env_steps_per_s"] = round(step_count / max(1e-9, time.time() - t_loop), 2)
        vram.append(gpu_mem_mib())
        env.close()

        n_ok = sum(r["success"] for r in results)
        summary.update({"trials": len(results), "successes": n_ok, "success_rate": n_ok / max(1, len(results)), "results": results,
                        "success_criterion": {"min_rise_m": picker.cfg.min_rise_m, "hold_s": picker.cfg.hold_s,
                                              "max_tcp_dist_m": picker.cfg.max_tcp_dist_m, "lift_m": picker.cfg.lift_m},
                        "timings": timings, "vram_peak_mib": max(vram), "vram_samples": len(vram),
                        "previews": sorted(p.name for p in PREV.glob("*.png"))})
        if frames and not args_cli.no_video:
            summary["video"] = write_video(frames)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        subprocess.run(["chmod", "-R", "a+rwX", str(OUT)], check=False)
        print(f"[env-check] success {n_ok}/{len(results)} -> {OUT / 'summary.json'}", flush=True)
        # launch_simulation closes Kit (and exits the process) when the block ends: leave through sys.exit inside it
        sys.exit(0 if summary["success_rate"] >= args_cli.min_success else 1)

    return 1


def write_video(frames):
    from PIL import Image

    def comp(f):
        h = 180
        ims = []
        for a in f:
            im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8) if a.dtype != np.uint8 else a[..., :3])
            ims.append(im.resize((int(im.width * h / im.height), h)))
        canvas = Image.new("RGB", (sum(i.width for i in ims), h))
        x = 0
        for i in ims:
            canvas.paste(i, (x, 0))
            x += i.width
        return canvas

    imgs = [comp(f) for f in frames]
    out = {}
    try:
        import imageio.v2 as iio

        iio.mimsave(OUT / "pick_lift.mp4", [np.asarray(i) for i in imgs], fps=10)
        out["mp4"] = "pick_lift.mp4"
    except Exception as e:  # imageio-ffmpeg is not in the Isaac image
        out["mp4_skipped"] = f"{type(e).__name__}: {str(e)[:80]}"
    imgs[0].save(OUT / "pick_lift.gif", save_all=True, append_images=imgs[1::2], duration=200, loop=0, optimize=True)
    out["gif"] = "pick_lift.gif"
    return out


if __name__ == "__main__":
    sys.exit(main())
