"""Synthetic Capture of the FR3 + Robotiq 2F-85 workcell, rendered in Isaac Sim, with ground truth.

Run inside the Isaac Lab image (see tools/synth_capture/run.sh). Writes the ingest ``frames/`` contract
(``color/``, ``depth/`` uint16 mm aligned to colour, ``frames.csv``, ``intrinsics.json``, ``keyframes.txt``, ``meta.json``) plus
ground truth into ``--out`` (default ``/captures/synth-robot``):

  gt_poses.csv      camera-to-world (OpenCV camera), world = robot base frame, metres  (ingest reads it: ATE)
  gt/robot_mask/    uint8 PNG, 255 = robot pixel (semantic segmentation of the rendered robot incl. gripper)
  synth_gt.json     board_to_base, table / pedestal / cube / clutter boxes, scan pose, camera model, trajectory phases

World (robot base = origin, z up): FR3 + 2F-85 at the scan pose on a 0.30 m pedestal (top at z = 0), a table (top z = -0.20)
in front of it, the ChArUco board flat on the table (frame as in the lab config ``labs/synth.yaml``), 3 cubes, clutter, a
textured room. Camera: handheld-like walk (orbits at knee/chest/head height, a close pass over the table, dwell at the
front-camera viewpoint, return to the start) with D435-like depth noise.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/r2s")
T0 = time.time()

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", default="/captures/synth-robot")
ap.add_argument("--lab", default="/opt/r2s/labs/synth.yaml")
ap.add_argument("--frames", type=int, default=600)
ap.add_argument("--size", default="848x480")
ap.add_argument("--fx", type=float, default=None, help="focal length px (default: 609.5 * width / 848)")
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rt-subframes", type=int, default=4, help="app updates per frame after moving the camera")
add_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
args.headless = True

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import gaussian_filter, gaussian_filter1d  # noqa: E402

from splat2sim.config import Box, Overrides, SpawnRegion, load_lab  # noqa: E402
from splat2sim.lab import ENV_ID  # noqa: E402
from splat2sim.lab.env_cfg import Splat2SimFR3CubesEnvCfg  # noqa: E402
from splat2sim.recon import usdutil  # noqa: E402
from synth_world import (BOARD_SHEET_MARGIN_M, FLOOR_Z, PEDESTAL, ROOM, TABLE, camera_path, clutter_specs, make_texture,  # noqa: E402
                         board_texture)

W, H = (int(v) for v in args.size.split("x"))
FX = args.fx or 609.5 * W / 848.0
OUT = Path(args.out)


def main():
    rng = np.random.default_rng(args.seed)
    lab = load_lab(args.lab)
    gen_lab = lab.model_copy(update={
        "overrides": Overrides(table_box=Box(center=TABLE["center"], size=TABLE["size"])),
        "cubes": lab.cubes.model_copy(update={"spawn_region": SpawnRegion(x_range=(0.32, 0.62), y_range=(0.08, 0.34), z=TABLE["top_z"])})})
    cfg = Splat2SimFR3CubesEnvCfg.from_lab(gen_lab, run_dir=Path("/tmp/synth_env"), scene_usd=None, table_collider="always",
                                          show_table=False, seed=args.seed)
    cfg.sim.use_fabric = False  # keep the USD stage in sync with physics so replicator renders the settled robot
    cfg.scene.ground.init_state.pos = (0.0, 0.0, FLOOR_Z)
    for d in ("color", "depth", "gt/robot_mask", "assets"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    for d in (OUT, OUT / "color", OUT / "depth", OUT / "gt", OUT / "gt/robot_mask", OUT / "assets"):
        d.chmod(0o777)  # the container runs as root: keep the output deletable by the host user even if the run fails

    with launch_simulation(cfg, args):
        import gymnasium as gym
        import omni.replicator.core as rep
        import omni.kit.app
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

        cfg.ensure_robot_asset()
        env = gym.make(ENV_ID, cfg=cfg)
        env.reset(seed=args.seed)
        u = env.unwrapped
        scan = torch.tensor([list(lab.robot.scan_pose_joints) + [0.0]], dtype=torch.float32, device=u.device)
        for _ in range(60):
            env.step(scan)  # hold the scan pose; cubes settle on the table
        cubes = {n: u.scene[n].data.root_pos_w.torch[0].cpu().numpy().tolist() for n in cfg.cube_names}
        print("[synth] cubes:", cubes, flush=True)
        kit_app = omni.kit.app.get_app()
        stage = omni.usd.get_context().get_stage()

        # ------------------------------------------------------------ stage dressing
        def mat(path, tex=None, color=(0.7, 0.7, 0.7), rough=0.7, metallic=0.0, uv_scale=1.0):
            m = UsdShade.Material.Define(stage, path)
            sh = UsdShade.Shader.Define(stage, path + "/Shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
            if tex is not None:
                rd = UsdShade.Shader.Define(stage, path + "/st")
                rd.CreateIdAttr("UsdPrimvarReader_float2")
                rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)
                t = UsdShade.Shader.Define(stage, path + "/tex")
                t.CreateIdAttr("UsdUVTexture")
                t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(tex))
                t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
                t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(t.ConnectableAPI(), "rgb")
            else:
                sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            return m

        def mesh(path, pts, faces, uvs, material):
            mg = UsdGeom.Mesh.Define(stage, path)
            mg.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
            mg.CreateFaceVertexCountsAttr([len(f) for f in faces])
            mg.CreateFaceVertexIndicesAttr([int(i) for f in faces for i in f])
            mg.CreateDoubleSidedAttr(True)
            mg.CreateSubdivisionSchemeAttr("none")
            pv = UsdGeom.PrimvarsAPI(mg).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
            pv.Set(Vt.Vec2fArray([Gf.Vec2f(*map(float, uv)) for uv in uvs]))
            UsdShade.MaterialBindingAPI.Apply(mg.GetPrim()).Bind(material)
            return mg

        def quad(path, c0, c1, c2, c3, material, uv_rep=(1.0, 1.0)):
            uv = [(0, 0), (uv_rep[0], 0), (uv_rep[0], uv_rep[1]), (0, uv_rep[1])]
            return mesh(path, [c0, c1, c2, c3], [[0, 1, 2, 3]], uv, material)

        def box(path, center, size, material, tile=0.5):
            cx, cy, cz = center
            hx, hy, hz = (s / 2 for s in size)
            P = {(i, j, k): (cx + (2 * i - 1) * hx, cy + (2 * j - 1) * hy, cz + (2 * k - 1) * hz) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
            faces = [[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
                     [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
                     [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)], [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]]
            pts, idx, uvs = [], [], []
            for f in faces:
                idx.append(list(range(len(pts), len(pts) + 4)))
                pts += [P[v] for v in f]
                uvs += [(0, 0), (max(size) / tile, 0), (max(size) / tile, max(size) / tile), (0, max(size) / tile)]
            return mesh(path, pts, idx, uvs, material)

        root = "/World/synth"
        UsdGeom.Xform.Define(stage, root)
        tex_dir = OUT / "assets"
        n_tex = [0]

        def tex(kind, size=1024, seed=None):
            n_tex[0] += 1
            p = tex_dir / f"tex_{n_tex[0]:02d}_{kind}.png"
            cv2.imwrite(str(p), cv2.cvtColor(make_texture(kind, size, rng if seed is None else np.random.default_rng(seed)), cv2.COLOR_RGB2BGR))
            return p

        def M(name, kind, **kw):
            return mat(f"{root}/mat_{name}", tex(kind), **kw)

        rz1 = ROOM["height_z"]
        x0, x1, y0, y1 = ROOM["x"][0], ROOM["x"][1], ROOM["y"][0], ROOM["y"][1]
        quad(f"{root}/floor", (x0, y0, FLOOR_Z + 0.002), (x1, y0, FLOOR_Z + 0.002), (x1, y1, FLOOR_Z + 0.002), (x0, y1, FLOOR_Z + 0.002),
             M("floor", "floor"), uv_rep=((x1 - x0) / 1.2, (y1 - y0) / 1.2))
        for i, (a, b) in enumerate([((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]):
            quad(f"{root}/wall{i}", (*a, FLOOR_Z), (*b, FLOOR_Z), (*b, rz1), (*a, rz1), M(f"wall{i}", "wall"), uv_rep=(1.0, 1.0))
        box(f"{root}/pedestal", PEDESTAL["center"], PEDESTAL["size"], M("pedestal", "metal"))
        T = TABLE
        box(f"{root}/table_top", T["center"], T["size"], M("table", "wood"))
        for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
            lx = T["center"][0] + sx * (T["size"][0] / 2 - 0.05)
            ly = T["center"][1] + sy * (T["size"][1] / 2 - 0.05)
            zt = T["center"][2] - T["size"][2] / 2
            box(f"{root}/table_leg{i}", (lx, ly, (zt + FLOOR_Z) / 2), (0.05, 0.05, zt - FLOOR_Z), M(f"leg{i}", "metal"))
        clutter = clutter_specs()
        for i, c in enumerate(clutter):
            box(f"{root}/clutter{i}", c["center"], c["size"], M(f"clutter{i}", c["tex"]))

        # ChArUco board: sheet = pattern + margin; board frame (origin corner 0, x along squares_x, y down the printed face, z into the board)
        import cv2 as _cv
        btex, sheet = board_texture(lab.charuco)
        bt = tex_dir / "board.png"
        _cv.imwrite(str(bt), btex)
        b2b = lab.board_to_base.matrix()
        sq = lab.charuco.square_length_m
        bx0, by0 = -sq - BOARD_SHEET_MARGIN_M, -sq - BOARD_SHEET_MARGIN_M
        bx1, by1 = bx0 + sheet[0], by0 + sheet[1]
        corners_b = np.array([[bx0, by0, 0.0], [bx1, by0, 0.0], [bx1, by1, 0.0], [bx0, by1, 0.0]])
        corners_w = corners_b @ b2b[:3, :3].T + b2b[:3, 3]
        bm = mat(f"{root}/mat_board", bt, rough=0.9)
        # texture row 0 is the top (smallest board y): USD v = 1 - row fraction
        mesh(f"{root}/board", corners_w, [[0, 1, 2, 3]], [(0, 1), (1, 1), (1, 0), (0, 0)], bm)

        # lights: dome from the env + a few area-like distant lights for shading
        for i, (rot, inten) in enumerate([((50, 0, 30), 1500), ((60, 0, 200), 900)]):
            dl = UsdLux.DistantLight.Define(stage, f"{root}/sun{i}")
            dl.CreateIntensityAttr(inten)
            dl.CreateAngleAttr(2.0)
            UsdGeom.Xformable(dl).AddRotateXYZOp().Set(Gf.Vec3f(*rot))

        # The env's own prims are not rendered by the Kit viewport under Isaac Lab 3 (and their USD xforms are stale), so author a
        # visual-only copy of the robot and the cubes at the physics poses (settled scan pose).
        from pxr import Usd, UsdPhysics

        from splat2sim.lab.visual import add_robot_visual_copy, quat_xyzw_to_R

        robot = u.scene["robot"]
        origin = u.scene.env_origins[0].cpu().numpy()
        bpos = robot.data.body_pos_w.torch[0].cpu().numpy() - origin
        bq = robot.data.body_quat_w.torch[0].cpu().numpy()  # xyzw
        names = list(robot.body_names)
        placed = add_robot_visual_copy(stage, cfg.robot_usd, f"{root}/robot_vis", names, bpos, bq)
        print(f"[synth] robot visual copy: {placed}/{len(names)} links placed", flush=True)
        from splat2sim.lab.env_cfg import CUBE_COLORS

        cs = lab.cubes.size_m
        for i, nme in enumerate(cfg.cube_names):
            cp = u.scene[nme].data.root_pos_w.torch[0].cpu().numpy() - origin
            cq = u.scene[nme].data.root_quat_w.torch[0].cpu().numpy()
            cm = box(f"{root}/{nme}", (0, 0, 0), (cs, cs, cs), mat(f"{root}/mat_{nme}", color=CUBE_COLORS[i % len(CUBE_COLORS)], rough=0.6))
            Mc = np.eye(4)
            Mc[:3, :3], Mc[:3, 3] = quat_xyzw_to_R(cq), cp
            cm.AddTransformOp().Set(Gf.Matrix4d(*[v for r in usdutil.np_to_usd_matrix(Mc) for v in r]))
        for _ in range(20):
            kit_app.update()

        # ------------------------------------------------------------ camera + annotators
        cam = UsdGeom.Camera.Define(stage, f"{root}/cam")
        ha = 20.955
        cam.GetHorizontalApertureAttr().Set(ha)
        cam.GetVerticalApertureAttr().Set(ha * H / W)
        cam.GetFocalLengthAttr().Set(FX * ha / W)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 30.0))
        op = cam.AddTransformOp()
        rp = rep.create.render_product(f"{root}/cam", (W, H))
        a_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        a_dep = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        a_seg = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
        for a in (a_rgb, a_dep, a_seg):
            a.attach([rp])

        # ------------------------------------------------------------ trajectory
        traj = camera_path(args.frames, rng, board_center_world=(b2b[:3, :3] @ np.array([3 * sq, 2 * sq, 0.0]) + b2b[:3, 3]))
        Ts = traj["T_world_cv"]
        t_ns0 = 1_800_000_000_000_000_000
        dt_ns = int(round(1e9 / args.fps))
        rows, poses, sharp, robot_frac = [], [], [], []
        F = np.diag([1.0, -1.0, -1.0, 1.0])
        t_render = time.time()
        for k in range(args.frames):
            T = Ts[k]
            op.Set(Gf.Matrix4d(*[v for r in usdutil.np_to_usd_matrix(T @ F) for v in r]))
            for _ in range(args.rt_subframes + (20 if k == 0 else 0)):
                kit_app.update()  # the camera moved: let the renderer converge (replicator's orchestrator.step cannot schedule under Isaac Lab)
            rgb = np.asarray(a_rgb.get_data())[..., :3]
            dep = np.asarray(a_dep.get_data(), np.float32).reshape(H, W)
            sg = a_seg.get_data()
            seg, info = sg["data"], sg["info"]["idToLabels"]
            rid = [int(i) for i, v in info.items() if (v.get("class") if isinstance(v, dict) else v) == "robot"]
            rmask = np.isin(np.asarray(seg).reshape(H, W), rid) if rid else np.zeros((H, W), bool)
            robot_frac.append(float(rmask.mean()))
            # --- sensor model: colour noise, depth noise / holes (D435-like), 16-bit mm
            col = np.clip(rgb.astype(np.float32) + rng.normal(0, 1.5, rgb.shape), 0, 255).astype(np.uint8)
            z = np.where(np.isfinite(dep) & (dep > 0.2) & (dep < 6.0), dep, 0.0)
            sig = 0.0005 + 0.0025 * z ** 2
            zn = z + rng.normal(0, 1, z.shape) * sig
            edge = cv2.dilate((np.abs(cv2.Laplacian(z, cv2.CV_32F)) > 0.15).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            zn[edge] = 0.0  # depth shadows / flying pixels at discontinuities are invalid on a real D435
            zn[rng.random(z.shape) < 0.01] = 0.0
            dmm = np.clip(np.round(zn * 1000), 0, 65535).astype(np.uint16)
            name = f"{k:06d}.png"
            cv2.imwrite(str(OUT / "color" / name), cv2.cvtColor(col, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(OUT / "depth" / name), dmm)
            cv2.imwrite(str(OUT / "gt/robot_mask" / name), (rmask * 255).astype(np.uint8))
            gray = cv2.cvtColor(col, cv2.COLOR_RGB2GRAY)
            sh = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            ts = t_ns0 + k * dt_ns
            rows.append([k, ts, f"color/{name}", f"depth/{name}", f"{sh:.2f}"])
            q = _quat(T[:3, :3])
            poses.append([ts, *T[:3, 3], *q])
            if k % 50 == 0:
                print(f"[synth] frame {k}/{args.frames}  {time.time() - t_render:.0f}s  robot px {robot_frac[-1]:.3f}", flush=True)

        # ------------------------------------------------------------ write the contract
        import csv

        with open(OUT / "frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "timestamp_ns", "color", "depth", "sharpness"])
            w.writerows(rows)
        with open(OUT / "gt_poses.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
            for p in poses:
                w.writerow([p[0], *(f"{v:.9g}" for v in p[1:])])
        (OUT / "keyframes.txt").write_text("".join(f"{i}\n" for i in range(args.frames)))
        (OUT / "intrinsics.json").write_text(json.dumps({
            "fx": FX, "fy": FX, "cx": W / 2 - 0.5, "cy": H / 2 - 0.5, "width": W, "height": H, "distortion_model": "plumb_bob",
            "distortion_coeffs": [0.0] * 5, "depth_scale_m": 0.001, "source": "synthetic (Isaac Sim pinhole, exact)"}, indent=2))
        gt = {
            "generator": "tools/synth_capture/generate.py", "seed": args.seed, "frames": args.frames, "size": [W, H], "fx": FX,
            "world": "robot base frame (FR3 base_link at origin), z up, metres",
            "board_to_base": {"xyz": list(lab.board_to_base.xyz), "rpy_deg": list(lab.board_to_base.rpy), "T": b2b.tolist()},
            "charuco": lab.charuco.model_dump(),
            "table_box": {"center": list(TABLE["center"]), "size": list(TABLE["size"]), "top_z": TABLE["top_z"]},
            "pedestal_box": {"center": list(PEDESTAL["center"]), "size": list(PEDESTAL["size"]), "top_z": PEDESTAL["center"][2] + PEDESTAL["size"][2] / 2},
            "floor_z": FLOOR_Z, "room": ROOM, "cubes": cubes, "cube_size_m": lab.cubes.size_m,
            "clutter": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()} for c in clutter],
            "scan_pose_joints": list(lab.robot.scan_pose_joints), "coupling_offset": lab.robot.coupling_offset.model_dump(),
            "phases": traj["phases"], "robot_pixel_fraction_mean": float(np.mean(robot_frac)),
            "render_seconds": round(time.time() - t_render, 1), "total_seconds": round(time.time() - T0, 1)}
        (OUT / "synth_gt.json").write_text(json.dumps(gt, indent=2))
        (OUT / "meta.json").write_text(json.dumps({"source_format": "synthetic", "generator": "tools/synth_capture"}, indent=2))
        import subprocess

        subprocess.run(["chmod", "-R", "a+rwX", str(OUT)], check=False)
        print(f"[synth] done: {args.frames} frames -> {OUT} in {time.time() - T0:.0f}s", flush=True)
        # preview contact sheet
        sel = np.linspace(0, args.frames - 1, 8).astype(int)
        tiles = [cv2.resize(cv2.imread(str(OUT / "color" / f"{i:06d}.png")), (W // 2, H // 2)) for i in sel]
        sheet_img = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
        cv2.imwrite(str(OUT / "preview_contact_sheet.jpg"), sheet_img)
        sys.exit(0)


def _quat(R):
    from splat2sim.recon.poseio import R_to_quat

    return R_to_quat(R)


if __name__ == "__main__":
    main()
