"""``collision`` stage (collision container): masked depth + poses -> ``collision/``.

Outputs (``/runs/<scene>/collision/``):
  tsdf_mesh.ply         full TSDF surface (recon frame)
  collision_mesh.ply    table-top carved out, small islands removed, decimated (recon frame)
  collision_mesh.usd    the same mesh as a UsdGeom.Mesh
  primitives.json       table box + pedestal box (anchor frame = world): center, size, rpy/yaw, source, fit stats
  collision.usda        layer referenced by compose at /World/Anchor/Collision: mesh + boxes as invisible static
                        colliders, authored in the **recon frame** so the Anchor transform moves them with the splat
  metrics.json          collision-vs-depth error (held-out frames), plane RMS, coverage, mesh stats, timings

Frames: the TSDF is integrated with the recon-frame poses (same frame as the splat). Fitting runs on the mesh mapped
into the anchor frame (z-up; ``poses/anchor.json`` when present else ``poses/fallback_anchor.json``, exactly the choice
compose makes), then the boxes are mapped back into the recon frame for the USD layer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..recon import usdutil
from ..recon.options import stage_options
from ..recon.poseio import read_frames_csv, read_poses_csv
from . import fitting
from .clearance import CLEARANCE_DEFAULTS, carve_clearance

COLLISION_DEFAULTS = {
    "voxel_m": 0.01,              # TSDF voxel edge
    "sdf_trunc_m": 0.04,          # TSDF truncation distance
    "depth_trunc_m": 2.5,         # depth beyond this is dropped (stereo depth noise grows with z^2; D455 <2 % at 4 m)
    "max_integrate_frames": 400,  # evenly subsampled from the posed frames (bounds time; map is static)
    "heldout_every": 8,           # every Nth posed frame is excluded from the TSDF and used to measure the error
    "heldout_max_frames": 40,
    "heldout_pixel_stride": 6,
    "target_triangles": 60000,    # decimation target for collision_mesh
    "min_component_triangles": 100,  # islands smaller than this (triangles) are dropped
    "plane_thresh_m": 0.015,
    "table_thickness_m": 0.04,
    "table_extent": "inscribed",  # inscribed (largest fully observed rectangle) | bounding (min-area rectangle)
    "table_level": False,         # False: table box follows the fitted plane tilt; True: roll = pitch = 0
    "table_z_range": None,        # [lo, hi] anchor-frame height window for the table-top plane; None = auto:
                                  # charuco anchor -> [-0.5, 0.3] around the robot base; fallback -> prefer an elevated plane
    "include_mesh": True,         # False: boxes only (e.g. scene not aligned to the robot, mesh would intrude on the arm)
    "carve_table": True,          # drop mesh triangles lying on the fitted table plane (the flat box replaces them)
    "carve_clutter_m": 0.12,    # also drop mesh within this height above the table top inside carve_clutter_radius_m of the base:
                                # the cubes / small objects lying on the table in the Capture (they stay in the splat) would
                                # otherwise overlap the cubes the env spawns. 0 = off
    "carve_clutter_radius_m": 0.8,
    "carve_tol_m": 0.025,       # removes triangles within [top-5 cm, top+tol] inside the box footprint
    "pedestal_search_radius_m": 0.45,
    **CLEARANCE_DEFAULTS,         # robot clearance volume (charuco anchor only); also settable in the lab config `collision:`
}


# ------------------------------------------------------------------ inputs
def collision_options(root: Path) -> dict:
    """defaults < lab config ``collision:`` section (lab.json) < ``--set collision.*`` (options.yaml)."""
    lab = {}
    p = root / "lab.json"
    if p.is_file():
        lab = {k: v for k, v in (json.loads(p.read_text()).get("collision") or {}).items() if v is not None}
    run = stage_options(root, "collision", COLLISION_DEFAULTS)
    from ..recon.options import load_run_options

    explicit = load_run_options(root).get("collision") or {}
    return {**run, **{k: v for k, v in lab.items() if k not in explicit}}


def scan_joints(root: Path):
    p = root / "lab.json"
    return (json.loads(p.read_text()).get("robot") or {}).get("scan_pose_joints") if p.is_file() else None


def load_anchor(run: Path) -> tuple[np.ndarray, str, dict]:
    """T_anchor_recon, kind ("charuco" | "fallback"), info. Same choice as docker/isaac/scripts/compose_scene.py."""
    p = run / "poses" / "anchor.json"
    if p.is_file():
        a = json.loads(p.read_text())
        if a.get("status", "charuco") == "charuco" and "T_base_recon" in a:  # fallback/failed carry no transform
            T, kind = usdutil.anchor_transform(a)
            return T, kind, {"anchor_status": a.get("status")}
    fb = run / "poses" / "fallback_anchor.json"
    if fb.is_file():
        T, kind = usdutil.anchor_transform(json.loads(fb.read_text()))
        return T, "fallback", {}
    raise FileNotFoundError("neither poses/anchor.json nor poses/fallback_anchor.json exists; run the poses stage")


def load_overrides(run: Path) -> dict:
    p = run / "lab.json"
    if not p.is_file():
        return {}
    ov = json.loads(p.read_text()).get("overrides") or {}
    return {k: v for k, v in ov.items() if v}


class DepthSource:
    """Posed frames with (masked) depth in uint16 mm."""

    def __init__(self, run: Path):
        self.run = run
        idx, _, col, dep = read_frames_csv(run / "frames" / "frames.csv")
        self.by_index = {int(i): (c, d) for i, c, d in zip(idx, col, dep)}
        self.intr = json.loads((run / "frames" / "intrinsics.json").read_text())
        self.masks = (run / "masks").is_dir() and any((run / "masks").glob("*.png"))

    def depth_mm(self, index: int) -> np.ndarray | None:
        from PIL import Image

        if index not in self.by_index:
            return None
        c, d = self.by_index[index]
        depth = np.array(Image.open(self.run / "frames" / d))
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.uint16)
        if self.masks:
            mp = self.run / "masks" / (Path(c).stem + ".png")
            if mp.is_file():
                m = np.array(Image.open(mp).convert("L"))
                if m.shape != depth.shape:
                    m = np.array(Image.fromarray(m).resize(depth.shape[::-1], Image.NEAREST))
                depth = np.where(m > 127, depth, 0).astype(np.uint16)  # 255 = keep, 0 = robot
        return depth


def backproject_mm(depth_mm: np.ndarray, intr: dict, trunc_m: float, stride: int, scale_m: float) -> np.ndarray:
    z = depth_mm[::stride, ::stride].astype(np.float64) * scale_m
    vs, us = np.mgrid[0 : depth_mm.shape[0] : stride, 0 : depth_mm.shape[1] : stride]
    ok = (z > 0.15) & (z < trunc_m)
    z, u, v = z[ok], us[ok], vs[ok]
    return np.stack([(u - intr["cx"]) * z / intr["fx"], (v - intr["cy"]) * z / intr["fy"], z], axis=1)


# ------------------------------------------------------------------ mesh
def integrate_tsdf(src: DepthSource, Ts: dict[int, np.ndarray], frames: list[int], o: dict):
    import open3d as o3d

    intr = src.intr
    w, h = int(intr["width"]), int(intr["height"])
    pin = o3d.camera.PinholeCameraIntrinsic(w, h, intr["fx"], intr["fy"], intr["cx"], intr["cy"])
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(o["voxel_m"]), sdf_trunc=float(o["sdf_trunc_m"]),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    gray = o3d.geometry.Image(np.full((h, w, 3), 128, np.uint8))
    scale = 1.0 / float(intr.get("depth_scale_m", 0.001))
    used = 0
    for i in frames:
        d = src.depth_mm(i)
        if d is None or not (d > 0).any():
            continue
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            gray, o3d.geometry.Image(np.ascontiguousarray(d)), depth_scale=scale, depth_trunc=float(o["depth_trunc_m"]),
            convert_rgb_to_intensity=False)
        vol.integrate(rgbd, pin, np.linalg.inv(Ts[i]))
        used += 1
    return vol.extract_triangle_mesh(), used


def clean_mesh(mesh, min_tris: int):
    import open3d as o3d  # noqa: F401

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    cl, n_per, _ = mesh.cluster_connected_triangles()
    cl, n_per = np.asarray(cl), np.asarray(n_per)
    if len(n_per):
        small = np.where(n_per < max(min_tris, 1))[0]
        if len(small) < len(n_per):  # never drop everything
            mesh.remove_triangles_by_mask(np.isin(cl, small))
            mesh.remove_unreferenced_vertices()
    return mesh


def carve_plane(mesh, T_anchor: np.ndarray, table: dict, tol: float, margin: float = 0.02):
    """Remove triangles with all 3 vertices on the table top (box frame: within [-5 cm, +tol] of the top face) and inside
    its footprint: the flat box replaces them, so cubes cannot rest on mesh bumps."""
    V = np.asarray(mesh.vertices)
    Va = V @ T_anchor[:3, :3].T + T_anchor[:3, 3]
    q = (Va - np.asarray(table["center"])) @ fitting.rpy_to_R(table["rpy_deg"])
    h = 0.5 * table["size"][2]
    on = (q[:, 2] > h - 0.05) & (q[:, 2] < h + tol) & (np.abs(q[:, 0]) < 0.5 * table["size"][0] + margin) \
        & (np.abs(q[:, 1]) < 0.5 * table["size"][1] + margin)
    tri = np.asarray(mesh.triangles)
    drop = on[tri].all(axis=1)
    mesh.remove_triangles_by_mask(drop)
    mesh.remove_unreferenced_vertices()
    return int(drop.sum())


def carve_clutter(mesh, T_anchor: np.ndarray, table: dict, height: float, radius: float, margin: float = 0.02) -> int:
    """Remove triangles with any vertex above the table top (box frame z in (top - 5 cm, top + ``height``)) inside its footprint and
    within ``radius`` (horizontal, anchor frame) of the robot base axis."""
    V = np.asarray(mesh.vertices)
    Va = V @ T_anchor[:3, :3].T + T_anchor[:3, 3]
    q = (Va - np.asarray(table["center"])) @ fitting.rpy_to_R(table["rpy_deg"])
    h = 0.5 * table["size"][2]
    on = (q[:, 2] > h - 0.05) & (q[:, 2] < h + height) & (np.abs(q[:, 0]) < 0.5 * table["size"][0] + margin) \
        & (np.abs(q[:, 1]) < 0.5 * table["size"][1] + margin) & (np.linalg.norm(Va[:, :2], axis=1) < radius)
    drop = on[np.asarray(mesh.triangles)].any(axis=1)
    mesh.remove_triangles_by_mask(drop)
    mesh.remove_unreferenced_vertices()
    return int(drop.sum())


# ------------------------------------------------------------------ metrics
def depth_error(mesh, boxes: list[dict], src: DepthSource, Ts: dict[int, np.ndarray], frames: list[int],
                o: dict) -> dict:
    """Distance from held-out depth points (recon frame) to the collision geometry (mesh and boxes, in recon)."""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    if len(mesh.triangles):
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    pts = []
    for i in frames:
        d = src.depth_mm(i)
        if d is None:
            continue
        p = backproject_mm(d, src.intr, float(o["depth_trunc_m"]), int(o["heldout_pixel_stride"]),
                           float(src.intr.get("depth_scale_m", 0.001)))
        pts.append(p @ Ts[i][:3, :3].T + Ts[i][:3, 3])
    if not pts:
        return {}
    P = np.concatenate(pts)
    d_mesh = scene.compute_distance(o3d.core.Tensor(P.astype(np.float32))).numpy() if len(mesh.triangles) \
        else np.full(len(P), np.inf)
    d_all = d_mesh.copy()
    for b in boxes:  # boxes given in the recon frame via fitting.box_to_frame
        M = b["matrix"]
        local = (P - M[:3, 3]) @ b["R"]
        half = 0.5 * b["size"]
        dd = np.abs(local) - half
        dist = np.abs(np.linalg.norm(np.maximum(dd, 0), axis=1) + np.minimum(dd.max(axis=1), 0)) * b["scale"]
        d_all = np.minimum(d_all, dist)

    def stats(d):
        return {"median_m": float(np.median(d)), "p90_m": float(np.percentile(d, 90)), "p95_m": float(np.percentile(d, 95)),
                "mean_m": float(d.mean()), "within_2cm": float((d < 0.02).mean()), "within_5cm": float((d < 0.05).mean())}

    return {"heldout_frames": len(pts), "points": int(len(P)), "mesh_only_carved": stats(d_mesh), "collision": stats(d_all)}


# ------------------------------------------------------------------ USD
def write_usd(out: Path, mesh, boxes: dict[str, dict], meta: dict) -> None:
    """collision_mesh.usd (mesh geometry) and collision.usda (physics layer referencing it). Recon frame."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)

    def mesh_stage(path: str, root: str):
        st = Usd.Stage.CreateNew(path)
        UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(st, 1.0)
        m = UsdGeom.Mesh.Define(st, root)
        m.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in V])
        m.CreateFaceVertexCountsAttr([3] * len(F))
        m.CreateFaceVertexIndicesAttr([int(i) for i in F.reshape(-1)])
        m.CreateSubdivisionSchemeAttr("none")
        m.CreateDoubleSidedAttr(True)
        st.SetDefaultPrim(m.GetPrim())
        return st

    mp = out / "collision_mesh.usd"
    mp.unlink(missing_ok=True)
    if len(F):
        mst = mesh_stage(str(mp), "/Mesh")
        mst.GetRootLayer().Save()
        del mst

    lp = out / "collision.usda"
    lp.unlink(missing_ok=True)
    st = Usd.Stage.CreateNew(str(lp))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/Collision")
    st.SetDefaultPrim(root.GetPrim())
    root.GetPrim().CreateAttribute("r2s:frame", Sdf.ValueTypeNames.String).Set(
        "recon: place under /World/Anchor (same frame as the splat)")
    st.GetRootLayer().customLayerData = {"r2s_collision": json.dumps(meta)}

    def make_invisible_static_collider(prim):
        UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(prim)  # no RigidBodyAPI anywhere: static

    if len(F):
        mesh_prim = st.DefinePrim("/Collision/Mesh", "Mesh")
        mesh_prim.GetReferences().AddReference("./collision_mesh.usd", "/Mesh")
        make_invisible_static_collider(mesh_prim)
        UsdPhysics.MeshCollisionAPI.Apply(mesh_prim).CreateApproximationAttr("none")  # exact triangle mesh (static)
    for name, b in boxes.items():
        cube = UsdGeom.Cube.Define(st, f"/Collision/{name}")
        cube.CreateSizeAttr(1.0)
        cube.AddTransformOp().Set(Gf.Matrix4d(*[v for row in usdutil.np_to_usd_matrix(b["matrix"]) for v in row]))
        make_invisible_static_collider(cube.GetPrim())
        cube.GetPrim().CreateAttribute("r2s:source", Sdf.ValueTypeNames.String).Set(b["source"])
    st.GetRootLayer().Save()


# ------------------------------------------------------------------ driver
def run(root: Path) -> dict:
    t0 = time.time()
    import open3d as o3d

    o = collision_options(root)
    out = root / "collision"
    out.mkdir(parents=True, exist_ok=True)
    for f in list(out.glob("*.ply")) + list(out.glob("*.usd*")) + list(out.glob("*.json")):
        f.unlink()

    src = DepthSource(root)
    idx, _, Tw = read_poses_csv(root / "poses" / "poses.csv")
    Ts = {int(i): T for i, T in zip(idx, Tw)}
    posed = [int(i) for i in idx if int(i) in src.by_index]
    if not posed:
        raise ValueError("no frame has both a pose and a depth image")
    he = int(o["heldout_every"])
    held = [i for k, i in enumerate(posed) if he > 0 and k % he == he // 2]
    train = [i for i in posed if i not in set(held)]
    sel = np.unique(np.linspace(0, len(train) - 1, min(int(o["max_integrate_frames"]), len(train))).round().astype(int))
    train = [train[k] for k in sel]
    held = [held[k] for k in np.unique(np.linspace(0, len(held) - 1, min(int(o["heldout_max_frames"]), len(held))).round().astype(int))] if held else []
    T_anchor, kind, ainfo = load_anchor(root)
    print(f"[collision] anchor={kind} frames: integrate {len(train)}, held-out {len(held)}, masks={src.masks}", flush=True)

    mesh, used = integrate_tsdf(src, Ts, train, o)
    if len(mesh.triangles) == 0:
        raise RuntimeError("TSDF produced an empty mesh (no valid depth?)")
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out / "tsdf_mesh.ply"), mesh)
    n_tsdf = len(mesh.triangles)
    t_tsdf = time.time()
    print(f"[collision] TSDF mesh {n_tsdf} triangles from {used} frames in {t_tsdf - t0:.0f}s", flush=True)

    # --- fit primitives in the anchor frame (z-up, metres) ---
    R_a = T_anchor[:3, :3]
    s = float(np.cbrt(abs(np.linalg.det(R_a))))
    V = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    Va = V @ R_a.T + T_anchor[:3, 3]
    Na = N @ R_a.T / s
    ov = load_overrides(root)
    zr = tuple(o["table_z_range"]) if o["table_z_range"] else ((-0.5, 0.3) if kind == "charuco" else None)
    elevated = kind != "charuco" and not o["table_z_range"]
    Tc = [np.append(T[:3, 3], 1.0) for T in Ts.values()]  # camera centres (recon) -> anchor heights
    cam_z = np.array([(T_anchor @ c)[2] for c in Tc])
    z_max = float(np.median(cam_z) - 0.15)
    fkw = dict(thickness=float(o["table_thickness_m"]), thresh=float(o["plane_thresh_m"]), z_range=zr,
               extent=str(o["table_extent"]), level=bool(o["table_level"]), prefer_elevated=elevated, z_max=z_max)
    prim: dict[str, dict | None] = {}
    if "table_box" in ov:
        prim["table_box"] = fitting.override_box(ov["table_box"])
        # still measure how well the floating-point plane fit agrees, for the report
        fit = fitting.fit_table_box(Va, Na, **fkw)
        prim["table_box"]["fit_stats"] = fit["stats"] if fit else None
        prim["table_box"]["ransac_box"] = {k: fit[k] for k in ("center", "size", "yaw_deg")} if fit else None
    else:
        fit = fitting.fit_table_box(Va, Na, **fkw)
        prim["table_box"] = fit
    table = prim["table_box"]
    if table is None:
        print("[collision] WARNING: no horizontal table plane found", flush=True)
    table_top = table["stats"]["top_z"] if (table and table.get("stats")) else (table["center"][2] + 0.5 * table["size"][2] if table else None)

    if "pedestal_box" in ov:
        prim["pedestal_box"] = fitting.override_box(ov["pedestal_box"])
        reason = None
    elif kind == "charuco":
        ped = fitting.fit_pedestal_box(Va, Na, table_top, search_radius=float(o["pedestal_search_radius_m"]),
                                       thresh=float(o["plane_thresh_m"]))
        prim["pedestal_box"] = ped
        reason = None if ped else "no horizontal patch near the robot base in the mesh"
    else:
        prim["pedestal_box"] = None
        reason = "no robot base location with the fallback anchor (set overrides.pedestal_box)"

    # --- collision mesh: carve the table top (flat box replaces it), drop islands, decimate ---
    cmesh = o3d.geometry.TriangleMesh(mesh)
    carved = carve_plane(cmesh, T_anchor, table, float(o["carve_tol_m"])) if (table and o["carve_table"]) else 0
    carved_clutter = 0
    if table and o["carve_table"] and float(o["carve_clutter_m"]) > 0 and len(cmesh.triangles):
        carved_clutter = carve_clutter(cmesh, T_anchor, table, float(o["carve_clutter_m"]), float(o["carve_clutter_radius_m"]))
        print(f"[collision] clutter above the table removed from the collision mesh: {carved_clutter} triangles", flush=True)
    clear_info = None
    if o["clearance"] and o["include_mesh"] and kind == "charuco" and len(cmesh.triangles):
        ped_top = (prim["pedestal_box"] or {}).get("stats", {}).get("top_z") if prim.get("pedestal_box") else None
        if ped_top is None and prim.get("pedestal_box"):
            pb = prim["pedestal_box"]
            ped_top = pb["center"][2] + 0.5 * pb["size"][2]
        clear_info = carve_clearance(cmesh, T_anchor, o, scan_joints(root), ped_top)
        print(f"[collision] robot clearance removed {clear_info['triangles_removed']} triangles", flush=True)
    elif o["clearance"] and kind != "charuco":
        clear_info = {"skipped": "anchor is not charuco (robot base unknown)"}
    if not o["include_mesh"]:
        cmesh = o3d.geometry.TriangleMesh()
    if len(cmesh.triangles):
        cmesh = clean_mesh(cmesh, int(o["min_component_triangles"]))
    n_before = len(cmesh.triangles)
    if n_before > int(o["target_triangles"]):
        cmesh = cmesh.simplify_quadric_decimation(target_number_of_triangles=int(o["target_triangles"]))
        cmesh.remove_degenerate_triangles()
        cmesh.remove_unreferenced_vertices()
    if len(cmesh.triangles):
        cmesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out / "collision_mesh.ply"), cmesh)
    else:
        print("[collision] WARNING: nothing left of the mesh after carving the table top; boxes only", flush=True)

    # --- metrics ---
    boxes_recon = {}
    for name, key in (("TableBox", "table_box"), ("PedestalBox", "pedestal_box")):
        b = prim.get(key)
        if b:
            boxes_recon[name] = {**fitting.box_to_frame(b, T_anchor), "source": b["source"]}
    metrics = {}
    if held:
        metrics["depth_error"] = depth_error(cmesh, list(boxes_recon.values()), src, Ts, held, o)
    Vc = np.asarray(cmesh.vertices) @ R_a.T + T_anchor[:3, 3]
    bb_lo, bb_hi = (Vc.min(0), Vc.max(0)) if len(Vc) else (np.full(3, np.nan), np.full(3, np.nan))

    meta = {"anchor": kind, **ainfo, "T_anchor_recon": T_anchor.tolist(), "frame": "recon"}
    write_usd(out, cmesh, boxes_recon, meta)

    prim_json = {"frame": "anchor (world, z-up, m)", "anchor": kind, "table_box": prim["table_box"],
                 "pedestal_box": prim["pedestal_box"] or {"source": "absent", "reason": reason}}
    (out / "primitives.json").write_text(json.dumps(prim_json, indent=2, default=float))
    ts = (table or {}).get("fit_stats") or (table or {}).get("stats") or {}
    result = {
        "anchor": kind, "options": o,
        "frames": {"integrated_requested": len(train), "integrated": used, "heldout": len(held), "masks_used": src.masks},
        "tsdf_mesh": {"triangles": n_tsdf, "vertices": int(len(mesh.vertices))},
        "collision_mesh": {"triangles": len(cmesh.triangles), "vertices": len(cmesh.vertices), "carved_table_triangles": carved, "carved_clutter_triangles": carved_clutter,
                           "robot_clearance": clear_info,
                           "bbox_anchor_min": bb_lo.tolist(), "bbox_anchor_max": bb_hi.tolist()},
        "table_plane": {"rms_m": ts.get("plane_rms_m"), "level_rms_m": ts.get("level_rms_m"), "tilt_deg": ts.get("tilt_deg"),
                        "inliers": ts.get("inliers"), "top_z": ts.get("top_z"), "fill_ratio": ts.get("fill_ratio")},
        "table_box": {k: (prim["table_box"] or {}).get(k) for k in ("center", "size", "yaw_deg", "source")},
        "pedestal_box": {k: (prim["pedestal_box"] or {}).get(k) for k in ("center", "size", "yaw_deg", "source")}
        if prim["pedestal_box"] else {"source": "absent", "reason": reason},
        **metrics,
        "seconds": {"tsdf": round(t_tsdf - t0, 1), "total": round(time.time() - t0, 1)},
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps({k: result[k] for k in ("table_box", "pedestal_box", "table_plane", "depth_error") if k in result},
                     indent=1, default=float), flush=True)
    return result
