"""Robot vs collision-layer safety check at reset (env stage).

The collision mesh (``collision/collision_mesh.usd``, recon frame) is mapped into the Scene frame with the anchor transform that
the collision stage stored in ``collision.usda``; the robot's collision geometry is sampled in each link frame (USD, needs ``pxr``)
and placed with the current link poses. Distances use a KD-tree on mesh points (vertices + triangle centroids, ~1 cm spacing, so
distances are accurate to ~5 mm) and are unsigned: ``min_distance_m`` below ``contact_m`` means contact or penetration.
The table box is checked too (all links except the base link, which legitimately stands on the pedestal).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .visual import quat_xyzw_to_R

CONTACT_M = 0.006   # below this: contact / penetration
NEAR_M = 0.03       # below this: near-contact (reported, not a warning by itself)
BASE_PLATE_M = 0.04  # base-link points below this height (base frame) are the mounting plate resting on the pedestal


def load_collision_mesh_world(run: Path) -> tuple[np.ndarray, int] | None:
    """Sample points (N,3) of the collision mesh in the Scene frame (anchor applied), and the triangle count."""
    from pxr import Usd, UsdGeom

    layer = run / "collision" / "collision.usda"
    mesh_usd = run / "collision" / "collision_mesh.usd"
    if not layer.is_file() or not mesh_usd.is_file():
        return None
    st = Usd.Stage.Open(str(layer))
    meta = json.loads((st.GetRootLayer().customLayerData or {}).get("r2s_collision", "{}"))
    T = np.array(meta.get("T_anchor_recon", np.eye(4).tolist()), float).reshape(4, 4)
    ms = Usd.Stage.Open(str(mesh_usd))
    prim = ms.GetDefaultPrim()
    mesh = UsdGeom.Mesh(prim)
    V = np.array(mesh.GetPointsAttr().Get(), float)
    F = np.array(mesh.GetFaceVertexIndicesAttr().Get(), int).reshape(-1, 3)
    P = np.vstack([V, V[F].mean(axis=1)])
    return P @ T[:3, :3].T + T[:3, 3], len(F)


def robot_collision_samples(stage, robot_root: str, body_names: list[str], max_pts_per_prim: int = 3000):
    """{link: (M,3) points in the link frame}: vertices (+ centroids) of every collision-enabled Mesh under the robot root and a
    5x5x5 grid of the bounding box of collision-enabled primitive shapes. Links without any collision-enabled geometry (the
    Robotiq's collisions are authored differently from the arm's) fall back to all their visual meshes. Instance proxies included."""
    from pxr import Usd, UsdGeom, UsdPhysics

    xc = UsdGeom.XformCache()
    root = stage.GetPrimAtPath(robot_root)
    pred = Usd.TraverseInstanceProxies()
    link_prims = {}
    for p in Usd.PrimRange(root, pred):
        if p.GetName() in body_names and p.GetName() not in link_prims:
            link_prims[p.GetName()] = p
    coll: dict[str, list[np.ndarray]] = {}
    vis: dict[str, list[np.ndarray]] = {}
    for p in Usd.PrimRange(root, pred):
        if not p.IsA(UsdGeom.Gprim):
            continue
        anc, link = p, None
        while anc and anc.GetPath() != root.GetPath():
            if anc.GetName() in link_prims and anc.GetPath() == link_prims[anc.GetName()].GetPath():
                link = anc.GetName()
                break
            anc = anc.GetParent()
        if link is None:
            continue
        M_p = np.array(xc.GetLocalToWorldTransform(p)).T
        M_l = np.array(xc.GetLocalToWorldTransform(link_prims[link])).T
        rel = np.linalg.inv(M_l) @ M_p
        if p.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(p)
            pts = m.GetPointsAttr().Get()
            if pts is None or len(pts) == 0:
                continue
            V = np.array(pts, float)
            fv, fc = m.GetFaceVertexIndicesAttr().Get(), m.GetFaceVertexCountsAttr().Get()
            if fv is not None and fc is not None and len(fc) and all(c == 3 for c in fc):
                F = np.array(fv, int).reshape(-1, 3)
                V = np.vstack([V, V[F].mean(axis=1)])
        else:
            b = UsdGeom.Gprim(p).ComputeLocalBound(0, "default").ComputeAlignedRange()
            lo, hi = np.array(b.GetMin(), float), np.array(b.GetMax(), float)
            V = np.stack(np.meshgrid(*[np.linspace(lo[i], hi[i], 5) for i in range(3)]), -1).reshape(-1, 3)
        if len(V) > max_pts_per_prim:
            V = V[np.random.default_rng(0).choice(len(V), max_pts_per_prim, replace=False)]
        Vl = V @ rel[:3, :3].T + rel[:3, 3]
        (coll if p.HasAPI(UsdPhysics.CollisionAPI) else vis).setdefault(link, []).append(Vl)
    out = {k: np.vstack(v) for k, v in vis.items()}
    out.update({k: np.vstack(v) for k, v in coll.items()})  # collision geometry wins where it exists
    return out


def signed_box_distance(points: np.ndarray, box: dict) -> np.ndarray:
    """Signed distance of points to an oriented box (negative inside = penetration depth)."""
    from ..collision.fitting import rpy_to_R

    q = (points - np.asarray(box["center"])) @ rpy_to_R(box.get("rpy_deg", [0, 0, box.get("yaw_deg", 0.0)]))
    d = np.abs(q) - 0.5 * np.asarray(box["size"])
    return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)


def check_robot_vs_collision(run: Path, stage, robot_root: str, body_names: list[str], body_pos: np.ndarray,
                             body_quat_xyzw: np.ndarray, contact_m: float = CONTACT_M, near_m: float = NEAR_M) -> dict:
    """``body_pos`` (B,3) / ``body_quat_xyzw`` (B,4): link poses in the Scene frame (env origin removed), order of ``body_names``."""
    from scipy.spatial import cKDTree

    mesh = load_collision_mesh_world(Path(run))
    samples = robot_collision_samples(stage, robot_root, body_names)
    res: dict = {"contact_threshold_m": contact_m, "near_threshold_m": near_m, "links_sampled": len(samples),
                 "robot_points": int(sum(len(v) for v in samples.values()))}
    if not samples:
        res["status"] = "skipped: no collision geometry found under the robot prim"
        return res
    world = {}
    for link, P in samples.items():
        i = body_names.index(link)
        W = P @ quat_xyzw_to_R(body_quat_xyzw[i]).T + body_pos[i]
        if link == body_names[0]:
            W = W[W[:, 2] > BASE_PLATE_M]  # the base link stands on the pedestal: ignore its mounting plate
        if len(W):
            world[link] = W
    per_link: dict[str, dict] = {}
    tree, n_tri = (cKDTree(mesh[0]), mesh[1]) if mesh else (None, 0)
    res["mesh_triangles"] = n_tri
    if tree is None:
        res["mesh"] = "no collision mesh in the run (boxes only)"
    for link, W in world.items():
        e: dict = {}
        if tree is not None:
            d, _ = tree.query(W)
            e["mesh_min_distance_m"] = float(d.min())
            e["mesh_points_within_contact"] = int((d < contact_m).sum())
        per_link[link] = e
    prim_path = Path(run) / "collision" / "primitives.json"
    if prim_path.is_file():
        prim = json.loads(prim_path.read_text())
        tb = prim.get("table_box")
        if tb and "center" in tb:
            for link, W in world.items():
                if link == body_names[0]:
                    continue
                sd = signed_box_distance(W, tb)
                per_link[link]["table_min_signed_distance_m"] = float(sd.min())
    res["per_link"] = per_link
    mesh_d = {k: v["mesh_min_distance_m"] for k, v in per_link.items() if "mesh_min_distance_m" in v}
    tab_d = {k: v["table_min_signed_distance_m"] for k, v in per_link.items() if "table_min_signed_distance_m" in v}
    res["mesh_min_distance_m"] = min(mesh_d.values()) if mesh_d else None
    res["table_min_signed_distance_m"] = min(tab_d.values()) if tab_d else None
    res["links_in_contact_with_mesh"] = sorted(k for k, v in mesh_d.items() if v < contact_m)
    res["links_near_mesh"] = sorted(k for k, v in mesh_d.items() if contact_m <= v < near_m)
    res["links_in_contact_with_table"] = sorted(k for k, v in tab_d.items() if v < contact_m)
    warns = []
    if res["links_in_contact_with_mesh"]:
        warns.append(f"collision mesh touches/penetrates robot link(s) {', '.join(res['links_in_contact_with_mesh'])} at reset "
                     f"(min distance {res['mesh_min_distance_m'] * 1000:.1f} mm): fragments near the arm; raise collision.clearance_* or fix the mask")
    if res["links_in_contact_with_table"]:
        warns.append(f"table box touches/penetrates robot link(s) {', '.join(res['links_in_contact_with_table'])} at reset "
                     f"(signed distance {res['table_min_signed_distance_m'] * 1000:.1f} mm): check board_to_base / table fit")
    res["warnings"] = warns
    res["status"] = "warning" if warns else "ok"
    return res
