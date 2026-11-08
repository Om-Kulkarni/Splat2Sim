"""Visual-only copy of the composed FR3 + 2F-85 posed at physics link poses (Kit ``pxr`` required).

Under Isaac Lab 3 the env's own prims are neither updated in USD nor drawn by a Kit viewport/replicator render product, so
renders of the robot (synthetic Capture, URDF silhouette for the report) author this copy: the robot asset is referenced,
physics/joint schemas are stripped and every link gets the world pose that the physics engine reports.
"""

from __future__ import annotations

import numpy as np


def quat_xyzw_to_R(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def add_robot_visual_copy(stage, robot_usd: str, vis_root: str, body_names: list[str], body_pos: np.ndarray,
                          body_quat_xyzw: np.ndarray, label: str | None = "robot") -> int:
    """Reference ``robot_usd`` at ``vis_root`` and pose its links; returns the number of links placed."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdSemantics

    from ..recon.usdutil import np_to_usd_matrix

    rv = stage.DefinePrim(vis_root, "Xform")
    rv.GetReferences().AddReference(str(robot_usd))
    for p in list(Usd.PrimRange(rv)):
        if p.IsA(UsdPhysics.Joint):
            p.SetActive(False)
            continue
        for api in (UsdPhysics.ArticulationRootAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI,
                    UsdPhysics.MeshCollisionAPI, UsdPhysics.MassAPI):
            if p.HasAPI(api):
                p.RemoveAPI(api)
    by_name = {}
    for p in Usd.PrimRange(rv):
        if p.GetName() in body_names and p.GetName() not in by_name:
            by_name[p.GetName()] = p
    placed = 0
    for nm in sorted(by_name, key=lambda n: len(by_name[n].GetPath().pathString.split("/"))):
        i = body_names.index(nm)
        Mw = np.eye(4)
        Mw[:3, :3], Mw[:3, 3] = quat_xyzw_to_R(body_quat_xyzw[i]), body_pos[i]
        pr = by_name[nm]
        P = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(pr.GetParent())).T
        xf = UsdGeom.Xformable(pr)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(Gf.Matrix4d(*[v for r in np_to_usd_matrix(np.linalg.inv(P) @ Mw) for v in r]))
        placed += 1
    if label:
        UsdSemantics.LabelsAPI.Apply(rv, "class").GetLabelsAttr().Set([label])
    return placed
