"""Robot clearance volume for the collision mesh (numpy only).

With a ChArUco anchor the robot base is the origin of the anchor frame. Mask leaks, dilation borders, the pedestal's
cables etc. leave mesh fragments around the arm, which would collide with the real robot in the Task environment. Triangles
with any vertex inside the clearance volume are removed: a vertical cylinder around the base axis (``radius_m``,
from the pedestal top + ``z_margin_m`` up to ``z_max_m``) plus, optionally, capsules around the arm and gripper at the
scan pose (``robot.scan_pose_joints``). The table/pedestal boxes are not affected.
"""

from __future__ import annotations

import numpy as np

from .. import robot_fk

CLEARANCE_DEFAULTS = {
    "clearance": True,             # only applied with a ChArUco anchor (robot base known)
    "clearance_radius_m": 0.25,
    "clearance_z_margin_m": 0.02,  # cylinder starts this far above the pedestal top (base frame z of the pedestal box top, else 0)
    "clearance_z_max_m": 1.2,
    "clearance_arm": True,         # + capsules around the arm at the scan pose
    "clearance_arm_radius_m": 0.10,
    "clearance_gripper_length_m": 0.22,
}


def clearance_mask(Va: np.ndarray, o: dict, scan_joints, pedestal_top_z: float | None) -> tuple[np.ndarray, dict]:
    """Vertices (anchor frame) inside the clearance volume -> (mask, info)."""
    z0 = (pedestal_top_z if pedestal_top_z is not None else 0.0) + float(o["clearance_z_margin_m"])
    z1 = float(o["clearance_z_max_m"])
    cyl = (np.linalg.norm(Va[:, :2], axis=1) < float(o["clearance_radius_m"])) & (Va[:, 2] > z0) & (Va[:, 2] < z1)
    mask = cyl.copy()
    info = {"cylinder": {"radius_m": float(o["clearance_radius_m"]), "z_min_m": z0, "z_max_m": z1}, "vertices_cylinder": int(cyl.sum())}
    if o["clearance_arm"] and scan_joints is not None:
        caps = robot_fk.arm_capsules(scan_joints, arm_radius=float(o["clearance_arm_radius_m"]),
                                     gripper_radius=float(o["clearance_arm_radius_m"]),
                                     gripper_length=float(o["clearance_gripper_length_m"]))
        arm = robot_fk.points_in_capsules(Va, caps)
        # the table is below the arm's lowest capsule surface in any sane setup; never let a capsule eat the pedestal top
        arm &= Va[:, 2] > z0 - 0.02
        mask |= arm
        info["vertices_arm"] = int(arm.sum())
        info["arm_capsules"] = [{"p0": a.tolist(), "p1": b.tolist(), "radius_m": r} for a, b, r in caps]
    return mask, info


def carve_clearance(mesh, T_anchor: np.ndarray, o: dict, scan_joints, pedestal_top_z: float | None) -> dict:
    """Remove triangles of an Open3D mesh (recon frame) that touch the clearance volume. Returns what was removed."""
    V = np.asarray(mesh.vertices)
    Va = V @ T_anchor[:3, :3].T + T_anchor[:3, 3]
    mask, info = clearance_mask(Va, o, scan_joints, pedestal_top_z)
    tri = np.asarray(mesh.triangles)
    drop = mask[tri].any(axis=1)
    removed = Va[tri[drop]].reshape(-1, 3) if drop.any() else np.zeros((0, 3))
    mesh.remove_triangles_by_mask(drop)
    mesh.remove_unreferenced_vertices()
    info.update({"triangles_removed": int(drop.sum()), "triangles_before": int(len(tri)),
                 "removed_bbox_min": removed.min(0).tolist() if len(removed) else None,
                 "removed_bbox_max": removed.max(0).tolist() if len(removed) else None})
    return info
