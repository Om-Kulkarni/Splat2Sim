"""FR3 + Robotiq 2F-85 composed into one articulation USD (``fr3_robotiq_2f85-<hash>.usda``).

NVIDIA ships ``FrankaPanda/franka.usd`` with a ``Robotiq_2F_85`` variant but no FR3 equivalent, so we compose it:

* ``/fr3``                          references ``Robots/FrankaRobotics/FrankaFR3/fr3.usd`` (articulation root, 7 arm DOF);
                                   the Franka hand (``fr3_hand``, fingers, tcp) and ``fr3_hand_joint`` are deactivated
* ``/fr3/Robotiq_2F_85_edit``       references ``Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd`` (``Physx_Mimic`` variant: ``finger_joint``
                                   driven, ``right_outer_knuckle_joint`` etc. mimic; the same referenced file the Panda variant uses),
                                   its nested articulation root is removed (same as NVIDIA's Panda variant)
* ``/fr3/fr3_link8/robotiq_coupling_joint``  PhysicsFixedJoint flange (``fr3_link8``) -> Robotiq ``base_link`` with the lab's
                                   ``robot.coupling_offset`` as the joint frame on the flange side

The asset only references remote assets (``ISAAC_NUCLEUS_DIR``, pinned to 6.1 by ``ISAACSIM_ASSET_ROOT``); the file name carries a
hash of coupling offset + source URLs, so it is rebuilt exactly when one of them changes. Building needs ``pxr`` (Kit running).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import Pose

FR3_REL = "Robots/FrankaRobotics/FrankaFR3/fr3.usd"
ROBOTIQ_REL = "Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"
ASSET_VERSION = 1  # bump when the composition recipe changes

ROOT = "/fr3"
ROBOTIQ_ROOT = f"{ROOT}/Robotiq_2F_85_edit"
BASE_LINK = f"{ROBOTIQ_ROOT}/Robotiq_2F_85/base_link"
FLANGE = f"{ROOT}/fr3_link8"
DEACTIVATE = ("fr3_hand", "fr3_leftfinger", "fr3_rightfinger", "fr3_hand_tcp", "fr3_link8/fr3_hand_joint")
AVOID_COLLISION_WITH = ("fr3_link6", "fr3_link7")  # arm links the coupling plate / gripper body may graze


def nucleus_dir() -> str:
    """Isaac assets root with the ``/Isaac`` suffix (pinned by ``ISAACSIM_ASSET_ROOT`` in compose.yaml)."""
    try:
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        return str(ISAAC_NUCLEUS_DIR)
    except Exception:  # not in the Isaac env (tests, host)
        import os

        root = os.environ.get("ISAACSIM_ASSET_ROOT", "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1")
        return root.rstrip("/") + "/Isaac"


def asset_key(coupling: Pose, nucl: str) -> str:
    blob = json.dumps({"v": ASSET_VERSION, "xyz": [round(v, 9) for v in coupling.xyz], "rpy": [round(v, 9) for v in coupling.rpy],
                       "fr3": f"{nucl}/{FR3_REL}", "robotiq": f"{nucl}/{ROBOTIQ_REL}"}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def asset_path(out_dir: Path | str, coupling: Pose, nucl: str | None = None) -> Path:
    return Path(out_dir) / f"fr3_robotiq_2f85-{asset_key(coupling, nucl or nucleus_dir())}.usda"


def quat_wxyz(pose: Pose) -> tuple[float, float, float, float]:
    x, y, z, w = pose.quat_xyzw()
    return (w, x, y, z)


def ensure_asset(out_dir: Path | str, coupling: Pose, nucl: str | None = None, force: bool = False) -> Path:
    """Return the path of the composed asset, building it first when missing (needs ``pxr``, i.e. Kit started)."""
    nucl = nucl or nucleus_dir()
    path = asset_path(out_dir, coupling, nucl)
    if force or not path.is_file():
        build_asset(path, coupling, nucl)
    return path


def build_asset(path: Path, coupling: Pose, nucl: str) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.usda")
    tmp.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(tmp))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = stage.DefinePrim(ROOT, "Xform")
    root.GetReferences().AddReference(f"{nucl}/{FR3_REL}")
    stage.SetDefaultPrim(root)

    # 1. drop the Franka hand (fingers, tcp) and the joint that attaches it to the flange
    for rel in DEACTIVATE:
        stage.OverridePrim(f"{ROOT}/{rel}").SetActive(False)

    # 2. Robotiq 2F-85, placed where the flange + coupling offset puts it (bodies must start coincident with the joint frames)
    rq = stage.DefinePrim(ROBOTIQ_ROOT, "Xform")
    rq.GetReferences().AddReference(f"{nucl}/{ROBOTIQ_REL}")
    T_c = Gf.Matrix4d(1.0)
    T_c.SetTransform(Gf.Rotation(Gf.Quatd(*quat_wxyz(coupling))), Gf.Vec3d(*coupling.xyz))
    M8 = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(FLANGE))
    xf = UsdGeom.Xformable(rq)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(T_c * M8)  # row-vector convention: local-to-world = offset * flange
    inner = stage.OverridePrim(f"{ROBOTIQ_ROOT}/Robotiq_2F_85")
    inner.RemoveAPI(UsdPhysics.ArticulationRootAPI)  # one articulation (/fr3) for the whole robot

    # 3. fixed joint flange -> Robotiq base_link
    j = UsdPhysics.FixedJoint.Define(stage, f"{FLANGE}/robotiq_coupling_joint")
    j.CreateBody0Rel().SetTargets([Sdf.Path(FLANGE)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(BASE_LINK)])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*coupling.xyz))
    j.CreateLocalRot0Attr().Set(Gf.Quatf(*quat_wxyz(coupling)))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    j.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    j.CreateCollisionEnabledAttr().Set(False)

    # 4. keep the gripper body from colliding with the wrist links it overlaps near the coupling plate
    fp = UsdPhysics.FilteredPairsAPI.Apply(stage.OverridePrim(BASE_LINK))
    fp.CreateFilteredPairsRel().SetTargets([Sdf.Path(f"{ROOT}/{n}") for n in AVOID_COLLISION_WITH])

    stage.GetRootLayer().customLayerData = {"r2s_json": json.dumps({
        "asset_version": ASSET_VERSION, "fr3": f"{nucl}/{FR3_REL}", "robotiq": f"{nucl}/{ROBOTIQ_REL}",
        "coupling_offset": {"xyz": list(coupling.xyz), "rpy_deg": list(coupling.rpy)}})}
    stage.GetRootLayer().Save()
    del stage
    tmp.replace(path)

    info = verify_asset(path, coupling)
    return info


def verify_asset(path: Path, coupling: Pose) -> dict:
    """Reopen the composed asset: one articulation root, no leftover hand, Robotiq base_link at flange * coupling."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics

    st = Usd.Stage.Open(str(path))
    roots = [p.GetPath().pathString for p in st.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    joints = [p.GetName() for p in st.Traverse() if p.IsA(UsdPhysics.Joint)]
    xc = UsdGeom.XformCache()
    m_fl = np.array(xc.GetLocalToWorldTransform(st.GetPrimAtPath(FLANGE))).T
    m_bl = np.array(xc.GetLocalToWorldTransform(st.GetPrimAtPath(BASE_LINK))).T
    want = m_fl @ coupling.matrix()
    err = float(np.abs(m_bl - want).max())
    info = {"articulation_roots": roots, "joints": joints, "base_link_pose_error": err}
    if roots != [ROOT]:
        raise RuntimeError(f"composed asset must have exactly one articulation root {ROOT}, got {roots}")
    if "fr3_hand" in " ".join(joints) or "fr3_finger" in " ".join(joints):
        raise RuntimeError(f"Franka hand joints still active in the composed asset: {joints}")
    if err > 1e-4:
        raise RuntimeError(f"Robotiq base_link is not at flange * coupling_offset (max abs error {err:.2e})")
    return info
