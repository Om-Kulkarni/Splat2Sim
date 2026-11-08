"""``Splat2Sim-FR3-Cubes-v0``: FR3 + Robotiq 2F-85 at the world origin, cubes on a table, two policy cameras,
optionally inside a reconstructed Scene (``scene.usda``). Manager-based Isaac Lab env, joint-position actions.

    from splat2sim.lab.env_cfg import Splat2SimFR3CubesEnvCfg
    cfg = Splat2SimFR3CubesEnvCfg.from_lab("robolab", run_dir="/runs/m2-demo")      # explicit
    env = gym.make("Splat2Sim-FR3-Cubes-v0", cfg=cfg)
    # or: gym.make("Splat2Sim-FR3-Cubes-v0")  -> lab from $R2S_LAB (yaml/json) or /runs/$R2S_SCENE/lab.json, run folder
    #                                           from $R2S_RUN_DIR or /runs/$R2S_SCENE

Action (8): absolute arm joint targets fr3_joint1..7 [rad] + ``finger_joint`` target [rad] (0 = open, ~0.7+ = closed on a cube).
Observations (``policy`` group, not concatenated): ``joint_pos`` (8), ``joint_vel`` (8), ``cube_pos`` (3*n), ``cube_quat`` (4*n, xyzw),
``wrist_rgb`` / ``front_rgb`` (H,W,3 float 0..1), ``wrist_depth`` / ``front_depth`` (H,W,1 metres).
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import clone, configclass

from ..config import LabConfig, load_lab
from . import ENV_ID, asset, geometry
from . import mdp as r2s_mdp

ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
GRIPPER_JOINT = "finger_joint"
CUBE_COLORS = [(0.85, 0.15, 0.12), (0.12, 0.45, 0.85), (0.95, 0.75, 0.10), (0.2, 0.7, 0.3), (0.6, 0.3, 0.7)]

_CTX: dict | None = None  # set by ``from_lab`` while the cfg is constructed


def robot_cfg(usd_path: str, home_joints: list[float]) -> ArticulationCfg:
    """Franka FR3 + Robotiq 2F-85, derived from Isaac Lab's ``FRANKA_ROBOTIQ_GRIPPER_CFG`` (same spawn props, gripper actuators).

    Deviations: composed USD (``asset.py``) instead of the Panda variant, ``fr3_joint*`` names, FR3 velocity limits
    (2.62 / 5.26 rad/s per the Franka spec), home pose from the lab config, base at the world origin.
    """
    from isaaclab_assets.robots.franka import FRANKA_ROBOTIQ_GRIPPER_CFG

    cfg: ArticulationCfg = clone(FRANKA_ROBOTIQ_GRIPPER_CFG)
    cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    cfg.spawn.usd_path = str(usd_path)
    cfg.spawn.variants = None
    cfg.init_state.pos = (0.0, 0.0, 0.0)
    cfg.init_state.joint_pos = {
        **{n: float(q) for n, q in zip(ARM_JOINTS, home_joints)},
        "finger_joint": 0.0,
        ".*_inner_finger_joint": 0.0,
        ".*_inner_finger_knuckle_joint": 0.0,
        ".*_outer_.*_joint": 0.0,
    }
    shoulder, forearm = cfg.actuators["panda_shoulder"], cfg.actuators["panda_forearm"]
    cfg.actuators["panda_shoulder"] = ImplicitActuatorCfg(
        joint_names_expr=["fr3_joint[1-4]"], joint_effort_limit=shoulder.joint_effort_limit, joint_velocity_limit=2.62,
        stiffness=shoulder.stiffness, damping=shoulder.damping)
    cfg.actuators["panda_forearm"] = ImplicitActuatorCfg(
        joint_names_expr=["fr3_joint[5-7]"], joint_effort_limit=forearm.joint_effort_limit, joint_velocity_limit=5.26,
        stiffness=forearm.stiffness, damping=forearm.damping)
    return cfg


def _camera_cfg(name: str, spec, parent_path: str, update_period: float = 0.0) -> CameraCfg:
    pos, rot = geometry.camera_offset(spec)
    f, ha, va = geometry.pinhole_aperture(spec.intrinsics)
    return CameraCfg(
        prim_path=f"{parent_path}/{name}_cam",
        update_period=update_period,
        update_latest_camera_pose=True,  # the wrist camera rides on the arm
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention=geometry.CAMERA_CONVENTION),
        data_types=["rgb", "distance_to_image_plane"],
        width=spec.intrinsics.width,
        height=spec.intrinsics.height,
        renderer_cfg=IsaacRtxRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(focal_length=f, horizontal_aperture=ha, vertical_aperture=va, clipping_range=(0.03, 10.0)),
    )


@configclass
class Splat2SimSceneCfg(InteractiveSceneCfg):
    """Scene entities are attached by :meth:`Splat2SimFR3CubesEnvCfg.configure` (cube count etc. come from the lab config)."""


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINTS, scale=1.0, use_default_offset=False)
    gripper_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[GRIPPER_JOINT], scale=1.0, use_default_offset=False)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS + [GRIPPER_JOINT])})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS + [GRIPPER_JOINT])})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False  # dict of tensors: states and images have different shapes

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.0)  # placeholder: this env defines no task reward


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


@configclass
class Splat2SimFR3CubesEnvCfg(ManagerBasedRLEnvCfg):
    scene: Splat2SimSceneCfg = Splat2SimSceneCfg(num_envs=1, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4  # 120 Hz physics, 30 Hz control
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        )
        ctx = _CTX if _CTX is not None else default_context()
        self.configure(**ctx)

    # ------------------------------------------------------------------------------------------------
    @classmethod
    def from_lab(cls, lab: LabConfig | str | Path, run_dir: str | Path | None = None, **kw) -> "Splat2SimFR3CubesEnvCfg":
        """Build the cfg for a lab config (object, name or yaml/json path) and an optional run folder.

        kwargs: ``num_envs`` (1), ``scene_usd`` ("auto" = ``<run>/scene/scene.usda`` if it exists, a path, or None for no Scene),
        ``asset_dir`` (default ``<run>/env/assets``), ``overview_camera`` (False), ``show_table`` (None = only without a Scene),
        ``table_collider`` ("auto": add our table collider unless the Scene carries colliders; "always"; "never"), ``seed``.
        """
        global _CTX
        _CTX = {"lab": lab, "run_dir": run_dir, **kw}
        try:
            return cls()
        finally:
            _CTX = None

    def configure(
        self,
        lab: LabConfig | str | Path,
        run_dir: str | Path | None = None,
        num_envs: int = 1,
        scene_usd: str | Path | None = "auto",
        asset_dir: str | Path | None = None,
        overview_camera: bool = False,
        show_table: bool | None = None,
        table_collider: str = "auto",
        seed: int | None = None,
    ) -> None:
        lab = _load_lab(lab)
        run_dir = Path(run_dir) if run_dir else None
        self.lab = lab
        self.run_dir = run_dir
        if seed is not None:
            self.seed = seed
        self.episode_length_s = lab.env.episode_length_s
        self.rerender_on_reset = True
        self.scene.num_envs = num_envs
        sc = self.scene

        # --- Scene (static background): splat + collision layer, referenced under each env ---
        if scene_usd == "auto":
            cand = run_dir / "scene" / "scene.usda" if run_dir else None
            scene_usd = cand if cand is not None and cand.is_file() else None
        self.scene_usd = str(scene_usd) if scene_usd else None
        if self.scene_usd:
            sc.background = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Scene", spawn=sim_utils.UsdFileCfg(usd_path=self.scene_usd))
        else:
            sc.ground = AssetBaseCfg(prim_path="/World/ground", init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
                                     spawn=sim_utils.GroundPlaneCfg(color=(0.35, 0.35, 0.37)))
        sc.light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.95, 0.95, 0.95)))

        # --- table: collider the cubes rest on (invisible when a splat shows the real one) ---
        self.table = geometry.read_table_box(lab, run_dir)
        self.add_table = table_collider == "always" or (table_collider == "auto" and not _scene_has_colliders(self.scene_usd))
        if self.add_table:
            visible = (self.scene_usd is None) if show_table is None else show_table
            t = self.table
            sc.table = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                init_state=AssetBaseCfg.InitialStateCfg(pos=t.center, rot=geometry.yaw_to_quat_xyzw(float(t.yaw_deg) * 3.141592653589793 / 180.0)),
                spawn=sim_utils.CuboidCfg(
                    size=t.size, visible=visible,
                    collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.30)),
                ),
            )

        # --- robot ---
        adir = Path(asset_dir) if asset_dir else (run_dir / "env" / "assets" if run_dir else Path(os.environ.get("R2S_ASSET_DIR", "/tmp/r2s_assets")))
        self.robot_usd = asset.asset_path(adir, lab.robot.coupling_offset)
        self.asset_dir = adir
        self.coupling_offset = lab.robot.coupling_offset
        sc.robot = robot_cfg(str(self.robot_usd), lab.robot.scan_pose_joints)

        # --- cubes ---
        c = lab.cubes
        self.cube_names = [f"cube_{i}" for i in range(c.count)]
        self.spawn_region = geometry.spawn_region(lab, self.table, lab.env.reach_min_m, lab.env.reach_max_m, lab.env.spawn_margin_m)
        for i, name in enumerate(self.cube_names):
            setattr(sc, name, RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.1 * (i - c.count / 2), self.spawn_region.z + c.size_m / 2)),
                spawn=sim_utils.CuboidCfg(
                    size=(c.size_m,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                                                                 max_depenetration_velocity=1.0),
                    mass_props=sim_utils.MassCfg(mass=c.mass_kg),
                    collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CUBE_COLORS[i % len(CUBE_COLORS)]),
                ),
            ))

        # --- cameras (policy): wrist on the flange link, front in the base frame ---
        for cam_name in ("wrist", "front"):
            spec = getattr(lab.policy_cameras, cam_name)
            parent = f"{{ENV_REGEX_NS}}/Robot/{geometry.camera_parent_link(spec.parent_frame)}"
            setattr(sc, f"{cam_name}_cam", _camera_cfg(cam_name, spec, parent))
        if overview_camera:  # not a policy input; for previews only (slow update)
            sc.overview_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/overview_cam", update_period=0.2, data_types=["rgb"], width=960, height=540,
                renderer_cfg=IsaacRtxRendererCfg(),
                offset=CameraCfg.OffsetCfg(pos=(1.5, -1.3, 1.1), rot=_look_at_quat((1.5, -1.3, 1.1), (0.35, 0.0, 0.1)), convention="world"),
                spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, horizontal_aperture=20.955, clipping_range=(0.05, 20.0)))

        # --- observations: state + cubes + images (policy group is a dict of tensors) ---
        pol = self.observations.policy
        pol.cube_pos = ObsTerm(func=r2s_mdp.cube_pos_b, params={"cube_names": self.cube_names})
        pol.cube_quat = ObsTerm(func=r2s_mdp.cube_quat_b, params={"cube_names": self.cube_names})
        for cam_name in ("wrist", "front"):
            setattr(pol, f"{cam_name}_rgb", ObsTerm(func=r2s_mdp.camera_image, params={"sensor_name": f"{cam_name}_cam", "data_type": "rgb"}))
            setattr(pol, f"{cam_name}_depth", ObsTerm(func=r2s_mdp.camera_image,
                                                       params={"sensor_name": f"{cam_name}_cam", "data_type": "distance_to_image_plane"}))

        # --- events ---
        self.events.reset_cubes = EventTerm(
            func=r2s_mdp.reset_cubes, mode="reset",
            params={"cube_names": self.cube_names, "region": self.spawn_region, "size": c.size_m})

    # -----------------------------------------------------------------------------------------------
    def ensure_robot_asset(self) -> Path:
        """Build the composed FR3 + 2F-85 USD if missing. Call after the app is launched (needs ``pxr``)."""
        return asset.ensure_asset(self.asset_dir, self.coupling_offset)


def _load_lab(lab: LabConfig | str | Path) -> LabConfig:
    """LabConfig, a lab name / yaml path (``splat2sim.config.load_lab``), or the ``lab.json`` the build writes into the run folder."""
    if isinstance(lab, LabConfig):
        return lab
    if str(lab).endswith(".json"):
        return LabConfig.model_validate(json.loads(Path(lab).read_text()))
    return load_lab(lab)


def _scene_has_colliders(scene_usd: str | None) -> bool:
    """True when ``scene.usda`` already carries static colliders under the Anchor's Collision scope (collision stage ran)."""
    if not scene_usd or not Path(scene_usd).is_file():
        return False
    # cheap textual check (no pxr needed): the compose stage references the collision layer only when it exists
    try:
        txt = Path(scene_usd).read_text(errors="ignore")
    except OSError:
        return False
    return "collision.usda" in txt or "PhysicsCollisionAPI" in txt


def _look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """xyzw quaternion of a camera in the 'world' convention (+x forward, +z up) at ``eye`` looking at ``target``."""
    import numpy as np

    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(up, f)
    r /= np.linalg.norm(r)  # +y (left) = up x forward
    u = np.cross(f, r)
    m = np.stack([f, r, u], axis=1)  # columns: x=f, y=left, z=up
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1.0) * 2
        v = [0.0] * 3
        v[i] = 0.25 * s
        v[j] = (m[j, i] + m[i, j]) / s
        v[k] = (m[k, i] + m[i, k]) / s
        q = (v[0], v[1], v[2], (m[k, j] - m[j, k]) / s)
    return tuple(float(x) for x in q)


def default_context() -> dict:
    """Lab config + run folder from the container contract / environment (used by ``gym.make`` without ``cfg=``)."""
    scene = os.environ.get("R2S_SCENE", "")
    run = Path(os.environ.get("R2S_RUN_DIR") or (Path("/runs") / scene if scene else ""))
    lab = os.environ.get("R2S_LAB")
    if not lab:
        lj = run / "lab.json"
        lab = str(lj) if lj.is_file() else "robolab"
    return {"lab": lab, "run_dir": run if scene or os.environ.get("R2S_RUN_DIR") else None}


def register() -> None:
    import gymnasium as gym

    if ENV_ID in gym.registry:
        return
    gym.register(
        id=ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}:Splat2SimFR3CubesEnvCfg"},
    )


with contextlib.suppress(Exception):  # importing the module registers the gym id
    register()
