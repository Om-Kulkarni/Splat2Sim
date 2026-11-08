"""Scripted pick-and-lift with Isaac Lab's differential IK (imports torch + isaaclab).

The TCP is a point on the gripper (``tcp_offset`` along the Robotiq ``base_link`` z axis, between the finger pads). A trial moves it above a cube,
descends to the cube centre, closes ``finger_joint``, lifts ``lift_m`` and holds ``hold_s``. Success = the cube centre rose >= ``min_rise_m``
above its rest height for the *whole* hold and stayed within ``max_tcp_dist_m`` of the TCP. All commands are absolute joint targets, i.e. the
env's native action space; the IK runs outside the env.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

import isaaclab.utils.math as math_utils
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from .env_cfg import ARM_JOINTS, GRIPPER_JOINT
from .geometry import grasp_yaw

TCP_OFFSET = (0.0, 0.0, 0.135)  # in Robotiq base_link: mid finger pads (pads span z 0.091..0.148 m, measured on the composed asset)


@dataclass
class PickCfg:
    tcp_offset: tuple[float, float, float] = TCP_OFFSET
    pregrasp_height_m: float = 0.15  # TCP above the cube centre before descending
    grasp_dz_m: float = 0.0  # TCP z relative to cube centre at grasp
    lift_m: float = 0.15  # margin over min_rise_m: IK undershoots ~2 cm when the table sits low
    close_rad: float = 0.75  # finger_joint target when closing (limit 0.82); the cube stops the fingers earlier
    t_settle: float = 0.5
    t_approach: float = 3.0
    t_descend: float = 2.0
    t_close: float = 1.5
    t_lift: float = 2.5
    hold_s: float = 1.0
    min_rise_m: float = 0.08
    max_tcp_dist_m: float = 0.05
    open_rad: float = 0.0


@dataclass
class TrialResult:
    cube: int = 0
    cube_yaw_deg: float = 0.0
    rest_xy: list = field(default_factory=list)
    rest_z: float = 0.0
    max_rise_m: float = 0.0
    final_rise_m: float = 0.0
    min_rise_in_hold_m: float = 0.0
    max_tcp_dist_in_hold_m: float = 0.0
    finger_at_grasp_rad: float = 0.0
    success: bool = False
    sim_seconds: float = 0.0
    note: str = ""
    trace: list = field(default_factory=list, repr=False)


class PickLift:
    def __init__(self, env, cfg: PickCfg | None = None):
        self.env = env
        self.u = env.unwrapped
        self.cfg = cfg or PickCfg()
        self.robot = self.u.scene["robot"]
        self.dev = self.u.device
        self.arm_ids = self.robot.find_joints(ARM_JOINTS)[0]
        self.grip_id = self.robot.find_joints(GRIPPER_JOINT)[0][0]
        self.body_id = self.robot.find_bodies("base_link")[0][0]
        self.jac_body = self.body_id - 1 if self.robot.is_fixed_base else self.body_id
        self.ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
                                           num_envs=self.u.num_envs, device=self.dev)
        self.off = torch.tensor(self.cfg.tcp_offset, device=self.dev).repeat(self.u.num_envs, 1)
        self.dt = self.u.step_dt

    # --- kinematics -------------------------------------------------------------------------------------
    def tcp_pose_b(self):
        d = self.robot.data
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            d.root_pos_w.torch, d.root_quat_w.torch, d.body_pos_w.torch[:, self.body_id], d.body_quat_w.torch[:, self.body_id])
        ident = torch.zeros_like(quat_b); ident[:, 3] = 1.0
        return math_utils.combine_frame_transforms(pos_b, quat_b, self.off, ident)

    def _jacobian_b(self):
        d = self.robot.data
        j = d.body_link_jacobian_w.torch[:, self.jac_body][:, :, self.arm_ids].clone()
        rot = math_utils.matrix_from_quat(math_utils.quat_inv(d.root_quat_w.torch))
        j[:, :3] = torch.bmm(rot, j[:, :3])
        j[:, 3:] = torch.bmm(rot, j[:, 3:])
        body_quat_b = math_utils.quat_mul(math_utils.quat_inv(d.root_quat_w.torch), d.body_quat_w.torch[:, self.body_id])
        off_b = math_utils.quat_apply(body_quat_b, self.off)
        j[:, :3] += torch.linalg.cross(j[:, 3:], off_b[:, :, None], dim=1)
        return j

    def ik_targets(self, pos_b: torch.Tensor, quat_b: torch.Tensor) -> torch.Tensor:
        ee_pos, ee_quat = self.tcp_pose_b()
        self.ik.set_command(torch.cat([pos_b, quat_b], dim=-1), ee_pos, ee_quat)
        q = self.robot.data.joint_pos.torch[:, self.arm_ids]
        return self.ik.compute(ee_pos, ee_quat, self._jacobian_b(), q)

    @staticmethod
    def down_quat(yaw: float, device) -> torch.Tensor:
        """TCP orientation, z pointing down, x axis horizontal at ``yaw``; the gripper closes along TCP y = (sin yaw, -cos yaw, 0)."""
        c, s = math.cos(yaw), math.sin(yaw)
        m = torch.tensor([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]], device=device)  # columns x, y, z
        return math_utils.quat_from_matrix(m.unsqueeze(0))

    def home_yaw(self) -> float:
        _, q = self.tcp_pose_b()
        m = math_utils.matrix_from_quat(q)[0]
        return float(math.atan2(m[1, 0], m[0, 0]))

    # --- stepping -----------------------------------------------------------------------------------------
    def step(self, arm_q: torch.Tensor, finger: float):
        a = torch.cat([arm_q, torch.full((arm_q.shape[0], 1), finger, device=self.dev)], dim=-1)
        out = self.env.step(a)
        return out[0]

    def hold_action(self, finger: float):
        return self.robot.data.joint_pos.torch[:, self.arm_ids].clone(), finger

    # --- one trial ------------------------------------------------------------------------------------------
    def run(self, obs, cube: int = 0, on_step=None) -> TrialResult:
        """Run from the current (freshly reset) state. ``on_step(phase, obs)`` is called after every env step."""
        c = self.cfg
        res = TrialResult(cube=cube)
        cube_obj = self.u.scene[f"cube_{cube}"]

        # settle: hold the home pose, let the cubes come to rest, read the rest height
        for _ in range(int(c.t_settle / self.dt)):
            obs = self._hold(0.0, "settle", on_step)
        rest = cube_obj.data.root_pos_w.torch[0] - self.u.scene.env_origins[0]
        res.rest_z = float(rest[2])
        res.rest_xy = [round(float(rest[0]), 4), round(float(rest[1]), 4)]
        q = float(cube_obj.data.root_quat_w.torch[0, 2]), float(cube_obj.data.root_quat_w.torch[0, 3])
        cube_yaw = 2.0 * math.atan2(q[0], q[1])
        res.cube_yaw_deg = math.degrees(cube_yaw)
        gyaw = grasp_yaw(cube_yaw, around=self.home_yaw())
        quat = self.down_quat(gyaw, self.dev)
        p0, q0 = self.tcp_pose_b()
        cx, cy, cz = float(rest[0]), float(rest[1]), float(rest[2])
        pre = torch.tensor([[cx, cy, cz + c.pregrasp_height_m]], device=self.dev)
        grasp = torch.tensor([[cx, cy, cz + c.grasp_dz_m]], device=self.dev)
        lift = grasp + torch.tensor([[0.0, 0.0, c.lift_m]], device=self.dev)

        self._move(p0, q0, pre, quat, c.t_approach, c.open_rad, "approach", on_step)
        self._move(pre, quat, grasp, quat, c.t_descend, c.open_rad, "descend", on_step)
        n = max(1, int(c.t_close / self.dt))
        for i in range(n):  # close: ramp the finger target; arm holds the grasp pose through the IK
            self._servo(grasp, quat, c.open_rad + (c.close_rad - c.open_rad) * (i + 1) / n, "close", on_step)
        res.finger_at_grasp_rad = float(self.robot.data.joint_pos.torch[0, self.grip_id])
        self._move(grasp, quat, lift, quat, c.t_lift, c.close_rad, "lift", on_step)
        rises, dists = [], []
        for _ in range(max(1, int(c.hold_s / self.dt))):
            self._servo(lift, quat, c.close_rad, "hold", on_step)
            cp = cube_obj.data.root_pos_w.torch[0] - self.u.scene.env_origins[0]
            tp = self.tcp_pose_b()[0][0]
            rises.append(float(cp[2]) - res.rest_z)
            dists.append(float(torch.linalg.norm(cp - tp)))
        res.max_rise_m = max(rises)
        res.final_rise_m = rises[-1]
        res.min_rise_in_hold_m = min(rises)
        res.max_tcp_dist_in_hold_m = max(dists)
        res.success = res.min_rise_in_hold_m >= c.min_rise_m and res.max_tcp_dist_in_hold_m <= c.max_tcp_dist_m
        res.sim_seconds = float(self.u.episode_length_buf[0]) * self.dt
        return res

    # internals
    def _hold(self, finger, phase, on_step):
        arm, f = self.hold_action(finger)
        obs = self.step(arm, f)
        if on_step:
            on_step(phase, obs)
        return obs

    def _servo(self, pos, quat, finger, phase, on_step):
        arm = self.ik_targets(pos, quat)
        obs = self.step(arm, finger)
        if on_step:
            on_step(phase, obs)
        return obs

    def _move(self, p_a, q_a, p_b, q_b, dur, finger, phase, on_step):
        n = max(1, int(dur / self.dt))
        for i in range(n):
            t = (i + 1) / n
            s = t * t * (3 - 2 * t)  # smoothstep
            pos = p_a + (p_b - p_a) * s
            quat = _slerp(q_a, q_b, s)
            self._servo(pos, quat, finger, phase, on_step)



def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    """xyzw quaternion slerp (shortest path), batch of 1."""
    d = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(d < 0, -q1, q1)
    d = d.abs().clamp(max=1.0)
    th = torch.acos(d)
    if float(th) < 1e-6:
        return q0
    s = torch.sin(th)
    return (torch.sin((1 - t) * th) / s) * q0 + (torch.sin(t * th) / s) * q1
