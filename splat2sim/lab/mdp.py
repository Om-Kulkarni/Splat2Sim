"""Observation / event functions for ``Splat2Sim-FR3-Cubes-v0`` (imports torch + isaaclab)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ..config import SpawnRegion
from .geometry import sample_cube_poses, yaw_to_quat_xyzw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def cube_pos_b(env: ManagerBasedEnv, cube_names: list[str]) -> torch.Tensor:
    """Cube centre positions in the robot-base (= env) frame, shape (num_envs, 3 * n_cubes)."""
    return torch.cat([env.scene[n].data.root_pos_w.torch - env.scene.env_origins for n in cube_names], dim=-1)


def cube_quat_b(env: ManagerBasedEnv, cube_names: list[str]) -> torch.Tensor:
    """Cube orientations (x, y, z, w; env frame has the world axes), shape (num_envs, 4 * n_cubes)."""
    return torch.cat([env.scene[n].data.root_quat_w.torch for n in cube_names], dim=-1)


def camera_image(env: ManagerBasedEnv, sensor_name: str, data_type: str = "rgb") -> torch.Tensor:
    """Camera output as float32 (num_envs, H, W, C): rgb in [0, 1] (alpha dropped), depth in metres (inf -> 0)."""
    out = env.scene[sensor_name].data.output[data_type]
    out = out.torch if hasattr(out, "torch") else out
    if data_type == "rgb":
        return out[..., :3].float() / 255.0
    out = out.float()
    if out.ndim == 3:
        out = out.unsqueeze(-1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def reset_cubes(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cube_names: list[str],
    region: SpawnRegion,
    size: float,
    min_gap: float = 0.04,
    settle_clearance: float = 0.002,
) -> None:
    """Drop the cubes at random non-overlapping (x, y, yaw) inside ``region`` just above the table, at rest."""
    rng = getattr(env, "_r2s_rng", None)
    if rng is None:
        rng = env._r2s_rng = np.random.default_rng(env.cfg.seed if env.cfg.seed is not None else 0)
    if isinstance(env_ids, slice):  # full reset passes slice(None)
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    env_ids = torch.as_tensor(env_ids, device=env.device)
    ids = env_ids.tolist()
    poses = np.stack([sample_cube_poses(rng, len(cube_names), region, size, min_gap) for _ in ids])  # (E, C, 4)
    poses[..., 2] += settle_clearance
    origins = env.scene.env_origins[env_ids]
    for ci, name in enumerate(cube_names):
        obj = env.scene[name]
        p = torch.as_tensor(poses[:, ci], dtype=torch.float32, device=env.device)
        quats = torch.tensor([yaw_to_quat_xyzw(float(y)) for y in p[:, 3]], dtype=torch.float32, device=env.device)
        pose = torch.cat([p[:, :3] + origins, quats], dim=-1)
        obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=env_ids)
