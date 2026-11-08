"""Lab config schema (``labs/<lab>.yaml``) plus per-run override merging.

Conventions (used everywhere in the lab config):

* Lengths in metres, masses in kg, joint angles in radians, fov/rpy in degrees.
* A **Pose** is ``xyz`` (m) + ``rpy`` (degrees). Rotation is
  ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (ROS/URDF fixed-axis rpy). Quaternions
  are derived (``Pose.quat_xyzw``), never written by hand.
* Frames are z-up. ``robot_base`` is the world origin of the Scene (FR3 base).
* Camera frames are ROS body convention: +x forward (optical axis), +y left,
  +z up. Positive pitch tilts the camera down.
* ``board_to_base`` is the transform T_base_board mapping points in the board
  frame to the robot-base frame (p_base = R p_board + t), i.e. the pose of the
  board in the base frame. Board frame: origin at the board's first inner
  corner (OpenCV chessboard corner 0), x along squares_x, y along squares_y (down the printed face), z = x cross y,
  i.e. INTO the board (right-handed).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Pose(_Model):
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)  # degrees

    def rotation_matrix(self) -> np.ndarray:
        r, p, y = (math.radians(a) for a in self.rpy)
        cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return rz @ ry @ rx

    def matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.rotation_matrix()
        m[:3, 3] = self.xyz
        return m

    def quat_xyzw(self) -> tuple[float, float, float, float]:
        m = self.rotation_matrix()
        t = np.trace(m)
        if t > 0:
            s = math.sqrt(t + 1.0) * 2
            q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
        else:
            i = int(np.argmax(np.diag(m)))
            j, k = (i + 1) % 3, (i + 2) % 3
            s = math.sqrt(m[i, i] - m[j, j] - m[k, k] + 1.0) * 2
            v = [0.0] * 3
            v[i] = 0.25 * s
            v[j] = (m[j, i] + m[i, j]) / s
            v[k] = (m[k, i] + m[i, k]) / s
            q = (v[0], v[1], v[2], (m[k, j] - m[j, k]) / s)
        return tuple(float(x) for x in q)  # type: ignore[return-value]


class CharucoSpec(_Model):
    dictionary: str = "DICT_4X4_50"
    squares_x: int = Field(7, gt=1)
    squares_y: int = Field(5, gt=1)
    square_length_m: float = Field(0.055, gt=0)
    marker_length_m: float = Field(0.041, gt=0)

    @model_validator(mode="after")
    def _marker_smaller(self) -> "CharucoSpec":
        if self.marker_length_m >= self.square_length_m:
            raise ValueError("marker_length_m must be smaller than square_length_m")
        return self


class Robot(_Model):
    model: Literal["fr3"] = "fr3"
    scan_pose_joints: list[float]
    gripper: Literal["robotiq_2f85"] = "robotiq_2f85"
    coupling_offset: Pose = Pose()

    @field_validator("scan_pose_joints")
    @classmethod
    def _seven(cls, v: list[float]) -> list[float]:
        if len(v) != 7:
            raise ValueError("scan_pose_joints needs exactly 7 values (FR3)")
        return v


class Intrinsics(_Model):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fov: float = Field(gt=0, lt=180)  # horizontal field of view, degrees


class CameraSpec(_Model):
    parent_frame: str
    pose: Pose
    intrinsics: Intrinsics


class PolicyCameras(_Model):
    wrist: CameraSpec
    front: CameraSpec


class SpawnRegion(_Model):
    """Cube spawn rectangle in robot-base frame. ``z`` is the table-top height."""

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z: float = 0.0


class Cubes(_Model):
    count: int = Field(3, ge=0)
    size_m: float = Field(0.05, gt=0)
    mass_kg: float = Field(0.05, gt=0)
    # None -> use the fitted table box from the collision stage.
    spawn_region: SpawnRegion | None = None


class Box(_Model):
    """Axis-aligned-in-its-own-frame box in robot-base frame."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)  # degrees


class Overrides(_Model):
    table_box: Box | None = None
    pedestal_box: Box | None = None


class MaskCfg(_Model):
    """Robot-mask stage (SAM 3.1). Output masks: 255 = keep, 0 = robot."""

    prompts: list[str] = Field(default_factory=lambda: ["robot arm", "robotic gripper"], min_length=1)
    dilate_px: int = Field(15, ge=0)
    fill_holes: bool = True


class EnvOptions(_Model):
    """Task environment (``Splat2Sim-FR3-Cubes-v0``) knobs that are not properties of the lab hardware."""

    episode_length_s: float = Field(20.0, gt=0)
    reach_min_m: float = Field(0.30, ge=0)  # cube spawn: horizontal distance from the base axis, lower bound
    reach_max_m: float = Field(0.65, gt=0)  # ... and upper bound (FR3 reach is 0.855 m; leave room for a top-down grasp)
    spawn_margin_m: float = Field(0.05, ge=0)  # keep cube centres this far (plus half a cube) from the table edge


class CollisionCfg(_Model):
    """Optional lab-level defaults for the collision stage (same keys as ``--set collision.*``; unset = stage default)."""

    clearance: bool | None = None
    clearance_radius_m: float | None = None
    clearance_z_margin_m: float | None = None
    clearance_z_max_m: float | None = None
    clearance_arm: bool | None = None
    clearance_arm_radius_m: float | None = None
    clearance_gripper_length_m: float | None = None


class LabConfig(_Model):
    lab: str
    charuco: CharucoSpec = CharucoSpec()
    board_to_base: Pose
    robot: Robot
    policy_cameras: PolicyCameras
    cubes: Cubes = Cubes()
    overrides: Overrides = Overrides()
    mask: MaskCfg = MaskCfg()
    env: EnvOptions = EnvOptions()
    collision: CollisionCfg = CollisionCfg()

    def canonical_json(self, sections: list[str] | None = None) -> str:
        d = self.model_dump(mode="json")
        if sections is not None:
            d = {k: d[k] for k in sections}
        return json.dumps(d, sort_keys=True)

    def digest(self, sections: list[str] | None = None) -> str:
        return hashlib.sha256(self.canonical_json(sections).encode()).hexdigest()[:16]


def repo_root() -> Path:
    return Path(os.environ.get("R2S_REPO") or Path(__file__).resolve().parents[1])


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``over`` onto ``base``. Dicts merge, everything else (incl. lists) replaces."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def load_lab(lab: str | Path, override: str | Path | None = None, labs_dir: Path | None = None) -> LabConfig:
    """Load ``labs/<lab>.yaml`` (or an explicit yaml path) and merge an optional override file on top."""
    p = Path(lab)
    if p.suffix not in (".yaml", ".yml"):
        p = (labs_dir or repo_root() / "labs") / f"{lab}.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"lab config not found: {p}")
    data = _read_yaml(p)
    data.setdefault("lab", p.stem)
    if override is not None:
        data = deep_merge(data, _read_yaml(Path(override)))
    return LabConfig.model_validate(data)
