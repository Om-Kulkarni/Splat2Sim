"""Task-environment helpers that run without Isaac: pose -> camera conversions, intrinsics, table box, cube spawning, asset naming."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from splat2sim.config import Box, Intrinsics, Pose, SpawnRegion, load_lab
from splat2sim.lab import asset, geometry

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def lab():
    return load_lab("robolab", labs_dir=REPO / "labs")


def test_robolab_camera_conventions_front_looks_at_the_base(lab):
    """front: 1 m ahead, 0.8 m up, yaw 180 + pitch 45 -> optical axis points back toward the base and down."""
    spec = lab.policy_cameras.front
    fwd = geometry.camera_forward_in_parent(spec.pose)
    assert np.allclose(fwd, [-math.sqrt(0.5), 0.0, -math.sqrt(0.5)], atol=1e-9)
    # the ray from the camera along fwd reaches z = 0 at x = 1 - 0.8 = 0.2 m (in front of the base)
    t = spec.pose.xyz[2] / -fwd[2]
    assert spec.pose.xyz[0] + t * fwd[0] == pytest.approx(0.2)


def test_wrist_camera_looks_along_the_flange_z_axis(lab):
    fwd = geometry.camera_forward_in_parent(lab.policy_cameras.wrist.pose)
    assert fwd[2] > 0.98 and abs(fwd[1]) < 1e-9  # pitch -80 deg: ~ +z of the flange, slightly toward +x


def test_positive_pitch_tilts_down_and_positive_yaw_turns_left():
    assert geometry.camera_forward_in_parent(Pose(rpy=(0, 30, 0)))[2] < 0
    assert geometry.camera_forward_in_parent(Pose(rpy=(0, 0, 30)))[1] > 0


def test_camera_offset_is_xyzw_unit_quaternion(lab):
    pos, quat = geometry.camera_offset(lab.policy_cameras.wrist)
    assert pos == (0.06, 0.0, 0.08)
    assert len(quat) == 4 and sum(q * q for q in quat) == pytest.approx(1.0)
    # rotation about +y by -80 deg -> quat (0, sin(-40), 0, cos(-40)) in xyzw
    assert quat == pytest.approx((0.0, math.sin(math.radians(-40)), 0.0, math.cos(math.radians(-40))), abs=1e-9)


def test_intrinsics_and_aperture_match_the_requested_fov(lab):
    intr = lab.policy_cameras.front.intrinsics
    k = geometry.intrinsic_matrix(intr)
    assert 2 * math.degrees(math.atan(intr.width / 2 / k[0, 0])) == pytest.approx(intr.fov)
    assert (k[0, 2], k[1, 2]) == (intr.width / 2, intr.height / 2)
    focal, ha, va = geometry.pinhole_aperture(intr)
    assert 2 * math.degrees(math.atan(ha / 2 / focal)) == pytest.approx(intr.fov)
    assert va / ha == pytest.approx(intr.height / intr.width)
    assert geometry.vertical_fov_deg(intr) < intr.fov


def test_parent_links_and_unknown_parent():
    assert geometry.camera_parent_link("flange") == "fr3_link8"
    assert geometry.camera_parent_link("robot_base") == "fr3_link0"
    with pytest.raises(ValueError):
        geometry.camera_parent_link("tool0")


def test_table_box_priority_default_override_primitives(lab, tmp_path):
    assert geometry.read_table_box(lab, tmp_path).source == "default"
    ov = lab.model_copy(update={"overrides": lab.overrides.model_copy(update={"table_box": Box(center=(0.4, 0, -0.03), size=(1.0, 0.6, 0.06))})})
    t = geometry.read_table_box(ov, tmp_path)
    assert t.source == "lab.overrides.table_box" and t.top_z == pytest.approx(0.0)
    (tmp_path / "collision").mkdir()
    (tmp_path / "collision" / "primitives.json").write_text(json.dumps(
        {"table_box": {"center": [0.5, 0.1, -0.02], "size": [1.1, 0.7, 0.04], "rpy_deg": [0, 0, 12.0], "yaw_deg": 12.0, "source": "ransac"}}))
    t = geometry.read_table_box(ov, tmp_path)
    assert t.source == "primitives.json:ransac" and t.yaw_deg == 12.0 and t.center == (0.5, 0.1, -0.02)
    # a null table_box (nothing found) falls through to the override
    (tmp_path / "collision" / "primitives.json").write_text(json.dumps({"table_box": None}))
    assert geometry.read_table_box(ov, tmp_path).source == "lab.overrides.table_box"


def test_spawn_region_is_inside_table_and_reach(lab):
    t = geometry.read_table_box(lab)  # default 1.2 x 0.8 table centred at x = 0.5
    r = geometry.spawn_region(lab, t, 0.30, 0.65, 0.05)
    assert r.z == pytest.approx(0.0)
    assert r.x_range[0] >= 0.30 and r.x_range[1] <= 0.65
    assert r.y_range[0] >= -0.65 and r.y_range[1] <= 0.65
    assert r.y_range[1] <= 0.4 - 0.05 - 0.025 + 1e-9  # table half-width minus margin minus half a cube


def test_spawn_region_config_wins_and_empty_region_errors(lab):
    explicit = SpawnRegion(x_range=(0.4, 0.5), y_range=(-0.1, 0.1), z=0.02)
    lab2 = lab.model_copy(update={"cubes": lab.cubes.model_copy(update={"spawn_region": explicit})})
    assert geometry.spawn_region(lab2, geometry.read_table_box(lab2), 0.3, 0.65, 0.05) == explicit
    far = geometry.TableBox(center=(2.0, 0.0, -0.02), size=(0.4, 0.4, 0.04))
    with pytest.raises(ValueError):
        geometry.spawn_region(lab, far, 0.30, 0.65, 0.05)


def test_sample_cube_poses_non_overlapping_and_reproducible():
    region = SpawnRegion(x_range=(0.3, 0.6), y_range=(-0.25, 0.25), z=0.0)
    a = geometry.sample_cube_poses(np.random.default_rng(1), 3, region, 0.05)
    b = geometry.sample_cube_poses(np.random.default_rng(1), 3, region, 0.05)
    assert np.array_equal(a, b) and a.shape == (3, 4)
    assert np.allclose(a[:, 2], 0.025)
    for i in range(3):
        for j in range(i):
            assert np.linalg.norm(a[i, :2] - a[j, :2]) >= 0.05 * math.sqrt(2)
    with pytest.raises(ValueError):
        geometry.sample_cube_poses(np.random.default_rng(0), 50, SpawnRegion(x_range=(0.3, 0.32), y_range=(0, 0.02)), 0.05)


def test_grasp_yaw_is_face_aligned_and_in_window():
    for cube_yaw in np.linspace(-math.pi, math.pi, 41):
        for around in (0.0, math.pi / 4):
            g = geometry.grasp_yaw(float(cube_yaw), around)
            assert around - math.pi / 4 - 1e-9 <= g < around + math.pi / 4 + 1e-9
            d = (g - cube_yaw) % (math.pi / 2)
            assert min(d, math.pi / 2 - d) < 1e-9  # differs from the cube yaw by a multiple of 90 deg


def test_asset_name_tracks_coupling_offset_and_sources():
    n = "https://x/Assets/Isaac/6.1/Isaac"
    a = asset.asset_path("/tmp/a", Pose(xyz=(0, 0, 0.008)), n)
    assert a == asset.asset_path("/tmp/a", Pose(xyz=(0, 0, 0.008)), n)
    assert a != asset.asset_path("/tmp/a", Pose(xyz=(0, 0, 0.010)), n)
    assert a != asset.asset_path("/tmp/a", Pose(xyz=(0, 0, 0.008), rpy=(0, 0, 45)), n)
    assert a != asset.asset_path("/tmp/a", Pose(xyz=(0, 0, 0.008)), "https://x/Assets/Isaac/6.2/Isaac")
    assert a.suffix == ".usda" and a.name.startswith("fr3_robotiq_2f85-")


def test_coupling_quaternion_wxyz_matches_pose():
    w, x, y, z = asset.quat_wxyz(Pose(rpy=(0, 0, 90)))
    assert (w, x, y, z) == pytest.approx((math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)))


def test_env_options_defaults_and_stage_wiring(lab):
    from splat2sim import stages as st

    assert lab.env.episode_length_s == 20.0 and lab.env.reach_min_m < lab.env.reach_max_m
    s = st.get("env")
    assert s.implemented and s.service == "isaac" and "overrides" in s.config_sections and "env" in s.config_sections
    with pytest.raises(Exception):
        Intrinsics(width=640, height=480, fov=190)


def test_projection_matches_camera_convention(lab):
    intr = lab.policy_cameras.front.intrinsics
    q = Pose(rpy=(0, 0, 0)).quat_xyzw()
    # a point straight ahead projects to the principal point; a point to the camera's left lands left of centre (u < cx)
    u, v, d = geometry.project_to_pixels([2, 0, 0], [0, 0, 0], q, intr)
    assert (u, v, d) == pytest.approx((intr.width / 2, intr.height / 2, 2.0))
    u, v, d = geometry.project_to_pixels([2, 0.5, 0.3], [0, 0, 0], q, intr)
    assert u < intr.width / 2 and v < intr.height / 2  # +y is left, +z is up
    assert math.isnan(geometry.project_to_pixels([-1, 0, 0], [0, 0, 0], q, intr)[0])
    # the horizontal fov edge
    half = math.tan(math.radians(intr.fov) / 2)
    assert geometry.project_to_pixels([1, half, 0], [0, 0, 0], q, intr)[0] == pytest.approx(0.0, abs=1e-6)


def test_front_camera_sees_the_table_in_front_of_the_base(lab):
    spec = lab.policy_cameras.front
    pos, quat = geometry.camera_offset(spec)
    u, v, d = geometry.project_to_pixels([0.2, 0.0, 0.0], pos, quat, spec.intrinsics)  # where the optical axis meets z = 0
    assert (u, v) == pytest.approx((spec.intrinsics.width / 2, spec.intrinsics.height / 2), abs=1e-6)
    u_l, _, _ = geometry.project_to_pixels([0.2, 0.1, 0.0], pos, quat, spec.intrinsics)
    # camera looks back toward the base (yaw 180), so world +y is on the camera's right -> larger u
    assert u_l > spec.intrinsics.width / 2


def test_compose_pose_and_quat_roundtrip():
    q = Pose(rpy=(10, 20, 30)).quat_xyzw()
    assert np.allclose(geometry.quat_xyzw_to_matrix(q), Pose(rpy=(10, 20, 30)).rotation_matrix())
    assert geometry.matrix_to_quat_xyzw(geometry.quat_xyzw_to_matrix(q)) == pytest.approx(q, abs=1e-9)
    pos, qq = geometry.compose_pose([1, 0, 0], Pose(rpy=(0, 0, 90)).quat_xyzw(), [1, 0, 0], (0, 0, 0, 1))
    assert pos == pytest.approx([1, 1, 0])


def test_color_centroid_finds_a_patch():
    img = np.full((60, 80, 3), 0.4)
    img[10:20, 30:40] = (0.85, 0.15, 0.12)
    c = geometry.color_centroid(img, (0.85, 0.15, 0.12))
    assert c is not None and c[0] == pytest.approx(34.5) and c[1] == pytest.approx(14.5) and c[2] == 100
    assert geometry.color_centroid(img, (0.12, 0.45, 0.85)) is None
