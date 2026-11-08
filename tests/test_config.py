import math
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from splat2sim.config import Pose, deep_merge, load_lab

REPO = Path(__file__).resolve().parents[1]


def test_robolab_loads_with_spec_defaults():
    lab = load_lab("robolab", labs_dir=REPO / "labs")
    assert lab.charuco.dictionary == "DICT_4X4_50"
    assert (lab.charuco.squares_x, lab.charuco.squares_y) == (7, 5)
    assert lab.charuco.square_length_m == 0.055 and lab.charuco.marker_length_m == 0.041
    assert len(lab.robot.scan_pose_joints) == 7
    assert lab.cubes.count == 3 and lab.cubes.size_m == 0.05 and lab.cubes.mass_kg == 0.05
    assert lab.overrides.table_box is None


def test_override_merges_deeply_and_replaces_lists(tmp_path):
    ov = tmp_path / "ov.yaml"
    ov.write_text(yaml.safe_dump({
        "charuco": {"square_length_m": 0.0551},
        "robot": {"scan_pose_joints": [0, 0, 0, -1, 0, 1, 0]},
        "overrides": {"table_box": {"center": [0.5, 0, -0.02], "size": [1.2, 0.8, 0.04]}},
    }))
    lab = load_lab("robolab", override=ov, labs_dir=REPO / "labs")
    assert lab.charuco.square_length_m == 0.0551
    assert lab.charuco.squares_x == 7  # untouched sibling keys survive
    assert lab.robot.scan_pose_joints == [0, 0, 0, -1, 0, 1, 0]
    assert lab.overrides.table_box.size == (1.2, 0.8, 0.04)


def test_deep_merge_does_not_mutate():
    a = {"x": {"y": 1, "z": [1]}}
    b = {"x": {"z": [2, 3]}}
    out = deep_merge(a, b)
    assert out == {"x": {"y": 1, "z": [2, 3]}} and a == {"x": {"y": 1, "z": [1]}}


def test_validation_errors(tmp_path):
    for bad in ({"robot": {"scan_pose_joints": [0, 0]}},
                {"charuco": {"marker_length_m": 0.1}},
                {"cubes": {"colour": "red"}}):  # unknown key rejected
        ov = tmp_path / "bad.yaml"
        ov.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValidationError):
            load_lab("robolab", override=ov, labs_dir=REPO / "labs")


def test_missing_lab():
    with pytest.raises(FileNotFoundError):
        load_lab("nope", labs_dir=REPO / "labs")


def test_pose_rpy_conventions():
    # pitch +90 deg tilts +x (forward) down to -z
    m = Pose(rpy=(0, 90, 0)).rotation_matrix()
    assert m @ [1, 0, 0] == pytest.approx([0, 0, -1], abs=1e-9)
    # yaw 180 flips x
    assert Pose(rpy=(0, 0, 180)).rotation_matrix() @ [1, 0, 0] == pytest.approx([-1, 0, 0], abs=1e-9)
    q = Pose(rpy=(0, 0, 90)).quat_xyzw()
    assert q == pytest.approx((0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)))


def test_digest_sections_isolated():
    lab = load_lab("robolab", labs_dir=REPO / "labs")
    other = lab.model_copy(update={"cubes": lab.cubes.model_copy(update={"count": 9})})
    assert lab.digest(["charuco"]) == other.digest(["charuco"])
    assert lab.digest(["cubes"]) != other.digest(["cubes"])
