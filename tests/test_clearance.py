import numpy as np

from splat2sim import robot_fk
from splat2sim.collision.clearance import CLEARANCE_DEFAULTS, clearance_mask

SCAN = [0.0, -0.7853981634, 0.0, -2.3561944902, 0.0, 1.5707963268, 0.7853981634]


def test_fk_zero_pose_matches_panda():
    f = robot_fk.fk_frames([0] * 7)[-1]
    assert np.allclose(f[:3, 3], [0.088, 0.0, 0.926], atol=1e-6)
    assert np.allclose(f[:3, 2], [0, 0, -1], atol=1e-9)  # flange points down


def test_fk_scan_pose_flange_in_front_of_base():
    f = robot_fk.fk_frames(SCAN)[-1]
    assert np.allclose(f[:3, 3], [0.307, 0.0, 0.59], atol=2e-3)  # Franka ready pose


def test_clearance_mask_cylinder_and_arm():
    o = dict(CLEARANCE_DEFAULTS)
    pts = np.array([[0.1, 0.0, 0.3],    # inside cylinder
                    [0.5, 0.5, 0.3],    # far away
                    [0.1, 0.0, -0.2],   # below the pedestal top: kept
                    [0.1, 0.0, 1.5],    # above z_max
                    [0.32, 0.0, 0.45]])  # below the gripper at the scan pose
    m, info = clearance_mask(pts, o, SCAN, pedestal_top_z=0.0)
    assert m.tolist()[:4] == [True, False, False, False]
    o["clearance_arm"] = False
    m2, _ = clearance_mask(pts, o, SCAN, pedestal_top_z=0.0)
    assert not m2[4] and m2[0]
