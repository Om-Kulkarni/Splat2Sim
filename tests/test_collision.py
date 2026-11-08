"""Collision-layer fitting (numpy only) and, when Open3D/pxr are importable, a synthetic end-to-end stage run."""
import json

import numpy as np
import pytest

from splat2sim.collision import fitting as fit


def rect_points(rng, center, size, yaw_deg, z, n=20000, noise=0.003):
    uv = rng.uniform(-0.5, 0.5, (n, 2)) * np.asarray(size)
    y = np.radians(yaw_deg)
    R = np.array([[np.cos(y), -np.sin(y)], [np.sin(y), np.cos(y)]])
    xy = uv @ R.T + np.asarray(center)
    return np.c_[xy, z + rng.normal(0, noise, n)]


def test_min_area_rect_recovers_rotated_rectangle():
    rng = np.random.default_rng(0)
    p = rect_points(rng, (0.3, -0.2), (1.2, 0.5), 30, 0.0)[:, :2]
    r = fit.min_area_rect(p)
    assert np.allclose(r["center"], (0.3, -0.2), atol=0.02)
    # u may be either side; compare sorted sizes and yaw modulo 90 degrees
    assert np.allclose(sorted(r["size"]), [0.5, 1.2], atol=0.03)
    yaw = np.degrees(r["yaw"]) % 90
    assert min(abs(yaw - 30), abs(yaw - 30 + 90), abs(yaw - 30 - 90)) < 2.0


def test_convex_hull_square():
    h = fit.convex_hull(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0.2, 0.7]]))
    assert len(h) == 4


def test_largest_component_drops_island():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 1, (3000, 2))
    b = rng.uniform(3, 3.1, (30, 2))
    keep = fit.largest_component(np.vstack([a, b]), cell=0.05)
    assert keep[:3000].mean() > 0.9 and not keep[3000:].any()


def test_table_box_on_noisy_scene_with_objects_and_floor():
    rng = np.random.default_rng(2)
    table = rect_points(rng, (0.5, 0.1), (1.0, 0.6), 25, 0.02, n=30000)
    floor = rect_points(rng, (0, 0), (4, 4), 0, -0.7, n=12000)
    obj = np.c_[rng.uniform(0.45, 0.55, (3000, 1)), rng.uniform(0.05, 0.15, (3000, 1)), rng.uniform(0.02, 0.12, 3000)]  # box-ish clutter on the table
    pts = np.vstack([table, floor, obj])
    box = fit.fit_table_box(pts, thickness=0.04, thresh=0.012, rng=np.random.default_rng(0))
    assert box["source"] == "ransac"
    assert box["stats"]["plane_rms_m"] < 0.006
    assert abs(box["center"][2] + 0.5 * box["size"][2] - 0.02) < 0.01  # top face at z=0.02
    assert abs(box["stats"]["top_z"] - 0.02) < 0.005
    assert np.allclose(sorted(box["size"][:2]), [0.6, 1.0], atol=0.06)
    assert np.allclose(box["center"][:2], (0.5, 0.1), atol=0.03)
    assert box["size"][2] == pytest.approx(0.04)


def test_inscribed_rect_on_l_shape():
    rng = np.random.default_rng(7)
    a = rng.uniform((0, 0), (2.0, 0.6), (20000, 2))
    b = rng.uniform((0, 0.6), (0.6, 1.8), (12000, 2))  # L: long leg 2.0 x 0.6 plus a 0.6 x 1.2 arm
    r = fit.inscribed_rect(np.vstack([a, b]), cell=0.03)
    assert r["area"] > 0.9 * 1.2 and r["area"] < 1.45  # biggest rectangle is ~2.0x0.6 = 1.2 (or 0.6x1.8 = 1.08)
    box = fit.fit_table_box(np.c_[np.vstack([a, b]), rng.normal(0, 0.002, 32000)], extent="bounding")
    assert box["stats"]["bounding_area_m2"] > 2.0 and box["stats"]["fill_ratio"] < 0.7


def test_prefer_elevated_picks_table_over_larger_floor():
    rng = np.random.default_rng(8)
    floor = rect_points(rng, (0, 0), (4, 4), 0, 0.0, n=30000)
    table = rect_points(rng, (-1.0, 0.5), (0.6, 0.9), 0, 0.88, n=9000)
    ceil = rect_points(rng, (0, 0), (4, 4), 0, 2.6, n=9000)
    pts = np.vstack([floor, table, ceil])
    assert fit.fit_table_box(pts, rng=np.random.default_rng(0))["stats"]["top_z"] < 0.05  # dominant = floor
    b = fit.fit_table_box(pts, prefer_elevated=True, z_max=1.2, rng=np.random.default_rng(0))
    assert b["stats"]["top_z"] == pytest.approx(0.88, abs=0.01)
    assert np.allclose(sorted(b["size"][:2]), [0.6, 0.9], atol=0.08)


def test_table_z_range_selects_higher_plane():
    rng = np.random.default_rng(3)
    table = rect_points(rng, (0, 0), (0.8, 0.5), 0, 0.0, n=6000)
    floor = rect_points(rng, (0, 0), (5, 5), 0, -0.75, n=20000)  # bigger
    pts = np.vstack([table, floor])
    assert fit.fit_table_box(pts, rng=np.random.default_rng(0))["stats"]["top_z"] < -0.7
    b = fit.fit_table_box(pts, z_range=(-0.2, 0.2), rng=np.random.default_rng(0))
    assert abs(b["stats"]["top_z"]) < 0.01


def test_tilted_plane_box_follows_tilt_or_stays_level():
    rng = np.random.default_rng(4)
    p = rect_points(rng, (0, 0), (1.0, 1.0), 0, 0.0, n=20000, noise=0.001)
    p[:, 2] += np.tan(np.radians(3.0)) * p[:, 0]  # 3 degree slope about y
    b = fit.fit_table_box(p, thresh=0.01, rng=np.random.default_rng(0))
    n = fit.rpy_to_R(b["rpy_deg"])[:, 2]
    assert np.degrees(np.arccos(n[2])) == pytest.approx(3.0, abs=0.3)
    assert b["stats"]["plane_rms_m"] < 0.003
    lv = fit.fit_table_box(p, thresh=0.04, level=True, rng=np.random.default_rng(0))
    assert lv["rpy_deg"][:2] == [0.0, 0.0] and lv["stats"]["tilt_deg"] == pytest.approx(3.0, abs=0.3)


def test_rpy_roundtrip():
    for rpy in ([10, -20, 30], [0, 0, 170], [-45, 5, -100]):
        assert np.allclose(fit.R_to_rpy_deg(fit.rpy_to_R(rpy)), rpy, atol=1e-6)


def test_no_plane_returns_none():
    rng = np.random.default_rng(5)
    assert fit.fit_table_box(rng.uniform(-1, 1, (100, 3))) is None


def test_pedestal_box_under_base():
    rng = np.random.default_rng(6)
    top = rect_points(rng, (0.0, 0.0), (0.3, 0.3), 0, 0.0, n=3000, noise=0.002)
    table = rect_points(rng, (0.6, 0), (1.0, 0.8), 0, -0.4, n=20000, noise=0.002)
    ped = fit.fit_pedestal_box(np.vstack([top, table]), None, table_top_z=-0.4, rng=np.random.default_rng(0))
    assert ped is not None
    assert np.allclose(ped["size"][:2], 0.3, atol=0.04)
    assert ped["size"][2] == pytest.approx(0.4, abs=0.02)
    assert ped["center"][2] == pytest.approx(-0.2, abs=0.02)
    # nothing near the base -> None
    assert fit.fit_pedestal_box(table, None, table_top_z=-0.4, rng=np.random.default_rng(0)) is None


def test_box_surface_distance_and_frame_roundtrip():
    box = fit.box_dict((0.1, 0.2, 0.3), (1.0, 0.5, 0.2), (0, 0, 40), "ransac")
    R = fit.rpy_to_R(box["rpy_deg"])
    # a point 0.1 above the top face centre
    p = np.array([0.1, 0.2, 0.3 + 0.1 + 0.1])
    assert fit.box_surface_distance(p[None], box["center"], box["size"], box["rpy_deg"])[0] == pytest.approx(0.1)
    # recon-frame form with a similarity anchor: unit cube corners map back onto the anchor-frame box
    T = np.eye(4)
    T[:3, :3] = 1.05 * fit.rpy_to_R((5, -3, 100))
    T[:3, 3] = (0.3, -1.0, 0.5)
    b = fit.box_to_frame(box, T)
    corner_recon = b["matrix"] @ np.array([0.5, 0.5, 0.5, 1.0])
    corner_anchor = T @ corner_recon
    expect = np.asarray(box["center"]) + R @ (0.5 * np.asarray(box["size"]))
    assert np.allclose(corner_anchor[:3], expect, atol=1e-9)


def test_override_box():
    b = fit.override_box({"center": [0.5, 0, -0.02], "size": [1.2, 0.8, 0.04], "rpy": [0, 0, 90]})
    assert b["source"] == "override" and b["yaw_deg"] == 90


# ------------------------------------------------------------------ end to end (Open3D + usd-core image only)
def _synth_run(tmp_path):
    from PIL import Image

    from splat2sim.recon.poseio import write_poses_csv

    root = tmp_path / "run"
    (root / "frames" / "color").mkdir(parents=True)
    (root / "frames" / "depth").mkdir()
    (root / "poses").mkdir()
    w, h, f = 160, 120, 150.0
    intr = {"fx": f, "fy": f, "cx": w / 2, "cy": h / 2, "width": w, "height": h, "depth_scale_m": 0.001}
    (root / "frames" / "intrinsics.json").write_text(json.dumps(intr))
    # world = anchor frame (identity anchor), z up; table plane z=0.0; cameras look down at it from 0.8 m, tilted
    vs, us = np.mgrid[0:h, 0:w]
    rows, Ts = ["index,timestamp_ns,color,depth,sharpness"], []
    for k in range(24):
        az = 2 * np.pi * k / 24
        eye = np.array([0.5 * np.cos(az), 0.5 * np.sin(az), 0.7])
        fwd = -eye / np.linalg.norm(eye)
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, down, fwd, eye
        rays = np.stack([(us - w / 2) / f, (vs - h / 2) / f, np.ones_like(us, float)], -1) @ T[:3, :3].T
        t = -eye[2] / rays[..., 2]
        pt = eye + rays * t[..., None]
        z_cam = t  # camera-z of the hit point = t since ray z-comp is 1 in cam frame
        inside = (np.abs(pt[..., 0]) < 0.6) & (np.abs(pt[..., 1]) < 0.4) & (t > 0)
        d = np.where(inside, z_cam * 1000, 0).astype(np.uint16)
        Image.fromarray(d).save(root / "frames" / "depth" / f"{k:06d}.png")
        rows.append(f"{k},{k},color/{k:06d}.png,depth/{k:06d}.png,1")
        Ts.append(T)
    (root / "frames" / "frames.csv").write_text("\n".join(rows) + "\n")
    write_poses_csv(root / "poses" / "poses.csv", range(24), range(24), np.array(Ts))
    (root / "poses" / "fallback_anchor.json").write_text(json.dumps({"T_anchor_recon": np.eye(4).tolist(), "method": "t"}))
    return root


def test_stage_end_to_end_synthetic(tmp_path):
    pytest.importorskip("open3d")
    pxr = pytest.importorskip("pxr")
    from splat2sim.collision import stage

    root = _synth_run(tmp_path)
    (root / "options.yaml").write_text("collision: {heldout_every: 6, voxel_m: 0.01, target_triangles: 5000, carve_table: false}\n")
    res = stage.run(root)
    tb = res["table_box"]
    assert tb["source"] == "ransac"
    assert abs(res["table_plane"]["top_z"]) < 0.01
    assert np.allclose(sorted(tb["size"][:2]), [0.8, 1.2], atol=0.1)
    assert res["pedestal_box"]["source"] == "absent"
    assert res["depth_error"]["collision"]["median_m"] < 0.01
    from pxr import Usd, UsdGeom, UsdPhysics

    st = Usd.Stage.Open(str(root / "collision" / "collision.usda"))
    for path in ("/Collision/Mesh", "/Collision/TableBox"):
        prim = st.GetPrimAtPath(path)
        assert prim and prim.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdGeom.Imageable(prim).GetVisibilityAttr().Get() == "invisible"
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MeshCollisionAPI(st.GetPrimAtPath("/Collision/Mesh")).GetApproximationAttr().Get() == "none"
    del st, prim  # release the layers so the rerun can overwrite them
    # override replaces the fit
    (root / "lab.json").write_text(json.dumps({"overrides": {"table_box": {"center": [0, 0, -0.02], "size": [1, 1, 0.04], "rpy": [0, 0, 0]}}}))
    res = stage.run(root)
    assert res["table_box"]["source"] == "override"
    assert json.loads((root / "collision" / "primitives.json").read_text())["table_box"]["source"] == "override"


def test_pedestal_ignores_table_top_close_below_the_base():
    rng = np.random.default_rng(3)
    ped = rect_points(rng, (0.0, 0.0), (0.3, 0.3), 0, 0.0, n=6000, noise=0.002)        # pedestal top at the base z = 0
    table = rect_points(rng, (0.6, 0.0), (0.9, 0.8), 0, -0.15, n=30000, noise=0.002)   # table top only 15 cm below
    pts = np.vstack([ped, table])
    box = fit.fit_pedestal_box(pts, None, table_top_z=-0.15, rng=np.random.default_rng(0))
    assert box is not None and np.allclose(box["size"][:2], [0.3, 0.3], atol=0.05)
    assert abs(box["stats"]["top_z"]) < 0.01 and abs(box["stats"]["bottom_z"] + 0.15) < 1e-6


def test_carve_clutter_and_clearance_on_open3d_mesh():
    o3d = pytest.importorskip("open3d")
    from splat2sim.collision.clearance import CLEARANCE_DEFAULTS, carve_clearance
    from splat2sim.collision.stage import carve_clutter

    def cube_mesh(c, s):
        m = o3d.geometry.TriangleMesh.create_box(s, s, s)
        m.translate(np.asarray(c) - s / 2)
        return m

    table = {"center": [0.7, 0.0, -0.22], "size": [1.0, 0.8, 0.04], "rpy_deg": [0, 0, 0]}  # top at -0.20
    cube_near = cube_mesh((0.5, 0.1, -0.175), 0.05)    # lying on the table, 50 cm from the base
    cube_far = cube_mesh((1.15, 0.0, -0.175), 0.05)    # beyond the carve radius
    arm = cube_mesh((0.1, 0.0, 0.4), 0.05)             # inside the clearance cylinder
    mesh = cube_near + cube_far + arm
    n0 = len(mesh.triangles)
    assert carve_clutter(mesh, np.eye(4), table, 0.12, 0.8) == 12
    info = carve_clearance(mesh, np.eye(4), dict(CLEARANCE_DEFAULTS), [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], pedestal_top_z=0.0)
    assert info["triangles_removed"] == 12 and len(mesh.triangles) == n0 - 24
