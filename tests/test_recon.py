"""Pure-python parts of the poses / splat / compose stages: pose IO, COLMAP writer, axis conversion, plane fit, ATE."""
import json

import numpy as np
import pytest

from splat2sim.recon import colmap, geometry as geo, options, poseio, usdutil
from splat2sim.recon.splat_stage import parse_metrics, train_command


def rand_pose(rng):
    q = rng.normal(size=4)
    T = poseio.pose_matrix(rng.normal(size=3), q)
    return T


def test_quat_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        R = poseio.quat_to_R(rng.normal(size=4))
        assert np.allclose(poseio.quat_to_R(poseio.R_to_quat(R)), R, atol=1e-9)
    # 180 degree rotations hit the non-trace branches
    for R in (np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1])):
        assert np.allclose(poseio.quat_to_R(poseio.R_to_quat(R)), R, atol=1e-9)


def test_poses_csv_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    Ts = np.array([rand_pose(rng) for _ in range(5)])
    poseio.write_poses_csv(tmp_path / "poses.csv", [0, 3, 4, 9, 10], [10, 20, 30, 40, 50], Ts)
    header = (tmp_path / "poses.csv").read_text().splitlines()[0]
    assert header == "index,timestamp_ns,tx,ty,tz,qx,qy,qz,qw"
    idx, ts, got = poseio.read_poses_csv(tmp_path / "poses.csv")
    assert list(idx) == [0, 3, 4, 9, 10] and list(ts) == [10, 20, 30, 40, 50]
    assert np.allclose(got, Ts, atol=1e-7)


def test_read_tum_poses(tmp_path):
    (tmp_path / "p.txt").write_text("#timestamp x y z qx qy qz qw id\n1.5 1 2 3 0 0 0 1 7\n2.5 0 0 0 0 0 1 0 8\n")
    t, T, ids = poseio.read_tum_poses(tmp_path / "p.txt")
    assert list(ids) == [7, 8] and np.allclose(T[0, :3, 3], [1, 2, 3])
    assert np.allclose(T[1, :3, :3], np.diag([-1, -1, 1]))


def test_colmap_images_are_world_to_camera_and_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    Ts = np.array([rand_pose(rng) for _ in range(4)])
    names = [f"{i:06d}.png" for i in range(4)]
    xyz = rng.normal(size=(10, 3))
    rgb = rng.integers(0, 255, size=(10, 3))
    sp = colmap.write_model(tmp_path, (848, 480), (600.0, 601.0, 424.0, 240.0), names, Ts, xyz, rgb)
    assert (sp / "cameras.txt").read_text().splitlines()[-1] == "1 PINHOLE 848 480 600.000000 601.000000 424.000000 240.000000"
    got_names, got = colmap.read_images_txt(sp / "images.txt")
    assert got_names == names and np.allclose(got, Ts, atol=1e-6)
    # first image line really stores T_cw: translation = -R^T t for camera-to-world pose
    v = [ln for ln in (sp / "images.txt").read_text().splitlines() if ln and not ln.startswith("#")][0].split()
    Tcw = np.linalg.inv(Ts[0])
    assert np.allclose([float(x) for x in v[5:8]], Tcw[:3, 3], atol=1e-6)
    assert len([ln for ln in (sp / "points3D.txt").read_text().splitlines() if not ln.startswith("#")]) == 10


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(4)
    src = rng.normal(size=(100, 3))
    R = poseio.quat_to_R(rng.normal(size=4))
    dst = 2.5 * src @ R.T + np.array([1.0, -2.0, 3.0])
    s, R2, t = geo.umeyama(src, dst, with_scale=True)
    assert s == pytest.approx(2.5) and np.allclose(R2, R) and np.allclose(t, [1, -2, 3])
    s1, _, _ = geo.umeyama(src, dst, with_scale=False)
    assert s1 == 1.0
    assert geo.ate_rmse(src, 1.0 * src @ R.T + 5)["rmse"] < 1e-9


def test_ate_lever_arm_removes_body_offset():
    rng = np.random.default_rng(5)
    t = np.linspace(0, 6, 300)
    est = np.stack([np.sin(t), np.cos(0.7 * t), 0.3 * t], 1)
    rots = np.array([poseio.quat_to_R(np.array([np.sin(a), 0, np.cos(2 * a) * 0.3, np.cos(a)])) for a in t])
    d = np.array([0.05, -0.03, 0.08])
    R_a = poseio.quat_to_R([0.1, 0.2, 0.3, 0.9])
    gt = est @ R_a.T + 4.0 + np.einsum("nij,j->ni", rots, d)
    assert geo.ate_rmse(est, gt)["rmse"] > 0.02
    out = geo.ate_with_lever_arm(est, gt, rots)
    assert out["rmse"] < 1e-3 and np.allclose(out["lever_arm_m"], d, atol=5e-3)


def test_backproject_pinhole():
    depth = np.full((10, 20), 2.0)
    p, u, v = geo.backproject(depth, 10.0, 10.0, 10.0, 5.0, stride=1, zmin=0.1, zmax=5)
    i = np.where((u == 10) & (v == 5))[0][0]
    assert np.allclose(p[i], [0, 0, 2.0])
    j = np.where((u == 15) & (v == 5))[0][0]
    assert np.allclose(p[j], [1.0, 0, 2.0])  # (15-10)/10 * 2


def make_room(rng, tilt_R=np.eye(3)):
    floor = np.c_[rng.uniform(-2, 2, (6000, 2)), rng.normal(0, 0.003, 6000)]
    wall = np.c_[np.full(3000, 2.0) + rng.normal(0, 0.003, 3000), rng.uniform(-2, 2, 3000), rng.uniform(0, 2, 3000)]
    clutter = rng.uniform([-2, -2, 0], [2, 2, 1.5], (1000, 3))
    P = np.vstack([floor, wall, clutter])
    return P @ tilt_R.T


def test_plane_fit_finds_floor():
    rng = np.random.default_rng(6)
    P = make_room(rng)
    fit = geo.fit_plane_ransac(P, rng=rng)
    assert abs(fit["normal"][2]) > 0.999 and fit["count"] > 5500


def test_gravity_plane_anchor_levels_a_tilted_recon():
    rng = np.random.default_rng(7)
    # recon frame = world rotated by an arbitrary rotation (no IMU: first camera frame, y down)
    Rr = poseio.quat_to_R([0.3, -0.5, 0.2, 0.8])
    t_shift = np.array([0.4, -1.0, 2.0])
    P_world = make_room(rng)
    P = P_world @ Rr.T + t_shift
    cams_world = np.c_[rng.uniform(-1, 1, (20, 2)), rng.uniform(0.8, 1.4, 20)]
    cams = cams_world @ Rr.T + t_shift
    fwd = Rr @ np.array([1.0, 0.2, -0.3])
    out = geo.gravity_plane_anchor(P, cams, fwd, rng=rng)
    T = out["T_anchor_recon"]
    A = P @ T[:3, :3].T + T[:3, 3]
    floor = A[:6000]
    assert abs(np.median(floor[:, 2])) < 0.01                       # floor at z=0
    assert np.allclose(np.median(floor[:, :2], axis=0), 0, atol=0.15)  # origin ~ floor centroid
    assert (cams @ T[:3, :3].T + T[:3, 3])[:, 2].min() > 0.5         # cameras above the plane (z-up)
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9) and np.linalg.det(T[:3, :3]) > 0
    assert out["up_source"] == "plane"
    # gravity hint restricts to horizontal planes: the wall must not win even with more points
    wall_heavy = np.vstack([P_world[:2000], P_world[6000:9000]]) @ Rr.T + t_shift
    out2 = geo.gravity_plane_anchor(wall_heavy, cams, fwd, up_hint=Rr @ np.array([0, 0, 1.0]), rng=rng)
    assert abs(out2["up_recon"] @ (Rr @ np.array([0, 0, 1.0]))) > 0.99 and out2["up_source"] == "imu+plane"


def test_usd_matrix_layout_and_roundtrip():
    T = np.eye(4)
    T[:3, 3] = [1, 2, 3]
    rows = usdutil.np_to_usd_matrix(T)
    assert rows[3] == (1.0, 2.0, 3.0, 1.0)  # USD is row-vector: translation in last row
    assert np.allclose(usdutil.usd_to_np_matrix(rows), T)


def test_yup_to_zup_axis_conversion():
    Y = usdutil.YUP_TO_ZUP
    assert np.allclose(Y[:3, :3] @ [0, 1, 0], [0, 0, 1])   # up -> up
    assert np.allclose(Y[:3, :3] @ [0, 0, 1], [0, -1, 0])  # +z (toward viewer) -> -y
    assert np.allclose(Y[:3, :3] @ [1, 0, 0], [1, 0, 0])
    assert np.linalg.det(Y[:3, :3]) == pytest.approx(1.0)
    assert np.allclose(usdutil.splat_local_transform("recon"), np.eye(4))
    assert np.allclose(usdutil.splat_local_transform("yup") @ Y, np.eye(4))
    with pytest.raises(ValueError):
        usdutil.splat_local_transform("zup")


def test_cv_camera_to_usd_looks_along_minus_z():
    # an OpenCV camera at the origin looking along world +x (z_cv = +x) with y_cv = -z (down)
    R = np.array([[0, 0, 1.0], [-1, 0, 0], [0, -1, 0]])  # columns: x_cv=-y, y_cv=-z, z_cv=+x in world
    T = np.eye(4)
    T[:3, :3] = R
    U = usdutil.cv_camera_to_usd(T)
    assert np.allclose(-U[:3, 2], [1, 0, 0])  # USD camera looks down its -Z => world +x
    assert np.allclose(U[:3, 1], [0, 0, 1])   # USD +Y (up) => world +z


def test_look_at_camera():
    T = usdutil.look_at_usd_camera([0, -2, 1], [0, 0, 1])
    assert np.allclose(-T[:3, 2], [0, 1, 0]) and np.allclose(T[:3, 1], [0, 0, 1]) and np.allclose(T[:3, 3], [0, -2, 1])
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)
    up_down = usdutil.look_at_usd_camera([0, 0, 3], [0, 0, 0])  # degenerate (parallel to up) still valid
    assert np.linalg.det(up_down[:3, :3]) == pytest.approx(1.0)


def test_anchor_transform_and_bbox():
    T = np.eye(4)
    T[:3, 3] = [0, 0, 1]
    assert usdutil.anchor_transform({"T_base_recon": T.tolist()})[1] == "charuco"
    assert usdutil.anchor_transform({"T_anchor_recon": T.tolist()})[1] == "fallback"
    with pytest.raises(ValueError):
        usdutil.anchor_transform({})
    lo, hi = usdutil.bbox_transform([0, 0, 0], [1, 1, 1], T)
    assert np.allclose(lo, [0, 0, 1]) and np.allclose(hi, [1, 1, 2])
    eyes = usdutil.overview_eyes([0, 0, 0], [2, 2, 1], n=2)
    assert len(eyes) == 2 and all(e[0][2] > 0.5 for e in eyes)


def test_options(tmp_path):
    assert options.stage_options(tmp_path, "splat", options.SPLAT_DEFAULTS)["iterations"] == 30000
    options.set_option(tmp_path, "splat.iterations", options.QUICK_ITERATIONS)
    options.set_option(tmp_path, "poses.use_imu", False)
    assert options.stage_options(tmp_path, "splat", options.SPLAT_DEFAULTS)["iterations"] == 3000
    assert options.stage_options(tmp_path, "poses", options.POSES_DEFAULTS)["use_imu"] is False
    options.set_option(tmp_path, "poses.bogus", 1)
    with pytest.raises(ValueError):
        options.stage_options(tmp_path, "poses", options.POSES_DEFAULTS)


def test_splat_command_disables_normalisation(tmp_path):
    cmd = train_command(tmp_path / "c", tmp_path / "o", tmp_path / "s.usdz", {**options.SPLAT_DEFAULTS, "iterations": 3000})
    assert "dataset.normalize_world_space=false" in cmd and "export_usd.apply_normalizing_transform=false" in cmd
    assert "n_iterations=3000" in cmd and "dataset.test_split_interval=8" in cmd


def test_parse_metrics():
    m = parse_metrics("noise\n\x1b[1mMean PSNR: 27.31\x1b[0m\nSSIM: 0.8123\nLPIPS: 0.2210\n")
    assert m == {"psnr": 27.31, "ssim": 0.8123, "lpips": 0.221}
    assert parse_metrics("nothing here") == {}


def test_auto_stride_and_read_metrics(tmp_path):
    import numpy as np
    from splat2sim.recon.poses_stage import auto_stride
    from splat2sim.recon.splat_stage import read_test_metrics

    ts = (np.arange(100) * 33_333_333).astype(np.int64)
    idx = np.arange(100)
    assert auto_stride({}, idx, ts, list(range(100))) == 3
    assert auto_stride({}, idx, ts, [0]) == 1
    d = tmp_path / "x" / "run"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text('{"mean_psnr": 24.3, "mean_ssim": 0.83, "mean_lpips": 0.33}')
    assert read_test_metrics(tmp_path) == {"psnr": 24.3, "ssim": 0.83, "lpips": 0.33}
    assert read_test_metrics(tmp_path / "none") == {}
