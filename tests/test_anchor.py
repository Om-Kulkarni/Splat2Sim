"""Anchor stage: Sim(3) solver unit tests (numpy/scipy) and end-to-end runs on synthetic ChArUco captures with known truth."""
import json

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")
from scipy.spatial.transform import Rotation  # noqa: E402

from splat2sim.anchor import solve, stage  # noqa: E402
from splat2sim.recon import usdutil  # noqa: E402

import synth_charuco as sc  # noqa: E402


def rand_sim(rng, s=None):
    return (float(s if s is not None else rng.uniform(0.8, 1.2)), Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix(),
            rng.normal(size=3))


def test_sim3_algebra():
    rng = np.random.default_rng(0)
    a, b = rand_sim(rng), rand_sim(rng)
    p = rng.normal(size=(7, 3))
    assert np.allclose(solve.apply(solve.invert(a), solve.apply(a, p)), p)
    assert np.allclose(solve.apply(solve.compose(a, b), p), solve.apply(a, solve.apply(b, p)))
    assert np.allclose(solve.to_matrix(a)[:3, :3] @ p.T + solve.to_matrix(a)[:3, 3:4], solve.apply(a, p).T)


def test_rpy_matrix_matches_config_convention():
    pytest.importorskip("pydantic")
    from splat2sim.config import Pose

    pose = Pose(xyz=(0.1, -0.2, 0.3), rpy=(10.0, -20.0, 35.0))
    assert np.allclose(solve.rpy_deg_matrix(pose.xyz, pose.rpy), pose.matrix())


def test_umeyama_exact_with_scale_and_planar_points():
    rng = np.random.default_rng(1)
    sim = rand_sim(rng, 1.07)
    q = np.c_[rng.uniform(0, 0.4, (30, 2)), np.zeros(30)]  # board-like planar data (rank 2)
    s, R, t = solve.umeyama(q, solve.apply(sim, q), True)
    assert abs(s - sim[0]) < 1e-9 and np.allclose(R, sim[1], atol=1e-9) and np.allclose(t, sim[2], atol=1e-9)


def test_ransac_survives_outliers():
    rng = np.random.default_rng(2)
    sim = rand_sim(rng, 1.0)
    q = np.c_[rng.uniform(0, 0.4, (200, 2)), np.zeros(200)]
    p = solve.apply(sim, q) + rng.normal(0, 0.004, (200, 3))
    bad = rng.random(200) < 0.4
    p[bad] += rng.uniform(-0.5, 0.5, (bad.sum(), 3))
    (s, R, t), inl = solve.ransac_sim3(q, p, 0.02, 1500, rng)
    err = solve.sim_error((s, R, t), sim)
    assert err["rotation_deg"] < 1.0 and err["scale_pct"] < 2.0 and np.linalg.norm(t - sim[2]) < 0.02
    assert inl[~bad].mean() > 0.9 and inl[bad].mean() < 0.1


def _synthetic_obs(rng, sim, K, n_frames=25, px_noise=0.3, outliers=0.05):
    q_all = np.array([[i * 0.055, j * 0.055, 0.0] for j in range(4) for i in range(6)])
    obs = []
    for _ in range(n_frames):
        centre = solve.apply(sim, q_all.mean(0))
        c = centre + rng.uniform(-1, 1, 3) * [0.4, 0.4, 0.2] + [0, 0, -1.0]
        T_wc = sc.look_at(c, centre, rng.uniform(-10, 10), down_hint=(0, 1, 0))
        T_cw = np.linalg.inv(T_wc)
        pc = solve.apply(sim, q_all) @ T_cw[:3, :3].T + T_cw[:3, 3]
        uv = np.c_[K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]]
        uv += rng.normal(0, px_noise, uv.shape)
        bad = rng.random(len(uv)) < outliers
        uv[bad] += rng.uniform(-30, 30, (bad.sum(), 2))
        obs.append(solve.FrameObs(T_cw=T_cw, uv=uv, q=q_all, z=pc[:, 2] + rng.normal(0, 0.003, len(pc))))
    return obs


@pytest.mark.parametrize("depth_weight", [0.0, 1.0])
def test_refine_improves_a_perturbed_start(depth_weight):
    rng = np.random.default_rng(3)
    K = np.array([[610.0, 0, 424], [0, 610.0, 240], [0, 0, 1]])
    sim = (1.0, Rotation.from_euler("zyx", [30, 5, -4], degrees=True).as_matrix(), np.array([0.3, -0.2, 0.1]))
    obs = _synthetic_obs(rng, sim, K)
    start = (1.03, Rotation.from_rotvec(np.radians([1.5, -1, 2])).as_matrix() @ sim[1], sim[2] + [0.02, -0.015, 0.01])
    est, info = solve.refine_sim3(start, obs, K, depth_weight=depth_weight)
    e0, e1 = solve.sim_error(start, sim), solve.sim_error(est, sim)
    assert e1["translation_mm"] < 0.3 * e0["translation_mm"] and e1["rotation_deg"] < 0.3 * e0["rotation_deg"]
    assert e1["translation_mm"] < 5 and e1["rotation_deg"] < 0.3 and abs(e1["scale_pct"]) < 1.0


def test_refine_fixed_scale_keeps_scale():
    rng = np.random.default_rng(4)
    K = np.array([[610.0, 0, 424], [0, 610.0, 240], [0, 0, 1]])
    sim = rand_sim(rng, 1.0)
    obs = _synthetic_obs(rng, sim, K, n_frames=10)
    est, _ = solve.refine_sim3(sim, obs, K, with_scale=False)
    assert est[0] == 1.0


# ---------------------------------------------------------------- end to end on synthetic captures
def _run(tmp_path, name="synth", options=None, **kw):
    run = tmp_path / name
    sc.make_synth(run, **kw)
    if options:
        import yaml

        (run / "options.yaml").write_text(yaml.safe_dump({"anchor": options}))
    return run, stage.run(run)


def test_e2e_synthetic_capture_recovers_ground_truth(tmp_path):
    run, res = _run(tmp_path, n_frames=40, seed=5)
    assert res["status"] == "charuco" and len(res["frames_used"]) >= 20
    err = sc.evaluate(run)
    # ground truth is perturbed by ~1 cm / 1 deg SLAM-like pose drift: that bounds what a single Sim(3) can recover
    assert err["rotation_deg"] < 1.5 and err["centre_mm"] < 30 and abs(err["scale_pct"]) < 1.5
    a = json.loads((run / "poses/anchor.json").read_text())
    T, kind = usdutil.anchor_transform(a)
    assert kind == "charuco" and np.allclose(T[:3, :3] @ T[:3, :3].T, a["scale"] ** 2 * np.eye(3), atol=1e-6)
    assert a["scale"] == pytest.approx(a["recon_to_board"]["scale"]) and a["reproj_rms_px"] < 4 and a["corner_3d_rms_mm"] < 25
    assert a["frames_considered"] <= 40 and a["board_to_base"]["T_base_board"]
    assert (run / "poses/anchor_overlays").is_dir()


def test_e2e_noise_free_is_nearly_exact(tmp_path):
    run, res = _run(tmp_path, n_frames=30, seed=6, pose_drift_cm=0, pose_drift_deg=0, pose_jitter_mm=0, pose_jitter_deg=0, px_noise=0)
    err = sc.evaluate(run)
    assert res["status"] == "charuco" and err["rotation_deg"] < 0.15 and err["centre_mm"] < 3 and abs(err["scale_pct"]) < 0.5
    assert res["reproj_rms_px"] < 1.0


def test_e2e_scale_deviation_is_estimated_and_warned(tmp_path):
    run, res = _run(tmp_path, n_frames=40, seed=7, scale_a=1.1, pose_drift_cm=0, pose_drift_deg=0)
    err = sc.evaluate(run)
    assert abs(err["scale_pct"]) < 1.5 and res["scale_estimate"] == pytest.approx(1 / 1.1, rel=0.015)
    assert any("scale" in w for w in res["warnings"])


def test_e2e_fixed_scale_option(tmp_path):
    _, res = _run(tmp_path, options={"fix_scale": True}, n_frames=30, seed=8)
    assert res["status"] == "charuco" and res["scale"] == 1.0


def test_e2e_no_board_falls_back_cleanly(tmp_path):
    run, res = _run(tmp_path, n_frames=15, seed=9, board_visible=False)
    a = json.loads((run / "poses/anchor.json").read_text())
    assert a["status"] == "fallback" and "no ChArUco board" in a["reason"] and "T_base_recon" not in a
    assert a["frames_with_board"] == 0


def test_e2e_too_few_frames_is_failed_not_charuco(tmp_path):
    _, res = _run(tmp_path, options={"min_frames": 500}, n_frames=15, seed=10)
    assert res["status"] == "failed" and "T_base_recon" not in res


def test_missing_poses_is_an_error(tmp_path):
    run = tmp_path / "r"
    sc.make_synth(run, n_frames=5, seed=1)
    (run / "poses/poses.csv").unlink()
    with pytest.raises(FileNotFoundError):
        stage.run(run)
