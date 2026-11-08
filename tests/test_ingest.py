import json
from pathlib import Path

import numpy as np
import pytest

from splat2sim.ingest import imu as imu_mod
from splat2sim.ingest.detect import BAG, MCAP, TUM, UnknownFormat, detect_format
from splat2sim.ingest.geometry import Intrinsics, invert_T, make_T, quat_to_R, register_depth_to_color
from splat2sim.ingest.intrinsics_io import parse_intrinsics
from splat2sim.ingest.pairing import StreamPairer
from splat2sim.ingest.readers.mcap_reader import discover_topics
from splat2sim.ingest.readers.tf import TfGraph
from splat2sim.ingest.readers.tum_reader import associate, parse_ts_ns

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])


# ---------- depth registration ----------
def test_register_identity():
    d = np.full((480, 640), 1000, np.uint16)
    d[100:200, 100:200] = 0
    out = register_depth_to_color(d, K, K, np.eye(4), (640, 480))
    assert np.array_equal(out, d)


def test_register_translation_shifts_pixels():
    """Plane at z=1 m, colour camera 0.1 m to the +x of depth => points shift by -fx*0.1/z = -50 px in colour u... (p_c = p_d + t)."""
    d = np.full((480, 640), 1000, np.uint16)
    T = make_T(np.eye(3), [0.1, 0, 0])  # p_color = p_depth + (0.1,0,0) -> u_c = u_d + 50
    out = register_depth_to_color(d, K, K, T, (640, 480))
    assert (out[:, :50] == 0).all()  # nothing lands in the first 50 columns
    assert (out[:, 50:] == 1000).all()


def test_register_zbuffer_nearest_wins():
    d = np.zeros((480, 640), np.uint16)
    d[240, 320] = 2000  # far, projects to (320,240) in colour
    # a near point that also lands on colour pixel (320,240): depth px (x,y) at z=1 -> shift so both coincide
    T = np.eye(4)
    d[240, 321] = 1000  # at z=1 m u=321 -> x=1/500 m; colour px = 321 (identity), so different pixel
    out = register_depth_to_color(d, K, K, T, (640, 480))
    assert out[240, 320] == 2000 and out[240, 321] == 1000
    # two depth pixels with different depth mapping to the same colour pixel through a tilted extrinsic
    T2 = make_T(np.eye(3), [0.002 * 1.0, 0, 0])  # at z=1 shifts by +1 px: (320 -> 321 for z=1), z=2 shifts 0.5
    d2 = np.zeros((480, 640), np.uint16)
    d2[240, 321] = 2000  # u_c = 321 + 0.5 -> rint 322 (banker: 321.5->322)
    d2[240, 320] = 1000  # u_c = 320 + 1 = 321
    out2 = register_depth_to_color(d2, K, K, T2, (640, 480))
    assert out2[240, 321] == 1000


def test_register_upsample_no_holes():
    """848x480-like low-res depth into a 2x larger colour image: splatting leaves no holes."""
    Kd = np.array([[250.0, 0, 160], [0, 250.0, 120], [0, 0, 1]])
    Kc = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    d = np.full((240, 320), 1500, np.uint16)
    out = register_depth_to_color(d, Kd, Kc, np.eye(4), (640, 480))
    assert (out[2:-2, 2:-2] == 1500).all()


def test_register_depth_scale_and_range():
    d = np.full((480, 640), 500, np.uint16)  # 0.5 mm units... scale 0.002 -> 1.0 m
    out = register_depth_to_color(d, K, K, np.eye(4), (640, 480), depth_scale_m=0.002)
    assert (out == 500).all()
    out = register_depth_to_color(d, K, K, np.eye(4), (640, 480), depth_scale_m=0.002, max_depth_m=0.5)
    assert (out == 0).all()


def test_quat_and_inverse():
    R = quat_to_R(0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4))  # 90 deg about z
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    T = make_T(R, [1, 2, 3])
    assert np.allclose(invert_T(T) @ T, np.eye(4), atol=1e-12)


# ---------- IMU ----------
def test_unite_uses_faster_stream_and_interpolates():
    gt = np.arange(0, 1_000_000_000, 5_000_000, dtype=np.int64) + 10**18  # 200 Hz
    at = np.arange(0, 1_000_000_000, 8_000_000, dtype=np.int64) + 10**18 + 1_000_000  # 125 Hz, offset
    gyro = np.stack([gt - gt[0], 0 * gt, 0 * gt], 1).astype(float) / 1e9
    accel = np.stack([0 * at, 0 * at, (at - at[0]) / 1e9 * 2.0], 1)  # linear ramp z = 2 t
    u = imu_mod.unite(gt, gyro, at, accel)
    assert len(u) < len(gt) and u.t_ns[0] >= at[0] and u.t_ns[-1] <= at[-1]
    expected = (u.t_ns - at[0]) / 1e9 * 2.0
    assert np.allclose(u.accel[:, 2], expected, atol=1e-9)
    assert np.allclose(u.gyro[:, 0], (u.t_ns - gt[0]) / 1e9)
    # swap which is faster
    u2 = imu_mod.unite(at, accel * 0, gt, gyro)
    assert len(u2) == len(u)


def test_unite_rejects_disjoint():
    with pytest.raises(ValueError):
        imu_mod.unite([0, 1], np.zeros((2, 3)), [10, 11], np.zeros((2, 3)))


def test_clip_and_dedupe():
    t = np.array([5, 1, 3, 3, 9], np.int64)
    z = np.zeros((5, 3))
    s = imu_mod.sort_dedupe(imu_mod.ImuData(t, z, z))
    assert list(s.t_ns) == [1, 3, 5, 9]
    c = imu_mod.clip(s, 3, 5, margin_ns=0)
    assert list(c.t_ns) == [3, 5]


# ---------- pairing ----------
def test_pairer_matches_and_counts_drops():
    p = StreamPairer(tol_ns=10, max_pending=3)
    got = []
    got += p.push("color", 100, "c0")
    got += p.push("depth", 103, "d0")
    got += p.push("color", 200, "c1")  # depth for c1 never arrives
    got += p.push("color", 300, "c2")
    got += p.push("depth", 302, "d2")
    assert [(t, c, d) for t, c, d in got] == [(100, "c0", "d0"), (300, "c2", "d2")]
    assert p.dropped["color"] == 1  # c1 overtaken by the c2/d2 match


# ---------- format detection ----------
def test_detect_formats(tmp_path: Path):
    m = tmp_path / "a.mcap"
    m.write_bytes(b"\x89MCAP0\r\n" + b"x" * 8)
    b = tmp_path / "b.bag"
    b.write_bytes(b"#ROSBAG V2.0\n" + b"x" * 8)
    tum = tmp_path / "tum"
    (tum / "rgb").mkdir(parents=True)
    (tum / "depth").mkdir()
    rosdir = tmp_path / "bagdir"
    rosdir.mkdir()
    (rosdir / "x_0.mcap").write_bytes(b"\x89MCAP0\r\n")
    assert detect_format(m) == MCAP
    assert detect_format(b) == BAG
    assert detect_format(tum) == TUM
    assert detect_format(rosdir) == MCAP
    mis = tmp_path / "renamed.dat"
    mis.write_bytes(b"#ROSBAG V2.0\n")
    assert detect_format(mis) == BAG  # magic wins over the extension
    junk = tmp_path / "x.db3"
    junk.write_bytes(b"SQLite format 3\0")
    with pytest.raises(UnknownFormat):
        detect_format(junk)
    with pytest.raises(UnknownFormat):
        detect_format(tmp_path / "missing")


# ---------- topic discovery, tf, tum ----------
def test_discover_topics_namespaces():
    for ns in ("/camera/camera", "/camera", "/d435i"):
        t = {f"{ns}/color/image_raw": "sensor_msgs/msg/Image", f"{ns}/aligned_depth_to_color/image_raw": "sensor_msgs/msg/Image",
             f"{ns}/depth/image_rect_raw": "sensor_msgs/msg/Image", f"{ns}/color/camera_info": "sensor_msgs/msg/CameraInfo",
             f"{ns}/imu": "sensor_msgs/msg/Imu", "/tf_static": "tf2_msgs/msg/TFMessage"}
        r = discover_topics(t)
        assert r["color"] == f"{ns}/color/image_raw" and r["depth_aligned"].endswith("aligned_depth_to_color/image_raw")
        assert r["imu"] == f"{ns}/imu" and r["gyro"] is None and r["tf_static"] == "/tf_static"


def test_discover_topics_oak():
    t = {"/oak/rgb/image_raw": "sensor_msgs/msg/Image", "/oak/rgb/image_raw/compressed": "sensor_msgs/msg/CompressedImage",
         "/oak/rgb/camera_info": "sensor_msgs/msg/CameraInfo", "/oak/stereo/image_raw": "sensor_msgs/msg/Image",
         "/oak/stereo/camera_info": "sensor_msgs/msg/CameraInfo", "/oak/imu/data": "sensor_msgs/msg/Imu",
         "/tf_static": "tf2_msgs/msg/TFMessage"}
    r = discover_topics(t)
    assert r["color"] == "/oak/rgb/image_raw" and r["color_info"] == "/oak/rgb/camera_info"
    assert r["depth_aligned"] == "/oak/stereo/image_raw" and r["aligned_info"] == "/oak/stereo/camera_info"
    assert r["imu"] == "/oak/imu/data" and r["depth_raw"] is None and r["tf_static"] == "/tf_static"


def test_tf_lookup_chain():
    g = TfGraph()
    g.add("link", "depth", make_T(np.eye(3), [0, 0, 0]))
    g.add("link", "color", make_T(np.eye(3), [0.015, 0, 0]))
    T = g.lookup("color", "depth")  # points in depth frame -> colour frame
    assert np.allclose(T[:3, 3], [-0.015, 0, 0])
    assert g.lookup("color", "nowhere") is None


def test_tum_timestamps_and_association():
    assert parse_ts_ns("1739373727.450895309") == 1739373727450895309
    rgb = [(100, "r0"), (200, "r1"), (300, "r2")]
    dep = [(101, "d0"), (205, "d1"), (900, "d9")]
    assert associate(rgb, dep, max_dt_ns=10) == [(100, "r0", "d0"), (200, "r1", "d1")]


def test_intrinsics_parsing_variants():
    a = parse_intrinsics({"fx": 600, "fy": 601, "cx": 424, "cy": 240, "width": 848, "height": 480})
    assert (a.fx, a.fy, a.width) == (600, 601, 848)
    b = parse_intrinsics({"camera_matrix": {"data": [600, 0, 424, 0, 601, 240, 0, 0, 1]},
                          "distortion_coefficients": {"data": [0.1, 0, 0, 0, 0]}, "image_width": 848, "image_height": 480})
    assert b.fy == 601 and b.distortion[0] == 0.1
    c = parse_intrinsics({"color": {"K": [600, 0, 424, 0, 601, 240, 0, 0, 1], "D": [0] * 5}}, default_size=(848, 480))
    assert c.width == 848 and c.cx == 424
    with pytest.raises(ValueError):
        parse_intrinsics({"foo": 1}, default_size=(1, 1))
    s = Intrinsics(600, 600, 424, 240, 848, 480).scaled(424, 240)
    assert abs(s.fx - 300) < 1e-9


# ---------- needs cv2 / mcap ----------
def test_keyframes_select():
    pytest.importorskip("cv2")
    from splat2sim.ingest import keyframes as kf

    sharp = [10, 50, 5, 60, 70, 2]
    assert kf.select(sharp) == [0, 1, 2, 3, 4, 5]
    assert kf.select(sharp, every=2) == [0, 2, 4]
    assert kf.select(sharp, min_sharpness=20) == [1, 3, 4]
    assert kf.select(sharp, every=3, min_sharpness=20) == [1, 4]  # sharpest of each window of 3
    flat = np.full((64, 64, 3), 128, np.uint8)
    noisy = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert kf.sharpness(noisy) > kf.sharpness(flat) + 100


@pytest.mark.parametrize("aligned,separate", [(True, False), (False, True)])
def test_end_to_end_synthetic_mcap(tmp_path, aligned, separate):
    pytest.importorskip("cv2")
    pytest.importorskip("mcap_ros2")
    pytest.importorskip("rosbags")
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from synth_mcap import write_mcap

    from splat2sim.ingest.pipeline import Options, run

    cap = tmp_path / "synth.mcap"
    write_mcap(cap, n_frames=8, aligned=aligned, separate_imu=separate)
    out = tmp_path / "frames"
    meta = run(cap, out, Options(every=2))
    assert meta["source_format"] == "ros2-mcap" and meta["frames"] == 8 and meta["keyframes"] == 4
    assert (out / "keyframes.txt").read_text().split() == ["0", "2", "4", "6"]
    intr = json.loads((out / "intrinsics.json").read_text())
    assert intr["depth_scale_m"] == 0.001 and intr["width"] == 160 and abs(intr["fx"] - 144) < 1e-9
    import cv2

    d = cv2.imread(str(out / "depth" / "000003.png"), cv2.IMREAD_UNCHANGED)
    c = cv2.imread(str(out / "color" / "000003.png"), cv2.IMREAD_UNCHANGED)
    assert d.dtype == np.uint16 and d.shape == (120, 160) and c.shape == (120, 160, 3) and c.dtype == np.uint8
    assert d[60, 80] in (600,) and d[5, 5] == 1000 or aligned is False
    rows = (out / "frames.csv").read_text().splitlines()
    assert rows[0].startswith("index,timestamp_ns,color,depth") and len(rows) == 9
    assert meta["imu"]["present"] and (out / "imu.csv").read_text().startswith("timestamp_ns,gx,gy,gz,ax,ay,az")
    assert (out / "preview").is_dir()
    if not aligned:
        assert meta["source"]["depth_mode"] == "register"
        assert 0.9 < meta["depth_valid_fraction_mean"] <= 1.0


def test_end_to_end_tum_folder(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from splat2sim.ingest.pipeline import Options, run

    cap = tmp_path / "tum"
    (cap / "rgb").mkdir(parents=True)
    (cap / "depth").mkdir()
    base = 1739373727
    assoc = ["#rgb_timestamp rgb_file depth_timestamp depth_file"]
    for i in range(5):
        ts = f"{base}.{i * 33:03d}000000"
        cv2.imwrite(str(cap / "rgb" / f"{ts}.png"), np.random.default_rng(i).integers(0, 255, (48, 64, 3), dtype=np.uint8))
        cv2.imwrite(str(cap / "depth" / f"{ts}.png"), np.full((48, 64), 1234, np.uint16))
        assoc.append(f"{ts} rgb/{ts}.png {ts} depth/{ts}.png")
    (cap / "associations.txt").write_text("\n".join(assoc) + "\n")
    (cap / "imu.txt").write_text("#t ax ay az gx gy gz\n" + "".join(
        f"{base}.{k * 5:03d}000000 0 0 9.81 0.1 0.2 0.3\n" for k in range(40)))
    (cap / "groundtruth.txt").write_text("#t qx qy qz qw tx ty tz\n" + "".join(
        f"{base}.{k * 8:03d}000000 0 0 0 1 {k * 10.0} 0 0\n" for k in range(30)))
    (cap / "intrinsics.yaml").write_text("fx: 50\nfy: 51\ncx: 32\ncy: 24\nwidth: 64\nheight: 48\n")
    out = tmp_path / "frames"
    meta = run(cap, out, Options())
    assert meta["source_format"] == "tum-folder" and meta["frames"] == 5
    intr = json.loads((out / "intrinsics.json").read_text())
    assert intr["fx"] == 50 and intr["fy"] == 51 and intr["source"].startswith("sidecar")
    first = (out / "frames.csv").read_text().splitlines()[1].split(",")
    assert first[1] == f"{base}000000000"  # exact ns, no float rounding
    imu = (out / "imu.csv").read_text().splitlines()
    assert imu[0] == "timestamp_ns,gx,gy,gz,ax,ay,az" and imu[1].split(",")[1:] == ["0.1", "0.2", "0.3", "0", "0", "9.81"]
    assert meta["gt_poses"]["translation_units"] == "mm" and (out / "gt_poses.csv").is_file()
    # no sidecar -> nominal intrinsics flagged as assumed
    (cap / "intrinsics.yaml").unlink()
    run(cap, tmp_path / "f2", Options(no_preview=True))
    assert "ASSUMED" in json.loads((tmp_path / "f2" / "intrinsics.json").read_text())["source"]


def test_frames_folder_reader_roundtrip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import json

    import numpy as np

    from splat2sim.ingest.detect import FRAMES, detect_format
    from splat2sim.ingest.readers import open_source

    (tmp_path / "color").mkdir()
    (tmp_path / "depth").mkdir()
    rows = ["index,timestamp_ns,color,depth,sharpness"]
    for i in range(3):
        cv2.imwrite(str(tmp_path / "color" / f"{i:06d}.png"), np.full((12, 16, 3), 40 * i, np.uint8))
        cv2.imwrite(str(tmp_path / "depth" / f"{i:06d}.png"), np.full((12, 16), 1000 + i, np.uint16))
        rows.append(f"{i},{1000 + i * 33},color/{i:06d}.png,depth/{i:06d}.png,1.0")
    (tmp_path / "frames.csv").write_text("\n".join(rows) + "\n")
    (tmp_path / "intrinsics.json").write_text(json.dumps({"fx": 10.0, "fy": 10.0, "cx": 8.0, "cy": 6.0, "width": 16, "height": 12}))
    (tmp_path / "gt_poses.csv").write_text("timestamp_ns,tx,ty,tz,qx,qy,qz,qw\n1000,0,0,0,0,0,0,1\n")
    assert detect_format(tmp_path) == FRAMES
    src = open_source(tmp_path)
    fr = list(src.frames())
    assert len(fr) == 3 and fr[2].depth[0, 0] == 1002 and fr[1].color.shape == (12, 16, 3)
    assert src.intrinsics().fx == 10.0 and src.gt_units == "m" and src.gt_poses()[1].shape == (1, 7)


# ---------- pinhole invariant: undistortion at ingest ----------
PLUMB = ("plumb_bob", [-0.30, 0.10, 0.001, -0.001, 0.0])
RATIONAL14 = ("rational_polynomial", [0.45, 0.12, 0.0004, -0.0004, 0.0, 0.55, 0.10, 0.0, 0, 0, 0, 0, 0, 0])


def _synth_K(w, h, hfov_deg=100.0):
    f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])


@pytest.mark.parametrize("model,D", [PLUMB, RATIONAL14])
def test_undistort_roundtrip_matches_ideal_pinhole(model, D):
    """Render a textured plane through the distorted lens, undistort, compare with the analytic ideal pinhole image."""
    pytest.importorskip("cv2")
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from synth_mcap import plane_pattern, render_plane

    from splat2sim.ingest.undistort import Undistorter, opencv_coeffs

    w, h = 320, 240
    K = _synth_K(w, h)
    img = render_plane(w, h, K, opencv_coeffs(model, D), 1.0)
    intr = Intrinsics.from_K(K, w, h, model, D)
    u = Undistorter.build(intr)
    out = u.color(img)
    nK = u.new_K
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    x, y = (uu - nK[0, 2]) / nK[0, 0], (vv - nK[1, 2]) / nK[1, 1]
    g = plane_pattern(x, y)
    expect = np.clip(np.stack([g, 255 - g, 0.5 * g + 60], -1), 0, 255)
    err = np.abs(out.astype(float) - expect)[h // 8: -h // 8, w // 8: -w // 8]  # periphery is minified by the lens: bilinear is coarse there
    assert np.median(err) < 1.5 and np.percentile(err, 99) < 10, (np.median(err), np.percentile(err, 99))
    # new intrinsics: zero distortion, same size, no black borders (alpha=0 -> every output pixel samples inside the source)
    ni = u.intrinsics("test")
    assert ni.distortion == [0.0] * 5 and (ni.width, ni.height) == (w, h)
    assert u.valid_fraction >= 0.9995
    assert (u.depth(np.full((h, w), 1000, np.uint16)) > 0).mean() > 0.999


def test_undistort_depth_is_nearest_and_passthrough_when_ideal():
    pytest.importorskip("cv2")
    from splat2sim.ingest.undistort import is_distorted, maybe_build

    w, h = 160, 120
    K = _synth_K(w, h)
    assert maybe_build(Intrinsics.from_K(K, w, h, "plumb_bob", [0.0] * 5)) is None
    assert maybe_build(Intrinsics.from_K(K, w, h, "rational_polynomial", [0.0] * 14)) is None
    assert maybe_build(Intrinsics.from_K(K, w, h, "plumb_bob", [1e-6] * 5)) is None  # numerical noise passes through
    assert is_distorted("plumb_bob", [0.01, 0, 0, 0, 0])
    u = maybe_build(Intrinsics.from_K(K, w, h, *PLUMB))
    assert u is not None
    d = np.full((h, w), 1000, np.uint16)
    d[:, w // 2:] = 600
    out = u.depth(d)
    assert set(np.unique(out)) <= {0, 600, 1000}  # nearest: no invented in-between values


def test_undistort_rejects_unsupported_models():
    pytest.importorskip("cv2")
    from splat2sim.ingest.undistort import opencv_coeffs

    with pytest.raises(ValueError, match="fisheye"):
        opencv_coeffs("equidistant", [0.1, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        opencv_coeffs("plumb_bob", [0.1, 0.2, 0.3])  # 3 coefficients is not a valid OpenCV vector
    assert len(opencv_coeffs("rational_polynomial", list(range(14)))) == 14


def _oak_mcap(tmp_path, **kw):
    pytest.importorskip("cv2")
    pytest.importorskip("mcap_ros2")
    pytest.importorskip("rosbags")
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from synth_mcap import write_mcap

    cap = tmp_path / "oak.mcap"
    write_mcap(cap, n_frames=6, layout="oak", color_size=(160, 120), depth_size=(160, 120), **kw)
    return cap


@pytest.mark.parametrize("depth_encoding", ["16UC1", "32FC1"])
def test_end_to_end_oak_already_undistorted(tmp_path, depth_encoding):
    """OAK with on-device undistortion: camera_info D == 0 -> pass-through; 32FC1 metres converted to uint16 mm; bgr8 -> RGB."""
    from splat2sim.ingest.pipeline import Options, run

    cap = _oak_mcap(tmp_path, depth_encoding=depth_encoding, color_encoding="bgr8",
                    capture_info={"product": "OAK-D-PRO-W", "driver": "depthai_ros_driver_v3", "serial": "X"})
    out = tmp_path / "frames"
    meta = run(cap, out, Options())
    assert meta["frames"] == 6 and meta["imu"]["present"]
    assert meta["source"]["topics"]["color"] == "/oak/rgb/image_raw" and meta["source"]["depth_topic"] == "/oak/stereo/image_raw"
    assert meta["undistorted_in_ingest"] is False
    assert meta["camera"]["product"] == "OAK-D-PRO-W" and meta["camera_serial"] == "X"
    intr = json.loads((out / "intrinsics.json").read_text())
    assert intr["undistorted_in_ingest"] is False and abs(intr["fx"] - 144) < 1e-9
    import cv2

    d = cv2.imread(str(out / "depth" / "000002.png"), cv2.IMREAD_UNCHANGED)
    assert d.dtype == np.uint16 and d[60, 80] == 600 and d[5, 5] == 1000
    c = cv2.cvtColor(cv2.imread(str(out / "color" / "000002.png")), cv2.COLOR_BGR2RGB)
    assert c.shape == (120, 160, 3)


@pytest.mark.parametrize("model,D", [PLUMB, RATIONAL14])
def test_end_to_end_oak_distorted_is_undistorted_in_ingest(tmp_path, model, D):
    """Recorded with rgb.i_undistorted=false: ingest removes the distortion and writes a pinhole intrinsics.json."""
    from splat2sim.ingest.pipeline import Options, run
    import cv2
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from synth_mcap import plane_pattern

    cap = _oak_mcap(tmp_path, distortion=(model, D), plane=True)
    out = tmp_path / "frames"
    meta = run(cap, out, Options())
    intr = json.loads((out / "intrinsics.json").read_text())
    assert intr["undistorted_in_ingest"] is True and meta["undistorted_in_ingest"] is True
    assert intr["original_camera_model"] == model and intr["original_distortion_coeffs"] == list(D)
    assert intr["distortion_model"] == "plumb_bob" and set(intr["distortion_coeffs"]) == {0.0}
    assert "UNDISTORTED" in intr["source"] and meta["camera_distortion_model"] == model
    assert (intr["width"], intr["height"]) == (160, 120)
    # the written K describes the written image: compare with the analytic ideal pinhole render
    c = cv2.cvtColor(cv2.imread(str(out / "color" / "000001.png")), cv2.COLOR_BGR2RGB).astype(float)
    uu, vv = np.meshgrid(np.arange(160), np.arange(120))
    g = plane_pattern((uu - intr["cx"]) / intr["fx"], (vv - intr["cy"]) / intr["fy"])
    assert np.median(np.abs(c[..., 0] - np.clip(g, 0, 255))) < 2.5
    d = cv2.imread(str(out / "depth" / "000001.png"), cv2.IMREAD_UNCHANGED)
    assert set(np.unique(d)) <= {0, 600, 1000} and (d > 0).mean() > 0.99
    assert meta["source"]["original_intrinsics"]["model"] == model
