import json

import numpy as np
import pytest

from splat2sim.report import coverage, summary


def test_table_grid_on_top_face():
    tb = {"center": [0.5, 0.0, -0.22], "size": [1.0, 0.6, 0.04], "rpy_deg": [0, 0, 0]}
    g = coverage.table_grid(tb, cell=0.1)
    assert np.allclose(g[:, 2], -0.20) and g[:, 0].min() > 0.0 and g[:, 0].max() < 1.0 and len(g) == 10 * 6


def test_view_counts_visible_vs_occluded():
    intr = {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240}
    pts = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [5.0, 0.0, 1.0]])  # in front, behind an occluder at 1 m, outside the image
    T = np.eye(4)
    d = np.full((240, 320), 1.0, np.float32)  # a wall at 1 m covers the image
    cnt = coverage.view_counts(pts, np.array([T, T]), lambda k: d, intr)
    assert cnt.tolist() == [2, 0, 0]


def test_checks_flag_redo_and_fix():
    S = {"anchor": "charuco", "anchor_detail": {"status": "charuco", "reproj_rms_px": 4.5, "corner_3d_rms_mm": 8.0,
                                                "scale_estimate": 1.0, "frames_used": list(range(20))},
         "board_seen_s": 12.0, "frames": {"slam_total": 100, "registered": 95, "loop_closures": 3},
         "splat": {"psnr": 24.0}, "collision": {"table_plane": {"rms_m": 0.02}}, "mask": {"frames": 10, "frames_with_any_mask": 9}}
    rows = {r["name"]: r for r in summary.build_checks(S)}
    assert rows["anchor_reproj_rms_px"]["level"] == "redo"
    assert rows["board_seen_seconds"]["level"] == "warn"
    assert rows["heldout_psnr_db"]["level"] == "ok"
    assert rows["table_plane_rms_mm"]["level"] == "fix"


def test_checks_fallback_anchor_is_redo():
    rows = summary.build_checks({"anchor": "fallback", "anchor_detail": {"status": "fallback", "reason": "no board"}})
    assert rows[0]["level"] == "redo"


def test_render_md_smoke(tmp_path):
    S = {"scene": "x", "anchor": "charuco", "anchor_detail": {"status": "charuco", "frames_used": [1]}, "frames": {}, "checks": [],
         "verdict": {"redo_take": False, "reasons_redo": [], "config_fixes": [], "warnings": []}, "previews": [], "flythrough": {"status": "x"}}
    md = summary.render_md(S, tmp_path)
    assert md.startswith("# Splat2Sim report: x") and "capture usable" in md
