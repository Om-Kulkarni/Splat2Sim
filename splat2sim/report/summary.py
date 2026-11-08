"""Collate stage outputs into ``report/summary.json`` + ``report/report.md`` (CPU; no Isaac).

Every check has a ``level``: ``ok`` | ``warn`` | ``redo`` (the Capture itself is probably not good enough: board, tracking, mask,
coverage, splat quality) | ``fix`` (a config / measurement problem: lab config, overrides, collision options). Thresholds in
THRESH, documented in docs/first-capture.md.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

THRESH = {
    "board_seen_s": (20.0, 10.0),            # warn below, redo below
    "anchor_reproj_px": (2.0, 4.0),          # warn above, redo above
    "anchor_corner3d_mm": (15.0, 30.0),
    "anchor_scale_dev": (0.02, 0.05),
    "anchor_frames": (8, 3),
    "poses_registered": (0.9, 0.7),          # fraction of frames; warn below, redo below
    "mask_frames_with_robot": (0.6, 0.2),
    "silhouette_covered": (0.9, 0.75),       # fraction of the URDF silhouette inside the SAM mask (median over frames)
    "silhouette_iou_dilated": (0.5, 0.3),
    "table_views_ge3": (0.8, 0.5),
    "psnr_db": (22.0, 18.0),
    "table_plane_rms_mm": (8.0, 15.0),
    "heldout_depth_median_cm": (2.0, 4.0),
    "clearance_removed_fraction": (0.25, 0.6),
    "env_success": (0.6, 0.0),
}


def _read(p: Path, default=None):
    try:
        return json.loads(p.read_text()) if p.is_file() else default
    except json.JSONDecodeError:
        return default


class Checks:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, name, value, level, msg, kind="capture"):
        self.rows.append({"name": name, "value": value, "level": level, "message": msg, "kind": kind})

    def low_is_bad(self, name, value, key, unit, msg, kind="capture", redo_level="redo"):
        if value is None:
            return
        warn, redo = THRESH[key]
        lvl = redo_level if value < redo else ("warn" if value < warn else "ok")
        self.add(name, round(float(value), 4), lvl, f"{msg} ({value:.3g}{unit}; warn < {warn}, {redo_level} < {redo})" if lvl != "ok" else msg, kind)

    def high_is_bad(self, name, value, key, unit, msg, kind="capture", redo_level="redo"):
        if value is None:
            return
        warn, redo = THRESH[key]
        lvl = redo_level if value > redo else ("warn" if value > warn else "ok")
        self.add(name, round(float(value), 4), lvl, f"{msg} ({value:.3g}{unit}; warn > {warn}, {redo_level} > {redo})" if lvl != "ok" else msg, kind)


def build_checks(S: dict) -> list[dict]:
    c = Checks()
    a = S.get("anchor_detail") or {}
    status = a.get("status")
    if status == "charuco":
        c.high_is_bad("anchor_reproj_rms_px", a.get("reproj_rms_px"), "anchor_reproj_px", " px", "ChArUco corner reprojection RMS")
        c.high_is_bad("anchor_corner_3d_rms_mm", a.get("corner_3d_rms_mm"), "anchor_corner3d_mm", " mm", "depth-backprojected corners vs board model")
        if a.get("scale_estimate") is not None:
            c.high_is_bad("anchor_scale_deviation", abs(a["scale_estimate"] - 1.0), "anchor_scale_dev", "", "recon->board scale vs 1 (wrong square_length_m or depth scale?)")
        c.low_is_bad("anchor_frames_used", len(a.get("frames_used") or []), "anchor_frames", "", "frames with a consistent board pose")
        sec = S.get("board_seen_s")
        if sec is not None:
            c.low_is_bad("board_seen_seconds", sec, "board_seen_s", " s", "time the board was in view (estimate)")
    elif status:
        c.add("anchor_status", status, "redo", f"no usable ChArUco anchor ({a.get('reason')}): the scene is NOT aligned to the robot")
    else:
        c.add("anchor_status", None, "warn", "no anchor.json")
    p = S.get("frames") or {}
    if p.get("slam_total") and p.get("registered") is not None:
        c.low_is_bad("poses_registered_fraction", p["registered"] / p["slam_total"], "poses_registered", "", "frames registered by RTAB-Map")
    if p.get("loop_closures") == 0:
        c.add("loop_closures", 0, "warn", "no loop closures: pose drift is unchecked (end the take where it started)")
    m = S.get("mask") or {}
    if m.get("frames"):
        c.low_is_bad("mask_frames_with_robot", m["frames_with_any_mask"] / m["frames"], "mask_frames_with_robot", "", "frames where SAM found the robot")
    sil = S.get("mask_vs_urdf") or {}
    if sil.get("status") == "ok" and sil.get("frames_robot_visible"):
        c.low_is_bad("silhouette_covered_by_mask", sil.get("median_silhouette_covered_by_mask"), "silhouette_covered", "",
                     "URDF silhouette inside the mask: robot ghosts / anchor misalignment otherwise", kind="config", redo_level="fix")
        c.low_is_bad("silhouette_iou_dilated", sil.get("median_iou_vs_dilated_silhouette"), "silhouette_iou_dilated", "",
                     "IoU of mask vs dilated URDF silhouette (board_to_base, scan pose, coupling offset)", kind="config", redo_level="fix")
    elif sil.get("status", "").startswith("skipped"):
        c.add("mask_vs_urdf", None, "warn", f"mask-vs-URDF IoU not computed ({sil['status']})", "config")
    cov = S.get("table_coverage") or {}
    if "fraction_ge_3_views" in cov:
        c.low_is_bad("table_views_ge3", cov["fraction_ge_3_views"], "table_views_ge3", "", "table top seen from >= 3 keyframes: do a slow close pass over the table")
    sp = S.get("splat") or {}
    if sp.get("psnr") is not None:
        c.low_is_bad("heldout_psnr_db", sp["psnr"], "psnr_db", " dB", "held-out PSNR of the splat")
    col = S.get("collision") or {}
    tp = col.get("table_plane") or {}
    if tp.get("rms_m") is not None:
        c.high_is_bad("table_plane_rms_mm", tp["rms_m"] * 1000, "table_plane_rms_mm", " mm", "RANSAC table-plane RMS", kind="config", redo_level="fix")
    de = ((col.get("depth_error") or {}).get("collision") or {})
    if de.get("median_m") is not None:
        c.high_is_bad("heldout_depth_median_cm", de["median_m"] * 100, "heldout_depth_median_cm", " cm", "held-out depth to collision geometry (median)", kind="config", redo_level="fix")
    if (col.get("table_box") or {}).get("source") == "override":
        c.add("table_box_source", "override", "ok", "table box is a lab-config override (fit not used)", "config")
    cl = (col.get("collision_mesh") or {}).get("robot_clearance") or {}
    if cl.get("triangles_before"):
        c.high_is_bad("clearance_removed_fraction", cl["triangles_removed"] / cl["triangles_before"], "clearance_removed_fraction", "",
                      "share of the collision mesh removed by the robot clearance volume", kind="config", redo_level="fix")
    if col and (col.get("pedestal_box") or {}).get("source") == "absent" and S.get("anchor") == "charuco":
        c.add("pedestal_box", "absent", "warn", "no pedestal box fitted: set overrides.pedestal_box", "config")
    env = S.get("env")
    if env and env.get("success_rate") is not None:
        lvl = "ok" if env["success_rate"] >= THRESH["env_success"][0] else "fix"
        c.add("env_pick_lift_success", env["success_rate"], lvl, f"scripted pick-and-lift {env.get('successes')}/{env.get('trials')}", "config")
        ss = env.get("scene_safety") or {}
        for w in ss.get("warnings", []):
            c.add("env_scene_safety", ss.get("mesh_min_distance_m"), "fix", w, "config")
    return c.rows


def finish(run: Path) -> dict:
    from . import coverage, silhouette, video

    t0 = time.time()
    run = Path(run)
    out = run / "report"
    out.mkdir(parents=True, exist_ok=True)
    isaac = _read(out / "isaac.json", {})
    poses_stats = _read(run / "poses" / "stats.json", {})
    frames_meta = _read(run / "frames" / "meta.json", {})
    anchor = _read(run / "poses" / "anchor.json")
    mask_sum = _read(run / "masks" / "summary.json")
    S: dict = {
        "scene": run.name,
        "anchor": (isaac.get("scene_meta") or {}).get("anchor"),
        "anchor_note": (isaac.get("scene_meta") or {}).get("note"),
        "anchor_detail": ({k: anchor.get(k) for k in ("status", "reason", "reproj_rms_px", "corner_3d_rms_mm", "scale_estimate",
                                                      "frames_used", "frames_with_board", "frames_considered", "ransac_inliers", "warnings")}
                          if anchor else None),
        "frames": {"ingested": frames_meta.get("frames"), "slam_total": poses_stats.get("frames_total"),
                   "registered": poses_stats.get("frames_registered"), "loop_closures": poses_stats.get("loop_closures"),
                   "colmap_images": (poses_stats.get("colmap") or {}).get("images"),
                   "masked_images": (poses_stats.get("colmap") or {}).get("masked_images")},
        "imu_used": (poses_stats.get("imu") or {}).get("used"),
        "ate": _read(run / "poses" / "eval.json"),
        "mask": mask_sum,
        "splat": _read(run / "splat" / "metrics.json"),
        "scene_check": (isaac.get("scene_meta") or {}).get("check"),
        "collision": _read(run / "collision" / "metrics.json"),
        "collision_check": _read(run / "collision" / "check.json"),
        "env": _read(run / "env" / "summary.json"),
    }
    # estimated time the board was in view (detections are on evenly sampled posed frames)
    if anchor and anchor.get("frames_considered") and frames_meta.get("duration_s"):
        S["board_seen_s"] = round(anchor.get("frames_with_board", 0) / anchor["frames_considered"] * frames_meta["duration_s"], 1)
    try:
        S["table_coverage"] = coverage.table_coverage(run)
    except Exception as e:  # report must not fail on a missing optional input
        S["table_coverage"] = {"status": f"error: {type(e).__name__}: {e}"}
    try:
        S["mask_vs_urdf"] = silhouette.evaluate(run)
    except Exception as e:
        S["mask_vs_urdf"] = {"status": f"error: {type(e).__name__}: {e}"}
    S["flythrough"] = video.encode(out / "flythrough", out / "flythrough.mp4")
    S["previews"] = isaac.get("previews", [])
    S["checks"] = build_checks(S)
    redo = [r for r in S["checks"] if r["level"] == "redo"]
    fix = [r for r in S["checks"] if r["level"] == "fix"]
    warn = [r for r in S["checks"] if r["level"] == "warn"]
    S["verdict"] = {"redo_take": bool(redo), "reasons_redo": [r["message"] for r in redo],
                    "config_fixes": [r["message"] for r in fix], "warnings": [r["message"] for r in warn]}
    S["warnings"] = (list(poses_stats.get("warnings") or []) + list((anchor or {}).get("warnings") or [])
                     + [r["message"] for r in warn + fix + redo]
                     + (["anchor is the gravity/plane fallback, not the ChArUco Anchor: scene is not aligned to the robot base"]
                        if S["anchor"] == "fallback" else []))
    S["timings_s"] = {"ingest": frames_meta.get("ingest_seconds"), "mask": (mask_sum or {}).get("timing", {}).get("inference_s"),
                      "poses": poses_stats.get("seconds"), "slam": poses_stats.get("slam_seconds"),
                      "splat": (S["splat"] or {}).get("wall_seconds"), "collision": ((S["collision"] or {}).get("seconds") or {}).get("total"),
                      "compose": (isaac.get("scene_meta") or {}).get("compose_seconds"), "report_isaac": isaac.get("seconds"),
                      "report_cpu": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(S, indent=2, default=str))
    (out / "report.md").write_text(render_md(S, out))
    print(f"[report] summary.json + report.md written: redo_take={S['verdict']['redo_take']}, "
          f"{len(redo)} redo / {len(fix)} fix / {len(warn)} warn checks", flush=True)
    try:
        import subprocess

        subprocess.run(["chmod", "-R", "a+rwX", str(out)], check=False)
    except Exception:
        pass
    return S


def _f(v, fmt="{:.3g}", none="n/a"):
    return none if v is None else fmt.format(v)


def render_md(S: dict, out: Path) -> str:
    a = S.get("anchor_detail") or {}
    sp = S.get("splat") or {}
    col = S.get("collision") or {}
    env = S.get("env") or {}
    sil = S.get("mask_vs_urdf") or {}
    cov = S.get("table_coverage") or {}
    v = S["verdict"]
    L = [f"# Splat2Sim report: {S['scene']}", ""]
    L.append(f"**Verdict: {'REDO THE TAKE' if v['redo_take'] else 'capture usable'}**"
             + (f" ({len(v['config_fixes'])} config issue(s))" if v["config_fixes"] else "") + "\n")
    for r in v["reasons_redo"]:
        L.append(f"- REDO: {r}")
    for r in v["config_fixes"]:
        L.append(f"- FIX: {r}")
    for r in v["warnings"]:
        L.append(f"- warn: {r}")
    L += ["", "## Headline numbers", "", "| | |", "|---|---|"]
    L.append(f"| Anchor | {S.get('anchor')} ({a.get('status')}) reproj {_f(a.get('reproj_rms_px'))} px, corners 3D {_f(a.get('corner_3d_rms_mm'))} mm, "
             f"scale {_f(a.get('scale_estimate'), '{:.4f}')}, {len(a.get('frames_used') or [])} frames, board in view ~{_f(S.get('board_seen_s'))} s |")
    fr = S.get("frames") or {}
    L.append(f"| Tracking | {fr.get('registered')}/{fr.get('slam_total')} frames registered, {fr.get('loop_closures')} loop closures, IMU used: {S.get('imu_used')} |")
    ate = S.get("ate") or {}
    if ate.get("ate_rmse_se3_m") is not None:
        L.append(f"| ATE vs GT | SE3 {ate['ate_rmse_se3_m'] * 1000:.1f} mm over {_f(ate.get('path_length_m'))} m |")
    m = S.get("mask") or {}
    if m:
        L.append(f"| Robot mask | {m.get('frames_with_any_mask')}/{m.get('frames')} frames with robot, mean masked {_f((m.get('masked_fraction') or {}).get('mean'), '{:.3%}')} |")
    if sil.get("status") == "ok":
        L.append(f"| Mask vs URDF silhouette | covered {_f(sil.get('median_silhouette_covered_by_mask'), '{:.2f}')}, IoU(dilated) {_f(sil.get('median_iou_vs_dilated_silhouette'), '{:.2f}')}, "
                 f"{sil.get('frames_robot_visible')} frames |")
    else:
        L.append(f"| Mask vs URDF silhouette | {sil.get('status')} |")
    L.append(f"| Splat | PSNR {_f(sp.get('psnr'), '{:.2f}')} dB, SSIM {_f(sp.get('ssim'), '{:.3f}')}, LPIPS {_f(sp.get('lpips'), '{:.3f}')}, {sp.get('num_gaussians')} gaussians, {_f(sp.get('wall_seconds'))} s |")
    tb = col.get("table_box") or {}
    rnd = lambda v: None if v is None else [round(float(x), 3) for x in v]  # noqa: E731
    L.append(f"| Table box | {tb.get('source')}: center {rnd(tb.get('center'))}, size {rnd(tb.get('size'))}, yaw {_f(tb.get('yaw_deg'))} deg |")
    de = ((col.get("depth_error") or {}).get("collision") or {})
    L.append(f"| Collision layer | held-out depth error median {_f(de.get('median_m', None) and de['median_m'] * 100)} cm, "
             f"mesh {(col.get('collision_mesh') or {}).get('triangles')} tris, robot clearance removed "
             f"{((col.get('collision_mesh') or {}).get('robot_clearance') or {}).get('triangles_removed')} |")
    if "fraction_ge_3_views" in cov:
        L.append(f"| Table coverage | {cov['fraction_ge_1_views']:.0%} seen, {cov['fraction_ge_3_views']:.0%} from >= 3 views, {cov['fraction_ge_5_views']:.0%} from >= 5 |")
    if env:
        ss = env.get("scene_safety") or {}
        L.append(f"| Task env | pick-and-lift {env.get('successes')}/{env.get('trials')}; reset safety: {ss.get('status')}"
                 f" (mesh min distance {_f(ss.get('mesh_min_distance_m') and ss['mesh_min_distance_m'] * 1000)} mm) |")
    L += ["", "## All checks", "", "| check | value | level | note |", "|---|---|---|---|"]
    for r in S["checks"]:
        L.append(f"| {r['name']} | {r['value']} | {r['level']} | {r['message']} |")
    L += ["", "## Previews", ""]
    shown = []
    for name in [p["name"] for p in S.get("previews", [])] + [f"silhouette_overlay_{r['frame']:06d}.png" for r in []]:
        shown.append(name)
    for o in (sil.get("overlays") or []):
        shown.append(o)
    if cov.get("heatmap"):
        shown.append(cov["heatmap"])
    for name in shown:
        if (out / name).is_file():
            L.append(f"![{name}]({name})")
    ft = S.get("flythrough") or {}
    L += ["", f"Fly-through: {ft.get('file') or ft.get('status')}" + (f" ({ft.get('frames')} frames)" if ft.get("frames") else ""), ""]
    L.append("Other outputs: `../poses/anchor_overlays/`, `../masks/preview/`, `../splat/heldout/`, `../collision/preview_*.png`, `../env/preview/`.")
    return "\n".join(L) + "\n"
