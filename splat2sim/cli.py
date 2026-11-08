"""``r2s`` command line interface."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import stages as st
from .config import LabConfig, load_lab, repo_root
from .recon.options import QUICK_ITERATIONS, set_option
from .runfolder import RunFolder, data_dir, ensure_data_dirs, git_rev

DOCTOR_GPU_IMAGES = ["python:3.12-slim", "nvidia/cuda:13.0.0-base-ubuntu24.04"]


def _scene_name(capture: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", capture.stem)
    return name or "scene"


def _load_run(run: RunFolder) -> tuple[LabConfig, Path | None] | None:
    meta = run.read_meta()
    if meta is None:
        return None
    lab = load_lab(meta["lab"], meta.get("override"))
    return lab, (Path(meta["capture"]) if meta.get("capture") else None)


def _stage_settings(a: argparse.Namespace) -> list[tuple[str, object]]:
    """``--quick`` and ``--set section.key=value`` -> run options (persisted in <run>/options.yaml)."""
    import yaml

    out: list[tuple[str, object]] = []
    if a.quick:
        out.append(("splat.iterations", QUICK_ITERATIONS))
    for item in a.set:
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects section.key=value, got {item!r}")
        out.append((key.strip(), yaml.safe_load(val)))
    return out


def cmd_build(a: argparse.Namespace) -> int:
    capture = Path(a.capture).expanduser().resolve()
    if not capture.exists():
        print(f"error: capture not found: {capture}", file=sys.stderr)
        return 2
    lab = load_lab(a.lab, a.override)
    run = RunFolder(a.scene or _scene_name(capture))
    ensure_data_dirs()
    run.ensure()
    (run.root / "lab.json").write_text(lab.canonical_json())  # container contract: /runs/<scene>/lab.json
    run.write_meta({"capture": str(capture), "lab": a.lab, "override": str(Path(a.override).resolve()) if a.override else None})
    try:
        selected = st.select(a.from_stage, a.to_stage)
        forced = {st.get(f).name for f in a.force}
    except (KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"scene {run.scene}  lab {lab.lab}  ->  {run.root}")
    try:
        for dotted, value in _stage_settings(a):
            set_option(run.root, dotted, value)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for stage in selected:
        if st.skippable(stage):
            print(f"[skip] {stage.name}: not implemented yet (optional)")
            continue
        if stage.name in forced:
            run.clear_marker(stage.name, stage.out_dir)
        expected = st.inputs_hash(stage, run, lab, capture)
        s = st.state(stage, run, expected)
        if s == "done":
            print(f"[skip] {stage.name}: up to date")
            continue
        missing = [d for d in stage.deps if run.read_marker(d, st.BY_NAME[d].out_dir) is None]
        if missing:
            print(f"error: stage '{stage.name}' needs {', '.join(missing)} to be done first", file=sys.stderr)
            return 1
        print(f"[run ] {stage.name} ({'stale' if s == 'stale' else 'pending'}) via service {stage.service}")
        try:
            st.execute(stage, run, dry_run=a.dry_run, capture=capture)
        except st.NotImplementedStage as e:
            print(f"error: {e}", file=sys.stderr)
            return 3
        except subprocess.CalledProcessError as e:
            print(f"error: stage '{stage.name}' failed (exit {e.returncode})", file=sys.stderr)
            return e.returncode or 1
        if not a.dry_run:
            run.write_marker(stage.name, stage.out_dir, expected)
    return 0


def cmd_status(a: argparse.Namespace) -> int:
    run = RunFolder(a.scene)
    loaded = _load_run(run)
    print(f"scene {a.scene}: {run.root}" + ("" if run.root.exists() else "  (does not exist yet)"))
    if loaded:
        lab, capture = loaded
        meta = run.read_meta() or {}
        print(f"  lab {lab.lab}  capture {meta.get('capture')}")
    for stage in st.STAGES:
        if loaded:
            expected = st.inputs_hash(stage, run, loaded[0], loaded[1])
        else:
            expected = None
        s = st.state(stage, run, expected)
        m = run.read_marker(stage.name, stage.out_dir)
        extra = f"  {m.timestamp}  {m.git_rev}" if m else ""
        impl = "" if stage.implemented else "  [not implemented yet]"
        print(f"  {stage.name:<10} {s:<8}{extra}{impl}")
    return 0


def cmd_view(a: argparse.Namespace) -> int:
    print("r2s view: not implemented yet (needs a logged-in X11 session; see docs). "
          f"Headless previews will be in {RunFolder(a.scene).dir('report')}")
    return 3


def _run(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return 1, str(e)


def cmd_doctor(a: argparse.Namespace) -> int:
    ok = True

    def report(name: str, good: bool, detail: str = "", warn: bool = False) -> None:
        nonlocal ok
        tag = "ok  " if good else ("warn" if warn else "FAIL")
        if not good and not warn:
            ok = False
        print(f"[{tag}] {name}" + (f": {detail}" if detail else ""))

    if shutil.which("docker"):
        rc, out = _run(["docker", "version", "--format", "client {{.Client.Version}} / server {{.Server.Version}}"])
        report("docker", rc == 0, out.splitlines()[-1] if out else "")
        rc, out = _run(["docker", "compose", "version", "--short"])
        report("docker compose", rc == 0, out)
    else:
        report("docker", False, "docker CLI not found")

    rc, out = _run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"])
    images = set(out.split()) if rc == 0 else set()
    gpu_img = next((i for i in DOCTOR_GPU_IMAGES if i in images), None)
    if gpu_img:
        rc, out = _run(["docker", "run", "--rm", "--gpus", "all", "--entrypoint", "", gpu_img, "nvidia-smi", "-L"])
        report("gpu (docker --gpus all)", rc == 0, out.splitlines()[0] if rc == 0 and out else out[-200:])
    else:
        report("gpu (docker --gpus all)", False, f"none of {DOCTOR_GPU_IMAGES} present locally (not pulling)")

    # nvidia-smi is injected into the container by the NVIDIA runtime, so any local image works.
    for img in ("r2s-isaaclab3-sim61:latest", "r2s-cli:latest"):
        report(f"image {img}", img in images, "" if img in images else "missing", warn=img == "r2s-cli:latest")

    d = data_dir()
    report("data dir", d.is_dir(), str(d))
    for sub in ("captures", "runs", "cache"):
        report(f"  {sub}/", (d / sub).is_dir(), "" if (d / sub).is_dir() else "will be created on first build", warn=True)
    env = d / ".env"
    if env.is_file():
        keys = [ln.split("=", 1)[0].strip() for ln in env.read_text().splitlines() if "=" in ln and not ln.lstrip().startswith("#")]
        empty = [ln.split("=", 1)[0].strip() for ln in env.read_text().splitlines()
                 if "=" in ln and not ln.lstrip().startswith("#") and not ln.split("=", 1)[1].strip()]
        report(".env", True, f"{len(keys)} keys present" + (f", empty: {', '.join(empty)}" if empty else ""))
    else:
        report(".env", False, f"{env} missing (copy .env.example; needed for HF_TOKEN)", warn=True)
    report("repo", (repo_root() / "compose.yaml").is_file(), f"{repo_root()} @ {git_rev()}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="r2s", description="Splat2Sim: Capture -> Scene -> Task environment")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="run the pipeline on a Capture")
    b.add_argument("capture", help="capture file/dir (MCAP, .bag, TUM folder)")
    b.add_argument("--lab", required=True, help="lab name (labs/<lab>.yaml) or path to a yaml")
    b.add_argument("--override", help="per-run override yaml merged on top of the lab config")
    b.add_argument("--scene", help="scene name (default: capture file stem)")
    b.add_argument("--from", dest="from_stage", metavar="STAGE", choices=st.NAMES)
    b.add_argument("--to", dest="to_stage", metavar="STAGE", choices=st.NAMES)
    b.add_argument("--force", action="append", default=[], metavar="STAGE", help="re-run this stage (repeatable)")
    b.add_argument("--quick", action="store_true", help=f"dev: train the splat for {QUICK_ITERATIONS} iterations")
    b.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="per-run stage option, e.g. poses.use_imu=false, splat.iterations=7000 (saved in <run>/options.yaml)")
    b.add_argument("--dry-run", action="store_true", help="print docker commands without running")
    b.set_defaults(fn=cmd_build)

    s = sub.add_parser("status", help="show per-stage state of a scene")
    s.add_argument("scene")
    s.set_defaults(fn=cmd_status)

    v = sub.add_parser("view", help="open a scene in Isaac Sim GUI (stub)")
    v.add_argument("scene")
    v.set_defaults(fn=cmd_view)

    d = sub.add_parser("doctor", help="check docker, GPU, data dir, .env")
    d.set_defaults(fn=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
