"""Stage registry: each stage maps to a docker compose service + command.

Stages whose ``implemented`` flag is False fail clearly instead of running.
Container contract: every service mounts ``<data>/runs`` at ``/runs`` (see
compose.yaml) and receives the scene name in ``$R2S_SCENE``, so its run folder
is ``/runs/$R2S_SCENE``; the lab config will be at ``/runs/$R2S_SCENE/lab.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import LabConfig, repo_root
from .runfolder import RunFolder, fingerprint_path, git_rev, sha


class NotImplementedStage(RuntimeError):
    pass


@dataclass(frozen=True)
class Stage:
    name: str
    service: str  # compose service
    command: tuple[str, ...]  # run inside the service container
    out_dir: str  # run-folder directory holding the outputs + marker
    deps: tuple[str, ...] = ()
    optional_deps: tuple[str, ...] = ()  # used when their marker exists (hashed), never required
    config_sections: tuple[str, ...] = ()  # LabConfig sections whose change invalidates the stage
    uses_capture: bool = False  # capture is bind-mounted read-only at /capture and exported as $R2S_CAPTURE
    run_as_user: bool = False  # run the container as the caller's uid:gid so run-folder files are not root-owned
    implemented: bool = False
    description: str = ""
    # extra (service, command, run_as_user) steps run after the main command, in order (e.g. report: silhouette render, CPU finish)
    post: tuple[tuple[str, tuple[str, ...], bool], ...] = ()


STAGES: list[Stage] = [
    Stage("ingest", "ingest", ("python", "-m", "splat2sim.ingest", "ingest"), "frames", uses_capture=True,
          run_as_user=True, implemented=True, description="Capture (MCAP/bag/TUM) -> frames/, intrinsics, imu"),
    Stage("mask", "sam3", ("python", "-m", "splat2sim.mask", "mask"), "masks", deps=("ingest",),
          config_sections=("mask",), run_as_user=True, implemented=True,
          description="SAM 3.1 Robot mask (masks/NNNNNN.png: 255 = keep, 0 = robot)"),
    Stage("poses", "rtabmap", ("python3", "-m", "splat2sim.recon", "poses"), "poses", deps=("ingest",),
          optional_deps=("mask",), run_as_user=True, implemented=True,
          description="metric camera poses (RTAB-Map), COLMAP model for 3DGRUT, ATE vs GT"),
    Stage("anchor", "ingest", ("python", "-m", "splat2sim.anchor", "anchor"), "poses", deps=("poses",),
          config_sections=("charuco", "board_to_base"), run_as_user=True, implemented=True,
          description="ChArUco Anchor: recon -> robot base Sim(3) (poses/anchor.json; status fallback without a board)"),
    Stage("splat", "3dgrut", ("python", "-m", "splat2sim.recon", "splat"), "splat", deps=("poses",),
          optional_deps=("mask",), run_as_user=True, implemented=True,  # anchor is applied at compose time, not trained on
          description="3DGUT training -> ParticleField USDZ + held-out PSNR"),
    Stage("collision", "collision", ("python", "-m", "splat2sim.collision", "collision"), "collision", deps=("poses",),
          optional_deps=("anchor", "mask"), config_sections=("overrides", "collision", "robot"), run_as_user=True, implemented=True,
          description="TSDF mesh (Open3D) + RANSAC table/pedestal boxes -> invisible static colliders (collision.usda)"),
    Stage("compose", "isaac", ("-p", "/workspace/r2s/compose_scene.py"), "scene", deps=("splat",),
          optional_deps=("anchor", "collision"), config_sections=("board_to_base",), implemented=True,
          description="scene.usda with /World/Anchor, splat, collision layer"),
    Stage("report", "isaac", ("-p", "/workspace/r2s/report.py"), "report", deps=("compose",),
          optional_deps=("anchor", "mask", "collision"), config_sections=("robot", "board_to_base", "charuco"), implemented=True,
          post=(("isaac", ("-p", "/workspace/r2s/robot_silhouette.py"), False),
                ("ingest", ("python", "-m", "splat2sim.report", "finish"), True)),
          description="previews, fly-through MP4, mask-vs-URDF IoU, coverage -> report/{summary.json,report.md}"),
    Stage("env", "isaac", ("-p", "/workspace/r2s/check_env.py"), "env", deps=("compose",),
          config_sections=("robot", "policy_cameras", "cubes", "overrides", "env"), implemented=True,
          post=(("ingest", ("python", "-m", "splat2sim.report", "finish"), True),),  # refresh report/ with the env results
          description="Splat2Sim-FR3-Cubes-v0 pick-and-lift check -> env/summary.json + previews"),
]
BY_NAME = {s.name: s for s in STAGES}
# Not-yet-implemented stages that `r2s build` skips instead of failing on (their outputs are optional inputs downstream).
OPTIONAL = {"mask", "collision", "env"}


def skippable(stage: Stage) -> bool:
    return not stage.implemented and stage.name in OPTIONAL
NAMES = [s.name for s in STAGES]


def get(name: str) -> Stage:
    if name not in BY_NAME:
        raise KeyError(f"unknown stage {name!r}; choose from: {', '.join(NAMES)}")
    return BY_NAME[name]


def select(from_stage: str | None, to_stage: str | None) -> list[Stage]:
    i = NAMES.index(get(from_stage).name) if from_stage else 0
    j = NAMES.index(get(to_stage).name) if to_stage else len(NAMES) - 1
    if i > j:
        raise ValueError(f"--from {from_stage} comes after --to {to_stage}")
    return STAGES[i : j + 1]


def stage_options_digest(run: RunFolder, name: str) -> str:
    """Hash of this stage's section of ``<run>/options.yaml`` ('' when absent)."""
    from .recon.options import load_run_options

    sec = load_run_options(run.root).get(name)
    return sha(name, json.dumps(sec, sort_keys=True)) if sec else ""


def inputs_hash(stage: Stage, run: RunFolder, lab: LabConfig, capture: Path | None) -> str:
    """Hash of everything the stage's output depends on: capture, config slice, upstream run ids."""
    parts = [stage.name]
    if stage.uses_capture and capture is not None:
        parts.append(fingerprint_path(capture))
    opts = run.root / "ingest.yaml"  # per-run ingest options (keyframes, intrinsics file, ...)
    if stage.name == "ingest" and opts.is_file():
        parts.append(opts.read_text())
    if stage.config_sections:
        parts.append(lab.digest(list(stage.config_sections)))
    opts = stage_options_digest(run, stage.name)
    if opts:
        parts.append(opts)
    for d in stage.deps:
        m = run.read_marker(d, BY_NAME[d].out_dir)
        parts.append(f"{d}={m.run_id if m else 'MISSING'}")
    for d in stage.optional_deps:
        m = run.read_marker(d, BY_NAME[d].out_dir)
        parts.append(f"{d}={m.run_id if m else 'none'}")
    return sha(*parts)


def state(stage: Stage, run: RunFolder, expected: str | None) -> str:
    """'done' | 'stale' | 'pending'."""
    m = run.read_marker(stage.name, stage.out_dir)
    if m is None:
        return "pending"
    return "done" if expected is not None and m.inputs_hash == expected else "stale"


def compose_cmd(stage: Stage, run: RunFolder, capture: Path | None = None, service: str | None = None,
                command: tuple[str, ...] | None = None, as_user: bool | None = None) -> list[str]:
    repo = repo_root()
    opts = ["-e", f"R2S_SCENE={run.scene}"]
    if stage.run_as_user if as_user is None else as_user:
        opts += ["--user", f"{os.getuid()}:{os.getgid()}"]
    if stage.uses_capture and capture is not None and service is None:
        opts += ["-v", f"{capture}:/capture:ro", "-e", "R2S_CAPTURE=/capture", "-e", f"R2S_CAPTURE_SRC={capture}"]
    return [
        "docker", "compose", "-f", str(repo / "compose.yaml"), "--project-directory", str(repo),
        "run", "--rm", *opts,
        service or stage.service, *(command or stage.command),
    ]


def execute(stage: Stage, run: RunFolder, dry_run: bool = False, capture: Path | None = None) -> None:
    if not stage.implemented:
        raise NotImplementedStage(f"stage '{stage.name}' ({stage.service}) is not implemented yet")
    cmds = [compose_cmd(stage, run, capture)] + [compose_cmd(stage, run, capture, svc, command, user) for svc, command, user in stage.post]
    env = {**os.environ, "R2S_GIT_REV": git_rev()}
    for cmd in cmds:
        if dry_run:
            print("   ", " ".join(cmd))
            continue
        subprocess.run(cmd, check=True, env=env)
