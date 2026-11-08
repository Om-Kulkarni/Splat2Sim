"""Run-folder contract: ``<data>/runs/<scene>/{frames,masks,poses,splat,collision,scene,report}``.

Every stage writes its outputs into its directory and, only on success, a
marker ``<dir>/.done.<stage>.json`` holding timestamp, inputs hash and git rev.
A stage is *done* when its marker exists and its inputs hash matches the one
computed now (otherwise it is *stale*). Stages sharing a directory (poses/anchor)
have separate markers.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

STAGE_DIRS = ["frames", "masks", "poses", "splat", "collision", "scene", "report"]
RUN_META = "run.json"


def data_dir() -> Path:
    return Path(os.environ.get("R2S_DATA") or Path.home() / "real2sim-data").expanduser()


ISAAC_CACHES = ["kit", "ov", "glcache", "computecache"]


def ensure_data_dirs(root: Path | None = None) -> None:
    """Create the data layout. Isaac cache dirs are world-writable (container uid 1234 / root)."""
    d = root or data_dir()
    for sub in ("captures", "runs", "cache"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    (d / "cache" / "hf").mkdir(parents=True, exist_ok=True)  # HF weights cache (sam3), written by the host uid
    for c in ISAAC_CACHES:
        p = d / "cache" / "isaac" / c
        p.mkdir(parents=True, exist_ok=True)
        try:
            p.chmod(0o777)
        except OSError:
            pass


def git_rev(repo: Path | None = None) -> str:
    if os.environ.get("R2S_GIT_REV"):
        return os.environ["R2S_GIT_REV"]
    try:
        out = subprocess.run(
            ["git", "-C", str(repo or Path(__file__).resolve().parents[1]), "describe", "--always", "--dirty"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def fingerprint_path(path: Path) -> str:
    """Cheap identity of a capture file/dir: resolved path + size + mtime (no content read)."""
    path = path.resolve()
    if path.is_file():
        st = path.stat()
        return f"{path}:{st.st_size}:{st.st_mtime_ns}"
    entries = []
    for f in sorted(path.rglob("*")):
        if f.is_file():
            st = f.stat()
            entries.append(f"{f.relative_to(path)}:{st.st_size}:{st.st_mtime_ns}")
    return f"{path}:" + sha(*entries)


@dataclass
class Marker:
    stage: str
    timestamp: str
    inputs_hash: str
    git_rev: str
    run_id: str


class RunFolder:
    def __init__(self, scene: str, root: Path | None = None):
        self.scene = scene
        self.root = (root or data_dir() / "runs") / scene

    def dir(self, name: str) -> Path:
        return self.root / name

    def ensure(self, extra: list[str] | None = None) -> None:
        for d in STAGE_DIRS + (extra or []):
            self.dir(d).mkdir(parents=True, exist_ok=True)

    # -- run metadata (capture, lab, override) --
    def meta_path(self) -> Path:
        return self.root / RUN_META

    def read_meta(self) -> dict | None:
        p = self.meta_path()
        return json.loads(p.read_text()) if p.is_file() else None

    def write_meta(self, meta: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path().write_text(json.dumps(meta, indent=2, sort_keys=True))

    # -- stage markers --
    def marker_path(self, stage: str, out_dir: str) -> Path:
        return self.dir(out_dir) / f".done.{stage}.json"

    def read_marker(self, stage: str, out_dir: str) -> Marker | None:
        p = self.marker_path(stage, out_dir)
        if not p.is_file():
            return None
        try:
            return Marker(**json.loads(p.read_text()))
        except (json.JSONDecodeError, TypeError):
            return None

    def write_marker(self, stage: str, out_dir: str, inputs_hash: str, rev: str | None = None) -> Marker:
        self.dir(out_dir).mkdir(parents=True, exist_ok=True)
        m = Marker(
            stage=stage,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            inputs_hash=inputs_hash,
            git_rev=rev or git_rev(),
            run_id=uuid.uuid4().hex[:12],
        )
        self.marker_path(stage, out_dir).write_text(json.dumps(m.__dict__, indent=2))
        return m

    def clear_marker(self, stage: str, out_dir: str) -> None:
        self.marker_path(stage, out_dir).unlink(missing_ok=True)
