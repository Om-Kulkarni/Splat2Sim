"""Shared bits for the Isaac-side stage scripts (run with isaaclab.sh -p). Stdlib + numpy + pxr only."""
import json
import os
import subprocess
from pathlib import Path

import numpy as np

SCENE = os.environ.get("R2S_SCENE", "")
RUN = Path("/runs") / SCENE


def start_app(headless_size=(1280, 720)):
    """pxr is only importable once Kit is up in the Isaac Python env; start a headless SimulationApp."""
    from isaacsim import SimulationApp

    return SimulationApp({"headless": True, "width": headless_size[0], "height": headless_size[1]})


def open_permissions(path: Path):
    """Container runs as root; leave the outputs deletable/writable by the host user."""
    subprocess.run(["chmod", "-R", "a+rwX", str(path)], check=False)


def read_json(p: Path, default=None):
    return json.loads(p.read_text()) if p.is_file() else default
