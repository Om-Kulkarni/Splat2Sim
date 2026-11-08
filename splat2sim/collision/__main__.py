"""``python -m splat2sim.collision collision`` (container entrypoint; scene from $R2S_SCENE, runs at /runs)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m splat2sim.collision")
    p.add_argument("stage", choices=["collision"])
    p.add_argument("--run", default=None, help="run folder (default /runs/$R2S_SCENE)")
    a = p.parse_args(argv)
    root = Path(a.run) if a.run else Path("/runs") / os.environ["R2S_SCENE"]
    try:
        from . import stage

        stage.run(root)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
