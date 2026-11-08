"""``python -m splat2sim.anchor anchor`` (ingest container; scene from $R2S_SCENE, runs at /runs)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m splat2sim.anchor")
    p.add_argument("stage", choices=["anchor"])
    p.add_argument("--run", default=None, help="run folder (default /runs/$R2S_SCENE)")
    a = p.parse_args(argv)
    root = Path(a.run) if a.run else Path("/runs") / os.environ["R2S_SCENE"]
    from . import stage

    try:
        res = stage.run(root)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    # fallback / failed are not errors: downstream keeps using the gravity/plane fallback
    print(f"[anchor] status={res['status']}" + (f" ({res['reason']})" if res.get("reason") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
