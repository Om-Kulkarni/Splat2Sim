import argparse
import os
import sys
from pathlib import Path

from .summary import finish


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m splat2sim.report")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("finish", help="coverage + silhouette IoU + MP4 + summary.json + report.md")
    f.add_argument("--run", default=None, help="run folder (default /runs/$R2S_SCENE)")
    a = p.parse_args(argv)
    run = Path(a.run) if a.run else Path("/runs") / os.environ.get("R2S_SCENE", "")
    finish(run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
