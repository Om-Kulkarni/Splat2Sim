"""Report stage, CPU part: coverage, mask-vs-URDF silhouette IoU, fly-through encoding, summary.json + report.md.

The Isaac part (``docker/isaac/scripts/report.py`` previews + fly-through frames, ``robot_silhouette.py`` robot-alone renders)
writes into ``report/``; ``python -m splat2sim.report finish`` (ingest image: numpy, opencv, imageio-ffmpeg) turns that into the
final files."""
