"""Reconstruction stages (poses, splat) and the pure-python helpers shared with compose/report.

Keep this package import-light: it runs inside the rtabmap, 3dgrut and Isaac containers (each with a
different python), so ``__init__`` imports nothing. Heavy deps (cv2) are imported lazily where used.
"""
