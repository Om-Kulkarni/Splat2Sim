"""Keyframe selection: every Nth frame and/or sharpness (variance of the Laplacian) filter."""

from __future__ import annotations

import cv2
import numpy as np


def sharpness(color_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select(sharp: list[float], every: int = 1, min_sharpness: float = 0.0) -> list[int]:
    """Indices kept: frame i is a keyframe if ``i % every == 0`` and its Laplacian variance >= ``min_sharpness``.
    With a blur threshold, a blurry frame is replaced by the sharpest frame in its window of ``every`` frames
    (if that one passes), so drops do not open large gaps."""
    n = len(sharp)
    every = max(1, int(every))
    out: list[int] = []
    for start in range(0, n, every):
        window = range(start, min(start + every, n))
        if min_sharpness <= 0:
            out.append(start)
            continue
        if every == 1:
            if sharp[start] >= min_sharpness:
                out.append(start)
            continue
        best = max(window, key=lambda i: sharp[i])
        if sharp[best] >= min_sharpness:
            out.append(best)
    return out
