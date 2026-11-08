"""Alignment sanity-check image: depth edges drawn over the colour frame (+ colourised depth beside it)."""

from __future__ import annotations

import cv2
import numpy as np


def depth_edges(depth_mm: np.ndarray) -> np.ndarray:
    valid = depth_mm > 0
    d = np.where(valid, depth_mm, 0).astype(np.float32)
    # log-ish normalisation so near and far edges both show
    norm = np.zeros_like(d)
    if valid.any():
        lo, hi = np.percentile(d[valid], [1, 99])
        norm = np.clip((d - lo) / max(hi - lo, 1.0), 0, 1)
    img8 = (norm * 255).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(img8, (3, 3), 0), 30, 90)
    # also mark validity borders only where valid region touches strong gradient: keep simple
    return edges > 0


def overlay(color_rgb: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
    """RGB uint8 image: [colour with red depth edges | colourised depth]."""
    edges = depth_edges(depth_mm)
    edges = cv2.dilate(edges.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    a = color_rgb.copy()
    a[edges] = (255, 0, 0)
    valid = depth_mm > 0
    v = np.zeros(depth_mm.shape, np.uint8)
    if valid.any():
        hi = np.percentile(depth_mm[valid], 98)
        v = np.clip(depth_mm.astype(np.float32) / max(hi, 1.0) * 255, 0, 255).astype(np.uint8)
    cm = cv2.cvtColor(cv2.applyColorMap(v, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    cm[~valid] = 0
    return np.concatenate([a, cm], axis=1)
