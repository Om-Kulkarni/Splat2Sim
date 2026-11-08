"""Pure numpy/OpenCV mask post-processing (no torch): union, fill holes, dilate, write, overlay."""

from __future__ import annotations

import cv2
import numpy as np


def fill_holes(m: np.ndarray) -> np.ndarray:
    """Fill enclosed background regions of a bool mask (regions not connected to the image border)."""
    u = np.pad(m.astype(np.uint8), 1)
    inv = (1 - u).astype(np.uint8)
    ff = inv.copy()
    h, w = ff.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 2)
    holes = ff == 1  # background pixels the flood from the border could not reach
    return (u.astype(bool) | holes)[1:-1, 1:-1]


def dilate(m: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(m.astype(np.uint8), k).astype(bool)


def finalize(robot: np.ndarray, dilate_px: int = 15, fill: bool = True) -> np.ndarray:
    """bool robot mask -> bool robot mask after hole fill (before and after dilation) and dilation."""
    if not robot.any():
        return robot
    if fill:
        robot = fill_holes(robot)
    robot = dilate(robot, dilate_px)
    if fill:
        robot = fill_holes(robot)
    return robot


def to_png_mask(robot: np.ndarray) -> np.ndarray:
    """bool robot mask -> uint8 file convention: 255 = keep, 0 = robot."""
    return np.where(robot, 0, 255).astype(np.uint8)


def overlay(bgr: np.ndarray, robot: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    out = bgr.copy()
    if robot.any():
        red = np.zeros_like(out)
        red[..., 2] = 255
        out[robot] = (out[robot] * (1 - alpha) + red[robot] * alpha).astype(np.uint8)
        cs, _ = cv2.findContours(robot.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (0, 255, 255), 2)
    return out
