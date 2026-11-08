#!/usr/bin/env python3
"""ChArUco board as a full-screen PNG, for when no printer is available (show it on a monitor or tablet).

Same board as the print (common.py: DICT_4X4_50, 7x5, marker/square = 41/55). The screen sets the physical size,
so MEASURE the square length on the screen (span of all 7 squares / 7) and put it in labs/<lab>.yaml
(charuco.square_length_m, marker_length_m = square * 41/55).

Run (no host install):
  docker run --rm -v "$PWD":/w -w /w python:3.12-slim bash -c \
    "pip install -q -r tools/charuco/requirements.txt && python tools/charuco/make_screen_board.py"
Open it full-screen at 100 % zoom (e.g. `feh -F docs/charuco-board-screen-1080p.png`, or F11 in a browser).
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SQUARES_X, SQUARES_Y, make_board  # noqa: E402


def render(w: int, h: int, fill: float = 0.86) -> np.ndarray:
    sq = int(min(fill * h / SQUARES_Y, fill * w / SQUARES_X))  # integer px per square: crisp edges
    W, H = SQUARES_X * sq, SQUARES_Y * sq
    img = make_board(55.0, 41.0).generateImage((W, H), marginSize=0, borderBits=1)
    out = np.full((h, w), 255, np.uint8)
    x0, y0 = (w - W) // 2, (h - H) // 2
    out[y0:y0 + H, x0:x0 + W] = img
    # ticks in the white margin pointing at corner id 0 (board-frame origin), as on the print
    cx, cy = x0 + sq, y0 + sq
    t = max(3, sq // 40)
    cv2.line(out, (cx, y0 - int(0.4 * y0)), (cx, y0 - 4), 0, t)
    cv2.line(out, (x0 - int(0.4 * x0), cy), (x0 - 4, cy), 0, t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()
    for name, (w, h) in {"1080p": (1920, 1080), "1200p": (1920, 1200), "4k": (3840, 2160)}.items():
        p = os.path.join(a.out, f"charuco-board-screen-{name}.png")
        cv2.imwrite(p, render(w, h))
        print("wrote", p)
    # laptop screen as the board (1920x1200): fill almost all of it, every mm of square size counts
    p = os.path.join(a.out, "charuco-board-laptop-1200p.png")
    cv2.imwrite(p, render(1920, 1200, fill=0.95))
    print("wrote", p)


if __name__ == "__main__":
    main()
