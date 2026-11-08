#!/usr/bin/env python3
"""Generate 1:1-scale ChArUco board PDFs (A3 preferred, A4 fallback).

Run (no host install):
  docker run --rm -v "$PWD":/w -w /w python:3.12-slim bash -c \
    "pip install -q -r tools/charuco/requirements.txt && python tools/charuco/make_board.py --out docs"
PRINT AT 100% / 'Actual size' (no fit-to-page). Verify with the 100 mm scale bar.
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DICT_NAME, SPECS, SQUARES_X, SQUARES_Y, make_board  # noqa: E402

DPI = 300
PX_MM = DPI / 25.4


def px(mm: float) -> int:
    return int(round(mm * PX_MM))


def render(name: str) -> np.ndarray:
    sw, sh, sq, mk = SPECS[name]
    board = make_board(sq, mk)
    bw, bh = SQUARES_X * sq, SQUARES_Y * sq
    ox, oy = (sw - bw) / 2, (sh - bh) / 2  # board top-left on sheet (mm)
    sheet = np.full((px(sh), px(sw)), 255, np.uint8)
    W, H = px(bw), px(bh)
    img = board.generateImage((W, H), marginSize=0, borderBits=1)
    x0, y0 = px(ox), px(oy)
    sheet[y0:y0 + H, x0:x0 + W] = img
    # exact px-per-mm actually achieved on the board (for the report)
    big = name == "a3"
    fs = 0.9 if big else 0.55
    th = 2 if big else 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    black = 0
    # board frame origin = charuco corner id 0 (first inner corner, (sq, sq) from the pattern's outer top-left); see splat2sim/anchor/detect.py
    cx, cy = x0 + px(sq), y0 + px(sq)
    tick = px(min(ox, oy) * 0.8)
    # ticks in margins aligned with corner id 0 (do not touch the pattern)
    cv2.line(sheet, (cx, y0 - tick), (cx, y0 - px(0.5)), black, th + 1)
    cv2.line(sheet, (x0 - tick, cy), (x0 - px(0.5), cy), black, th + 1)
    # axis arrows in margins: +X along top margin, +Y down left margin (board frame; z into board)
    ay = y0 - px(oy / 2)
    cv2.arrowedLine(sheet, (x0 + px(sq * 2), ay), (x0 + px(sq * 4), ay), black, th, tipLength=0.08)
    cv2.putText(sheet, "+X", (x0 + px(sq * 4) + 8, ay + px(1.5)), font, fs, black, th, cv2.LINE_AA)
    ax = x0 - px(ox / 2)
    cv2.arrowedLine(sheet, (ax, y0 + px(sq * 2)), (ax, y0 + px(sq * 4)), black, th, tipLength=0.08)
    cv2.putText(sheet, "+Y", (ax - px(2), y0 + px(sq * 4) + px(5 if big else 3.5)), font, fs, black, th, cv2.LINE_AA)
    lab = "ORIGIN = corner id 0" if big else "ORIGIN id0"
    cv2.putText(sheet, lab + "", (x0 + px(sq * 4.6), ay + px(1.5)), font, fs * 0.8, black, th, cv2.LINE_AA)
    # 100 mm scale bar + spec text in bottom margin
    by = y0 + H + px(oy / 2)
    cv2.line(sheet, (x0, by), (x0 + px(100), by), black, th + 2)
    for t in (0, 100):
        cv2.line(sheet, (x0 + px(t), by - px(oy * 0.3)), (x0 + px(t), by + px(oy * 0.3)), black, th + 1)
    cv2.putText(sheet, "100 mm - measure me! (print at 100%)", (x0 + px(105), by + px(1.5)), font, fs, black, th, cv2.LINE_AA)
    spec = (f"{DICT_NAME} {SQUARES_X}x{SQUARES_Y} sq={sq:g}mm marker={mk:g}mm "
            f"OpenCV {cv2.__version__} z=into board")
    (tw, _), _ = cv2.getTextSize(spec, font, fs * 0.7, th)
    cv2.putText(sheet, spec, (x0 + W - tw, by + px(1.5)), font, fs * 0.7, black, th, cv2.LINE_AA)
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs")
    ap.add_argument("--png", action="store_true", help="also write PNG rasters next to the PDFs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name in SPECS:
        im = render(name)
        pil = Image.fromarray(im)
        p = os.path.join(a.out, f"charuco-board-{name}.pdf")
        pil.save(p, "PDF", resolution=DPI)
        if a.png:
            pil.save(p.replace(".pdf", ".png"), dpi=(DPI, DPI))
        print("wrote", p, im.shape[::-1], "px @", DPI, "dpi")


if __name__ == "__main__":
    main()
