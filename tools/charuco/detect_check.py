#!/usr/bin/env python3
"""Sanity-check a printed board: detect ChArUco corners in an image or webcam and draw axes.

  detect_check.py photo.jpg [--out annotated.png] [--square-mm 55 --marker-mm 41] [--hfov 69.4]
  detect_check.py --camera 0            (live window; needs X11/display)
Prints corners found (max 24 for 7x5) and, for a rough pinhole model, the board pose.
After printing, measure the 7-square span with a ruler and pass --square-mm <measured>.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SQUARES_X, SQUARES_Y, make_board  # noqa: E402

N_CORNERS = (SQUARES_X - 1) * (SQUARES_Y - 1)


def make_detector(board):
    dp = cv2.aruco.DetectorParameters()
    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), dp)


def process(img, det, board, hfov_deg, axis_len):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    ch_c, ch_i, m_c, m_i = det.detectBoard(gray)
    n = 0 if ch_i is None else len(ch_i)
    nm = 0 if m_i is None else len(m_i)
    if nm:
        cv2.aruco.drawDetectedMarkers(out, m_c, m_i)
    msg = f"markers {nm}/{(SQUARES_X * SQUARES_Y) // 2}  corners {n}/{N_CORNERS}"
    pose = None
    if n >= 4:
        cv2.aruco.drawDetectedCornersCharuco(out, ch_c, ch_i, (0, 255, 0))
        h, w = gray.shape
        f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        obj, imgp = board.matchImagePoints(ch_c, ch_i)
        if len(obj) >= 6 and not board.checkCharucoCornersCollinear(ch_i):
            ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, None, flags=cv2.SOLVEPNP_IPPE)
            if ok:
                cv2.drawFrameAxes(out, K, None, rvec, tvec, axis_len)
                pose = tvec.ravel()
                msg += f"  dist~{np.linalg.norm(pose):.2f} m (rough, assumed FOV)"
    cv2.putText(out, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return out, n, nm, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--square-mm", type=float, default=55.0)
    ap.add_argument("--marker-mm", type=float, default=None, help="default = 0.7455 x square")
    ap.add_argument("--hfov", type=float, default=90.0, help="horizontal FOV for rough pose (D455 colour, 1280x720)")
    a = ap.parse_args()
    mk = a.marker_mm or a.square_mm * 41.0 / 55.0
    board = make_board(a.square_mm, mk)
    det = make_detector(board)
    axis = 3 * a.square_mm / 1000.0
    if a.camera is not None:
        cap = cv2.VideoCapture(a.camera)
        while cap.isOpened():
            ok, fr = cap.read()
            if not ok:
                break
            out, *_ = process(fr, det, board, a.hfov, axis)
            cv2.imshow("charuco check (q quits)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        return 0
    if not a.image:
        ap.error("give an image or --camera N")
    img = cv2.imread(a.image)
    if img is None:
        print("cannot read", a.image, file=sys.stderr)
        return 2
    out, n, nm, msg = process(img, det, board, a.hfov, axis)
    print(msg)
    if a.out:
        cv2.imwrite(a.out, out)
        print("wrote", a.out)
    return 0 if n == N_CORNERS else 1


if __name__ == "__main__":
    sys.exit(main())
