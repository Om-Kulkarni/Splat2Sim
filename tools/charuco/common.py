"""Shared ChArUco board definition (single source of truth for print + detect)."""
import cv2

DICT_NAME = "DICT_4X4_50"
SQUARES_X, SQUARES_Y = 7, 5

# name -> (sheet w mm, sheet h mm, square mm, marker mm); landscape
SPECS = {
    "a3": (420.0, 297.0, 55.0, 41.0),
    "a4": (297.0, 210.0, 40.0, 30.0),
}


def make_board(square_mm: float, marker_mm: float) -> "cv2.aruco.CharucoBoard":
    """Lengths in metres. 7x5 is odd/odd so the post-4.6 legacy-pattern change should
    not matter, but we never call setLegacyPattern: same code prints and detects."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
    return cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), square_mm / 1000.0, marker_mm / 1000.0, d)
