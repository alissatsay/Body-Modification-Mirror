"""
Output-frame display helpers for run_installation.py: rotating a camera
frame to match the installation's physical screen orientation
(rotate_frame_for_output) and setting up/showing the fullscreen window on
the second monitor the mesh is displayed on.
"""

import cv2

PRIMARY_MONITOR_WIDTH = 1920
WINDOW_NAME = "Hand brush drag on mesh"

ROTATE_DEG = 90
ROTATE_DIR = "ccw"   # "ccw" or "cw"


def rotate_frame_for_output(frame_bgr, deg=ROTATE_DEG, direction=ROTATE_DIR):
    if frame_bgr is None:
        return None

    deg = float(deg) % 360.0
    direction = str(direction).lower().strip()
    if direction not in ("ccw", "cw"):
        direction = "ccw"

    ccw_deg = deg if direction == "ccw" else (360.0 - deg) % 360.0

    if abs(ccw_deg - 0.0) < 1e-6:
        return frame_bgr
    if abs(ccw_deg - 90.0) < 1e-6:
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if abs(ccw_deg - 180.0) < 1e-6:
        return cv2.rotate(frame_bgr, cv2.ROTATE_180)
    if abs(ccw_deg - 270.0) < 1e-6:
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

    h, w = frame_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), ccw_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    return cv2.warpAffine(
        frame_bgr,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


def setup_fullscreen_second_monitor_window(
    window_name,
    primary_monitor_width=PRIMARY_MONITOR_WIDTH,
    y=0,
):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, primary_monitor_width, y)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def show_output_frame(window_name, frame_bgr, rotate_deg=ROTATE_DEG, rotate_dir=ROTATE_DIR):
    final_frame = rotate_frame_for_output(frame_bgr, deg=rotate_deg, direction=rotate_dir)
    cv2.imshow(window_name, final_frame)