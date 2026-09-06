"""
clinical_mirror/dataset_generation/generate_nobackground_dataset.py

Captures a single person photo from the webcam, then batch-warps it
across a sweep of uGain values with no background compositing at all
(just the raw warped frame, saved as-is). Writes the sweep to
WARPED_DIR.

Run from the Digital_Mirror_Code repo root (paths below are relative to
the current working directory, same as the original script).

Previously the top-level DynamicImageWarp_noBackground.py -- only the
imports and the now-shared helper functions changed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np
import mediapipe as mp

from helpers import (
    build_warp_maps,
    warp_frame,
    get_hip_center_and_peakY_from_pose,
    ensure_dirs,
    capture_after_countdown,
)

mp_pose = mp.solutions.pose

# --------------------------
# Config
# --------------------------
NUM_PERS = 2

IMAGES_DIR = "images_for_warping"
WARPED_DIR = "warped_dataset_nobackground"

PERSON_IMAGE_PATH = os.path.join(IMAGES_DIR, f"pers{NUM_PERS}.png")

SIGMA_Y = 0.30
MIRROR = True

UGAINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
          0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

CAPTURE_DELAY_SEC = 5
CAMERA_INDEX = 0


def main():
    ensure_dirs(IMAGES_DIR, WARPED_DIR)

    window_name = "Camera Capture"

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    try:
        print("Capturing person...")
        person_bgr = capture_after_countdown(
            cap, window_name, "Stand still.", CAPTURE_DELAY_SEC, mirror=MIRROR
        )
        cv2.imwrite(PERSON_IMAGE_PATH, person_bgr)
        print("Saved person image.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    h, w = person_bgr.shape[:2]

    fallback_centerX = 0.5
    fallback_peakY = 0.55

    with mp_pose.Pose(static_image_mode=True) as pose:
        rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        uCenterX, uPeakY = get_hip_center_and_peakY_from_pose(results)
        if uCenterX is None:
            uCenterX, uPeakY = fallback_centerX, fallback_peakY

        for uGain in UGAINS:
            map_x, map_y = build_warp_maps(w, h, uCenterX, uPeakY, uGain, SIGMA_Y)
            warped = warp_frame(person_bgr, map_x, map_y)

            out_name = f"pers{NUM_PERS}_uGain{int(uGain*100)}.png"
            out_path = os.path.join(WARPED_DIR, out_name)

            cv2.imwrite(out_path, warped)
            print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
