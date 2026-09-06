"""
clinical_mirror/dataset_generation/generate_greenscreen_dataset.py

Captures a background + person-in-front-of-greenscreen photo from the
webcam, chroma-keys the person out via HSV thresholding (instead of
MediaPipe segmentation), then batch-warps the keyed person across a
sweep of uGain values, compositing each over the captured background.
Writes the full uGain sweep to WARPED_DIR.

Run from the Digital_Mirror_Code repo root (paths below are relative to
the current working directory, same as the original script).

Previously the top-level DynamicImageWarp_greenscreen.py -- only the
imports and the now-shared helper functions changed. The green-screen
tuning constants (GREEN_LOWER_HSV, morphology kernel sizes, edge
feathering) moved to helpers.py as function defaults with the same
values, since this script only ever used the defaults -- they're no
longer duplicated here.
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
    make_greenscreen_mask,
    despill_green,
    composite_with_alpha_soft,
    ensure_dirs,
    capture_after_countdown,
    show_preview,
)

mp_pose = mp.solutions.pose

# --------------------------
# Config
# --------------------------
NUM_PERS = 16

IMAGES_DIR = "images_for_warping"
WARPED_DIR = "warped_dataset_greenscreen"

PERSON_IMAGE_PATH = os.path.join(IMAGES_DIR, f"pers{NUM_PERS}.png")
BACKGROUND_IMAGE_PATH = os.path.join(IMAGES_DIR, f"pers{NUM_PERS}bg.png")
MASK_IMAGE_PATH = os.path.join(IMAGES_DIR, f"pers{NUM_PERS}_mask.png")

SIGMA_Y = 0.30
MIRROR = True

UGAINS = [
    0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15,
    0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00
]

CAPTURE_DELAY_SEC = 5
CAMERA_INDEX = 0

DESPILL_STRENGTH = 0.35


def main():
    ensure_dirs(IMAGES_DIR, WARPED_DIR)

    window_name = "Camera Capture"

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    try:
        print("Please move out of the frame, capturing background...")
        bg_bgr = capture_after_countdown(
            cap, window_name, "Please move out of the frame.", CAPTURE_DELAY_SEC, mirror=MIRROR
        )
        cv2.imwrite(BACKGROUND_IMAGE_PATH, bg_bgr)
        print(f"Saved background -> {os.path.abspath(BACKGROUND_IMAGE_PATH)}")

        print("Please stand still in front of the green screen...")
        person_bgr = capture_after_countdown(
            cap, window_name, "Please stand still.", CAPTURE_DELAY_SEC, mirror=MIRROR
        )
        cv2.imwrite(PERSON_IMAGE_PATH, person_bgr)
        print(f"Saved person -> {os.path.abspath(PERSON_IMAGE_PATH)}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    bg_h, bg_w = bg_bgr.shape[:2]
    person_bgr = cv2.resize(person_bgr, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
    h, w = person_bgr.shape[:2]

    fg_mask = make_greenscreen_mask(person_bgr)
    cv2.imwrite(MASK_IMAGE_PATH, fg_mask)
    print(f"Saved mask -> {os.path.abspath(MASK_IMAGE_PATH)}")

    person_despilled = despill_green(person_bgr, fg_mask, strength=DESPILL_STRENGTH)

    preview_comp = composite_with_alpha_soft(person_despilled, fg_mask, bg_bgr)
    show_preview("Foreground mask", fg_mask)
    show_preview("Keyed preview", preview_comp)
    print("Press any key in an OpenCV window to continue to warping...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    fallback_centerX = 0.5
    fallback_peakY = 0.55

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        rgb_for_pose = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_for_pose)

        uCenterX, uPeakY = get_hip_center_and_peakY_from_pose(pose_results)
        if uCenterX is None:
            uCenterX, uPeakY = fallback_centerX, fallback_peakY
            print("Pose not detected. Using fallback warp center.")
        else:
            print(f"Pose center: uCenterX={uCenterX:.3f}, uPeakY={uPeakY:.3f}")

        for uGain in UGAINS:
            map_x, map_y = build_warp_maps(w, h, uCenterX, uPeakY, uGain, SIGMA_Y)

            warped_person = warp_frame(
                person_despilled, map_x, map_y,
                border_mode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_LINEAR
            )

            warped_mask = warp_frame(
                fg_mask, map_x, map_y,
                border_mode=cv2.BORDER_CONSTANT, border_value=0, interpolation=cv2.INTER_NEAREST
            )
            warped_mask = np.where(warped_mask > 0, 255, 0).astype(np.uint8)

            final = composite_with_alpha_soft(warped_person, warped_mask, bg_bgr)

            gain_str = f"{uGain * 100:.0f}"
            out_name = f"pers{NUM_PERS}uGain{gain_str}.png"
            out_path = os.path.join(WARPED_DIR, out_name)

            cv2.imwrite(out_path, final)
            print(f"Saved -> {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
