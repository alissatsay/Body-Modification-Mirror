"""
clinical_mirror/dataset_generation/generate_dynamic_dataset.py

Captures a background + person photo from the webcam, then batch-warps
the person across a sweep of uGain values, compositing each over the
captured background via MediaPipe segmentation. Writes the full uGain
sweep to WARPED_DIR.

Run from the repo root (paths below are relative to
the current working directory, same as the original script).

Previously the top-level DynamicImageWarp.py -- only the imports and the
now-shared helper functions changed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import mediapipe as mp

from helpers import (
    build_warp_maps,
    warp_frame,
    get_hip_center_and_peakY_from_pose,
    composite_person_over_bg,
    ensure_dirs,
    capture_after_countdown,
)

mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# --------------------------
# Config
# --------------------------
NUM_PERS = 2

IMAGES_DIR = "images_for_warping"
WARPED_DIR = "warped_dataset"

PERSON_IMAGE_PATH = os.path.join(IMAGES_DIR, f"pers{NUM_PERS}.png")
BACKGROUND_IMAGE_PATH = os.path.join(IMAGES_DIR, f"pers{NUM_PERS}bg.png")

SIGMA_Y = 0.30       # Vertical spread of warp
SEG_THRESH = 0.5     # Segmentation threshold
FEATHER_PX = 5       # Feathering radius
MIRROR = True        # Selfie-style mirror

UGAINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
          0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

CAPTURE_DELAY_SEC = 5
CAMERA_INDEX = 0     # change to 1 if you have multiple cameras


def main():
    ensure_dirs(IMAGES_DIR, WARPED_DIR)

    window_name = "Camera Capture"

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    try:
        # 1) Capture background
        print("Please move out of the frame, capturing background in 3 secs...")
        bg_bgr = capture_after_countdown(
            cap, window_name, "Please move out of the frame.", CAPTURE_DELAY_SEC, mirror=MIRROR
        )
        cv2.imwrite(BACKGROUND_IMAGE_PATH, bg_bgr)
        print(f"Saved background -> {os.path.abspath(BACKGROUND_IMAGE_PATH)}")

        # 2) Capture person
        print("Please stand still, capturing you in 3 secs...")
        person_bgr = capture_after_countdown(
            cap, window_name, "Please stand still.", CAPTURE_DELAY_SEC, mirror=MIRROR
        )
        cv2.imwrite(PERSON_IMAGE_PATH, person_bgr)
        print(f"Saved person -> {os.path.abspath(PERSON_IMAGE_PATH)}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # 3) Warp person for multiple uGains and composite over background
    bg_h, bg_w = bg_bgr.shape[:2]
    person_bgr = cv2.resize(person_bgr, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
    h, w = person_bgr.shape[:2]

    fallback_centerX = 0.5
    fallback_peakY = 0.55

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose, mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:

        # Pose once on the (captured) person image to get consistent center/peak across uGains
        pose_input = person_bgr  # already mirrored if MIRROR True
        rgb_for_pose = cv2.cvtColor(pose_input, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_for_pose)

        uCenterX, uPeakY = get_hip_center_and_peakY_from_pose(pose_results)
        if uCenterX is None:
            uCenterX, uPeakY = fallback_centerX, fallback_peakY

        for uGain in UGAINS:
            map_x, map_y = build_warp_maps(w, h, uCenterX, uPeakY, uGain, SIGMA_Y)
            warped = warp_frame(pose_input, map_x, map_y)

            rgb_for_seg = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            seg_results = segmenter.process(rgb_for_seg)
            seg_mask = seg_results.segmentation_mask

            final = composite_person_over_bg(warped, seg_mask, bg_bgr, thresh=SEG_THRESH, feather_px=FEATHER_PX)

            out_name = f"pers{NUM_PERS}uGain{uGain * 100}.png"
            out_path = os.path.join(WARPED_DIR, out_name)
            cv2.imwrite(out_path, final)
            print(f"Saved -> {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
