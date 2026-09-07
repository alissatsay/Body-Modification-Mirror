"""
clinical_mirror/dataset_generation/generate_still_warp.py

Loads a single already-captured person + background image pair from
disk (does not touch the webcam), applies one warp at UGAIN, composites
over the background via MediaPipe segmentation, and saves/shows the
single result.

Run from the repo root (paths below are relative to
the current working directory, same as the original script).

Previously the top-level StillImageWarp.py -- only the imports and the
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
)

mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation

NUM_PERS = 1
PERSON_IMAGE_PATH = "images_for_warping/pers" + str(NUM_PERS) + ".png"
BACKGROUND_IMAGE_PATH = "images_for_warping/pers" + str(NUM_PERS) + "bg.png"
OUTPUT_PATH = "images_for_warping/pers" + str(NUM_PERS) + "W.png"

UGAIN = 0.30        # Warp strength
SIGMA_Y = 0.30       # Vertical spread of warp
SEG_THRESH = 0.5     # Segmentation threshold
FEATHER_PX = 5       # Feathering radius
MIRROR = False       # Selfie-style mirror


def main():
    person_bgr = cv2.imread(PERSON_IMAGE_PATH)
    bg_bgr = cv2.imread(BACKGROUND_IMAGE_PATH)

    if person_bgr is None:
        raise FileNotFoundError(f"Could not read {PERSON_IMAGE_PATH}")
    if bg_bgr is None:
        raise FileNotFoundError(f"Could not read {BACKGROUND_IMAGE_PATH}")

    # Resize person to match background
    bg_h, bg_w = bg_bgr.shape[:2]
    person_bgr = cv2.resize(person_bgr, (bg_w, bg_h))

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

        pose_input = cv2.flip(person_bgr, 1) if MIRROR else person_bgr

        rgb_for_pose = cv2.cvtColor(pose_input, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_for_pose)

        uCenterX, uPeakY = get_hip_center_and_peakY_from_pose(pose_results)

        if uCenterX is None:
            uCenterX = fallback_centerX
            uPeakY = fallback_peakY

        map_x, map_y = build_warp_maps(w, h, uCenterX, uPeakY, UGAIN, SIGMA_Y)
        warped = warp_frame(pose_input, map_x, map_y)

        rgb_for_seg = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        seg_results = segmenter.process(rgb_for_seg)
        seg_mask = seg_results.segmentation_mask

        final = composite_person_over_bg(warped, seg_mask, bg_bgr, thresh=SEG_THRESH, feather_px=FEATHER_PX)

    cv2.imwrite(OUTPUT_PATH, final)
    cv2.imshow("Warped Composite", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Saved to: {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
