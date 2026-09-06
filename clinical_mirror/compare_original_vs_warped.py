"""
clinical_mirror/compare_original_vs_warped.py

Side-by-side comparison view: original camera feed (left) vs. warped
output (right), each cropped to the middle half of the frame width. Not
a "run the mirror" script -- this is a documentation/figure tool, used
to generate comparison captures for the paper/presentations.

Consolidates what used to be two separate files
(Old_code/GLSLside_by_side.py and GLSLside_by_side_saveframe.py), which
differed only in whether/how often a frame gets saved to disk; that's
now the SAVE_MODE setting below (default changed to "none" -- neither
original script defaulted to off, but nothing else here should silently
write to disk unless you ask it to).
"""

import os
import time

import cv2
import numpy as np
import mediapipe as mp

from helpers import build_warp_maps, warp_frame, get_hip_center_and_peakY_from_pose

mp_pose = mp.solutions.pose

# SAVE_MODE:
#   "none"         - don't save anything (just show the comparison view)
#   "interval"     - save the raw original frame every SAVE_INTERVAL_SEC seconds
#   "every_frame"  - save every displayed side-by-side comparison frame
SAVE_MODE = "none"
SAVE_INTERVAL_SEC = 5.0
SAVE_DIR = os.path.join("saved_frames", "compare_original_vs_warped")


def main():
    if SAVE_MODE != "none":
        os.makedirs(SAVE_DIR, exist_ok=True)
    last_save_time = time.time()
    frame_idx = 0  # counter for "every_frame" filenames

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: couldn't read initial frame.")
        cap.release()
        return
    height, width = frame.shape[:2]

    uGain = 0.30
    sigma_y = 0.30
    fallback_centerX = 0.5
    fallback_peakY = 0.55

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_for_pose = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_for_pose)

            uCenterX, uPeakY = get_hip_center_and_peakY_from_pose(pose_results)
            if uCenterX is None or uPeakY is None:
                uCenterX = fallback_centerX
                uPeakY = fallback_peakY

            map_x, map_y = build_warp_maps(width, height, uCenterX, uPeakY, uGain, sigma_y)
            warped = warp_frame(frame, map_x, map_y)

            mirrored_orig = cv2.flip(frame, 1)
            mirrored_warp = cv2.flip(warped, 1)

            half_width = width // 2
            start_x = (width - half_width) // 2
            end_x = start_x + half_width

            crop_orig = mirrored_orig[:, start_x:end_x, :]
            crop_warp = mirrored_warp[:, start_x:end_x, :]

            min_w = min(crop_orig.shape[1], crop_warp.shape[1])
            crop_orig = crop_orig[:, :min_w, :]
            crop_warp = crop_warp[:, :min_w, :]

            side_by_side = np.hstack([crop_orig, crop_warp])
            cv2.imshow("Middle Half: Original (left) vs Warped (right)", side_by_side)

            if SAVE_MODE == "interval":
                now = time.time()
                if now - last_save_time >= SAVE_INTERVAL_SEC:
                    filename = os.path.join(SAVE_DIR, f"frame_{int(now)}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame to: {filename}")
                    last_save_time = now
            elif SAVE_MODE == "every_frame":
                filename = os.path.join(SAVE_DIR, f"frame_{frame_idx:06d}.png")
                cv2.imwrite(filename, side_by_side)
                frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
