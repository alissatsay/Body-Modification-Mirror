"""
clinical_mirror/live_mirror.py

The primary, demo-ready version of the camera-based digital mirror.

Consolidates what used to be five separate near-duplicate scripts in
Old_code/ (GLSL_controls.py, GLSLwarpBackground.py,
GLSLwarpCombinationBackground.py, GLSLCombinationBackgroundSaveFrame.py,
GLSL_warpBackground_optimized_for_speed.py) into one script, with runtime
flags for the features that used to each require a different file:

  - BACKGROUND_MODE: "still" (composite the warped person over one
    captured background) or "combination" (top of frame = captured
    background, bottom = the live original frame, split at a
    hand-derived cut line). Press 'b' to toggle live.
  - SAVE_FRAMES: off by default; when on, periodically saves raw camera
    frames to disk. Press 's' to toggle live.

Everything else -- threaded capture, frame-skipped pose/segmentation,
interactive uGain/uPeak controls -- is unchanged from GLSL_controls.py,
which was the most current/complete version of this lineage and is the
version used for demos.

Note: because this now shares one optimized architecture (frame-skipped
pose at model_complexity=0), "combination" mode's hand-tracking updates
on the same cadence as everything else here, rather than running pose
every frame at model_complexity=1 the way the original standalone
GLSLCombinationBackgroundSaveFrame.py did. In practice this trades a
small amount of hand-tracking responsiveness for a meaningfully higher
frame rate; adjust POSE_EVERY_N below if you want it more responsive.
"""

import os
import queue
import threading
import time

import cv2
import numpy as np
import mediapipe as mp

from helpers import (
    build_warp_maps,
    warp_frame,
    get_hip_center_and_peakY_from_pose,
    get_lowest_hand_related_y_from_pose,
    composite_person_over_bg,
    capture_thread,
)

mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation


# ── Config (edit these, or toggle live with the keys printed at startup) ──
INITIAL_BACKGROUND_MODE = "still"   # "still" or "combination"
INITIAL_SAVE_FRAMES     = False     # periodic raw-frame saving; off by default
SAVE_INTERVAL_SEC       = 5.0
SAVE_DIR                = os.path.join("saved_frames", "live_mirror")

# Maps digit keys 0-6 to their uGain values
GAIN_PRESETS = {
    ord('0'): 0.00, ord('1'): 0.10, ord('2'): 0.20, ord('3'): 0.30,
    ord('4'): 0.40, ord('5'): 0.50, ord('6'): 0.60,
}


def build_combination_background(captured_bg, mirrored_orig, cut_line, height, width):
    """
    Top of frame (above cut_line) = the captured clean background.
    Bottom (at/below cut_line) = the live, unwarped original frame.
    cut_line is a row index derived from a hand-related pose landmark.
    """
    if captured_bg is not None:
        combined_bg = captured_bg.copy()
    else:
        combined_bg = mirrored_orig.copy()

    if combined_bg.shape[:2] != (height, width):
        combined_bg = cv2.resize(combined_bg, (width, height), interpolation=cv2.INTER_LINEAR)

    if 0 <= cut_line < height:
        combined_bg[cut_line:, :, :] = mirrored_orig[cut_line:, :, :]

    return combined_bg


def main():
    # ── tunables ──────────────────────────────────────────────────────────
    uGain                    = 0.30
    uPeakY_manual            = None   # None = let pose drive it; set by u/d keys
    sigma_y                  = 0.30
    fallback_centerX         = 0.5
    fallback_peakY           = 0.55
    fallback_cutline_y_norm  = 0.5
    POSE_EVERY_N             = 3       # run pose every N frames (also drives cut-line updates)
    SEG_EVERY_N              = 2       # run segmentation every N frames
    SEG_SCALE                = 0.5     # downscale factor for segmentation inference

    GAIN_STEP                = 0.05
    PEAK_STEP                = 0.05
    # ─────────────────────────────────────────────────────────────────────

    background_mode = INITIAL_BACKGROUND_MODE
    save_frames = INITIAL_SAVE_FRAMES
    if save_frames:
        os.makedirs(SAVE_DIR, exist_ok=True)
    last_save_time = time.time()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    ret, frame = cap.read()
    if not ret:
        print("Error: couldn't read initial frame.")
        cap.release()
        return
    height, width = frame.shape[:2]
    seg_w = int(width  * SEG_SCALE)
    seg_h = int(height * SEG_SCALE)

    print("Please move out of the frame. Capturing background in 3 seconds...")
    cv2.waitKey(3000)
    ret, captured_bg = cap.read()
    if ret:
        captured_bg = cv2.flip(captured_bg, 1)
        print("Background captured.")
    else:
        captured_bg = None
        print("Warning: background capture failed. Falling back to white.")

    print("\nControls:")
    print("  +  /  -   : increase / decrease uGain by 0.05")
    print("  0 … 6     : set uGain directly (0.00, 0.10, 0.20, … 0.60)")
    print("  u  /  d   : increase / decrease uPeak by 0.05 (overrides pose tracking)")
    print("  b         : toggle background mode (still <-> combination)")
    print("  s         : toggle periodic frame-saving on/off")
    print("  q         : quit\n")
    print(f"Starting background_mode = {background_mode}, save_frames = {save_frames}\n")

    frame_queue = queue.Queue(maxsize=2)
    t = threading.Thread(target=capture_thread, args=(cap, frame_queue), daemon=True)
    t.start()

    map_x, map_y        = build_warp_maps(width, height, fallback_centerX, fallback_peakY, uGain, sigma_y)
    last_seg_mask        = np.zeros((height, width), dtype=np.float32)
    last_uCenterX        = fallback_centerX
    last_uPeakY          = fallback_peakY
    last_cutline_y_norm  = fallback_cutline_y_norm
    maps_dirty           = False
    frame_idx            = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose, mp_selfie_segmentation.SelfieSegmentation(
        model_selection=0
    ) as segmenter:

        while True:
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Pose (every POSE_EVERY_N frames) ──────────────────────────────
            if frame_idx % POSE_EVERY_N == 0:
                pose_results = pose.process(rgb)

                cx, py = get_hip_center_and_peakY_from_pose(pose_results)
                if cx is not None:
                    effective_peakY = uPeakY_manual if uPeakY_manual is not None else py
                    if (abs(cx - last_uCenterX) > 0.01 or
                            abs(effective_peakY - last_uPeakY) > 0.01 or
                            maps_dirty):
                        map_x, map_y = build_warp_maps(width, height, cx, effective_peakY, uGain, sigma_y)
                        last_uCenterX = cx
                        last_uPeakY   = effective_peakY
                        maps_dirty    = False
                elif maps_dirty:
                    effective_peakY = uPeakY_manual if uPeakY_manual is not None else last_uPeakY
                    map_x, map_y = build_warp_maps(width, height, last_uCenterX, effective_peakY, uGain, sigma_y)
                    last_uPeakY = effective_peakY
                    maps_dirty  = False

                new_cutline = get_lowest_hand_related_y_from_pose(pose_results)
                if new_cutline is not None:
                    last_cutline_y_norm = new_cutline

            # ── Warp + mirror ─────────────────────────────────────────────────
            warped   = warp_frame(frame, map_x, map_y)
            mirrored = cv2.flip(warped, 1)

            # ── Segmentation (every SEG_EVERY_N frames, on downscaled input) ──
            if frame_idx % SEG_EVERY_N == 0:
                small_rgb = cv2.resize(
                    cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB),
                    (seg_w, seg_h),
                    interpolation=cv2.INTER_LINEAR
                )
                seg_results  = segmenter.process(small_rgb)
                last_seg_mask = cv2.resize(
                    seg_results.segmentation_mask,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR
                )

            # ── Background (mode-dependent) ────────────────────────────────────
            if background_mode == "combination":
                cut_line = int((last_cutline_y_norm + 0.1) * (height - 1))
                mirrored_orig = cv2.flip(frame, 1)
                bg_for_composite = build_combination_background(
                    captured_bg, mirrored_orig, cut_line, height, width
                )
            else:  # "still"
                bg_for_composite = captured_bg

            # ── Composite ─────────────────────────────────────────────────────
            final_frame = composite_person_over_bg(
                mirrored, last_seg_mask,
                bg_bgr=bg_for_composite, thresh=0.5, feather_px=5
            )

            cv2.imshow("Warped Mirror", final_frame)
            frame_idx += 1

            # ── Periodic frame saving (off by default) ─────────────────────────
            if save_frames:
                now = time.time()
                if now - last_save_time >= SAVE_INTERVAL_SEC:
                    filename = os.path.join(SAVE_DIR, f"frame_{int(now)}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame to: {filename}")
                    last_save_time = now

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('+') or key == ord('='):
                uGain = round(min(uGain + GAIN_STEP, 2.0), 4)
                maps_dirty = True
                print(f"uGain → {uGain:.2f}")

            elif key == ord('-'):
                uGain = round(max(uGain - GAIN_STEP, 0.0), 4)
                maps_dirty = True
                print(f"uGain → {uGain:.2f}")

            elif key in GAIN_PRESETS:
                uGain = GAIN_PRESETS[key]
                maps_dirty = True
                print(f"uGain → {uGain:.2f}  [preset {chr(key)}]")

            elif key == ord('u'):
                base = uPeakY_manual if uPeakY_manual is not None else last_uPeakY
                uPeakY_manual = round(max(base - PEAK_STEP, 0.0), 4)
                maps_dirty = True
                print(f"uPeakY → {uPeakY_manual:.2f} [manual]")

            elif key == ord('d'):
                base = uPeakY_manual if uPeakY_manual is not None else last_uPeakY
                uPeakY_manual = round(min(base + PEAK_STEP, 1.0), 4)
                maps_dirty = True
                print(f"uPeakY → {uPeakY_manual:.2f} [manual]")

            elif key == ord('b'):
                background_mode = "combination" if background_mode == "still" else "still"
                print(f"background_mode → {background_mode}")

            elif key == ord('s'):
                save_frames = not save_frames
                if save_frames:
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    last_save_time = time.time()
                print(f"save_frames → {save_frames}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
