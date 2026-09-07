"""
clinical_mirror/conditional_warp.py

An alternate warping technique, distinct from live_mirror.py: instead of
warping the whole frame and then compositing the person over a captured
background, this blends the WARP MAP itself against an identity map,
weighted by the segmentation alpha -- so only the person's pixels get
warped, and the background (whatever is actually behind them, live) is
left completely untouched. No captured background image is used at all.

Kept as a separate script because it's a genuinely different approach,
not a redundant copy of anything else here. Previously
Old_code/GLSL_conditional.py -- only the imports changed.
"""

import cv2
import mediapipe as mp

from helpers import (
    build_warp_maps,
    make_identity_maps,
    alpha_from_segmentation,
    get_hip_center_and_peakY_from_pose,
)

mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation


def main():
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

    id_map_x, id_map_y = make_identity_maps(width, height)

    uGain = 0.50
    sigma_y = 0.30
    fallback_centerX = 0.5
    fallback_peakY = 0.55

    seg_thresh = 0.5
    feather_px = 11
    use_soft_alpha = False
    mirror_view = True

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose, mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb)

            uCenterX, uPeakY = get_hip_center_and_peakY_from_pose(pose_results)
            if uCenterX is None or uPeakY is None:
                uCenterX = fallback_centerX
                uPeakY = fallback_peakY

            seg_results = segmenter.process(rgb)
            seg_mask = seg_results.segmentation_mask

            warp_map_x, warp_map_y = build_warp_maps(width, height, uCenterX, uPeakY, uGain, sigma_y)
            alpha = alpha_from_segmentation(seg_mask, thresh=seg_thresh, feather_px=feather_px, soft=use_soft_alpha)

            map_x = alpha * warp_map_x + (1.0 - alpha) * id_map_x
            map_y = alpha * warp_map_y + (1.0 - alpha) * id_map_y

            out = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            if mirror_view:
                out = cv2.flip(out, 1)

            cv2.imshow("Conditional Warp (Person Only via Map Blending)", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(']'):
                uGain = min(1.0, uGain + 0.02)
                print(f"uGain={uGain:.2f}")
            elif key == ord('['):
                uGain = max(0.0, uGain - 0.02)
                print(f"uGain={uGain:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
