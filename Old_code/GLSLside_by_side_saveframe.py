import cv2
import numpy as np
import mediapipe as mp
import os
import time

mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation


def build_warp_maps(width, height, uCenterX, uPeakY, uGain, sigma_y=0.2):
    """
    Build the remap grids (map_x, map_y) that tell cv2.remap()
    where to sample the source image for each output pixel.
    """
    x_norm = np.linspace(0.0, 1.0, width,  dtype=np.float32)
    y_norm = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xv_norm, yv_norm = np.meshgrid(x_norm, y_norm)  # (h, w)

    dy = (yv_norm - uPeakY) / max(sigma_y, 1e-6)
    vertical_profile = np.exp(-(dy ** 2))  # 1 at peak

    scale = 1.0 + uGain * vertical_profile

    dx = xv_norm - uCenterX
    srcx_norm = uCenterX + dx / scale

    map_x = (srcx_norm * (width  - 1)).astype(np.float32)
    map_y = (yv_norm   * (height - 1)).astype(np.float32)
    return map_x, map_y


def warp_frame(frame_bgr, map_x, map_y):
    return cv2.remap(
        frame_bgr, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


def get_hip_center_and_peakY_from_pose(results):
    """
    Returns (uCenterX, uPeakY) in [0,1] or (None, None) if not available.
    """
    if not results.pose_landmarks:
        return None, None

    lm = results.pose_landmarks.landmark
    left_hip  = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

    uCenterX = 0.5 * (left_hip.x + right_hip.x)
    uPeakY   = 0.5 * (left_hip.y + right_hip.y) - 0.1

    uCenterX = max(0.0, min(1.0, uCenterX))
    uPeakY   = max(0.0, min(1.0, uPeakY))
    return uCenterX, uPeakY


def main():
    save_dir = os.path.join("saved_frames", "exp_sidebyside")
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    # Prime stream & get size
    ret, frame = cap.read()
    if not ret:
        print("Error: couldn't read initial frame.")
        cap.release()
        return
    height, width = frame.shape[:2]

    # (Background capture kept, but not used now)
    print("Please move out of the frame. Capturing background in 3 seconds...")
    cv2.waitKey(3000)
    ret, captured_bg = cap.read()
    if ret:
        captured_bg = cv2.flip(captured_bg, 1)
        print("Background captured.")
    else:
        captured_bg = None
        print("Warning: background capture failed. Continuing anyway.")

    # params
    uGain    = 0.30
    sigma_y  = 0.30
    fallback_centerX = 0.5
    fallback_peakY   = 0.55

    frame_idx = 0  # counter for saved frames

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

            # Pose on original frame (non-mirrored)
            rgb_for_pose = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_for_pose)

            # Warp center from hips
            uCenterX, uPeakY = get_hip_center_and_peakY_from_pose(pose_results)
            if uCenterX is None or uPeakY is None:
                uCenterX = fallback_centerX
                uPeakY   = fallback_peakY

            # Build warp & warp
            map_x, map_y = build_warp_maps(width, height, uCenterX, uPeakY, uGain, sigma_y)
            warped = warp_frame(frame, map_x, map_y)

            # Mirror both original and warped to keep consistent view
            mirrored_orig = cv2.flip(frame, 1)
            mirrored_warp = cv2.flip(warped, 1)

            # --- Crop middle half of the width for both frames ---
            half_width = width // 2                      # 1/2 of original width
            start_x = (width - half_width) // 2          # left bound so that the crop is centered
            end_x = start_x + half_width                 # right bound

            crop_orig = mirrored_orig[:, start_x:end_x, :]
            crop_warp = mirrored_warp[:, start_x:end_x, :]

            # Safety: make sure they have same width (off-by-one issues)
            min_w = min(crop_orig.shape[1], crop_warp.shape[1])
            crop_orig = crop_orig[:, :min_w, :]
            crop_warp = crop_warp[:, :min_w, :]

            # Side-by-side: left = original, right = warped
            side_by_side = np.hstack([crop_orig, crop_warp])

            cv2.imshow("Middle Half: Original (left) vs Warped (right)", side_by_side)

            # Save EVERY side-by-side frame
            filename = os.path.join(save_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(filename, side_by_side)
            frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
