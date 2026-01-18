import cv2
import numpy as np
import mediapipe as mp

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

    scale = 1.0 + uGain * vertical_profile  # >1 near hips -> wider

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
    uPeakY   = 0.5 * (left_hip.y + right_hip.y) - 0.1  # slight upward bias

    uCenterX = max(0.0, min(1.0, uCenterX))
    uPeakY   = max(0.0, min(1.0, uPeakY))
    return uCenterX, uPeakY


def composite_person_over_bg(person_bgr, seg_mask, bg_bgr=None, thresh=0.5, feather_px=5):
    """
    person_bgr : (H,W,3) warped+mirrored person frame
    seg_mask   : (H,W) float32 [0..1] from MediaPipe segmenter
    bg_bgr     : (H,W,3) background image to fill (if None, uses white)
    """
    h, w = person_bgr.shape[:2]

    # Prepare background
    if bg_bgr is None:
        bg_f32 = np.ones_like(person_bgr, dtype=np.float32) * 255.0  # white
    else:
        if bg_bgr.shape[:2] != (h, w):
            bg_bgr = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        bg_f32 = bg_bgr.astype(np.float32)

    # Person mask
    person_mask = (seg_mask >= thresh).astype(np.float32)
    if feather_px > 0:
        k = max(1, int(feather_px))
        if k % 2 == 0:
            k += 1
        person_mask = cv2.GaussianBlur(person_mask, (k, k), 0)

    mask_3 = np.dstack([person_mask]*3)

    person_f32 = person_bgr.astype(np.float32)
    out = mask_3 * person_f32 + (1.0 - mask_3) * bg_f32
    return out.astype(np.uint8)


def main():
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

    # Capture a clean background 
    print("Please move out of the frame. Capturing background in 3 seconds...")
    cv2.waitKey(3000)
    ret, captured_bg = cap.read()
    if ret:
        # Mirror the background to match our final mirrored output
        captured_bg = cv2.flip(captured_bg, 1)
        print("Background captured.")
    else:
        captured_bg = None
        print("Warning: background capture failed. Falling back to white.")

    # Tunable visual parameters 
    uGain    = 0.30
    sigma_y  = 0.30
    fallback_centerX = 0.5
    fallback_peakY   = 0.55

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose, mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1
    ) as segmenter:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Pose on original frame
            rgb_for_pose = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_for_pose)

            uCenterX, uPeakY = get_hip_center_and_peakY_from_pose(pose_results)
            if uCenterX is None or uPeakY is None:
                uCenterX = fallback_centerX
                uPeakY   = fallback_peakY

            # Build warp & warp
            map_x, map_y = build_warp_maps(width, height, uCenterX, uPeakY, uGain, sigma_y)
            warped = warp_frame(frame, map_x, map_y)

            # Mirror (selfie/mirror view)
            mirrored = cv2.flip(warped, 1)

            # Segmentation on mirrored
            rgb_for_seg = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
            seg_results = segmenter.process(rgb_for_seg)
            seg_mask = seg_results.segmentation_mask  # (H,W) float32

            # Composite person over captured background (or white fallback)
            final_frame = composite_person_over_bg(
                mirrored, seg_mask, bg_bgr=captured_bg, thresh=0.5, feather_px=5
            )

            cv2.imshow("Warped Mirror over Captured Background", final_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
