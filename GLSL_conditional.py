import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation


def build_warp_maps(width, height, uCenterX, uPeakY, uGain, sigma_y=0.2):
    """
    Build remap grids (map_x, map_y) for inverse mapping:
    For each output pixel (x_out, y_out), map_x/map_y specify (x_src, y_src) to sample.
    """
    x_norm = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y_norm = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xv_norm, yv_norm = np.meshgrid(x_norm, y_norm)  # (h, w)

    dy = (yv_norm - uPeakY) / max(sigma_y, 1e-6)
    vertical_profile = np.exp(-(dy ** 2))

    scale = 1.0 + uGain * vertical_profile  # >1 near peak -> wider appearance

    dx = xv_norm - uCenterX
    srcx_norm = uCenterX + dx / scale

    map_x = (srcx_norm * (width - 1)).astype(np.float32)
    map_y = (yv_norm * (height - 1)).astype(np.float32)
    return map_x, map_y


def get_hip_center_and_peakY_from_pose(results):
    if not results.pose_landmarks:
        return None, None

    lm = results.pose_landmarks.landmark
    left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

    uCenterX = 0.5 * (left_hip.x + right_hip.x)
    uPeakY = 0.5 * (left_hip.y + right_hip.y) - 0.1  # slight upward bias

    uCenterX = float(np.clip(uCenterX, 0.0, 1.0))
    uPeakY = float(np.clip(uPeakY, 0.0, 1.0))
    return uCenterX, uPeakY


def make_identity_maps(width, height):
    # Identity inverse-map: output pixel (x,y) samples input (x,y)
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    id_map_x, id_map_y = np.meshgrid(x, y)
    return id_map_x, id_map_y


def alpha_from_segmentation(seg_mask, thresh=0.5, feather_px=11, soft=False):
    """
    seg_mask: float32 (H,W) in [0..1] from MediaPipe
    Returns alpha (H,W) in [0..1]
    - soft=False: hard threshold then feather
    - soft=True : use seg probabilities then feather
    """
    if soft:
        alpha = seg_mask.astype(np.float32)
    else:
        alpha = (seg_mask >= thresh).astype(np.float32)

    if feather_px and feather_px > 0:
        k = int(feather_px)
        if k % 2 == 0:
            k += 1
        k = max(1, k)
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)

    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


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

    # Precompute identity maps once
    id_map_x, id_map_y = make_identity_maps(width, height)

    # Tunables
    uGain = 0.50
    sigma_y = 0.30
    fallback_centerX = 0.5
    fallback_peakY = 0.55

    # Segmentation blending controls
    seg_thresh = 0.5
    feather_px = 11      # increase for smoother boundary, decrease for sharper
    use_soft_alpha = False  # True uses raw probabilities as alpha (often smoother)
    mirror_view = True      # flip final frame for selfie/mirror view

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

            # Pose on original frame (camera space)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb)

            uCenterX, uPeakY = get_hip_center_and_peakY_from_pose(pose_results)
            if uCenterX is None or uPeakY is None:
                uCenterX = fallback_centerX
                uPeakY = fallback_peakY

            # Segmentation on original frame (camera space)
            seg_results = segmenter.process(rgb)
            seg_mask = seg_results.segmentation_mask  # (H,W) float32 in [0..1]

            # Build warp maps (camera space)
            warp_map_x, warp_map_y = build_warp_maps(width, height, uCenterX, uPeakY, uGain, sigma_y)

            # Alpha: 1=person => use warped maps; 0=background => use identity maps
            alpha = alpha_from_segmentation(seg_mask, thresh=seg_thresh, feather_px=feather_px, soft=use_soft_alpha)

            # Blend maps (IMPORTANT: alpha is in output-pixel space)
            map_x = alpha * warp_map_x + (1.0 - alpha) * id_map_x
            map_y = alpha * warp_map_y + (1.0 - alpha) * id_map_y

            # Apply conditional remap
            out = cv2.remap(
                frame, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )

            if mirror_view:
                out = cv2.flip(out, 1)

            cv2.imshow("Conditional Warp (Person Only via Map Blending)", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # Optional quick tuning:
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
