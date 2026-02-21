import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation


def rot90ccw(frame_bgr):
    return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)


def ema(prev, new, alpha):
    return new if prev is None else (1 - alpha) * prev + alpha * new


def clamp_delta(prev, new, max_step):
    if prev is None:
        return new
    return max(prev - max_step, min(prev + max_step, new))


def build_warp_maps(width, height, uCenterX, uPeakY, uGain, sigma_y=0.2):
    """
    Build remap grids (map_x, map_y) that tell cv2.remap() where to sample
    the source image for each output pixel.
    uCenterX, uPeakY in [0,1]
    """
    x_norm = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y_norm = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xv_norm, yv_norm = np.meshgrid(x_norm, y_norm)

    dy = (yv_norm - uPeakY) / max(sigma_y, 1e-6)
    vertical_profile = np.exp(-(dy ** 2))  # 1 at peak

    scale = 1.0 + uGain * vertical_profile  # >1 near peak => wider
    dx = xv_norm - uCenterX
    srcx_norm = uCenterX + dx / scale

    map_x = (srcx_norm * (width - 1)).astype(np.float32)
    map_y = (yv_norm * (height - 1)).astype(np.float32)
    return map_x, map_y


def warp_frame(frame_bgr, map_x, map_y):
    return cv2.remap(
        frame_bgr, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


def get_hip_center_and_peakY_from_pose(results, vis_thresh=0.6):
    """
    Returns (uCenterX, uPeakY) in [0,1] or (None, None) if not available/reliable.
    Uses hip landmarks and requires both hips to meet visibility threshold.
    """
    if not results.pose_landmarks:
        return None, None

    lm = results.pose_landmarks.landmark
    left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Require sufficient visibility to avoid "tracking the wall"
    if left_hip.visibility < vis_thresh or right_hip.visibility < vis_thresh:
        return None, None

    uCenterX = 0.5 * (left_hip.x + right_hip.x)
    uPeakY = 0.5 * (left_hip.y + right_hip.y) - 0.1  # slight upward bias

    uCenterX = max(0.0, min(1.0, float(uCenterX)))
    uPeakY = max(0.0, min(1.0, float(uPeakY)))
    return uCenterX, uPeakY


def get_index_y_from_pose(results, vis_thresh=0.6):
    """
    Returns normalized y in [0,1] for the index finger (using Pose landmarks),
    or None if not available / not reliable (visibility below threshold).
    Uses RIGHT_INDEX and LEFT_INDEX if present and visible.
    """
    if not results.pose_landmarks:
        return None

    lm = results.pose_landmarks.landmark
    candidates = []

    for idx in [mp_pose.PoseLandmark.RIGHT_INDEX.value,
                mp_pose.PoseLandmark.LEFT_INDEX.value]:
        pt = lm[idx]
        if pt.visibility >= vis_thresh:
            candidates.append(pt.y)

    if not candidates:
        return None

    y_norm = float(min(candidates))
    y_norm = max(0.0, min(1.0, y_norm))
    return y_norm


# -----------------------------
# Compositing
# -----------------------------
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

    # Feather edges
    if feather_px > 0:
        k = max(1, int(feather_px))
        if k % 2 == 0:
            k += 1
        person_mask = cv2.GaussianBlur(person_mask, (k, k), 0)

    mask_3 = np.dstack([person_mask] * 3)

    person_f32 = person_bgr.astype(np.float32)
    out = mask_3 * person_f32 + (1.0 - mask_3) * bg_f32
    return out.astype(np.uint8)



def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    # Prime stream & get rotated size
    ret, frame = cap.read()
    if not ret:
        print("Error: couldn't read initial frame.")
        cap.release()
        return

    # Rotate every incoming frame BEFORE anything else
    frame = rot90ccw(frame)
    height, width = frame.shape[:2]

    # Capture a clean background (also rotated before anything else)
    print("Please move out of the frame. Capturing background in 3 seconds...")
    cv2.waitKey(3000)
    ret, captured_bg = cap.read()
    if ret:
        captured_bg = rot90ccw(captured_bg)          # rotate background capture
        captured_bg = cv2.flip(captured_bg, 1)       # keep your mirror convention
        print("Background captured.")
    else:
        captured_bg = None
        print("Warning: background capture failed. Falling back to white.")

    # params
    uGain = 0.50
    sigma_y = 0.30
    fallback_centerX = 0.5
    fallback_peakY = 0.55
    fallback_index_y_norm = 0.5

    # Stabilization state
    prev_centerX = None
    prev_peakY = None
    prev_indexY = None

    alpha_pose = 0.15
    max_step_x = 0.03
    max_step_y = 0.03
    max_step_idx = 0.05

    # Visibility gating threshold for trusting landmarks
    vis_thresh = 0.6

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose, mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Rotate every incoming live frame BEFORE any further processing
            frame = rot90ccw(frame)

            # Pose on the rotated frame (so coords match what you display)
            rgb_for_pose = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_for_pose)

            uCenterX_raw, uPeakY_raw = get_hip_center_and_peakY_from_pose(
                pose_results, vis_thresh=vis_thresh
            )

            if uCenterX_raw is None or uPeakY_raw is None:
                uCenterX_raw = prev_centerX if prev_centerX is not None else fallback_centerX
                uPeakY_raw = prev_peakY if prev_peakY is not None else fallback_peakY

            uCenterX_raw = clamp_delta(prev_centerX, uCenterX_raw, max_step_x)
            uPeakY_raw = clamp_delta(prev_peakY, uPeakY_raw, max_step_y)

            uCenterX = ema(prev_centerX, uCenterX_raw, alpha_pose)
            uPeakY = ema(prev_peakY, uPeakY_raw, alpha_pose)

            prev_centerX, prev_peakY = uCenterX, uPeakY

            index_y_raw = get_index_y_from_pose(pose_results, vis_thresh=vis_thresh)
            if index_y_raw is None:
                index_y_raw = prev_indexY if prev_indexY is not None else fallback_index_y_norm

            index_y_raw = clamp_delta(prev_indexY, index_y_raw, max_step_idx)
            index_y_norm = ema(prev_indexY, index_y_raw, alpha_pose)
            prev_indexY = index_y_norm

            cut_line = int((index_y_norm + 0.1) * (height - 1))

            # Build warp & warp (using rotated frame dimensions)
            map_x, map_y = build_warp_maps(width, height, uCenterX, uPeakY, uGain, sigma_y)
            warped = warp_frame(frame, map_x, map_y)

            # Mirror (selfie/mirror view) for both warped person and original frame
            mirrored = cv2.flip(warped, 1)
            mirrored_orig = cv2.flip(frame, 1)

            # build background
            if captured_bg is not None:
                combined_bg = captured_bg.copy()
            else:
                combined_bg = mirrored_orig.copy()

            if combined_bg.shape[:2] != (height, width):
                combined_bg = cv2.resize(combined_bg, (width, height), interpolation=cv2.INTER_LINEAR)

            # Whatever is BELOW the index y coordinate is the ORIGINAL FRAME
            # Whatever is ABOVE is the captured background (already in combined_bg)
            if 0 <= cut_line < height:
                combined_bg[cut_line:, :, :] = mirrored_orig[cut_line:, :, :]

            # Segmentation on mirrored (warped person) [unchanged]
            rgb_for_seg = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
            seg_results = segmenter.process(rgb_for_seg)
            seg_mask = seg_results.segmentation_mask  # (H,W) float32

            # Composite warped person over the combined background
            final_frame = composite_person_over_bg(
                mirrored, seg_mask, bg_bgr=combined_bg, thresh=0.5, feather_px=5
            )

            cv2.imshow("Warped Mirror over Combined Background", final_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
