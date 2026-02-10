import cv2
import numpy as np
import mediapipe as mp
import time

# -----------------------------
# Global rotation controls
# -----------------------------
ROTATE_DEG = 0                 # any angle (degrees)
ROTATE_DIR = "ccw"              # "ccw" or "cw"

mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation


# -----------------------------
# Frame rotation (FIRST function)
# -----------------------------
def rotate_frame(frame_bgr, deg=ROTATE_DEG, direction=ROTATE_DIR):
    """
    Rotate an image by deg degrees in the requested direction.
    - For exact multiples of 90, uses fast cv2.rotate (no cropping).
    - Otherwise uses warpAffine (keeps full image with a larger canvas).
    direction: "ccw" or "cw"
    """
    if frame_bgr is None:
        return None

    deg = float(deg) % 360.0
    direction = str(direction).lower().strip()
    if direction not in ("ccw", "cw"):
        direction = "ccw"

    # Convert into an equivalent CCW angle in [0, 360)
    ccw_deg = deg if direction == "ccw" else (360.0 - deg) % 360.0

    # Fast paths for multiples of 90 (exact)
    if abs(ccw_deg - 0.0) < 1e-6:
        return frame_bgr
    if abs(ccw_deg - 90.0) < 1e-6:
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if abs(ccw_deg - 180.0) < 1e-6:
        return cv2.rotate(frame_bgr, cv2.ROTATE_180)
    if abs(ccw_deg - 270.0) < 1e-6:
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

    # General angle: rotate with expanded canvas to avoid cropping
    h, w = frame_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), ccw_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    return cv2.warpAffine(
        frame_bgr,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


# -----------------------------
# Stabilization helpers
# -----------------------------
def ema(prev, new, alpha):
    return new if prev is None else (1 - alpha) * prev + alpha * new


def clamp_delta(prev, new, max_step):
    if prev is None:
        return new
    return max(prev - max_step, min(prev + max_step, new))


def clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))


# -----------------------------
# Warp / remap
# -----------------------------
def build_warp_maps(width, height, uCenterX, uPeakY, uGain, sigma_y=0.2):
    x_norm = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y_norm = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xv_norm, yv_norm = np.meshgrid(x_norm, y_norm)

    dy = (yv_norm - uPeakY) / max(sigma_y, 1e-6)
    vertical_profile = np.exp(-(dy ** 2))

    scale = 1.0 + uGain * vertical_profile
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


# -----------------------------
# Pose-derived control signals
# -----------------------------
def get_hip_center_and_peakY_from_pose(results, vis_thresh=0.6):
    if not results.pose_landmarks:
        return None, None

    lm = results.pose_landmarks.landmark
    left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

    if left_hip.visibility < vis_thresh or right_hip.visibility < vis_thresh:
        return None, None

    uCenterX = 0.5 * (left_hip.x + right_hip.x)
    uPeakY = 0.5 * (left_hip.y + right_hip.y) - 0.1

    uCenterX = max(0.0, min(1.0, float(uCenterX)))
    uPeakY = max(0.0, min(1.0, float(uPeakY)))
    return uCenterX, uPeakY


def get_index_y_from_pose(results, vis_thresh=0.6):
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
# Gesture: hand distance (Pose)
# -----------------------------
def get_hand_distance_from_pose(results, vis_thresh=0.6, use_wrist=False):
    if not results.pose_landmarks:
        return None

    lm = results.pose_landmarks.landmark

    if use_wrist:
        li = mp_pose.PoseLandmark.LEFT_WRIST.value
        ri = mp_pose.PoseLandmark.RIGHT_WRIST.value
    else:
        li = mp_pose.PoseLandmark.LEFT_INDEX.value
        ri = mp_pose.PoseLandmark.RIGHT_INDEX.value

    L = lm[li]
    R = lm[ri]

    if L.visibility < vis_thresh or R.visibility < vis_thresh:
        return None

    dx = float(L.x - R.x)
    dy = float(L.y - R.y)
    return (dx * dx + dy * dy) ** 0.5


# -----------------------------
# Compositing
# -----------------------------
def composite_person_over_bg(person_bgr, seg_mask, bg_bgr=None, thresh=0.5, feather_px=5):
    h, w = person_bgr.shape[:2]

    if bg_bgr is None:
        bg_f32 = np.ones_like(person_bgr, dtype=np.float32) * 255.0
    else:
        if bg_bgr.shape[:2] != (h, w):
            bg_bgr = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        bg_f32 = bg_bgr.astype(np.float32)

    person_mask = (seg_mask >= thresh).astype(np.float32)

    if feather_px > 0:
        k = max(1, int(feather_px))
        if k % 2 == 0:
            k += 1
        person_mask = cv2.GaussianBlur(person_mask, (k, k), 0)

    mask_3 = np.dstack([person_mask] * 3)

    person_f32 = person_bgr.astype(np.float32)
    out = mask_3 * person_f32 + (1.0 - mask_3) * bg_f32
    return out.astype(np.uint8)


# -----------------------------
# UI helper: transient banner
# -----------------------------
def start_banner(text, duration_sec=1.2):
    return {"text": text, "until": time.time() + duration_sec}


def draw_banner(frame_bgr, banner, y):
    """
    Draw a centered banner at a y position in the *current* frame coordinates.
    (Since we rotate the frame first, this is always parallel to the frame.)
    """
    if banner is None or time.time() > banner["until"]:
        return

    h, w = frame_bgr.shape[:2]
    text = banner["text"]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 3

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x = (w - tw) // 2
    y_text = int(y)

    pad_x, pad_y = 18, 14
    x1 = x - pad_x
    y1 = y_text - th - pad_y
    x2 = x + tw + pad_x
    y2 = y_text + baseline + pad_y

    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w - 1, x2); y2 = min(h - 1, y2)

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(frame_bgr, text, (x, y_text), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


# -----------------------------
# Main
# -----------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    # Prime stream & get size (rotate BEFORE size is determined)
    ret, frame = cap.read()
    if not ret:
        print("Error: couldn't read initial frame.")
        cap.release()
        return

    frame = rotate_frame(frame)
    height, width = frame.shape[:2]

    # Capture a clean background (rotate, then mirror)
    print("Please move out of the frame. Capturing background in 3 seconds...")
    cv2.waitKey(3000)
    ret, captured_bg = cap.read()
    if ret:
        captured_bg = rotate_frame(captured_bg)
        captured_bg = cv2.flip(captured_bg, 1)
        print("Background captured.")
    else:
        captured_bg = None
        print("Warning: background capture failed. Falling back to white.")

    # params
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

    vis_thresh = 0.6

    # -----------------------------
    # Incremental uGain control state
    # -----------------------------
    uGain_live = 0.50
    uGain_min = -0.70
    uGain_max = 2.50

    prev_hand_dist = None

    gain_sensitivity = 6.0
    dist_deadzone = 0.005
    max_gain_step = 0.10

    alpha_gain = 0.25
    prev_gain_smoothed = None

    # -----------------------------
    # Mode gate: idle -> ready -> active
    # -----------------------------
    mode = "idle"
    prev_mode = mode

    stable_required_sec = 2.0

    stable_eps = 0.006
    activate_eps = 0.012
    stop_eps = 0.006
    stop_hold_sec = 0.40

    stable_start_t = None
    stop_start_t = None

    banner = None

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

            # Rotate each incoming frame BEFORE any processing
            frame = rotate_frame(frame)

            now = time.time()

            # Pose on the rotated frame
            rgb_for_pose = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_for_pose)

            # --- Stabilized hips ---
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

            # --- Stabilized index y (kept for compatibility) ---
            index_y_raw = get_index_y_from_pose(pose_results, vis_thresh=vis_thresh)
            if index_y_raw is None:
                index_y_raw = prev_indexY if prev_indexY is not None else fallback_index_y_norm
            index_y_raw = clamp_delta(prev_indexY, index_y_raw, max_step_idx)
            index_y_norm = ema(prev_indexY, index_y_raw, alpha_pose)
            prev_indexY = index_y_norm

            # --- Hand distance & mode gate ---
            hand_dist = get_hand_distance_from_pose(
                pose_results, vis_thresh=vis_thresh, use_wrist=False
            )

            move = None
            d_dist = None
            if hand_dist is not None and prev_hand_dist is not None:
                d_dist = float(hand_dist - prev_hand_dist)
                move = abs(d_dist)

            if hand_dist is None:
                mode = "idle"
                stable_start_t = None
                stop_start_t = None
            else:
                if mode == "idle":
                    if move is not None and move < stable_eps:
                        if stable_start_t is None:
                            stable_start_t = now
                        elif (now - stable_start_t) >= stable_required_sec:
                            mode = "ready"
                            stop_start_t = None
                    else:
                        stable_start_t = None

                elif mode == "ready":
                    if move is not None and move >= activate_eps:
                        mode = "active"
                        stop_start_t = None

                elif mode == "active":
                    if d_dist is not None:
                        dd = d_dist
                        if abs(dd) < dist_deadzone:
                            dd = 0.0

                        d_gain = gain_sensitivity * dd
                        d_gain = clamp(d_gain, -max_gain_step, max_gain_step)

                        uGain_live += d_gain
                        uGain_live = clamp(uGain_live, uGain_min, uGain_max)

                    if move is not None and move < stop_eps:
                        if stop_start_t is None:
                            stop_start_t = now
                        elif (now - stop_start_t) >= stop_hold_sec:
                            mode = "idle"
                            stable_start_t = None
                            stop_start_t = None
                    else:
                        stop_start_t = None

            if hand_dist is not None:
                prev_hand_dist = hand_dist

            # Detect transitions and start banners
            if prev_mode != mode:
                if mode == "active":
                    banner = start_banner("MODIFICATION MODE: ON", duration_sec=1.2)
                elif prev_mode == "active" and mode != "active":
                    banner = start_banner("MODIFICATION MODE: OFF", duration_sec=1.2)
                prev_mode = mode

            # Smooth the used gain slightly
            uGain_used = ema(prev_gain_smoothed, uGain_live, alpha_gain)
            prev_gain_smoothed = uGain_used

            # Build warp & warp
            map_x, map_y = build_warp_maps(width, height, uCenterX, uPeakY, uGain_used, sigma_y)
            warped = warp_frame(frame, map_x, map_y)

            # Mirror warped person
            mirrored = cv2.flip(warped, 1)

            # Ensure captured background matches size
            if captured_bg is None:
                bg = None
            else:
                bg = captured_bg
                if bg.shape[:2] != (height, width):
                    bg = cv2.resize(bg, (width, height), interpolation=cv2.INTER_LINEAR)

            # Segmentation on mirrored
            rgb_for_seg = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
            seg_results = segmenter.process(rgb_for_seg)
            seg_mask = seg_results.segmentation_mask

            # Composite over captured background ONLY
            final_frame = composite_person_over_bg(
                mirrored, seg_mask, bg_bgr=bg, thresh=0.5, feather_px=5
            )

            # -----------------------------
            # NEW: Dynamic overlay placement (always "top" of the rotated frame)
            # -----------------------------
            h, w = final_frame.shape[:2]

            banner_y = int(0.12 * h)      # ~12% from top
            draw_banner(final_frame, banner, y=banner_y)

            x = int(0.03 * w)             # 3% left margin
            y1 = int(0.06 * h)            # line 1 ~6% from top
            y2 = int(0.11 * h)            # line 2 ~11% from top

            hd = hand_dist if hand_dist is not None else -1.0
            mv = move if move is not None else -1.0

            cv2.putText(
                final_frame,
                f"MODE: {mode.upper()}  uGain: {uGain_live:.2f}  hand_dist: {hd:.3f}  move: {mv:.4f}",
                (x, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                final_frame,
                "Hold hands still ~3s to ARM, then move to MODIFY. Stop moving to EXIT. (r reset, q quit)",
                (x, y2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Warped Mirror over Captured Background", final_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                uGain_live = 0.50
                prev_gain_smoothed = None
                mode = "idle"
                prev_mode = "idle"
                stable_start_t = None
                stop_start_t = None
                banner = start_banner("RESET", duration_sec=0.8)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
