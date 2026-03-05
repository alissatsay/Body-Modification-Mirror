import cv2
import numpy as np
import mediapipe as mp
import os
import time

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
MIRROR = True       # Selfie-style mirror

UGAINS = [0.20, 0.30, 0.40, 0.50, 0.60]

CAPTURE_DELAY_SEC = 5
CAMERA_INDEX = 0     # change to 1 if you have multiple cameras

# --------------------------
# Warp / Composite helpers (unchanged)
# --------------------------
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


def get_hip_center_and_peakY_from_pose(results):
    if not results.pose_landmarks:
        return None, None

    lm = results.pose_landmarks.landmark
    left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

    uCenterX = 0.5 * (left_hip.x + right_hip.x)
    uPeakY = 0.5 * (left_hip.y + right_hip.y) - 0.1

    uCenterX = np.clip(uCenterX, 0.0, 1.0)
    uPeakY = np.clip(uPeakY, 0.0, 1.0)

    return uCenterX, uPeakY


def composite_person_over_bg(person_bgr, seg_mask, bg_bgr):
    h, w = person_bgr.shape[:2]

    if bg_bgr.shape[:2] != (h, w):
        bg_bgr = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    bg_f32 = bg_bgr.astype(np.float32)

    person_mask = (seg_mask >= SEG_THRESH).astype(np.float32)

    if FEATHER_PX > 0:
        k = max(1, int(FEATHER_PX))
        if k % 2 == 0:
            k += 1
        person_mask = cv2.GaussianBlur(person_mask, (k, k), 0)

    mask_3 = np.dstack([person_mask] * 3)

    person_f32 = person_bgr.astype(np.float32)
    out = mask_3 * person_f32 + (1.0 - mask_3) * bg_f32

    return out.astype(np.uint8)

# --------------------------
# Camera capture helpers
# --------------------------
def ensure_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(WARPED_DIR, exist_ok=True)


def put_center_text(img, text, y_offset=0, scale=0.9, thickness=2):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    x = max(10, (w - tw) // 2)
    y = max(th + 10, (h // 2) + y_offset)

    # outline
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def capture_after_countdown(cap, window_name, message, delay_sec):
    """
    Shows live feed + message. Captures one frame after delay_sec seconds.
    Press ESC to abort.
    """
    start = time.time()
    last_frame = None

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Failed to read from camera.")
        
        if MIRROR:
            frame = cv2.flip(frame, 1)
        last_frame = frame

        frame_show = frame.copy()
        elapsed = time.time() - start
        remaining = max(0, int(np.ceil(delay_sec - elapsed)))

        put_center_text(frame_show, message, y_offset=-40)
        put_center_text(frame_show, f"Capturing in {remaining}...", y_offset=40, scale=0.8)

        cv2.imshow(window_name, frame_show)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            raise KeyboardInterrupt("User aborted (ESC).")

        if elapsed >= delay_sec:
            return last_frame


# --------------------------
# Main
# --------------------------
def main():
    ensure_dirs()

    window_name = "Camera Capture"

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    try:
        # 1) Capture background
        print("Please move out of the frame, capturing background in 3 secs...")
        bg_bgr = capture_after_countdown(
            cap,
            window_name,
            "Please move out of the frame.",
            CAPTURE_DELAY_SEC
        )

        cv2.imwrite(BACKGROUND_IMAGE_PATH, bg_bgr)
        print(f"Saved background -> {os.path.abspath(BACKGROUND_IMAGE_PATH)}")

        # 2) Capture person
        print("Please stand still, capturing you in 3 secs...")
        person_bgr = capture_after_countdown(
            cap,
            window_name,
            "Please stand still.",
            CAPTURE_DELAY_SEC
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

            final = composite_person_over_bg(warped, seg_mask, bg_bgr)

            out_name = f"pers{NUM_PERS}uGain{uGain * 100}.png"
            out_path = os.path.join(WARPED_DIR, out_name)
            cv2.imwrite(out_path, final)
            print(f"Saved -> {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()