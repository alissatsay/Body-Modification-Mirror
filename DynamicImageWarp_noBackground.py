import cv2
import numpy as np
import mediapipe as mp
import os
import time

mp_pose = mp.solutions.pose

# --------------------------
# Config
# --------------------------
NUM_PERS = 2

IMAGES_DIR = "images_for_warping"
WARPED_DIR = "warped_dataset_nobackground"

PERSON_IMAGE_PATH = os.path.join(IMAGES_DIR, f"pers{NUM_PERS}.png")

SIGMA_Y = 0.30
MIRROR = True

UGAINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
          0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

CAPTURE_DELAY_SEC = 5
CAMERA_INDEX = 0

# --------------------------
# Warp helpers
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

    return np.clip(uCenterX, 0, 1), np.clip(uPeakY, 0, 1)

# --------------------------
# Camera helpers
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

    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def capture_after_countdown(cap, window_name, message, delay_sec):
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

        remaining = max(0, int(np.ceil(delay_sec - (time.time() - start))))

        put_center_text(frame_show, message, -40)
        put_center_text(frame_show, f"Capturing in {remaining}...", 40, 0.8)

        cv2.imshow(window_name, frame_show)
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

        if time.time() - start >= delay_sec:
            return last_frame

# --------------------------
# Main
# --------------------------
def main():
    ensure_dirs()

    window_name = "Camera Capture"

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    try:
        print("Capturing person...")
        person_bgr = capture_after_countdown(
            cap,
            window_name,
            "Stand still.",
            CAPTURE_DELAY_SEC
        )

        cv2.imwrite(PERSON_IMAGE_PATH, person_bgr)
        print("Saved person image.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    h, w = person_bgr.shape[:2]

    fallback_centerX = 0.5
    fallback_peakY = 0.55

    with mp_pose.Pose(static_image_mode=True) as pose:

        rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        uCenterX, uPeakY = get_hip_center_and_peakY_from_pose(results)

        if uCenterX is None:
            uCenterX, uPeakY = fallback_centerX, fallback_peakY

        for uGain in UGAINS:
            map_x, map_y = build_warp_maps(w, h, uCenterX, uPeakY, uGain, SIGMA_Y)
            warped = warp_frame(person_bgr, map_x, map_y)

            out_name = f"pers{NUM_PERS}_uGain{int(uGain*100)}.png"
            out_path = os.path.join(WARPED_DIR, out_name)

            cv2.imwrite(out_path, warped)
            print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()