import cv2
import numpy as np
import mediapipe as mp
import os
import time

mp_pose = mp.solutions.pose

# --------------------------
# Config
# --------------------------
NUM_PERS = 5

IMAGES_DIR = "images_for_warping"
WARPED_DIR = "warped_dataset_greenscreen"

PERSON_IMAGE_PATH = os.path.join(IMAGES_DIR, f"pers{NUM_PERS}.png")
BACKGROUND_IMAGE_PATH = os.path.join(IMAGES_DIR, f"pers{NUM_PERS}bg.png")
MASK_IMAGE_PATH = os.path.join(IMAGES_DIR, f"pers{NUM_PERS}_mask.png")

SIGMA_Y = 0.30
MIRROR = True

UGAINS = [
    0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00
]

CAPTURE_DELAY_SEC = 5
CAMERA_INDEX = 0

# --------------------------
# Green screen tuning
# --------------------------
GREEN_LOWER_HSV = (55, 140, 80)
GREEN_UPPER_HSV = (85, 255, 255)

OPEN_KSIZE = 3
CLOSE_KSIZE = 3
BLUR_KSIZE = 0

DESPILL_STRENGTH = 0.35

# Very minimal feathering settings
EDGE_ERODE_KSIZE = 3   # size of erosion used to define edge band
EDGE_BLUR_KSIZE = 3    # tiny blur only on the edge band

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


def warp_frame(frame, map_x, map_y, border_mode=cv2.BORDER_REPLICATE, border_value=0, interpolation=cv2.INTER_LINEAR):
    return cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=border_value
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

# --------------------------
# Green screen helpers
# --------------------------
def make_greenscreen_mask(
    frame_bgr,
    lower_hsv=GREEN_LOWER_HSV,
    upper_hsv=GREEN_UPPER_HSV,
    open_ksize=OPEN_KSIZE,
    close_ksize=CLOSE_KSIZE,
    blur_ksize=BLUR_KSIZE
):
    """
    Returns:
        fg_mask_u8: uint8 mask in {0,255}, where foreground/person = 255
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    bg_mask = cv2.inRange(
        hsv,
        np.array(lower_hsv, dtype=np.uint8),
        np.array(upper_hsv, dtype=np.uint8)
    )

    fg_mask = 255 - bg_mask

    if open_ksize and open_ksize > 0:
        if open_ksize % 2 == 0:
            open_ksize += 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, k)

    if close_ksize and close_ksize > 0:
        if close_ksize % 2 == 0:
            close_ksize += 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k)

    if blur_ksize and blur_ksize > 0:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        fg_mask = cv2.GaussianBlur(fg_mask, (blur_ksize, blur_ksize), 0)

    fg_mask = np.where(fg_mask >= 128, 255, 0).astype(np.uint8)
    return fg_mask


def despill_green(frame_bgr, fg_mask_u8, strength=0.35):
    frame = frame_bgr.astype(np.float32)
    b, g, r = cv2.split(frame)

    fg = fg_mask_u8.astype(np.float32) / 255.0
    max_rb = np.maximum(r, b)
    excess_green = np.maximum(g - max_rb, 0.0)

    g = g - strength * excess_green * fg

    out = cv2.merge([
        np.clip(b, 0, 255),
        np.clip(g, 0, 255),
        np.clip(r, 0, 255)
    ])
    return out.astype(np.uint8)


def composite_with_alpha_soft(
    fg_bgr,
    mask_u8,
    bg_bgr,
    edge_erode_ksize=EDGE_ERODE_KSIZE,
    edge_blur_ksize=EDGE_BLUR_KSIZE
):
    """
    Minimal feathering:
    - interior stays fully opaque
    - exterior stays fully transparent
    - only a very thin boundary band is softened
    """
    h, w = fg_bgr.shape[:2]

    if bg_bgr.shape[:2] != (h, w):
        bg_bgr = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    mask_bin = (mask_u8 > 0).astype(np.uint8)

    if edge_erode_ksize % 2 == 0:
        edge_erode_ksize += 1
    if edge_blur_ksize % 2 == 0:
        edge_blur_ksize += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (edge_erode_ksize, edge_erode_ksize)
    )

    eroded = cv2.erode(mask_bin, kernel)
    edge_band = mask_bin - eroded

    edge_soft = edge_band.astype(np.float32)
    if edge_blur_ksize > 1:
        edge_soft = cv2.GaussianBlur(edge_soft, (edge_blur_ksize, edge_blur_ksize), 0)

    alpha = mask_bin.astype(np.float32)
    alpha = np.where(edge_band > 0, edge_soft, alpha)
    alpha = np.clip(alpha, 0.0, 1.0)

    alpha_3 = np.dstack([alpha, alpha, alpha])

    fg = fg_bgr.astype(np.float32)
    bg = bg_bgr.astype(np.float32)

    out = alpha_3 * fg + (1.0 - alpha_3) * bg
    return np.clip(out, 0, 255).astype(np.uint8)

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

        elapsed = time.time() - start
        remaining = max(0, int(np.ceil(delay_sec - elapsed)))

        put_center_text(frame_show, message, y_offset=-40)
        put_center_text(frame_show, f"Capturing in {remaining}...", y_offset=40, scale=0.8)

        cv2.imshow(window_name, frame_show)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            raise KeyboardInterrupt("User aborted (ESC).")

        if elapsed >= delay_sec:
            return last_frame

# --------------------------
# Preview helpers
# --------------------------
def show_preview(title, image, max_width=700, max_height=900):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    preview = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(title, preview)

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
        print("Please move out of the frame, capturing background...")
        bg_bgr = capture_after_countdown(
            cap,
            window_name,
            "Please move out of the frame.",
            CAPTURE_DELAY_SEC
        )
        cv2.imwrite(BACKGROUND_IMAGE_PATH, bg_bgr)
        print(f"Saved background -> {os.path.abspath(BACKGROUND_IMAGE_PATH)}")

        print("Please stand still in front of the green screen...")
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

    bg_h, bg_w = bg_bgr.shape[:2]
    person_bgr = cv2.resize(person_bgr, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
    h, w = person_bgr.shape[:2]

    fg_mask = make_greenscreen_mask(person_bgr)
    cv2.imwrite(MASK_IMAGE_PATH, fg_mask)
    print(f"Saved mask -> {os.path.abspath(MASK_IMAGE_PATH)}")

    person_despilled = despill_green(person_bgr, fg_mask, strength=DESPILL_STRENGTH)

    preview_comp = composite_with_alpha_soft(person_despilled, fg_mask, bg_bgr)
    show_preview("Foreground mask", fg_mask)
    show_preview("Keyed preview", preview_comp)
    print("Press any key in an OpenCV window to continue to warping...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    fallback_centerX = 0.5
    fallback_peakY = 0.55

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        rgb_for_pose = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_for_pose)

        uCenterX, uPeakY = get_hip_center_and_peakY_from_pose(pose_results)
        if uCenterX is None:
            uCenterX, uPeakY = fallback_centerX, fallback_peakY
            print("Pose not detected. Using fallback warp center.")
        else:
            print(f"Pose center: uCenterX={uCenterX:.3f}, uPeakY={uPeakY:.3f}")

        for uGain in UGAINS:
            map_x, map_y = build_warp_maps(w, h, uCenterX, uPeakY, uGain, SIGMA_Y)

            warped_person = warp_frame(
                person_despilled,
                map_x,
                map_y,
                border_mode=cv2.BORDER_REPLICATE,
                interpolation=cv2.INTER_LINEAR
            )

            warped_mask = warp_frame(
                fg_mask,
                map_x,
                map_y,
                border_mode=cv2.BORDER_CONSTANT,
                border_value=0,
                interpolation=cv2.INTER_NEAREST
            )

            warped_mask = np.where(warped_mask > 0, 255, 0).astype(np.uint8)

            final = composite_with_alpha_soft(warped_person, warped_mask, bg_bgr)

            gain_str = f"{uGain * 100:.0f}"
            out_name = f"pers{NUM_PERS}uGain{gain_str}.png"
            out_path = os.path.join(WARPED_DIR, out_name)

            cv2.imwrite(out_path, final)
            print(f"Saved -> {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()