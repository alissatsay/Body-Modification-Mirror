"""
clinical_mirror/helpers.py

Shared helper functions used across the clinical/Body-Reshaping-Mirror
scripts: the live mirror (live_mirror.py), the alternate map-blending
technique (conditional_warp.py), the comparison tool
(compare_original_vs_warped.py), and the offline dataset-generation
scripts (dataset_generation/*.py).

These were previously defined as near-identical copies inside each of
Old_code/GLSL*.py, DynamicImageWarp*.py and StillImageWarp.py. Behavior
is unchanged from the originals; only the duplication has been removed,
and a few hard-coded module-level constants have become parameters
with the same default values so each caller can still tune them locally.
"""

import os
import time

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation


# ---------------------------------------------------------------------------
# Core warp math
# ---------------------------------------------------------------------------

def build_warp_maps(width, height, uCenterX, uPeakY, uGain, sigma_y=0.30):
    """
    Build the remap grids (map_x, map_y) that tell cv2.remap() where to
    sample the source image for each output pixel: a Gaussian-weighted
    horizontal stretch centered at (uCenterX, uPeakY).
    """
    x_norm = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y_norm = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xv_norm, yv_norm = np.meshgrid(x_norm, y_norm)

    dy = (yv_norm - uPeakY) / max(sigma_y, 1e-6)
    vertical_profile = np.exp(-(dy ** 2))  # 1 at peak

    scale = 1.0 + uGain * vertical_profile  # >1 near peak -> wider

    dx = xv_norm - uCenterX
    srcx_norm = uCenterX + dx / scale

    map_x = (srcx_norm * (width - 1)).astype(np.float32)
    map_y = (yv_norm * (height - 1)).astype(np.float32)
    return map_x, map_y


def make_identity_maps(width, height):
    """Identity inverse-map: output pixel (x, y) samples input (x, y)."""
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    id_map_x, id_map_y = np.meshgrid(x, y)
    return id_map_x, id_map_y


def warp_frame(frame_bgr, map_x, map_y,
               interpolation=cv2.INTER_LINEAR,
               border_mode=cv2.BORDER_REPLICATE,
               border_value=0):
    """
    Apply a remap. Defaults match every original script's plain call
    (linear interpolation, replicated border); the green-screen dataset
    script overrides interpolation/border_mode/border_value per-call
    (e.g. nearest-neighbor + constant border when warping a binary mask).
    """
    return cv2.remap(
        frame_bgr, map_x, map_y,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    )


# ---------------------------------------------------------------------------
# Pose-derived measurements
# ---------------------------------------------------------------------------

def get_hip_center_and_peakY_from_pose(results):
    """
    Returns (uCenterX, uPeakY) in [0, 1], or (None, None) if pose isn't
    available. This is the warp center used by every clinical-mirror script.
    """
    if not results.pose_landmarks:
        return None, None

    lm = results.pose_landmarks.landmark
    left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

    uCenterX = 0.5 * (left_hip.x + right_hip.x)
    uPeakY = 0.5 * (left_hip.y + right_hip.y) - 0.1  # slight upward bias

    uCenterX = max(0.0, min(1.0, uCenterX))
    uPeakY = max(0.0, min(1.0, uPeakY))
    return uCenterX, uPeakY


def get_index_y_from_pose(results):
    """
    Normalized y in [0, 1] for the index finger (via Pose landmarks), or
    None if unavailable. One of two "cut line" strategies used by the
    combination-background mode (see also get_lowest_hand_related_y_from_pose,
    which live_mirror.py uses by default as the more robust of the two).
    Kept for reference/future tuning — not currently called anywhere.
    """
    if not results.pose_landmarks:
        return None

    lm = results.pose_landmarks.landmark
    candidates = []
    for idx in [mp_pose.PoseLandmark.RIGHT_INDEX.value,
                mp_pose.PoseLandmark.LEFT_INDEX.value]:
        pt = lm[idx]
        if pt.visibility > 0.5:
            candidates.append(pt.y)

    if not candidates:
        return None

    y_norm = float(min(candidates))
    return max(0.0, min(1.0, y_norm))


def get_lowest_hand_related_y_from_pose(results):
    """
    Normalized y in [0, 1] for the *lowest* (largest y) of the wrist,
    index, elbow or shoulder landmarks, or None if none are visible
    enough. The more robust of the two "cut line" strategies for
    combination-background mode (multiple candidate landmarks instead of
    just the index finger) -- this is the one live_mirror.py uses.
    """
    if not results.pose_landmarks:
        return None

    lm = results.pose_landmarks.landmark
    candidate_indices = [
        mp_pose.PoseLandmark.RIGHT_WRIST.value,
        mp_pose.PoseLandmark.LEFT_INDEX.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    ]

    candidates = [lm[idx].y for idx in candidate_indices if lm[idx].visibility > 0.5]
    if not candidates:
        return None

    y_norm = float(max(candidates))
    return max(0.0, min(1.0, y_norm))


# ---------------------------------------------------------------------------
# Segmentation / compositing (MediaPipe selfie-segmentation based)
# ---------------------------------------------------------------------------

def composite_person_over_bg(person_bgr, seg_mask, bg_bgr=None, thresh=0.5, feather_px=5):
    """
    person_bgr : (H,W,3) warped(+mirrored) person frame
    seg_mask   : (H,W) float32 [0..1] from MediaPipe's selfie segmenter
    bg_bgr     : (H,W,3) background to composite over; None => white
    """
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
    out = mask_3 * person_bgr.astype(np.float32) + (1.0 - mask_3) * bg_f32
    return out.astype(np.uint8)


def alpha_from_segmentation(seg_mask, thresh=0.5, feather_px=11, soft=False):
    """
    Turn a raw MediaPipe segmentation mask into a feathered alpha in
    [0, 1]. Used by conditional_warp.py's map-blending technique.
    soft=False: hard-threshold then feather. soft=True: feather the raw
    probabilities directly (often smoother, less crisp edges).
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


# ---------------------------------------------------------------------------
# Green-screen chroma-key helpers (dataset generation only)
# ---------------------------------------------------------------------------

GREEN_LOWER_HSV = (55, 140, 80)
GREEN_UPPER_HSV = (85, 255, 255)


def make_greenscreen_mask(frame_bgr,
                           lower_hsv=GREEN_LOWER_HSV,
                           upper_hsv=GREEN_UPPER_HSV,
                           open_ksize=3,
                           close_ksize=3,
                           blur_ksize=0):
    """Returns a uint8 mask in {0, 255} where the foreground/person = 255."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    bg_mask = cv2.inRange(hsv, np.array(lower_hsv, dtype=np.uint8), np.array(upper_hsv, dtype=np.uint8))
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

    return np.where(fg_mask >= 128, 255, 0).astype(np.uint8)


def despill_green(frame_bgr, fg_mask_u8, strength=0.35):
    """Suppress green spill/bleed on the foreground's edges after chroma-keying."""
    frame = frame_bgr.astype(np.float32)
    b, g, r = cv2.split(frame)

    fg = fg_mask_u8.astype(np.float32) / 255.0
    max_rb = np.maximum(r, b)
    excess_green = np.maximum(g - max_rb, 0.0)
    g = g - strength * excess_green * fg

    out = cv2.merge([np.clip(b, 0, 255), np.clip(g, 0, 255), np.clip(r, 0, 255)])
    return out.astype(np.uint8)


def composite_with_alpha_soft(fg_bgr, mask_u8, bg_bgr,
                               edge_erode_ksize=3, edge_blur_ksize=3):
    """
    Minimal-feathering composite: interior stays fully opaque, exterior
    fully transparent, only a thin boundary band is softened. Used with
    the green-screen mask (as opposed to composite_person_over_bg's
    full-mask Gaussian feather, used with the MediaPipe segmentation mask).
    """
    h, w = fg_bgr.shape[:2]
    if bg_bgr.shape[:2] != (h, w):
        bg_bgr = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    mask_bin = (mask_u8 > 0).astype(np.uint8)

    if edge_erode_ksize % 2 == 0:
        edge_erode_ksize += 1
    if edge_blur_ksize % 2 == 0:
        edge_blur_ksize += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_erode_ksize, edge_erode_ksize))
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


# ---------------------------------------------------------------------------
# Live-capture helpers (threaded webcam reads, countdown capture)
# ---------------------------------------------------------------------------

def capture_thread(cap, frame_queue):
    """Continuously grab frames into a queue so the main thread never waits on I/O."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)


def capture_after_countdown(cap, window_name, message, delay_sec, mirror=True):
    """
    Shows the live feed with an on-screen countdown message, and returns
    one frame after delay_sec seconds. Press ESC to abort. Used by the
    dataset-generation scripts to capture a clean background / person shot.
    """
    start = time.time()
    last_frame = None

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Failed to read from camera.")

        if mirror:
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


# ---------------------------------------------------------------------------
# Display / misc utilities
# ---------------------------------------------------------------------------

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def put_center_text(img, text, y_offset=0, scale=0.9, thickness=2):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    x = max(10, (w - tw) // 2)
    y = max(th + 10, (h // 2) + y_offset)

    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def show_preview(title, image, max_width=700, max_height=900):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    preview = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(title, preview)
