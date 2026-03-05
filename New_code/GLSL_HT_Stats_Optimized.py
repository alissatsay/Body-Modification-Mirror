"""
Warped Mirror (optimized)

What’s optimized (FPS):
1) Process at lower internal resolution (PROC_W/PROC_H), upscale only for display
2) Warp map generation: precompute grids + reuse buffers (no meshgrid/linspace per frame)
3) Compositing: uint8 alpha blend (no float32 + dstack)
4) Preallocate overlays and reuse (no per-frame np.zeros allocations)
5) Decimate heavy models (pose/hands/seg) with frame skipping + reuse last results
6) Optional: avoid expensive fullscreen resize path by letting OS upscale

Profiler included:
- Prints every ~2s with FPS (from P50 frame_ms) + stage P50/P95 + robustness stats
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from collections import defaultdict, deque, Counter

# -----------------------
# Config
# -----------------------
ROTATE_DEG = 90
ROTATE_DIR = "ccw"  # "ccw" or "cw"

# Display resolution (what you show on screen)
OUTPUT_W = 1080
OUTPUT_H = 1920

# Internal processing resolution (major FPS lever)
# Start with 540x960; try 720x1280 if you can afford it.
PROC_W = 540
PROC_H = 960

PRIMARY_MONITOR_WIDTH = 1920
WINDOW_NAME = "Warped Mirror"

OPEN_HAND_REQUIRE_EXTENDED = 4

IDLE_COLOR = (247, 94, 77)      # BGR
IDLE_FILL_ALPHA = 0.22
IDLE_RING_ALPHA = 0.55

ACTIVE_COLOR = (0, 220, 0)      # BGR
ACTIVE_FILL_ALPHA = 0.18

RING_THICKNESS = 6
PALM_RADIUS_SCALE = 0.85
RING_START_DEG = -90

# Decimation (run models every N frames; reuse last results in between)
POSE_EVERY = 3
HANDS_EVERY = 3
SEG_EVERY = 4

# If True, skip upscaling to OUTPUT_W/H and display at PROC resolution (often faster)
DISPLAY_AT_PROC_RES = False

# -----------------------
# MediaPipe
# -----------------------
mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_hands = mp.solutions.hands

# -----------------------
# Stats / profiler
# -----------------------
MODE_COUNTS = Counter()
MODE_TRANSITIONS = Counter()
_last_mode = None

class RollingStats:
    def __init__(self, window=600):
        self.data = defaultdict(lambda: deque(maxlen=window))

    def add(self, key, value):
        self.data[key].append(float(value))

    def pct(self, key, p):
        arr = np.asarray(self.data[key], dtype=np.float32)
        if arr.size == 0:
            return None
        return float(np.percentile(arr, p))

    def mean(self, key):
        arr = np.asarray(self.data[key], dtype=np.float32)
        if arr.size == 0:
            return None
        return float(arr.mean())

def now_ms():
    return time.perf_counter() * 1000.0

STATS = RollingStats(window=600)
_last_report_t = time.perf_counter()
_report_every_sec = 2.0

def report_stats():
    p50_ft = STATS.pct("frame_ms", 50)
    p95_ft = STATS.pct("frame_ms", 95)
    fps = 1000.0 / p50_ft if p50_ft and p50_ft > 0 else None

    def fmt_ms(k):
        p50 = STATS.pct(k, 50)
        p95 = STATS.pct(k, 95)
        if p50 is None:
            return f"{k}: n/a"
        return f"{k}: {p50:.1f}/{p95:.1f}ms"

    pose_ok = STATS.mean("pose_ok") or 0.0
    hd_ok = STATS.mean("hand_dist_ok") or 0.0
    hands_open = STATS.mean("hands_open_ok") or 0.0
    fg_ratio = STATS.mean("fg_ratio") or 0.0

    lines = [
        f"FPS~{fps:.1f}  frame_ms P50/P95={p50_ft:.1f}/{p95_ft:.1f}",
        fmt_ms("cap_ms"),
        fmt_ms("resize_in_ms"),
        fmt_ms("pose_ms"),
        fmt_ms("hands_ms"),
        fmt_ms("maps_ms"),
        fmt_ms("remap_ms"),
        fmt_ms("seg_ms"),
        fmt_ms("comp_ms"),
        fmt_ms("render_ms"),
        f"pose_ok={pose_ok:.2f} hand_dist_ok={hd_ok:.2f} hands_open_ok={hands_open:.2f} fg_ratio={fg_ratio:.2f}",
    ]
    print(" | ".join(lines))

# -----------------------
# Utils
# -----------------------
def rotate_frame(frame_bgr, deg=ROTATE_DEG, direction=ROTATE_DIR):
    if frame_bgr is None:
        return None
    deg = float(deg) % 360.0
    direction = str(direction).lower().strip()
    if direction not in ("ccw", "cw"):
        direction = "ccw"

    ccw_deg = deg if direction == "ccw" else (360.0 - deg) % 360.0

    if abs(ccw_deg - 0.0) < 1e-6:
        return frame_bgr
    if abs(ccw_deg - 90.0) < 1e-6:
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if abs(ccw_deg - 180.0) < 1e-6:
        return cv2.rotate(frame_bgr, cv2.ROTATE_180)
    if abs(ccw_deg - 270.0) < 1e-6:
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

    h, w = frame_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), ccw_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy
    return cv2.warpAffine(frame_bgr, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

def ema(prev, new, alpha):
    return new if prev is None else (1 - alpha) * prev + alpha * new

def clamp_delta(prev, new, max_step):
    if prev is None:
        return new
    return max(prev - max_step, min(prev + max_step, new))

def clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))

# -----------------------
# Warp map builder (optimized maps_ms)
# -----------------------
class WarpMapBuilder:
    """
    Builds map_x/map_y for cv2.remap.
    Optimized:
      - Precomputes normalized grids once
      - Reuses buffers; no meshgrid/linspace allocations per frame
      - map_y is constant for this warp (y unchanged)
    """
    def __init__(self, width, height, sigma_y=0.30):
        self.w = int(width)
        self.h = int(height)
        self.sigma_y = max(float(sigma_y), 1e-6)

        x = np.linspace(0.0, 1.0, self.w, dtype=np.float32)
        y = np.linspace(0.0, 1.0, self.h, dtype=np.float32)

        # Precompute grids
        self.xv = np.tile(x, (self.h, 1))
        self.yv = np.tile(y.reshape(-1, 1), (1, self.w))

        # Constant map_y (pixel coords)
        self.map_y = (self.yv * (self.h - 1)).astype(np.float32)

        # Reusable buffers
        self.vertical_profile = np.empty((self.h, self.w), dtype=np.float32)
        self.scale = np.empty((self.h, self.w), dtype=np.float32)
        self.map_x = np.empty((self.h, self.w), dtype=np.float32)

    def build(self, uCenterX, uPeakY, uGain):
        uCenterX = float(uCenterX)
        uPeakY = float(uPeakY)
        uGain = float(uGain)

        # dy = (yv - peak)/sigma
        dy = (self.yv - uPeakY) / self.sigma_y

        # vertical_profile = exp(-(dy^2))
        np.square(dy, out=dy)
        np.negative(dy, out=dy)
        np.exp(dy, out=self.vertical_profile)

        # scale = 1 + gain*vertical_profile
        self.scale[:] = 1.0 + uGain * self.vertical_profile

        # srcx = center + (x-center)/scale
        dx = self.xv - uCenterX
        srcx = uCenterX + (dx / self.scale)

        # map_x in pixel coords
        np.multiply(srcx, (self.w - 1), out=self.map_x)
        return self.map_x, self.map_y

def warp_frame(frame_bgr, map_x, map_y, border=cv2.BORDER_REPLICATE):
    return cv2.remap(frame_bgr, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=border)

# -----------------------
# Pose helpers
# -----------------------
def get_hip_center_and_peakY_from_pose(results, vis_thresh=0.6):
    if not results or not results.pose_landmarks:
        return None, None
    lm = results.pose_landmarks.landmark
    left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
    if left_hip.visibility < vis_thresh or right_hip.visibility < vis_thresh:
        return None, None

    uCenterX = 0.5 * (left_hip.x + right_hip.x)
    uPeakY = 0.5 * (left_hip.y + right_hip.y) - 0.1
    return clamp(uCenterX, 0.0, 1.0), clamp(uPeakY, 0.0, 1.0)

def get_index_y_from_pose(results, vis_thresh=0.6):
    if not results or not results.pose_landmarks:
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
    return clamp(min(candidates), 0.0, 1.0)

def get_hand_distance_from_pose(results, vis_thresh=0.6, use_wrist=False):
    if not results or not results.pose_landmarks:
        return None
    lm = results.pose_landmarks.landmark
    if use_wrist:
        li = mp_pose.PoseLandmark.LEFT_WRIST.value
        ri = mp_pose.PoseLandmark.RIGHT_WRIST.value
    else:
        li = mp_pose.PoseLandmark.LEFT_INDEX.value
        ri = mp_pose.PoseLandmark.RIGHT_INDEX.value

    L, R = lm[li], lm[ri]
    if L.visibility < vis_thresh or R.visibility < vis_thresh:
        return None
    dx = float(L.x - R.x)
    dy = float(L.y - R.y)
    return (dx * dx + dy * dy) ** 0.5

# -----------------------
# Segmentation + composite (optimized comp_ms)
# -----------------------
def composite_person_over_bg_fast(person_bgr, seg_mask, bg_bgr=None, thresh=0.5, feather_px=5):
    """
    Faster than float32 + dstack blend:
      - alpha mask uint8 (0..255)
      - optional blur on alpha
      - bitwise compose (integer ops)
    """
    if bg_bgr is None:
        bg_bgr = np.full_like(person_bgr, 255, dtype=np.uint8)

    # seg_mask is float32 [0..1]
    alpha = (seg_mask >= thresh).astype(np.uint8) * 255

    if feather_px > 0:
        k = int(feather_px)
        if k % 2 == 0:
            k += 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)

    alpha3 = cv2.merge([alpha, alpha, alpha])
    inv = cv2.bitwise_not(alpha3)
    fg = cv2.bitwise_and(person_bgr, alpha3)
    bg = cv2.bitwise_and(bg_bgr, inv)
    return cv2.add(fg, bg)

# -----------------------
# UI helpers
# -----------------------
def start_banner(text, duration_sec=1.2):
    return {"text": text, "until": time.time() + duration_sec}

def draw_banner(frame_bgr, banner, y):
    if banner is None or time.time() > banner["until"]:
        return
    h, w = frame_bgr.shape[:2]
    text = banner["text"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.15
    thickness = 3
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y_text = int(y)
    pad_x, pad_y = 22, 16
    x1 = max(0, x - pad_x)
    y1 = max(0, y_text - th - pad_y)
    x2 = min(w - 1, x + tw + pad_x)
    y2 = min(h - 1, y_text + baseline + pad_y)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(frame_bgr, text, (x, y_text), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def count_extended_fingers(hand_lm, handedness_label="Right"):
    THUMB_TIP, THUMB_IP = 4, 3
    INDEX_TIP, INDEX_PIP = 8, 6
    MIDDLE_TIP, MIDDLE_PIP = 12, 10
    RING_TIP, RING_PIP = 16, 14
    PINKY_TIP, PINKY_PIP = 20, 18
    extended = 0
    if hand_lm.landmark[INDEX_TIP].y < hand_lm.landmark[INDEX_PIP].y:
        extended += 1
    if hand_lm.landmark[MIDDLE_TIP].y < hand_lm.landmark[MIDDLE_PIP].y:
        extended += 1
    if hand_lm.landmark[RING_TIP].y < hand_lm.landmark[RING_PIP].y:
        extended += 1
    if hand_lm.landmark[PINKY_TIP].y < hand_lm.landmark[PINKY_PIP].y:
        extended += 1
    if handedness_label.lower().startswith("right"):
        if hand_lm.landmark[THUMB_TIP].x > hand_lm.landmark[THUMB_IP].x:
            extended += 1
    else:
        if hand_lm.landmark[THUMB_TIP].x < hand_lm.landmark[THUMB_IP].x:
            extended += 1
    return extended

def both_hands_open(hands_results, min_extended=4):
    if not hands_results or not hands_results.multi_hand_landmarks or not hands_results.multi_handedness:
        return False
    if len(hands_results.multi_hand_landmarks) < 2:
        return False
    ok = 0
    for hlm, hinfo in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
        label = hinfo.classification[0].label
        if count_extended_fingers(hlm, label) >= min_extended:
            ok += 1
    return ok >= 2

def _landmark_xy_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def get_palm_circles_from_pose(results, w, h, vis_thresh=0.6):
    if not results or not results.pose_landmarks:
        return []
    lm = results.pose_landmarks.landmark
    circles = []
    hands_triplets = [
        (mp_pose.PoseLandmark.LEFT_WRIST.value,
         mp_pose.PoseLandmark.LEFT_INDEX.value,
         mp_pose.PoseLandmark.LEFT_PINKY.value),
        (mp_pose.PoseLandmark.RIGHT_WRIST.value,
         mp_pose.PoseLandmark.RIGHT_INDEX.value,
         mp_pose.PoseLandmark.RIGHT_PINKY.value),
    ]
    for wi, ii, pi in hands_triplets:
        W, I, P = lm[wi], lm[ii], lm[pi]
        if W.visibility < vis_thresh or I.visibility < vis_thresh or P.visibility < vis_thresh:
            continue
        wx, wy = _landmark_xy_px(W, w, h)
        ix, iy = _landmark_xy_px(I, w, h)
        px, py = _landmark_xy_px(P, w, h)
        cx = int((wx + ix + px) / 3.0)
        cy = int((wy + iy + py) / 3.0)
        d_wi = ((wx - ix) ** 2 + (wy - iy) ** 2) ** 0.5
        d_wp = ((wx - px) ** 2 + (wy - py) ** 2) ** 0.5
        r = int(PALM_RADIUS_SCALE * max(d_wi, d_wp))
        if r < 8:
            r = 8
        circles.append((cx, cy, r))
    return circles

def draw_palm_ui_to_overlay(overlay_bgr, overlay_alpha, circles,
                           fill_color_bgr, fill_alpha,
                           ring_progress=None, ring_alpha=0.0, ring_thickness=6):
    if not circles:
        return
    h, w = overlay_alpha.shape[:2]
    if fill_alpha > 0:
        for (cx, cy, r) in circles:
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(overlay_bgr, (int(cx), int(cy)), int(r), fill_color_bgr, -1, lineType=cv2.LINE_AA)
                cv2.circle(overlay_alpha, (int(cx), int(cy)), int(r), float(fill_alpha), -1, lineType=cv2.LINE_AA)

    if ring_progress is not None and ring_alpha > 0 and ring_thickness > 0:
        p = clamp(ring_progress, 0.0, 1.0)
        if p > 0:
            end_angle = RING_START_DEG + int(360.0 * p)
            for (cx, cy, r) in circles:
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.ellipse(overlay_bgr, (int(cx), int(cy)), (int(r), int(r)),
                                0, RING_START_DEG, end_angle,
                                fill_color_bgr, int(ring_thickness), lineType=cv2.LINE_AA)
                    cv2.ellipse(overlay_alpha, (int(cx), int(cy)), (int(r), int(r)),
                                0, RING_START_DEG, end_angle,
                                float(ring_alpha), int(ring_thickness), lineType=cv2.LINE_AA)

def alpha_blend_inplace(dst_bgr, overlay_bgr, overlay_alpha):
    # overlay_alpha: float32 0..1
    a = np.clip(overlay_alpha, 0.0, 1.0).astype(np.float32)
    a3 = np.dstack([a, a, a])
    dst_f = dst_bgr.astype(np.float32)
    ov_f = overlay_bgr.astype(np.float32)
    out = a3 * ov_f + (1.0 - a3) * dst_f
    dst_bgr[:] = np.clip(out, 0, 255).astype(np.uint8)

# -----------------------
# Main
# -----------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    ret, frame0 = cap.read()
    if not ret:
        print("Error: couldn't read initial frame.")
        cap.release()
        return

    frame0 = rotate_frame(frame0)
    # resize to processing resolution immediately
    frame0 = cv2.resize(frame0, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
    height, width = frame0.shape[:2]  # should match PROC_H/PROC_W

    print("Please move out of the frame. Capturing background in 3 seconds...")
    cv2.waitKey(3000)
    ret, captured_bg = cap.read()
    if ret:
        captured_bg = rotate_frame(captured_bg)
        captured_bg = cv2.resize(captured_bg, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
        captured_bg = cv2.flip(captured_bg, 1)
        print("Background captured.")
    else:
        captured_bg = None
        print("Warning: background capture failed. Falling back to white.")

    # Warp params
    sigma_y = 0.30
    warp_builder = WarpMapBuilder(PROC_W, PROC_H, sigma_y=sigma_y)

    fallback_centerX = 0.5
    fallback_peakY = 0.55
    fallback_index_y_norm = 0.5

    prev_centerX = None
    prev_peakY = None
    prev_indexY = None

    alpha_pose = 0.15
    max_step_x = 0.03
    max_step_y = 0.03
    max_step_idx = 0.05

    vis_thresh = 0.6

    uGain_live = 0.50
    uGain_min = -0.70
    uGain_max = 2.50

    prev_hand_dist = None
    gain_sensitivity = 6.0
    dist_deadzone = 0.005
    max_gain_step = 0.10

    alpha_gain = 0.25
    prev_gain_smoothed = None

    mode = "idle"
    stable_required_sec = 2.0
    stable_eps = 0.006
    activate_eps = 0.012
    stop_eps = 0.006
    stop_hold_sec = 0.40

    stable_start_t = None
    stop_start_t = None

    banner = start_banner("HOLD YOUR HANDS STILL", duration_sec=2.0)

    # Preallocate overlays (avoid per-frame allocation)
    ov_bgr = np.empty((PROC_H, PROC_W, 3), dtype=np.uint8)
    ov_a = np.empty((PROC_H, PROC_W), dtype=np.float32)

    # Results caches for decimation
    frame_idx = 0
    last_pose_results = None
    last_hands_results = None
    last_seg_mask = None

    # Window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_NAME, PRIMARY_MONITOR_WIDTH, 0)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose, mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            t0 = now_ms()
            now = time.time()

            # 1) Capture + rotate
            t = now_ms()
            ret, frame = cap.read()
            STATS.add("cap_read_ok", 1.0 if ret else 0.0)
            if not ret:
                break
            frame = rotate_frame(frame)
            STATS.add("cap_ms", now_ms() - t)

            # resize to processing resolution ASAP
            t = now_ms()
            frame = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
            STATS.add("resize_in_ms", now_ms() - t)

            # 2) Pose + Hands RGB (one conversion)
            rgb_for_pose = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Pose decimation
            t = now_ms()
            if frame_idx % POSE_EVERY == 0:
                pose_results = pose.process(rgb_for_pose)
                last_pose_results = pose_results
            else:
                pose_results = last_pose_results
            STATS.add("pose_ms", now_ms() - t)

            # Hands decimation
            t = now_ms()
            if frame_idx % HANDS_EVERY == 0:
                hands_results = hands.process(rgb_for_pose)
                last_hands_results = hands_results
            else:
                hands_results = last_hands_results
            STATS.add("hands_ms", now_ms() - t)

            hands_open_ok = both_hands_open(hands_results, min_extended=OPEN_HAND_REQUIRE_EXTENDED)
            STATS.add("hands_open_ok", 1.0 if hands_open_ok else 0.0)

            # Pose validity + smoothed control params
            uCenterX_raw, uPeakY_raw = get_hip_center_and_peakY_from_pose(pose_results, vis_thresh=vis_thresh)
            pose_ok = (uCenterX_raw is not None and uPeakY_raw is not None)
            STATS.add("pose_ok", 1.0 if pose_ok else 0.0)

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
            prev_indexY = ema(prev_indexY, index_y_raw, alpha_pose)

            hand_dist = get_hand_distance_from_pose(pose_results, vis_thresh=vis_thresh, use_wrist=False)
            STATS.add("hand_dist_ok", 1.0 if (hand_dist is not None) else 0.0)

            move = None
            d_dist = None
            if hand_dist is not None and prev_hand_dist is not None:
                d_dist = float(hand_dist - prev_hand_dist)
                move = abs(d_dist)

            # 3) Mode machine + transition stats
            global _last_mode
            if _last_mode is None:
                _last_mode = mode
            elif mode != _last_mode:
                MODE_TRANSITIONS[(_last_mode, mode)] += 1
                _last_mode = mode
            MODE_COUNTS[mode] += 1

            if hand_dist is None:
                mode = "idle"
                stable_start_t = None
                stop_start_t = None
            else:
                if mode == "idle":
                    if hands_open_ok and (move is not None) and (move < stable_eps):
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
                        dd = 0.0 if abs(d_dist) < dist_deadzone else d_dist
                        d_gain = clamp(gain_sensitivity * dd, -max_gain_step, max_gain_step)
                        uGain_live = clamp(uGain_live + d_gain, uGain_min, uGain_max)

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

            uGain_used = ema(prev_gain_smoothed, uGain_live, alpha_gain)
            prev_gain_smoothed = uGain_used

            # 4) Background (already PROC size)
            bg = captured_bg  # may be None

            # 5) Warp maps (optimized)
            t = now_ms()
            map_x, map_y = warp_builder.build(uCenterX, uPeakY, uGain_used)
            STATS.add("maps_ms", now_ms() - t)

            # 6) UI overlay (preallocated buffers)
            t = now_ms()
            ov_bgr.fill(0)
            ov_a.fill(0.0)

            circles_frame = get_palm_circles_from_pose(pose_results, PROC_W, PROC_H, vis_thresh=vis_thresh)

            if mode in ("idle", "ready"):
                draw_palm_ui_to_overlay(ov_bgr, ov_a, circles_frame, IDLE_COLOR, IDLE_FILL_ALPHA)
                if mode == "idle" and stable_start_t is not None and hands_open_ok:
                    charge_progress = clamp((now - stable_start_t) / max(stable_required_sec, 1e-6), 0.0, 1.0)
                    draw_palm_ui_to_overlay(
                        ov_bgr, ov_a, circles_frame,
                        fill_color_bgr=IDLE_COLOR,
                        fill_alpha=0.0,
                        ring_progress=charge_progress,
                        ring_alpha=IDLE_RING_ALPHA,
                        ring_thickness=RING_THICKNESS
                    )
            elif mode == "active":
                draw_palm_ui_to_overlay(ov_bgr, ov_a, circles_frame, ACTIVE_COLOR, ACTIVE_FILL_ALPHA)
            # ui_ms not tracked separately now; keep remap/comp/render as primary
            # STATS.add("ui_ms", now_ms() - t)

            # 7) Remap (warp) + flip
            t = now_ms()
            warped = warp_frame(frame, map_x, map_y, border=cv2.BORDER_REPLICATE)
            ov_bgr_warp = warp_frame(ov_bgr, map_x, map_y, border=cv2.BORDER_REPLICATE)
            ov_a_warp = cv2.remap(ov_a, map_x, map_y,
                                  interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
            STATS.add("remap_ms", now_ms() - t)

            mirrored = cv2.flip(warped, 1)
            ov_bgr_warp = cv2.flip(ov_bgr_warp, 1)
            ov_a_warp = cv2.flip(ov_a_warp, 1)

            # 8) Segmentation decimation
            t = now_ms()
            if frame_idx % SEG_EVERY == 0:
                rgb_for_seg = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
                seg_results = segmenter.process(rgb_for_seg)
                last_seg_mask = seg_results.segmentation_mask
            seg_mask = last_seg_mask
            STATS.add("seg_ms", now_ms() - t)

            if seg_mask is not None:
                STATS.add("fg_ratio", float((seg_mask >= 0.5).mean()))
            else:
                STATS.add("fg_ratio", 0.0)

            # 9) Composite + overlay blend + (optional) upscale
            t = now_ms()
            if seg_mask is None:
                # fallback: if seg not ready yet, just show mirrored
                prefinal = mirrored
            else:
                prefinal = composite_person_over_bg_fast(mirrored, seg_mask, bg_bgr=bg, thresh=0.5, feather_px=5)

            alpha_blend_inplace(prefinal, ov_bgr_warp, ov_a_warp)

            if DISPLAY_AT_PROC_RES:
                final_frame = prefinal
            else:
                final_frame = cv2.resize(prefinal, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)

            STATS.add("comp_ms", now_ms() - t)

            # 10) Banner + render
            t = now_ms()
            banner_y = int(0.10 * final_frame.shape[0])
            draw_banner(final_frame, banner, y=banner_y)
            cv2.imshow(WINDOW_NAME, final_frame)
            key = cv2.waitKey(1) & 0xFF
            STATS.add("render_ms", now_ms() - t)

            # 11) Whole-frame timing
            STATS.add("frame_ms", now_ms() - t0)

            # Periodic report
            global _last_report_t
            if time.perf_counter() - _last_report_t >= _report_every_sec:
                total = sum(MODE_COUNTS.values()) or 1
                idle_pct = MODE_COUNTS["idle"] / total
                ready_pct = MODE_COUNTS["ready"] / total
                active_pct = MODE_COUNTS["active"] / total
                print(f"mode% idle={idle_pct:.2f} ready={ready_pct:.2f} active={active_pct:.2f} transitions={dict(MODE_TRANSITIONS)}")
                report_stats()
                _last_report_t = time.perf_counter()

            # Keys
            if key == ord('q'):
                break
            if key == ord('r'):
                uGain_live = 0.50
                prev_gain_smoothed = None
                mode = "idle"
                stable_start_t = None
                stop_start_t = None
                banner = start_banner("HOLD YOUR HANDS STILL", duration_sec=1.6)

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()