import os
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import random
from scipy.spatial import Delaunay

mp_pose = mp.solutions.pose

import Triangle_Mesh_helpers as TMh
import Pose_Tracking_helpers as PTh

# Gesture tracker objects — created once, live for the whole session
_peace_detector = PTh.PeaceSignHoldDetector()
_pinch_tracker  = PTh.PinchGestureTracker()
import Loop_helpers as Lh
import Display_helpres as Dh

ROTATE_DEG = 270
ROTATE_DIR = "ccw"

OUTPUT_W = 1080
OUTPUT_H = 1920

PRIMARY_MONITOR_WIDTH = 1920

SEG_EVERY_N = 2

# Welcome screen duration (seconds). Set to 0 to skip.
WELCOME_SCREEN_DURATION = 6.0

# Path to the welcome background image (relative to script or cwd).
WELCOME_BG_PATH = "welcome_background.png"

# Animated gradient background
# When True the captured/loaded bg_plate is replaced each frame by a
# moving diagonal gradient.  Cheap: ~1-2 ms per frame (vectorised numpy).
USE_ANIMATED_BACKGROUND = False

# Two BGR colours the gradient moves between.
ANIM_BG_COLOR_A = (80,  10,  30)   # BGR — left/top colour
ANIM_BG_COLOR_B = (20, 100, 180)   # BGR — right/bottom colour

# Speed: higher = faster sweep. 0.3 ≈ one full cycle every ~3 s at 30 fps.
ANIM_BG_SPEED = 0.3

# Set to True to capture a clean background plate before initialization.
CAPTURE_BACKGROUND_BEFORE_INIT = False

# Set by the selection screen — holds the name of the chosen image
# (e.g. 'MBS_2' or 'FBS_3').  Empty string means nothing was chosen.
current_beauty_standard = ""

POSE_IDS = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

TRACKED_POSE_NAMES = list(POSE_IDS.keys())

SEGMENT_NAMES = [
    "torso",
    "head",
    "left_upper_arm",
    "left_lower_arm",
    "left_palm",
    "right_upper_arm",
    "right_lower_arm",
    "right_palm",
    "left_thigh",
    "left_calf",
    "right_thigh",
    "right_calf",
]

SEGMENT_INDEX = {name: i for i, name in enumerate(SEGMENT_NAMES)}

REQUIRED_INIT_LANDMARKS = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

RENDER_GROUP_NAMES = [
    "torso",
    "head",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
]

RENDER_GROUP_INDEX = {name: i for i, name in enumerate(RENDER_GROUP_NAMES)}

RENDER_GROUP_COLORS = {
    "torso": (255, 220, 0),
    "head": (255, 0, 255),
    "left_arm": (0, 255, 0),
    "right_arm": (0, 180, 255),
    "left_leg": (255, 0, 0),
    "right_leg": (0, 0, 255),
}

SHOW_RENDER_GROUP_DEBUG = False
RENDER_GROUP_DEBUG_ALPHA = 0.28

SHOW_MESH_OUTLINE = False
SHOW_LAYERED_RENDER = False

FIXED_RENDER_ORDER = [
    "left_leg",
    "right_leg",
    "torso",
    "head",
    "left_arm",
    "right_arm",
]

LAYER_MASK_DILATE_KSIZE = 0
LAYER_MASK_BLUR_KSIZE = 0

USE_SOFT_LAYER_MASKS = False
USE_FLOAT_ALPHA_COMPOSITING = False
USE_DYNAMIC_YAW_RENDER_ORDER = True

YAW_ENTER_THRESHOLD = 0.22
YAW_EXIT_THRESHOLD = 0.12
YAW_SIGN_SMOOTH_ALPHA = 0.30
YAW_SIGN_DEADBAND = 0.08

FRONTAL_RENDER_ORDER = [
    "left_leg", "right_leg", "torso", "head", "left_arm", "right_arm",
]
LEFT_SIDE_FRONT_RENDER_ORDER = [
    "right_leg", "left_leg", "torso", "head", "right_arm", "left_arm",
]
RIGHT_SIDE_FRONT_RENDER_ORDER = [
    "left_leg", "right_leg", "torso", "head", "left_arm", "right_arm",
]

# ── Gesture / interaction constants ──────────────────────────────────────
BRUSH_RADIUS_MIN = 25
BRUSH_RADIUS_MAX = 300

# Finish button geometry (top-right corner of OUTPUT canvas)
_RESTART_MARGIN  = 30    # px from top and right edges
_RESTART_W       = 220   # button width  (px, output coords)
_RESTART_H       = 90    # button height (px, output coords)
_RESTART_CORNER  = 18    # rounded corner radius
_RESTART_OUTLINE = 5     # outline thickness
_RESTART_FRAMES  = 50    # frames to dwell before triggering finish

_MODE_BTN_W = 220
_MODE_BTN_H = 90
_MODE_BTN_MARGIN = 30
_MODE_BTN_GAP = 20

# ── Beauty-standard image directory ──────────────────────────────────────
_BS_IMAGE_DIR = "beauty_standard_images"

# ── Initialization pose image ─────────────────────────────────────────────
_INIT_POSE_PATH = os.path.join("UI_gestures", "initialization_pose.png")
_INIT_POSE_ALPHA = 0.35   # same opacity as BS overlay

SESSION_DURATION_SECONDS = 180.0   # 3 minutes per session
TIMEOUT_MESSAGE_DURATION = 5.0     # seconds to show the message

_BS_ALL_NAMES = {
    "MBS": ["MBS_1", "MBS_2", "MBS_3"],
    "FBS": ["FBS_1", "FBS_2", "FBS_3"],
}


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE BODY MESH
# ══════════════════════════════════════════════════════════════════════════════

def build_adaptive_body_mesh(
    w, h,
    body_mask,
    interior_step=60,
    contour_step=8,
    contour_inset=3,
    min_contour_pts=80,
):
    H, W = h, w
    mask_u8 = (body_mask >= 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contour_pts = []
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        pts_raw = main_contour[:, 0, :].astype(np.float32)

        n_contour = len(pts_raw)
        step_px = max(1, min(contour_step, n_contour // max(min_contour_pts, 1)))
        sampled = pts_raw[::step_px]

        if contour_inset > 0 and len(sampled) >= 3:
            cx = float(np.mean(sampled[:, 0]))
            cy = float(np.mean(sampled[:, 1]))
            d = sampled - np.array([cx, cy], dtype=np.float32)
            norms = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
            sampled = sampled - contour_inset * (d / norms)
            sampled[:, 0] = np.clip(sampled[:, 0], 0, W - 1)
            sampled[:, 1] = np.clip(sampled[:, 1], 0, H - 1)

        contour_pts = sampled.tolist()

    interior_pts = []
    for y in range(0, H, interior_step):
        for x in range(0, W, interior_step):
            if mask_u8[y, x] > 0:
                interior_pts.append([float(x), float(y)])

    frame_pts = []
    for x in range(0, W + 1, interior_step):
        xc = float(min(x, W - 1))
        frame_pts.append([xc, 0.0])
        frame_pts.append([xc, float(H - 1)])
    for y in range(interior_step, H, interior_step):
        frame_pts.append([0.0, float(y)])
        frame_pts.append([float(W - 1), float(y)])
    frame_pts += [[0.0, 0.0], [float(W-1), 0.0],
                  [0.0, float(H-1)], [float(W-1), float(H-1)]]

    all_pts = np.array(contour_pts + interior_pts + frame_pts, dtype=np.float32)

    if len(all_pts) < 3:
        print("build_adaptive_body_mesh: not enough points, falling back to grid")
        V_fb, T_fb, _, _ = TMh.build_grid_mesh(w, h, step=interior_step)
        active_fb = np.ones(len(T_fb), dtype=bool)
        return V_fb.astype(np.float32), T_fb, active_fb

    rounded = np.round(all_pts / 2.0).astype(np.int32)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    all_pts = all_pts[unique_idx]

    if len(all_pts) < 3:
        V_fb, T_fb, _, _ = TMh.build_grid_mesh(w, h, step=interior_step)
        active_fb = np.ones(len(T_fb), dtype=bool)
        return V_fb.astype(np.float32), T_fb, active_fb

    tri = Delaunay(all_pts)
    T   = tri.simplices.astype(np.int32)
    V   = all_pts.astype(np.float32)

    centroids = (V[T[:, 0]] + V[T[:, 1]] + V[T[:, 2]]) / 3.0
    cx = np.clip(np.round(centroids[:, 0]).astype(np.int32), 0, W - 1)
    cy = np.clip(np.round(centroids[:, 1]).astype(np.int32), 0, H - 1)
    active = mask_u8[cy, cx] > 0

    print(f"build_adaptive_body_mesh: {len(V)} vertices, {len(T)} triangles, "
          f"{int(active.sum())} active ({len(contour_pts)} contour pts, "
          f"{len(interior_pts)} interior pts)")

    return V, T, active

def _pick_new_beauty_standard(current):
    if not current:
        return current
    prefix = current[:3].upper()   # "MBS" or "FBS"
    candidates = _BS_ALL_NAMES.get(prefix, [])
    alternatives = [c for c in candidates if c != current]
    if not alternatives:
        return current
    return random.choice(alternatives)

def _draw_timeout_message(canvas, new_bs_name):
    from PIL import Image as _PI, ImageDraw as _PD, ImageFont as _PF
    BOLD_PATHS = ["C:/Windows/Fonts/segoeuisb.ttf","C:/Windows/Fonts/segoeuib.ttf",
                  "C:/Windows/Fonts/calibrib.ttf","C:/Windows/Fonts/arialbd.ttf","C:/Windows/Fonts/arial.ttf"]
    REG_PATHS  = ["C:/Windows/Fonts/segoeui.ttf","C:/Windows/Fonts/calibri.ttf","C:/Windows/Fonts/arial.ttf"]
    W, H = OUTPUT_W, OUTPUT_H
    usable_w = W - 120

    def _first(paths):
        for p in paths:
            if os.path.exists(p): return p
        return None

    def _fit(text, path, start=60, mn=18):
        if path is None: return _PF.load_default()
        tmp = _PD.Draw(_PI.new("RGB", (1, 1)))
        for sz in range(start, mn - 1, -1):
            try:
                f = _PF.truetype(path, sz)
                bb = tmp.textbbox((0, 0), text, font=f)
                if (bb[2] - bb[0]) <= usable_w: return f
            except Exception: continue
        return _PF.load_default()

    line1 = "Ooops, we're sorry but you ran out of time."
    line2 = "The body shape you were trying to match is no longer deemed beautiful."
    line3 = f"The current beauty standard is:"

    font1 = _fit(line1, _first(BOLD_PATHS), 60)
    font2 = _fit(line2, _first(REG_PATHS),  52)
    font3 = _fit(line3, _first(BOLD_PATHS), 56)

    img  = _PI.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = _PD.Draw(img)

    def _measure(text, font):
        try:
            bb = draw.textbbox((0, 0), text, font=font)
            return bb[2]-bb[0], bb[3]-bb[1]
        except AttributeError:
            return draw.textsize(text, font=font)

    w1,h1 = _measure(line1, font1)
    w2,h2 = _measure(line2, font2)
    w3,h3 = _measure(line3, font3)
    gap = int(H * 0.022)
    total_h = h1 + gap + h2 + gap + h3
    y = (H - total_h) // 2

    def _draw_line(text, font, y, color=(255,255,255)):
        w_, h_ = _measure(text, font)
        x = (W - w_) // 2
        draw.text((x+2, y+2), text, font=font, fill=(0,0,0))
        draw.text((x,   y  ), text, font=font, fill=color)
        return h_

    h_ = _draw_line(line1, font1, y, (255,255,255));  y += h_ + gap
    h_ = _draw_line(line2, font2, y, (220,220,220));  y += h_ + gap
    _draw_line(line3, font3, y, (220,200,255))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND INFERENCE THREAD
# ══════════════════════════════════════════════════════════════════════════════

class PoseInferenceThread:
    def __init__(self, pose, hands, segmenter, feather=0, thresh=0.5, w=640, h=480):
        self.pose      = pose
        self.hands     = hands
        self.segmenter = segmenter
        self.feather   = feather
        self.thresh    = thresh
        self.w         = w
        self.h         = h

        self._lock        = threading.Lock()
        self._result      = None
        self._frame       = None
        self._frame_ready = threading.Event()
        self._stop        = threading.Event()
        self._thread      = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_frame(self, frame_bgr):
        with self._lock:
            self._frame = frame_bgr.copy()
        self._frame_ready.set()

    def latest_result(self):
        with self._lock:
            return self._result

    def stop(self):
        self._stop.set()
        self._frame_ready.set()

    def _run(self):
        while not self._stop.is_set():
            self._frame_ready.wait()
            self._frame_ready.clear()
            if self._stop.is_set():
                break
            with self._lock:
                frame = self._frame
            if frame is None:
                continue
            seg_mask, rgb = PTh.get_segmentation_mask(
                frame, self.segmenter, feather=self.feather
            )
            cur_pts    = extract_pose_points(rgb, self.pose, self.w, self.h, min_vis=0.45)
            hand_state = PTh.get_hand_state(rgb, self.hands, seg_mask, self.thresh, self.w, self.h)
            with self._lock:
                self._result = (seg_mask, rgb, cur_pts, hand_state)


# ══════════════════════════════════════════════════════════════════════════════
# VECTORIZED MESH RECONSTRUCTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_frame_arrays(frames, seg_ids, num_vertices):
    origins = np.zeros((num_vertices, 2), dtype=np.float32)
    xhats   = np.zeros((num_vertices, 2), dtype=np.float32)
    yhats   = np.zeros((num_vertices, 2), dtype=np.float32)
    lengths = np.zeros(num_vertices,      dtype=np.float32)
    for i in range(num_vertices):
        si = seg_ids[i]
        if si < 0:
            continue
        f = frames[SEGMENT_NAMES[si]]
        origins[i] = f["origin"]
        xhats[i]   = f["xhat"]
        yhats[i]   = f["yhat"]
        lengths[i] = f["length"]
    return origins, xhats, yhats, lengths


def reconstruct_tracked_mesh_vectorized(binding, frame_arrays):
    origins, xhats, yhats, lengths = frame_arrays
    rest_uv = binding["vertex_local_uv_rest"]
    seg_ids = binding["vertex_segment"]
    u = rest_uv[:, 0:1]
    v = rest_uv[:, 1:2]
    V = origins + u * lengths[:, None] * xhats \
                + v * lengths[:, None] * yhats
    V[seg_ids < 0] = 0.0
    return V.astype(np.float32)


def reconstruct_deformed_mesh_vectorized(binding, frame_arrays):
    origins, xhats, yhats, lengths = frame_arrays
    rest_uv = binding["vertex_local_uv_rest"]
    off_uv  = binding["vertex_local_uv_offset"]
    seg_ids = binding["vertex_segment"]
    uv = rest_uv + off_uv
    u  = uv[:, 0:1]
    v  = uv[:, 1:2]
    V = origins + u * lengths[:, None] * xhats \
                + v * lengths[:, None] * yhats
    V[seg_ids < 0] = 0.0
    return V.astype(np.float32)


def update_local_offsets_vectorized(binding, frame_arrays, V_new):
    origins, xhats, yhats, lengths = frame_arrays
    seg_ids = binding["vertex_segment"]
    rest_uv = binding["vertex_local_uv_rest"]
    valid   = seg_ids >= 0
    safe_len = np.where(lengths > 1e-6, lengths, 1.0)
    d = V_new - origins
    u = np.sum(d * xhats, axis=1) / safe_len
    v = np.sum(d * yhats, axis=1) / safe_len
    uv_now = np.stack([u, v], axis=1)
    binding["vertex_local_uv_offset"][valid] = (uv_now - rest_uv)[valid]


def frames_moved_enough(prev_frames, new_frames, threshold_px=1.5):
    if prev_frames is None:
        return True
    for seg in SEGMENT_NAMES:
        if seg not in prev_frames or seg not in new_frames:
            return True
        if np.linalg.norm(new_frames[seg]["origin"] - prev_frames[seg]["origin"]) > threshold_px:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# MISC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

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
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos); new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2.0) - cx; M[1, 2] += (new_h / 2.0) - cy
    return cv2.warpAffine(frame_bgr, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def capture_background_plate(cap, h, w, n_frames=20, window_name="Background capture"):
    acc = None
    got = 0
    for k in range(n_frames):
        ok, fr = cap.read()
        if not ok:
            continue
        fr = rotate_frame(fr)
        if fr.shape[:2] != (h, w):
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_LINEAR)
        fr = cv2.flip(fr, 1)
        acc = fr.astype(np.float32) if acc is None else acc + fr.astype(np.float32)
        got += 1
        vis_display = cv2.resize(fr, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)
        # ── CHANGE 1: no frame counter in countdown text ──────────────────
        vis_display = _draw_countdown_text(
            vis_display,
            "Capturing background...",
            OUTPUT_W, OUTPUT_H)
        cv2.imshow(window_name, vis_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if acc is None or got == 0:
        return None
    return np.clip(acc / float(got), 0, 255).astype(np.uint8)


def _draw_countdown_text(frame_bgr, text, out_w, out_h):
    """
    Render countdown / status text onto frame_bgr using PIL.
    Returns a new BGR image.
    """
    from PIL import Image as _PILImg, ImageDraw as _PILDraw, ImageFont as _PILFont
    BOLD_PATHS = [
        "C:/Windows/Fonts/segoeuisb.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    usable_w = out_w - 120
    font = None
    for fp in BOLD_PATHS:
        if not os.path.exists(fp):
            continue
        for sz in range(72, 18, -1):
            try:
                f  = _PILFont.truetype(fp, sz)
                _d = _PILDraw.Draw(_PILImg.new("RGB", (1, 1)))
                bb = _d.textbbox((0, 0), text, font=f)
                if (bb[2] - bb[0]) <= usable_w:
                    font = f
                    break
            except Exception:
                continue
        if font:
            break
    if font is None:
        font = _PILFont.load_default()

    img  = _PILImg.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = _PILDraw.Draw(img)
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except AttributeError:
        tw, th = draw.textsize(text, font=font)
    x = (out_w - tw) // 2
    y = (out_h - th) // 2
    draw.text((x + 3, y + 3), text, font=font, fill=(0, 0, 0))
    draw.text((x,     y    ), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def draw_pinch_hint_under_button(vis_display, pinch_icon, button_x0, button_y1, button_w):
    hint_text = "Pinch to click"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(hint_text, font, font_scale, thickness)

    gap = 10
    icon_gap = 8

    icon_w = pinch_icon.shape[1] if pinch_icon is not None else 0
    icon_h = pinch_icon.shape[0] if pinch_icon is not None else 0

    total_w = icon_w + icon_gap + tw

    group_x = button_x0 + (button_w - total_w) // 2
    group_y = button_y1 + gap

    icon_x = group_x
    icon_y = group_y

    text_x = group_x + icon_w + icon_gap
    text_y = group_y + (icon_h + th) // 2

    if pinch_icon is not None:
        vis_display = overlay_bgra(vis_display, pinch_icon, icon_x, icon_y)

    cv2.putText(
        vis_display,
        hint_text,
        (text_x + 2, text_y + 2),
        font,
        font_scale,
        (0, 0, 0),
        thickness + 1,
        cv2.LINE_AA
    )

    cv2.putText(
        vis_display,
        hint_text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    return vis_display


def _draw_two_line_countdown_text(frame_bgr, line1, line2, out_w, out_h):
    """
    Render two lines of text centred on frame_bgr using PIL.
    line1 is drawn above line2 with a small gap.
    Returns a new BGR image.
    """
    from PIL import Image as _PILImg, ImageDraw as _PILDraw, ImageFont as _PILFont
    BOLD_PATHS = [
        "C:/Windows/Fonts/segoeuisb.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    usable_w = out_w - 120

    def _pick_font(text):
        for fp in BOLD_PATHS:
            if not os.path.exists(fp):
                continue
            for sz in range(72, 18, -1):
                try:
                    f  = _PILFont.truetype(fp, sz)
                    _d = _PILDraw.Draw(_PILImg.new("RGB", (1, 1)))
                    bb = _d.textbbox((0, 0), text, font=f)
                    if (bb[2] - bb[0]) <= usable_w:
                        return f
                except Exception:
                    continue
        return _PILFont.load_default()

    font1 = _pick_font(line1)
    font2 = _pick_font(line2)

    img  = _PILImg.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = _PILDraw.Draw(img)

    def _measure(text, font):
        try:
            bb = draw.textbbox((0, 0), text, font=font)
            return bb[2] - bb[0], bb[3] - bb[1]
        except AttributeError:
            return draw.textsize(text, font=font)

    tw1, th1 = _measure(line1, font1)
    tw2, th2 = _measure(line2, font2)
    gap = int(out_h * 0.015)
    total_h = th1 + gap + th2

    y1 = (out_h - total_h) // 2
    y2 = y1 + th1 + gap

    x1 = (out_w - tw1) // 2
    x2 = (out_w - tw2) // 2

    # line1 — instruction text, slightly smaller / lighter colour
    draw.text((x1 + 3, y1 + 3), line1, font=font1, fill=(0, 0, 0))
    draw.text((x1,     y1    ), line1, font=font1, fill=(220, 200, 255))

    # line2 — countdown text, white
    draw.text((x2 + 3, y2 + 3), line2, font=font2, fill=(0, 0, 0))
    draw.text((x2,     y2    ), line2, font=font2, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _composite_init_pose_overlay(canvas, init_pose_img, alpha=_INIT_POSE_ALPHA):
    """
    Composite init_pose_img onto canvas using the same logic as the BS overlay:
    horizontally centred, bottom-aligned with a 60px margin, at `alpha` opacity.
    init_pose_img may be 3-channel BGR or 4-channel BGRA.
    """
    if init_pose_img is None:
        return canvas
    oh_, ow = init_pose_img.shape[:2]
    _BS_BOTTOM_MARGIN = 60
    ox = (OUTPUT_W - ow) // 2
    oy = OUTPUT_H - oh_ - _BS_BOTTOM_MARGIN

    x0 = max(ox, 0);              y0 = max(oy, 0)
    x1 = min(ox + ow, OUTPUT_W);  y1 = min(oy + oh_, OUTPUT_H)
    sx0 = x0 - ox;  sy0 = y0 - oy
    sx1 = sx0 + (x1 - x0);        sy1 = sy0 + (y1 - y0)

    if x1 <= x0 or y1 <= y0:
        return canvas

    roi   = canvas[y0:y1, x0:x1]
    patch = init_pose_img[sy0:sy1, sx0:sx1]

    if patch.shape[2] == 4:
        a   = patch[:, :, 3:4].astype(np.float32) / 255.0
        a  *= alpha
        bgr = patch[:, :, :3].astype(np.float32)
    else:
        a   = np.full((y1 - y0, x1 - x0, 1), alpha, dtype=np.float32)
        bgr = patch.astype(np.float32)

    canvas[y0:y1, x0:x1] = np.clip(
        a * bgr + (1.0 - a) * roi.astype(np.float32), 0, 255
    ).astype(np.uint8)
    return canvas


def show_countdown(cap, h, w, seconds, message, window_name,
                   bs_img=None, bs_size=120, bs_margin=18,
                   overlay_timer_start=None, overlay_timer_duration=180.0):
    start_time = time.time()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = rotate_frame(fr)
        if fr.shape[:2] != (h, w):
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_LINEAR)
        fr = cv2.flip(fr, 1)
        elapsed   = time.time() - start_time
        remaining = int(np.ceil(seconds - elapsed))
        if remaining <= 0:
            break
        # ── CHANGE 1: no frame counter ────────────────────────────────────
        text        = f"{message} in {remaining}"
        vis_display = cv2.resize(fr, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)
        vis_display = _composite_init_pose_overlay(vis_display, bs_img, _INIT_POSE_ALPHA)
        vis_display = _draw_countdown_text(vis_display, text, OUTPUT_W, OUTPUT_H)
        cv2.imshow(window_name, vis_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def build_active_mesh_mask_fast(h, w, V_def, T, active):
    return rasterize_triangle_mask_from_indices(h, w, V_def, T, np.flatnonzero(active).astype(np.int32))


def composite_mesh_with_background_holefill(frame_raw, bg_plate, warped_mesh_bgr,
                                             mesh_mask_u8, seg_mask, body_thresh=0.50,
                                             dilate_ksize=7, blur_ksize=5, apply_mesh=True):
    out = frame_raw.copy()
    body_mask = (seg_mask >= body_thresh).astype(np.uint8) * 255
    if dilate_ksize is not None and dilate_ksize > 1:
        k = int(dilate_ksize); k = k if k % 2 == 1 else k + 1
        body_mask = cv2.dilate(body_mask, np.ones((k, k), np.uint8), iterations=1)
    hole_mask = cv2.bitwise_and(body_mask, cv2.bitwise_not(mesh_mask_u8))
    if blur_ksize is not None and blur_ksize > 1:
        k = int(blur_ksize); k = k if k % 2 == 1 else k + 1
        hole_mask = cv2.GaussianBlur(hole_mask, (k, k), 0)
    if hole_mask.ndim == 2:
        out = alpha_composite_bgr(out, bg_plate, hole_mask.astype(np.float32) / 255.0)
    if apply_mesh:
        out = composite_bgr_hard_mask(out, warped_mesh_bgr, mesh_mask_u8)
    return out, hole_mask


def filter_selection_disallow_same_side_arm(selection_vertices, selection_triangles,
                                             T, binding, handedness_label):
    if handedness_label not in ("Left", "Right"):
        return selection_vertices, selection_triangles
    seg_ids = binding["vertex_segment"]
    if handedness_label == "Left":
        blocked_seg_ids = {SEGMENT_INDEX["left_upper_arm"],
                           SEGMENT_INDEX["left_lower_arm"], SEGMENT_INDEX["left_palm"]}
    else:
        blocked_seg_ids = {SEGMENT_INDEX["right_upper_arm"],
                           SEGMENT_INDEX["right_lower_arm"], SEGMENT_INDEX["right_palm"]}
    blocked_vertices = np.isin(seg_ids, list(blocked_seg_ids))
    tri_touches_blocked = np.any(blocked_vertices[T], axis=1)
    filtered_triangles = selection_triangles & (~tri_touches_blocked)
    filtered_vertices = np.zeros_like(selection_vertices)
    if np.any(filtered_triangles):
        filtered_vertices[np.unique(T[filtered_triangles].ravel())] = True
    return filtered_vertices, filtered_triangles


def compute_arm_front_score(cur_pts, arm_name, yaw_state_txt):
    ls = cur_pts.get("left_shoulder"); rs = cur_pts.get("right_shoulder")
    lh = cur_pts.get("left_hip");     rh = cur_pts.get("right_hip")
    if any(p is None for p in [ls, rs, lh, rh]):
        return 0.0
    torso_mid_x = 0.25 * (ls[0] + rs[0] + lh[0] + rh[0])
    e = cur_pts.get("left_elbow" if arm_name == "left_arm" else "right_elbow")
    w = cur_pts.get("left_wrist" if arm_name == "left_arm" else "right_wrist")
    if e is None or w is None:
        return 0.0
    if yaw_state_txt == "left side front":
        side_bias = 10.0 if arm_name == "left_arm" else -10.0
    elif yaw_state_txt == "right side front":
        side_bias = 10.0 if arm_name == "right_arm" else -10.0
    else:
        side_bias = 0.0
    center_cross = (-abs(e[0] - torso_mid_x) - abs(w[0] - torso_mid_x)) * 0.15
    return float(side_bias + center_cross)


def override_arm_order(base_order, left_arm_front=None, right_arm_front=None):
    leg_block = [x for x in base_order if x in ("left_leg", "right_leg")]
    arm_block  = [x for x in base_order if x in ("left_arm", "right_arm")]
    back_arms, front_arms = [], []
    for arm_name in arm_block:
        flag = left_arm_front if arm_name == "left_arm" else right_arm_front
        (back_arms if flag is False else front_arms).append(arm_name)
    return leg_block + back_arms + ["torso", "head"] + front_arms


def compute_render_order_with_arm_override(base_order, layers, stable_pts, yaw_state_txt):
    if layers is None:
        return base_order, 0.0, 0.0, 0.0, 0.0, False
    torso_mask     = layers["torso"]["mask_u8"]
    left_arm_mask  = layers["left_arm"]["mask_u8"]
    right_arm_mask = layers["right_arm"]["mask_u8"]
    left_ov  = compute_mask_overlap_fraction(left_arm_mask, torso_mask)
    right_ov = compute_mask_overlap_fraction(right_arm_mask, torso_mask)
    ls, rs, lf, rf, used = 0.0, 0.0, None, None, False
    if left_ov >= ARM_OVERLAP_TRIGGER_FRAC:
        ls = compute_arm_front_score(stable_pts, "left_arm", yaw_state_txt); used = True
        if ls > ARM_FRONT_SCORE_DEADBAND: lf = True
        elif ls < -ARM_FRONT_SCORE_DEADBAND: lf = False
    if right_ov >= ARM_OVERLAP_TRIGGER_FRAC:
        rs = compute_arm_front_score(stable_pts, "right_arm", yaw_state_txt); used = True
        if rs > ARM_FRONT_SCORE_DEADBAND: rf = True
        elif rs < -ARM_FRONT_SCORE_DEADBAND: rf = False
    return override_arm_order(base_order, lf, rf), left_ov, right_ov, ls, rs, used


def composite_prebuilt_layers(base_bgr, layers, render_order):
    out = base_bgr.copy()
    for group_name in render_order:
        layer = layers[group_name]
        if USE_FLOAT_ALPHA_COMPOSITING:
            out = alpha_composite_bgr(out, layer["color"],
                                      layer["mask_u8"].astype(np.float32) / 255.0)
        else:
            out = composite_bgr_hard_mask(out, layer["color"], layer["mask_u8"])
    return out


def compute_mask_overlap_fraction(mask_a_u8, mask_b_u8):
    a = mask_a_u8 > 0; b = mask_b_u8 > 0
    area_a = int(np.count_nonzero(a)); area_b = int(np.count_nonzero(b))
    if area_a == 0 or area_b == 0:
        return 0.0
    return int(np.count_nonzero(a & b)) / float(max(1, min(area_a, area_b)))


def compute_leg_front_score(cur_pts, yaw_state_txt):
    lk = cur_pts.get("left_knee");  rk = cur_pts.get("right_knee")
    la = cur_pts.get("left_ankle"); ra = cur_pts.get("right_ankle")
    lh = cur_pts.get("left_hip");   rh = cur_pts.get("right_hip")
    if any(p is None for p in [lk, rk, la, ra, lh, rh]):
        return 0.0
    hip_mid_x = 0.5 * (lh[0] + rh[0])
    if yaw_state_txt == "left side front":
        score = ((hip_mid_x - lk[0]) + (hip_mid_x - la[0])) - \
                ((hip_mid_x - rk[0]) + (hip_mid_x - ra[0]))
    elif yaw_state_txt == "right side front":
        score = ((lk[0] - hip_mid_x) + (la[0] - hip_mid_x)) - \
                ((rk[0] - hip_mid_x) + (ra[0] - hip_mid_x))
    else:
        score = (abs(lk[0] - hip_mid_x) + abs(la[0] - hip_mid_x)) - \
                (abs(rk[0] - hip_mid_x) + abs(ra[0] - hip_mid_x))
    return float(score)


def override_leg_order(base_order, left_leg_front):
    order = [x for x in base_order if x not in ("left_leg", "right_leg")]
    return (["right_leg", "left_leg"] if left_leg_front else ["left_leg", "right_leg"]) + order


def compute_render_order_with_leg_override(base_order, layers, stable_pts, yaw_state_txt):
    if layers is None:
        return base_order, 0.0, 0.0, False
    ov = compute_mask_overlap_fraction(layers["left_leg"]["mask_u8"], layers["right_leg"]["mask_u8"])
    if ov < LEG_OVERLAP_TRIGGER_FRAC:
        return base_order, ov, 0.0, False
    score = compute_leg_front_score(stable_pts, yaw_state_txt)
    if score > LEG_FRONT_SCORE_DEADBAND:
        return override_leg_order(base_order, True),  ov, score, True
    elif score < -LEG_FRONT_SCORE_DEADBAND:
        return override_leg_order(base_order, False), ov, score, True
    return base_order, ov, score, True


def compute_yaw_sign_components(cur_pts):
    ls = cur_pts.get("left_shoulder"); rs = cur_pts.get("right_shoulder")
    lh = cur_pts.get("left_hip");     rh = cur_pts.get("right_hip")
    nose = cur_pts.get("nose")
    if ls is None or rs is None or lh is None or rh is None:
        return 0.0, {"nose_term": 0.0, "torso_term": 0.0, "shoulder_term": 0.0}
    shoulder_mid = 0.5 * (ls + rs); hip_mid = 0.5 * (lh + rh)
    torso_mid = 0.5 * (shoulder_mid + hip_mid)
    shoulder_x, shoulder_len = safe_normalize(rs - ls)
    nose_term = 0.0
    if nose is not None and shoulder_len > 1e-6:
        nose_term = float(np.dot(nose - shoulder_mid, shoulder_x) / max(0.35 * shoulder_len, 1.0))
    torso_term    = float((np.dot(lh - torso_mid, shoulder_x) + np.dot(rh - torso_mid, shoulder_x))
                          / max(0.35 * shoulder_len, 1.0))
    shoulder_term = float((np.dot(ls - torso_mid, shoulder_x) + np.dot(rs - torso_mid, shoulder_x))
                          / max(0.35 * shoulder_len, 1.0))
    sign_value = 1.0 * nose_term + 0.8 * torso_term + 0.5 * shoulder_term
    return float(sign_value), {"nose_term": float(nose_term),
                                "torso_term": float(torso_term),
                                "shoulder_term": float(shoulder_term)}


def compute_render_order_from_yaw(cur_pts, ref_metrics, interaction_state):
    yaw_amount, _, yaw_debug = estimate_body_yaw(cur_pts, ref_metrics)
    prev_s = float(interaction_state.get("yaw_sign_value_smooth", 0.0))
    sign_value_smooth = (1.0 - YAW_SIGN_SMOOTH_ALPHA) * prev_s + \
                         YAW_SIGN_SMOOTH_ALPHA * float(yaw_debug["sign_value"])
    interaction_state["yaw_sign_value_smooth"] = float(sign_value_smooth)
    if sign_value_smooth > YAW_SIGN_DEADBAND: yaw_sign = 1.0
    elif sign_value_smooth < -YAW_SIGN_DEADBAND: yaw_sign = -1.0
    else: yaw_sign = 0.0
    prev_state = interaction_state.get("yaw_side_state", "frontal")
    state = prev_state
    if prev_state == "frontal":
        if yaw_amount >= YAW_ENTER_THRESHOLD:
            state = "left_front" if yaw_sign > 0 else ("right_front" if yaw_sign < 0 else "frontal")
    elif prev_state == "left_front":
        if yaw_amount <= YAW_EXIT_THRESHOLD: state = "frontal"
        elif yaw_sign < 0 and yaw_amount >= YAW_ENTER_THRESHOLD: state = "right_front"
    elif prev_state == "right_front":
        if yaw_amount <= YAW_EXIT_THRESHOLD: state = "frontal"
        elif yaw_sign > 0 and yaw_amount >= YAW_ENTER_THRESHOLD: state = "left_front"
    interaction_state["yaw_side_state"] = state
    if state == "left_front":    render_order = LEFT_SIDE_FRONT_RENDER_ORDER;  yaw_state_txt = "left side front"
    elif state == "right_front": render_order = RIGHT_SIDE_FRONT_RENDER_ORDER; yaw_state_txt = "right side front"
    else:                        render_order = FRONTAL_RENDER_ORDER;           yaw_state_txt = "frontal"
    yaw_debug["sign_value_smooth"] = float(sign_value_smooth)
    return render_order, yaw_amount, yaw_sign, yaw_state_txt, yaw_debug


def build_render_group_triangle_index_cache(binding):
    tri_active = binding["tri_active"]; tri_render_group = binding["tri_render_group"]
    return {name: np.flatnonzero(tri_active & (tri_render_group == RENDER_GROUP_INDEX[name])).astype(np.int32)
            for name in RENDER_GROUP_NAMES}


def rasterize_triangle_mask_from_indices(h, w, V_dst, T, tri_indices):
    mask = np.zeros((h, w), dtype=np.uint8)
    if tri_indices is None or len(tri_indices) == 0:
        return mask
    for k in tri_indices:
        tri = V_dst[T[k]].astype(np.float32)
        if not np.isfinite(tri).all(): continue
        if triangle_area2(tri) < 1.0: continue
        if not _triangle_inside_image(tri, w, h): continue
        cv2.fillConvexPoly(mask, np.round(tri).astype(np.int32), 255, lineType=cv2.LINE_AA)
    return mask


def warp_mesh_piecewise_to_blank_indices(src_img, V_src, V_dst, T, tri_indices,
                                          min_area=1.0, min_bbox=2.0):
    h, w = src_img.shape[:2]
    dst_img = np.zeros_like(src_img)
    if tri_indices is None or len(tri_indices) == 0:
        return dst_img
    for k in tri_indices:
        tri_idx = T[k]
        t_src = V_src[tri_idx].astype(np.float32)
        t_dst = V_dst[tri_idx].astype(np.float32)
        if not np.isfinite(t_src).all() or not np.isfinite(t_dst).all(): continue
        if triangle_area2(t_src) < min_area or triangle_area2(t_dst) < min_area: continue
        src_bw, src_bh = _triangle_bbox_size(t_src)
        dst_bw, dst_bh = _triangle_bbox_size(t_dst)
        if src_bw < min_bbox or src_bh < min_bbox or dst_bw < min_bbox or dst_bh < min_bbox: continue
        if not _triangle_inside_image(t_src, w, h): continue
        try:
            TMh.warp_triangle(src_img, dst_img, t_src, t_dst)
        except cv2.error:
            continue
    return dst_img


def render_group_layer_fast(src_img, V_src, V_dst, T, tri_indices,
                             min_area=4.0, min_bbox=3.0,
                             mask_dilate_ksize=0, mask_blur_ksize=0):
    h, w = src_img.shape[:2]
    if tri_indices is None or len(tri_indices) == 0:
        return np.zeros_like(src_img), np.zeros((h, w), dtype=np.uint8)
    layer_color = warp_mesh_piecewise_to_blank_indices(src_img, V_src, V_dst, T, tri_indices, min_area, min_bbox)
    layer_mask_u8 = rasterize_triangle_mask_from_indices(h, w, V_dst, T, tri_indices)
    if USE_SOFT_LAYER_MASKS:
        layer_mask_u8 = soften_layer_mask(layer_mask_u8, mask_dilate_ksize, mask_blur_ksize)
    return layer_color, layer_mask_u8


def composite_bgr_hard_mask(base_bgr, over_bgr, mask_u8):
    out = base_bgr.copy(); out[mask_u8 > 0] = over_bgr[mask_u8 > 0]; return out


def build_layered_body_layers_fast(frame_raw, V_track, V_def, T, binding,
                                    min_area=4.0, min_bbox=3.0,
                                    mask_dilate_ksize=0, mask_blur_ksize=0):
    layers = {}
    group_tri_indices = binding.get("group_tri_indices") or build_render_group_triangle_index_cache(binding)
    binding["group_tri_indices"] = group_tri_indices
    for group_name in RENDER_GROUP_NAMES:
        color, mask_u8 = render_group_layer_fast(frame_raw, V_track, V_def, T,
                                                  group_tri_indices[group_name],
                                                  min_area, min_bbox,
                                                  mask_dilate_ksize, mask_blur_ksize)
        layers[group_name] = {"color": color, "mask_u8": mask_u8}
    return layers


def render_layered_body_fixed_order_fast(frame_raw, V_track, V_def, T, binding,
                                          render_order=None, min_area=4.0, min_bbox=3.0,
                                          mask_dilate_ksize=0, mask_blur_ksize=0):
    if render_order is None: render_order = FIXED_RENDER_ORDER
    layers = build_layered_body_layers_fast(frame_raw, V_track, V_def, T, binding,
                                             min_area, min_bbox, mask_dilate_ksize, mask_blur_ksize)
    return composite_prebuilt_layers(frame_raw, layers, render_order), layers


def get_tri_mask_for_render_group(binding, group_name):
    gid = RENDER_GROUP_INDEX[group_name]
    return binding["tri_active"] & (binding["tri_render_group"] == gid)


def rasterize_triangle_mask(h, w, V_dst, T, tri_mask):
    mask = np.zeros((h, w), dtype=np.uint8)
    for k, tri_idx in enumerate(T):
        if not tri_mask[k]: continue
        tri = V_dst[tri_idx].astype(np.float32)
        if not np.isfinite(tri).all(): continue
        if triangle_area2(tri) < 1.0: continue
        if not _triangle_inside_image(tri, w, h): continue
        cv2.fillConvexPoly(mask, np.round(tri).astype(np.int32), 255, lineType=cv2.LINE_AA)
    return mask


def soften_layer_mask(mask_u8, dilate_ksize=5, blur_ksize=5):
    out = mask_u8.copy()
    if dilate_ksize is not None and dilate_ksize > 1:
        k = int(dilate_ksize); k = k if k % 2 == 1 else k + 1
        out = cv2.dilate(out, np.ones((k, k), np.uint8), iterations=1)
    if blur_ksize is not None and blur_ksize > 1:
        k = int(blur_ksize); k = k if k % 2 == 1 else k + 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def warp_mesh_piecewise_to_blank(src_img, V_src, V_dst, T, active_mask=None,
                                  min_area=1.0, min_bbox=2.0):
    h, w = src_img.shape[:2]
    dst_img = np.zeros_like(src_img)
    if active_mask is None: active_mask = np.ones(len(T), dtype=bool)
    for k, tri_idx in enumerate(T):
        if not active_mask[k]: continue
        t_src = V_src[tri_idx].astype(np.float32); t_dst = V_dst[tri_idx].astype(np.float32)
        if not np.isfinite(t_src).all() or not np.isfinite(t_dst).all(): continue
        if triangle_area2(t_src) < min_area or triangle_area2(t_dst) < min_area: continue
        src_bw, src_bh = _triangle_bbox_size(t_src); dst_bw, dst_bh = _triangle_bbox_size(t_dst)
        if src_bw < min_bbox or src_bh < min_bbox or dst_bw < min_bbox or dst_bh < min_bbox: continue
        if not _triangle_inside_image(t_src, w, h): continue
        try: TMh.warp_triangle(src_img, dst_img, t_src, t_dst)
        except cv2.error: continue
    return dst_img


def render_group_layer(src_img, V_src, V_dst, T, tri_mask,
                       min_area=4.0, min_bbox=3.0,
                       mask_dilate_ksize=5, mask_blur_ksize=5):
    h, w = src_img.shape[:2]
    if tri_mask is None or not np.any(tri_mask):
        return np.zeros_like(src_img), np.zeros((h, w), dtype=np.float32)
    layer_color = warp_mesh_piecewise_to_blank(src_img, V_src, V_dst, T, tri_mask, min_area, min_bbox)
    layer_mask_u8 = soften_layer_mask(rasterize_triangle_mask(h, w, V_dst, T, tri_mask),
                                       mask_dilate_ksize, mask_blur_ksize)
    return layer_color, layer_mask_u8.astype(np.float32) / 255.0


def alpha_composite_bgr(base_bgr, over_bgr, alpha):
    alpha3 = alpha[:, :, None] if alpha.ndim == 2 else alpha
    return np.clip(alpha3 * over_bgr.astype(np.float32) +
                   (1.0 - alpha3) * base_bgr.astype(np.float32), 0, 255).astype(np.uint8)


def render_layered_body_fixed_order(frame_raw, V_track, V_def, T, binding,
                                     render_order=None, min_area=4.0, min_bbox=3.0,
                                     mask_dilate_ksize=5, mask_blur_ksize=5):
    if render_order is None: render_order = FIXED_RENDER_ORDER
    out = frame_raw.copy(); layers = {}
    for group_name in RENDER_GROUP_NAMES:
        tri_mask = get_tri_mask_for_render_group(binding, group_name)
        color, alpha = render_group_layer(frame_raw, V_track, V_def, T, tri_mask,
                                          min_area, min_bbox, mask_dilate_ksize, mask_blur_ksize)
        layers[group_name] = {"color": color, "alpha": alpha}
    for group_name in render_order:
        out = alpha_composite_bgr(out, layers[group_name]["color"], layers[group_name]["alpha"])
    return out, layers


def fine_segment_to_render_group(seg_idx):
    if seg_idx is None or seg_idx < 0 or seg_idx >= len(SEGMENT_NAMES): return -1
    seg_name = SEGMENT_NAMES[int(seg_idx)]
    if seg_name == "torso": return RENDER_GROUP_INDEX["torso"]
    if seg_name == "head":  return RENDER_GROUP_INDEX["head"]
    if seg_name in ("left_upper_arm",  "left_lower_arm",  "left_palm"):  return RENDER_GROUP_INDEX["left_arm"]
    if seg_name in ("right_upper_arm", "right_lower_arm", "right_palm"): return RENDER_GROUP_INDEX["right_arm"]
    if seg_name in ("left_thigh",  "left_calf"):  return RENDER_GROUP_INDEX["left_leg"]
    if seg_name in ("right_thigh", "right_calf"): return RENDER_GROUP_INDEX["right_leg"]
    return -1


def build_triangle_render_groups(binding, T):
    vertex_segment = binding["vertex_segment"]; tri_active = binding["tri_active"]
    tri_render_group = -np.ones(len(T), dtype=np.int32)
    for k, tri in enumerate(T):
        if not tri_active[k]: continue
        fine_ids = vertex_segment[tri]
        if np.any(fine_ids < 0): continue
        coarse_ids = [fine_segment_to_render_group(int(s)) for s in fine_ids]
        if np.any(np.array(coarse_ids) < 0): continue
        vals, counts = np.unique(np.array(coarse_ids, dtype=np.int32), return_counts=True)
        tri_render_group[k] = int(vals[np.argmax(counts)])
    return tri_render_group


def draw_triangle_render_group_overlay(frame, V, T, tri_active, tri_render_group,
                                        alpha=0.28, line_thickness=1):
    out = frame.copy(); overlay = frame.copy()
    for k, tri_idx in enumerate(T):
        if not tri_active[k]: continue
        group_id = int(tri_render_group[k])
        if group_id < 0 or group_id >= len(RENDER_GROUP_NAMES): continue
        color = RENDER_GROUP_COLORS[RENDER_GROUP_NAMES[group_id]]
        tri = np.round(V[tri_idx]).astype(np.int32)
        cv2.fillConvexPoly(overlay, tri, color, lineType=cv2.LINE_AA)
        cv2.polylines(overlay, [tri.reshape(-1, 1, 2)], True, color, line_thickness, cv2.LINE_AA)
    return cv2.addWeighted(overlay, float(alpha), out, 1.0 - float(alpha), 0.0)


def draw_filled_triangle_highlight(frame, V, T, tri_mask,
                                    fill_color=(0,255,0), fill_alpha=0.16,
                                    edge_color=(0,255,0), edge_thickness=2, edge_alpha=0.95):
    out = frame.copy()
    if tri_mask is None or not np.any(tri_mask): return out
    tri_ids = np.flatnonzero(tri_mask)
    fill_overlay = np.zeros_like(frame)
    for k in tri_ids:
        cv2.fillConvexPoly(fill_overlay, np.round(V[T[k]]).astype(np.int32), fill_color, cv2.LINE_AA)
    out = cv2.addWeighted(fill_overlay, float(fill_alpha), out, 1.0, 0.0)
    edge_overlay = np.zeros_like(frame)
    for k in tri_ids:
        cv2.polylines(edge_overlay, [np.round(V[T[k]]).astype(np.int32).reshape(-1,1,2)],
                      True, edge_color, edge_thickness, cv2.LINE_AA)
    out = cv2.addWeighted(edge_overlay, float(edge_alpha), out, 1.0, 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def print_render_group_triangle_stats(binding):
    tri_active = binding["tri_active"]; tri_render_group = binding["tri_render_group"]
    print("\n--- Triangle render-group stats ---")
    print(f"active triangles total: {int(np.sum(tri_active))}/{len(tri_active)}")
    for group_name in RENDER_GROUP_NAMES:
        gid = RENDER_GROUP_INDEX[group_name]
        print(f"{group_name:>10s}: {int(np.sum(tri_active & (tri_render_group == gid)))}")
    print(f"{'unassigned':>10s}: {int(np.sum(tri_active & (tri_render_group < 0)))}")
    print("-----------------------------------\n")


def estimate_body_yaw(cur_pts, ref_metrics):
    ls = cur_pts.get("left_shoulder"); rs = cur_pts.get("right_shoulder")
    lh = cur_pts.get("left_hip");     rh = cur_pts.get("right_hip")
    if ls is None or rs is None or lh is None or rh is None:
        return 0.0, 0.0, {"width_ratio": 1.0, "sign_value": 0.0}
    shoulder_ratio = np.clip(np.linalg.norm(rs-ls) / max(ref_metrics["shoulder_width"], 1.0), 0.0, 1.2)
    hip_ratio      = np.clip(np.linalg.norm(rh-lh) / max(ref_metrics["hip_width"], 1.0), 0.0, 1.2)
    width_ratio    = 0.5 * (shoulder_ratio + hip_ratio)
    yaw_amount     = np.clip((1.0 - width_ratio) / 0.55, 0.0, 1.0)
    sign_value, sign_debug = compute_yaw_sign_components(cur_pts)
    yaw_sign = 1.0 if sign_value > 0 else (-1.0 if sign_value < 0 else 0.0)
    return float(yaw_amount), float(yaw_sign), {"width_ratio": float(width_ratio),
                                                  "sign_value": float(sign_value), **sign_debug}


def triangle_area2(tri):
    a, b, c = tri
    return abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))


def _triangle_bbox_size(tri):
    return float(np.max(tri[:,0])-np.min(tri[:,0])), float(np.max(tri[:,1])-np.min(tri[:,1]))


def _triangle_inside_image(tri, w, h, pad=2.0):
    x0=np.min(tri[:,0]); x1=np.max(tri[:,0]); y0=np.min(tri[:,1]); y1=np.max(tri[:,1])
    return not (x1 < -pad or y1 < -pad or x0 > w-1+pad or y0 > h-1+pad)


def warp_mesh_piecewise(src_img, V_src, V_dst, T, active_mask=None, dst_img=None,
                         min_area=1.0, min_bbox=2.0):
    if dst_img is None: dst_img = src_img.copy()
    if active_mask is None: active_mask = np.ones(len(T), dtype=bool)
    h, w = src_img.shape[:2]
    for k, tri_idx in enumerate(T):
        if not active_mask[k]: continue
        t_src = V_src[tri_idx].astype(np.float32); t_dst = V_dst[tri_idx].astype(np.float32)
        if not np.isfinite(t_src).all() or not np.isfinite(t_dst).all(): continue
        if triangle_area2(t_src) < min_area or triangle_area2(t_dst) < min_area: continue
        src_bw, src_bh = _triangle_bbox_size(t_src); dst_bw, dst_bh = _triangle_bbox_size(t_dst)
        if src_bw < min_bbox or src_bh < min_bbox or dst_bw < min_bbox or dst_bh < min_bbox: continue
        if not _triangle_inside_image(t_src, w, h): continue
        try: TMh.warp_triangle(src_img, dst_img, t_src, t_dst)
        except cv2.error: continue
    return dst_img


def reconstruct_tracked_mesh_from_skeleton(binding, cur_pts, V_base):
    frames = build_segment_frames(cur_pts)
    if frames is None: return None, None
    V_track = V_base.copy().astype(np.float32)
    seg_ids = binding["vertex_segment"]; rest_uv = binding["vertex_local_uv_rest"]
    for i in range(len(V_track)):
        si = seg_ids[i]
        if si < 0: continue
        V_track[i] = world_from_local_in_frame(rest_uv[i], frames[SEGMENT_NAMES[si]])
    return V_track, frames


def reconstruct_deformed_mesh_from_skeleton(binding, cur_pts, V_base):
    frames = build_segment_frames(cur_pts)
    if frames is None: return None, None
    V_def = V_base.copy().astype(np.float32)
    seg_ids = binding["vertex_segment"]
    rest_uv = binding["vertex_local_uv_rest"]; off_uv = binding["vertex_local_uv_offset"]
    for i in range(len(V_def)):
        si = seg_ids[i]
        if si < 0: continue
        V_def[i] = world_from_local_in_frame(rest_uv[i] + off_uv[i], frames[SEGMENT_NAMES[si]])
    return V_def, frames


def has_required_landmarks(pts, required_names=REQUIRED_INIT_LANDMARKS):
    if pts is None: return False
    return all(pts.get(name) is not None for name in required_names)


def _midpoint(a, b):
    return None if (a is None or b is None) else 0.5 * (a + b)


def _segment_length(a, b, fallback):
    return float(fallback) if (a is None or b is None) else float(max(np.linalg.norm(b - a), 1.0))


def extract_pose_points(rgb, pose, w, h, min_vis=0.45):
    res = pose.process(rgb)
    if not res.pose_landmarks: return None
    lm = res.pose_landmarks.landmark; pts = {}
    for name, idx in POSE_IDS.items():
        p = lm[idx]
        if p.visibility is not None and p.visibility < min_vis: pts[name] = None; continue
        pts[name] = np.array([float(np.clip(p.x*w, 0, w-1)), float(np.clip(p.y*h, 0, h-1))], dtype=np.float32)
    if not all(pts.get(k) is not None for k in ["left_shoulder","right_shoulder","left_hip","right_hip"]):
        return None
    return pts


def sample_mask_at_points(mask, pts_xy, thresh=0.5):
    h, w = mask.shape[:2]
    x = np.clip(np.round(pts_xy[:,0]).astype(np.int32), 0, w-1)
    y = np.clip(np.round(pts_xy[:,1]).astype(np.int32), 0, h-1)
    return mask[y, x] >= thresh


def triangle_centroids(V, T):
    return (V[T[:,0]] + V[T[:,1]] + V[T[:,2]]) / 3.0


def expand_mask(mask, ksize=9):
    k = max(1, int(ksize)); k = k if k%2==1 else k+1
    return cv2.dilate((mask>0.5).astype(np.uint8), np.ones((k,k),np.uint8), iterations=1).astype(np.float32)


def build_vertex_neighbors_from_triangles(num_vertices, T):
    neighbors = [set() for _ in range(num_vertices)]
    for tri in T:
        a, b, c = map(int, tri)
        neighbors[a].update((b,c)); neighbors[b].update((a,c)); neighbors[c].update((a,b))
    return neighbors


def extract_largest_mask_contour(mask, thresh=0.5):
    m = (mask >= thresh).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return None
    contour = max(contours, key=cv2.contourArea)
    return None if (contour is None or len(contour) < 3) else contour[:,0,:].astype(np.float32)


def shape_mesh_boundary_to_mask(V_base, T, init_mask, mask_thresh=0.5, snap_dist_px=24.0, smooth_iters=1):
    if init_mask is None: return V_base.copy().astype(np.float32)
    V_new = V_base.copy().astype(np.float32)
    inside = sample_mask_at_points(init_mask, V_base, thresh=mask_thresh)
    if not np.any(inside): return V_new
    contour_pts = extract_largest_mask_contour(init_mask, thresh=mask_thresh)
    if contour_pts is None or len(contour_pts) == 0: return V_new
    neighbors = build_vertex_neighbors_from_triangles(len(V_base), T)
    boundary = np.zeros(len(V_base), dtype=bool)
    for i in range(len(V_base)):
        if not inside[i]: continue
        for j in neighbors[i]:
            if not inside[j]: boundary[i] = True; break
    boundary_ids = np.flatnonzero(boundary)
    if len(boundary_ids) == 0: return V_new
    snap_dist2 = float(max(snap_dist_px, 1.0)) ** 2
    for i in boundary_ids:
        d2 = np.sum((contour_pts - V_new[i][None,:])**2, axis=1)
        j = int(np.argmin(d2))
        if d2[j] <= snap_dist2: V_new[i] = contour_pts[j]
    for _ in range(max(0, int(smooth_iters))):
        prev = V_new.copy()
        for i in boundary_ids:
            nbrs = [j for j in neighbors[i] if inside[j]]
            if nbrs: V_new[i] = 0.7*prev[i] + 0.3*np.mean(prev[nbrs], axis=0)
    return V_new.astype(np.float32)


def remove_small_components(mask, min_area=250):
    m = (mask > 0.5).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area: out[labels == label] = 1
    return out.astype(np.float32)


def trim_mask_arms_combined(mask, pts, shoulder_keep_arm_frac=1.0, shoulder_cap_scale=0.24,
                             elbow_keep_forearm_frac=1.0, min_component_area=250):
    return remove_small_components(mask.copy(), min_area=min_component_area)


def point_segment_distance(pt, a, b):
    ab = b - a; denom = float(np.dot(ab, ab))
    if denom < 1e-8: return np.linalg.norm(pt - a), 0.0, a.copy()
    t = float(np.clip(np.dot(pt - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return np.linalg.norm(pt - proj), t, proj


def compute_body_metrics(pts):
    shoulder_mid = _midpoint(pts.get("left_shoulder"), pts.get("right_shoulder"))
    hip_mid      = _midpoint(pts.get("left_hip"),      pts.get("right_hip"))
    torso_len    = _segment_length(shoulder_mid, hip_mid, 120.0)
    shoulder_width = _segment_length(pts.get("left_shoulder"), pts.get("right_shoulder"), 60.0)
    hip_width      = _segment_length(pts.get("left_hip"),      pts.get("right_hip"),      50.0)
    return {
        "torso_len": torso_len, "shoulder_width": shoulder_width, "hip_width": hip_width,
        "left_upper_arm_len":  _segment_length(pts.get("left_shoulder"),  pts.get("left_elbow"),   0.55*torso_len),
        "left_lower_arm_len":  _segment_length(pts.get("left_elbow"),     pts.get("left_wrist"),   0.55*torso_len),
        "right_upper_arm_len": _segment_length(pts.get("right_shoulder"), pts.get("right_elbow"),  0.55*torso_len),
        "right_lower_arm_len": _segment_length(pts.get("right_elbow"),    pts.get("right_wrist"),  0.55*torso_len),
        "left_thigh_len":  _segment_length(pts.get("left_hip"),   pts.get("left_knee"),   0.75*torso_len),
        "left_calf_len":   _segment_length(pts.get("left_knee"),  pts.get("left_ankle"),  0.75*torso_len),
        "right_thigh_len": _segment_length(pts.get("right_hip"),  pts.get("right_knee"),  0.75*torso_len),
        "right_calf_len":  _segment_length(pts.get("right_knee"), pts.get("right_ankle"), 0.75*torso_len),
    }


def clamp_point_jump(prev_pt, cur_pt, max_jump_px=25.0):
    if prev_pt is None or cur_pt is None: return cur_pt
    d = cur_pt - prev_pt; dist = np.linalg.norm(d)
    return cur_pt if (dist <= max_jump_px or dist < 1e-6) else prev_pt + d * (max_jump_px / dist)


def rasterize_pose_capsules(mask_shape, pts):
    h, w = mask_shape[:2]; out = np.zeros((h, w), dtype=np.uint8)
    segments = [("left_shoulder","left_elbow",28),("left_elbow","left_wrist",24),
                ("right_shoulder","right_elbow",28),("right_elbow","right_wrist",24),
                ("left_hip","left_knee",30),("left_knee","left_ankle",26),
                ("right_hip","right_knee",30),("right_knee","right_ankle",26)]
    for a_name, b_name, thick in segments:
        a = pts.get(a_name); b = pts.get(b_name)
        if a is None or b is None: continue
        cv2.line(out, tuple(np.round(a).astype(np.int32)), tuple(np.round(b).astype(np.int32)),
                 255, thickness=thick, lineType=cv2.LINE_AA)
    req = ["left_shoulder","right_shoulder","right_hip","left_hip"]
    if all(pts.get(k) is not None for k in req):
        cv2.fillConvexPoly(out, np.array([pts[k] for k in req], dtype=np.int32), 255)
    nose = pts.get("nose"); ls = pts.get("left_shoulder"); rs = pts.get("right_shoulder")
    if nose is not None and ls is not None and rs is not None:
        r = max(20, int(0.32 * np.linalg.norm(rs - ls)))
        cv2.circle(out, tuple(np.round(nose).astype(np.int32)), r, 255, -1, cv2.LINE_AA)
    return (out > 0).astype(np.float32)


def make_torso_frame(pts):
    ls=pts["left_shoulder"]; rs=pts["right_shoulder"]; lh=pts["left_hip"]; rh=pts["right_hip"]
    shoulder_mid = 0.5*(ls+rs); hip_mid = 0.5*(lh+rh)
    yhat, torso_len = safe_normalize(hip_mid - shoulder_mid)
    shoulder_width = float(np.linalg.norm(rs-ls)) or 40.0
    xhat = np.array([yhat[1], -yhat[0]], dtype=np.float32)
    return {"origin": shoulder_mid.astype(np.float32), "xhat": xhat, "yhat": yhat,
            "length": float(max(torso_len, 20.0)), "width": float(max(shoulder_width, 20.0))}


def build_segment_frames(pts):
    if pts is None: return None
    if not all(pts.get(k) is not None for k in ["left_shoulder","right_shoulder","left_hip","right_hip"]):
        return None
    frames = {}
    frames["torso"] = make_torso_frame(pts)
    shoulder_mid = 0.5*(pts["left_shoulder"]+pts["right_shoulder"])
    hip_mid      = 0.5*(pts["left_hip"]+pts["right_hip"])
    torso_axis   = hip_mid - shoulder_mid

    def _arm_frame(shoulder, elbow, wrist, fallback_dir):
        frames_out = {}
        up_dir = (elbow - shoulder) if (shoulder is not None and elbow is not None) else fallback_dir
        frames_out["upper"] = make_frame_from_points(shoulder, elbow, fallback_origin=shoulder, fallback_dir=up_dir)
        low_dir = ((wrist - elbow) if (elbow is not None and wrist is not None) else
                   ((elbow - shoulder) if (elbow is not None and shoulder is not None) else up_dir))
        low_origin = elbow if elbow is not None else shoulder
        frames_out["lower"] = make_frame_from_points(elbow, wrist, fallback_origin=low_origin, fallback_dir=low_dir)
        palm_origin = wrist
        palm_dir = (wrist - elbow) if (elbow is not None and wrist is not None) else low_dir
        palm_len = max(18.0, 0.75 * np.linalg.norm(palm_dir))
        palm_tip = (palm_origin + (palm_dir / np.linalg.norm(palm_dir)) * palm_len
                    if (np.linalg.norm(palm_dir) > 1e-6 and palm_origin is not None) else None)
        frames_out["palm"] = make_frame_from_points(palm_origin, palm_tip, fallback_origin=palm_origin, fallback_dir=palm_dir)
        return frames_out

    nose = pts.get("nose")
    frames["head"] = make_frame_from_points(shoulder_mid, nose, fallback_origin=shoulder_mid,
                                             fallback_dir=np.array([0.0,-40.0],dtype=np.float32))

    lf = _arm_frame(pts.get("left_shoulder"), pts.get("left_elbow"), pts.get("left_wrist"),
                    np.array([-1.0,0.0],dtype=np.float32)*max(30.0, np.linalg.norm(torso_axis)*0.35))
    frames["left_upper_arm"] = lf["upper"]; frames["left_lower_arm"] = lf["lower"]; frames["left_palm"] = lf["palm"]

    rf = _arm_frame(pts.get("right_shoulder"), pts.get("right_elbow"), pts.get("right_wrist"),
                    np.array([1.0,0.0],dtype=np.float32)*max(30.0, np.linalg.norm(torso_axis)*0.35))
    frames["right_upper_arm"] = rf["upper"]; frames["right_lower_arm"] = rf["lower"]; frames["right_palm"] = rf["palm"]

    def _leg_frames(hip, knee, ankle, fallback_dir):
        thigh_dir = (knee - hip) if (hip is not None and knee is not None) else fallback_dir
        fr_thigh = make_frame_from_points(hip, knee, fallback_origin=hip, fallback_dir=thigh_dir)
        calf_dir = (ankle - knee) if (knee is not None and ankle is not None) else thigh_dir
        calf_origin = knee if knee is not None else hip
        fr_calf = make_frame_from_points(knee, ankle, fallback_origin=calf_origin, fallback_dir=calf_dir)
        return fr_thigh, fr_calf

    lt, lc = _leg_frames(pts.get("left_hip"),  pts.get("left_knee"),  pts.get("left_ankle"),  torso_axis)
    rt, rc = _leg_frames(pts.get("right_hip"), pts.get("right_knee"), pts.get("right_ankle"), torso_axis)
    frames["left_thigh"] = lt; frames["left_calf"] = lc
    frames["right_thigh"] = rt; frames["right_calf"] = rc
    return frames


def point_in_quad(pt, quad):
    return cv2.pointPolygonTest(quad.astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0


def make_frame_from_points(p0, p1, fallback_origin=None, fallback_dir=None, min_len=10.0):
    if p0 is not None and p1 is not None:
        xhat, length = safe_normalize(p1 - p0); length = max(length, min_len)
        return {"origin": p0.astype(np.float32), "xhat": xhat,
                "yhat": np.array([-xhat[1], xhat[0]], dtype=np.float32), "length": float(length)}
    origin = np.array([0.0,0.0],dtype=np.float32) if fallback_origin is None else fallback_origin.astype(np.float32)
    fdir   = np.array([1.0,0.0],dtype=np.float32) if fallback_dir   is None else fallback_dir.astype(np.float32)
    xhat, length = safe_normalize(fdir); length = max(length, min_len)
    return {"origin": origin, "xhat": xhat,
            "yhat": np.array([-xhat[1], xhat[0]], dtype=np.float32), "length": float(length)}


def safe_normalize(v, eps=1e-6):
    n = np.linalg.norm(v)
    return (np.array([1.0,0.0],dtype=np.float32), 1.0) if n < eps else ((v/n).astype(np.float32), float(n))


def torso_quad_from_pts(pts, expand_x=0.22, expand_y_top=0.10, expand_y_bottom=0.10):
    ls=pts["left_shoulder"]; rs=pts["right_shoulder"]; lh=pts["left_hip"]; rh=pts["right_hip"]
    shoulder_mid=0.5*(ls+rs); hip_mid=0.5*(lh+rh)
    yhat, torso_h = safe_normalize(hip_mid - shoulder_mid)
    xhat = np.array([yhat[1], -yhat[0]], dtype=np.float32)
    half_w = 0.5 * max(np.linalg.norm(rs-ls), np.linalg.norm(rh-lh)) * (1.0 + expand_x)
    top = shoulder_mid - expand_y_top    * torso_h * yhat
    bot = hip_mid      + expand_y_bottom * torso_h * yhat
    return np.stack([top-half_w*xhat, top+half_w*xhat, bot+half_w*xhat, bot-half_w*xhat], axis=0).astype(np.float32)


def localize_point_in_frame(pt, frame):
    d = pt - frame["origin"]
    return np.array([np.dot(d, frame["xhat"]) / frame["length"],
                     np.dot(d, frame["yhat"]) / frame["length"]], dtype=np.float32)


def world_from_local_in_frame(uv, frame):
    return (frame["origin"] + uv[0]*frame["length"]*frame["xhat"]
                            + uv[1]*frame["length"]*frame["yhat"]).astype(np.float32)


def bind_mesh_to_skeleton(V_base, T, ref_pts, init_seg_mask, mask_thresh=0.5):
    ref_frames = build_segment_frames(ref_pts)
    if ref_frames is None or init_seg_mask is None: return None
    init_mask = expand_mask(init_seg_mask, ksize=9)
    vertex_in_mask = sample_mask_at_points(init_mask, V_base, thresh=mask_thresh)
    vertex_segment = -np.ones(len(V_base), dtype=np.int32)
    vertex_local_uv = np.zeros((len(V_base), 2), dtype=np.float32)

    shoulder_mid = 0.5*(ref_pts["left_shoulder"]+ref_pts["right_shoulder"])
    hip_mid      = 0.5*(ref_pts["left_hip"]+ref_pts["right_hip"])
    shoulder_w   = np.linalg.norm(ref_pts["right_shoulder"]-ref_pts["left_shoulder"])
    shoulder_y   = min(ref_pts["left_shoulder"][1], ref_pts["right_shoulder"][1])
    hip_y        = max(ref_pts["left_hip"][1],      ref_pts["right_hip"][1])
    torso_quad   = torso_quad_from_pts(ref_pts, expand_x=0.28, expand_y_top=0.22, expand_y_bottom=0.12)

    head_center = (ref_pts["nose"] + np.array([0.0,-0.18*shoulder_w],dtype=np.float32)
                   if ref_pts.get("nose") is not None
                   else shoulder_mid + np.array([0.0,-0.75*shoulder_w],dtype=np.float32))
    head_radius  = max(36.0, 0.70*shoulder_w)
    torso_left   = min(ref_pts["left_shoulder"][0],  ref_pts["left_hip"][0])  - 0.22*shoulder_w
    torso_right  = max(ref_pts["right_shoulder"][0], ref_pts["right_hip"][0]) + 0.22*shoulder_w
    left_shoulder  = ref_pts.get("left_shoulder")
    right_shoulder = ref_pts.get("right_shoulder")
    shoulder_cap_r = max(24.0, 0.30*shoulder_w)

    arm_capsules = []
    for s_key, e_key, w_key, ua, la, pa, ud, ld in [
        ("left_shoulder",  "left_elbow",  "left_wrist",
         "left_upper_arm",  "left_lower_arm",  "left_palm",  0.26, 0.22),
        ("right_shoulder", "right_elbow", "right_wrist",
         "right_upper_arm", "right_lower_arm", "right_palm", 0.26, 0.22),
    ]:
        s=ref_pts.get(s_key); e=ref_pts.get(e_key); w=ref_pts.get(w_key)
        if s is not None and e is not None: arm_capsules.append((ua, s, e, ud))
        if e is not None and w is not None:
            arm_capsules.append((la, e, w, ld))
            lw_dir = w - e; lw_len = np.linalg.norm(lw_dir)
            if lw_len > 1e-6:
                arm_capsules.append((pa, w, w + (lw_dir/lw_len)*0.75*lw_len, 0.42))

    leg_capsules = []
    for h_key, k_key, a_key, th, ca, thrad, carad in [
        ("left_hip",  "left_knee",  "left_ankle",  "left_thigh",  "left_calf",  0.28, 0.24),
        ("right_hip", "right_knee", "right_ankle", "right_thigh", "right_calf", 0.28, 0.24),
    ]:
        h=ref_pts.get(h_key); k=ref_pts.get(k_key); a=ref_pts.get(a_key)
        if h is not None and k is not None: leg_capsules.append((th, h, k, thrad))
        if k is not None and a is not None: leg_capsules.append((ca, k, a, carad))

    def assign_vertex(i, seg):
        vertex_segment[i] = SEGMENT_INDEX[seg]
        vertex_local_uv[i] = localize_point_in_frame(V_base[i], ref_frames[seg])

    def best_capsule_match(p, capsule_defs, min_radius_px, accept_scale,
                            allowed_sides=None, y_min=None, y_max=None):
        best_seg, best_dist = None, np.inf
        for seg, a, b, radius_scale in capsule_defs:
            seg_len = np.linalg.norm(b - a)
            radius  = max(min_radius_px, radius_scale * seg_len)
            if allowed_sides is not None:
                if "left"  in seg and "left"  not in allowed_sides: continue
                if "right" in seg and "right" not in allowed_sides: continue
            if y_min is not None and p[1] < y_min: continue
            if y_max is not None and p[1] > y_max: continue
            dist, _, _ = point_segment_distance(p, a, b)
            if dist <= accept_scale * radius and dist < best_dist:
                best_dist, best_seg = dist, seg
        return best_seg

    arm_y_max = hip_y + 0.10 * shoulder_w
    leg_y_min = shoulder_y + 0.35 * shoulder_w

    for i, p in enumerate(V_base):
        if not vertex_in_mask[i]: continue
        assigned = False
        if point_in_quad(p, torso_quad):
            assign_vertex(i, "torso"); assigned = True
        if not assigned:
            in_l = left_shoulder  is not None and np.linalg.norm(p-left_shoulder)  <= shoulder_cap_r
            in_r = right_shoulder is not None and np.linalg.norm(p-right_shoulder) <= shoulder_cap_r
            if in_l and not in_r:   assign_vertex(i,"left_upper_arm");  assigned=True
            elif in_r and not in_l: assign_vertex(i,"right_upper_arm"); assigned=True
            elif in_l and in_r:
                assign_vertex(i, "left_upper_arm" if np.linalg.norm(p-left_shoulder) <= np.linalg.norm(p-right_shoulder) else "right_upper_arm")
                assigned=True
        if not assigned and np.linalg.norm(p-head_center) <= head_radius:
            assign_vertex(i,"head"); assigned=True
        if not assigned:
            neck_top=shoulder_y-0.24*shoulder_w; neck_bot=shoulder_y+0.22*shoulder_w
            if neck_top<=p[1]<=neck_bot and torso_left<=p[0]<=torso_right:
                assign_vertex(i,"torso"); assigned=True
        if not assigned and leg_capsules:
            seg=best_capsule_match(p, leg_capsules, 12.0, 1.15, y_min=leg_y_min)
            if seg: assign_vertex(i,seg); assigned=True
        if not assigned and arm_capsules:
            palm_caps=[c for c in arm_capsules if c[0].endswith("palm")]
            if palm_caps:
                seg=best_capsule_match(p, palm_caps, 18.0, 1.60, y_max=arm_y_max+0.35*shoulder_w)
                if seg: assign_vertex(i,seg); assigned=True
        if not assigned and arm_capsules:
            non_palm=[c for c in arm_capsules if not c[0].endswith("palm")]
            if non_palm:
                seg=best_capsule_match(p, non_palm, 10.0, 1.25, y_max=arm_y_max+0.15*shoulder_w)
                if seg: assign_vertex(i,seg); assigned=True

    tri_vertices_in_mask = np.all(vertex_in_mask[T], axis=1)
    tri_centroid_in_mask = sample_mask_at_points(init_mask, triangle_centroids(V_base, T), thresh=mask_thresh)
    tri_assigned = np.all(vertex_segment[T] >= 0, axis=1)
    tri_active   = tri_vertices_in_mask & tri_centroid_in_mask & tri_assigned
    used_vertices = np.zeros(len(V_base), dtype=bool)
    if np.any(tri_active):
        used_vertices[np.unique(T[tri_active].reshape(-1))] = True
    vertex_segment[~used_vertices] = -1
    return {
        "ref_pts": ref_pts, "ref_frames": ref_frames,
        "vertex_segment": vertex_segment,
        "vertex_local_uv_rest": vertex_local_uv.copy(),
        "vertex_local_uv_offset": np.zeros_like(vertex_local_uv),
        "tri_active": tri_active, "vertex_in_mask": vertex_in_mask,
    }


def update_local_offsets_from_world(binding, frames, V_new):
    seg_ids=binding["vertex_segment"]; rest_uv=binding["vertex_local_uv_rest"]; off_uv=binding["vertex_local_uv_offset"]
    for i in range(len(V_new)):
        si=seg_ids[i]
        if si<0: continue
        off_uv[i] = localize_point_in_frame(V_new[i], frames[SEGMENT_NAMES[si]]) - rest_uv[i]


def smooth_pose_points(cur_pts, state, alpha=0.35, max_jump_px=25.0, hold_frames=6):
    if cur_pts is None:
        state["pose_missing_count"] += 1
        return state["pose_last_good_pts"] if state["pose_missing_count"] <= hold_frames else None
    state["pose_missing_count"] = 0
    prev_pts = state.get("pose_prev_pts")
    smoothed = {}
    for name in TRACKED_POSE_NAMES:
        cur  = cur_pts.get(name)
        prev = None if prev_pts is None else prev_pts.get(name)
        if cur is None:  smoothed[name] = prev.copy() if prev is not None else None; continue
        cur = clamp_point_jump(prev, cur, max_jump_px)
        smoothed[name] = cur.copy() if prev is None else (alpha*cur + (1.0-alpha)*prev).astype(np.float32)
    state["pose_prev_pts"]      = {k: (None if v is None else v.copy()) for k,v in smoothed.items()}
    state["pose_last_good_pts"] = {k: (None if v is None else v.copy()) for k,v in smoothed.items()}
    return smoothed


def finalize_init_mask(mask_accum, count, avg_thresh=0.30, dilate_ksize=11, close_ksize=11):
    if count <= 0: return None
    mask_bin = (mask_accum / float(count) >= avg_thresh).astype(np.uint8)
    if close_ksize > 0:
        k=close_ksize if close_ksize%2==1 else close_ksize+1
        mask_bin=cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, np.ones((k,k),np.uint8))
    if dilate_ksize > 0:
        k=dilate_ksize if dilate_ksize%2==1 else dilate_ksize+1
        mask_bin=cv2.dilate(mask_bin, np.ones((k,k),np.uint8), iterations=1)
    return mask_bin.astype(np.float32)


def _resolve_beauty_standard_image_path(filename="beauty_standard.png"):
    candidates = []
    if "__file__" in globals():
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "beauty_standard_images", filename))
    candidates.append(os.path.join(os.getcwd(), "beauty_standard_images", filename))
    for p in candidates:
        if os.path.exists(p): return p
    return candidates[0] if candidates else os.path.join("beauty_standard_images", filename)


def load_beauty_standard_overlay(size=120, filename="beauty_standard.png"):
    path = _resolve_beauty_standard_image_path(filename)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None, path
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA), path


def _draw_bs_timer_overlay(canvas, bs_img, size, margin, elapsed, duration):
    H, W = canvas.shape[:2]; x0=W-margin-size; y0=margin
    if bs_img is not None:
        roi = canvas[y0:y0+size, x0:x0+size]
        if bs_img.shape[2] == 4:
            alpha=bs_img[:,:,3:4].astype(np.float32)/255.0; rgb=bs_img[:,:,:3].astype(np.float32)
            canvas[y0:y0+size, x0:x0+size] = np.clip(alpha*rgb + (1.0-alpha)*roi.astype(np.float32), 0,255).astype(np.uint8)
        else:
            canvas[y0:y0+size, x0:x0+size] = bs_img
    frac=float(np.clip(elapsed/max(duration,1.0),0.0,1.0)); color=(203,120,255); thick=8; pad=thick//2+2
    rx0=x0-pad; ry0=y0-pad; rx1=x0+size+pad; ry1=y0+size+pad
    perimeter=4*(size+2*pad); draw_len=frac*perimeter
    segs=[((rx0,ry0),(rx1,ry0),rx1-rx0),((rx1,ry0),(rx1,ry1),ry1-ry0),
          ((rx1,ry1),(rx0,ry1),rx1-rx0),((rx0,ry1),(rx0,ry0),ry1-ry0)]
    remaining=draw_len
    for (sx,sy),(ex,ey),seg_len in segs:
        if remaining<=0: break
        t=min(remaining,seg_len)/seg_len
        cv2.line(canvas,(int(sx),int(sy)),(int(round(sx+t*(ex-sx))),int(round(sy+t*(ey-sy)))),
                 color,thick,lineType=cv2.LINE_AA)
        remaining-=seg_len
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# ANIMATED GRADIENT BACKGROUND
# ══════════════════════════════════════════════════════════════════════════════

def make_gradient_background(h, w, t,
                              color_a=ANIM_BG_COLOR_A,
                              color_b=ANIM_BG_COLOR_B):
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    d = 0.5 * (xv + yv)
    phase = np.pi * (d - t % 1.0)
    alpha = np.sin(phase) ** 2
    a = np.array(color_a, dtype=np.float32)
    b = np.array(color_b, dtype=np.float32)
    frame = (alpha[:, :, None] * b[None, None, :] +
             (1.0 - alpha[:, :, None]) * a[None, None, :])
    return np.clip(frame, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# WELCOME SCREEN
# ══════════════════════════════════════════════════════════════════════════════

def _load_welcome_fonts(out_w, out_h, margin=90):
    from PIL import ImageFont, Image, ImageDraw

    usable_w = out_w - 2 * margin

    BOLD_PATHS = [
        "C:/Windows/Fonts/segoeuisb.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    REG_PATHS = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    LIGHT_PATHS = [
        "C:/Windows/Fonts/segoeuil.ttf",
        "C:/Windows/Fonts/calibril.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]

    def first_path(candidates):
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    bold_path  = first_path(BOLD_PATHS)
    reg_path   = first_path(REG_PATHS)
    light_path = first_path(LIGHT_PATHS)

    def fit_font(text, path, start_size, min_size=20):
        if path is None:
            return ImageFont.load_default()
        tmp_img  = Image.new("RGB", (1, 1))
        tmp_draw = ImageDraw.Draw(tmp_img)
        for sz in range(start_size, min_size - 1, -1):
            try:
                f  = ImageFont.truetype(path, sz)
                bb = tmp_draw.textbbox((0, 0), text, font=f)
                if (bb[2] - bb[0]) <= usable_w:
                    return f
            except Exception:
                continue
        return ImageFont.load_default()

    font_bold    = fit_font("Welcome",                                       bold_path,  160)
    font_regular = fit_font("to My Magic Mirror",                            reg_path,    72)
    font_body    = fit_font("to emulate a beauty standard silhouette.",      light_path,  60)
    font_warn    = fit_font("WARNING: the display may cause you distress.",  light_path,
                            font_body.size if hasattr(font_body, "size") else 60)

    return font_bold, font_regular, font_body, font_warn


_WELCOME_FONTS      = None
_WELCOME_FONTS_SIZE = None


def _make_welcome_frame(bg_base, t, out_w, out_h):
    from PIL import Image, ImageDraw

    global _WELCOME_FONTS, _WELCOME_FONTS_SIZE
    if _WELCOME_FONTS is None or _WELCOME_FONTS_SIZE != (out_w, out_h):
        _WELCOME_FONTS      = _load_welcome_fonts(out_w, out_h, margin=90)
        _WELCOME_FONTS_SIZE = (out_w, out_h)
    font_bold, font_regular, font_body, font_warn = _WELCOME_FONTS

    H, W = out_h, out_w

    xs = np.linspace(0.0, 1.0, W, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, H, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    d     = 0.5 * (xv + yv)
    speed = 0.18
    phase = np.pi * (d - (t * speed) % 1.0)
    alpha = (np.sin(phase) ** 2) * 0.28
    shimmer_bgr = np.array([180.0, 50.0, 170.0], dtype=np.float32)
    shimmer = (alpha[:, :, None] * shimmer_bgr[None, None, :]).astype(np.float32)
    frame_bgr = np.clip(bg_base.astype(np.float32) + shimmer, 0, 255).astype(np.uint8)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img  = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)

    def draw_centred(text, y, font, color=(255, 255, 255)):
        try:
            bb = draw.textbbox((0, 0), text, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except AttributeError:
            tw, th = draw.textsize(text, font=font)
        x = (W - tw) // 2
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0))
        draw.text((x,     y    ), text, font=font, fill=color)
        return th

    line_gap = int(H * 0.018)
    para_gap = int(H * 0.050)
    y = int(H * 0.28)

    h = draw_centred("Welcome",           y, font_bold);    y += h + line_gap
    h = draw_centred("to My Magic Mirror", y, font_regular); y += h + para_gap

    for line in ("You are invited to use hand gestures",
                 "to emulate a beauty standard silhouette."):
        h = draw_centred(line, y, font_body); y += h + line_gap

    y += para_gap
    draw_centred(
        "WARNING: the display may cause you distress.",
        y, font_warn,
        color=(220, 200, 255),
    )

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def show_welcome_screen(window_name, duration=WELCOME_SCREEN_DURATION):
    if duration <= 0:
        return

    bg_img = cv2.imread(WELCOME_BG_PATH)
    if bg_img is None:
        bg_img = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)
        bg_img[:, :] = (60, 20, 10)
        print(f"WARNING: welcome background not found at '{WELCOME_BG_PATH}', using plain colour.")
    else:
        bg_img = cv2.resize(bg_img, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)

    start = time.time()
    while True:
        t       = time.time() - start
        elapsed = t

        if elapsed >= duration:
            break

        frame = _make_welcome_frame(bg_img, t, OUTPUT_W, OUTPUT_H)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(16) & 0xFF
        if key != 255:
            break


# ══════════════════════════════════════════════════════════════════════════════
# BEAUTY STANDARD SELECTION SCREEN
# ══════════════════════════════════════════════════════════════════════════════

_SEL_MARGIN_X    = 60
_SEL_MARGIN_Y_TOP = 40
_SEL_MARGIN_Y_BOT = 60
_SEL_GAP_X       = 28
_SEL_GAP_Y       = 28
_SEL_COLS        = 2
_SEL_ROWS        = 3
_SEL_CORNER_R    = 22
_SEL_OUTLINE     = 6

_SEL_NAMES = [
    ["MBS_1", "FBS_1"],
    ["MBS_2", "FBS_2"],
    ["MBS_3", "FBS_3"],
]


def _load_bs_thumbnails(cell_w, cell_h):
    from PIL import Image as PILImage
    thumbs = []
    for row in range(_SEL_ROWS):
        row_imgs = []
        for col in range(_SEL_COLS):
            name = _SEL_NAMES[row][col]
            fpath = os.path.join(_BS_IMAGE_DIR, f"{name}.png")
            if not os.path.exists(fpath):
                row_imgs.append(None)
                print(f"WARNING: beauty standard image not found: {fpath}")
                continue
            img = PILImage.open(fpath).convert("RGBA")
            img_w, img_h = img.size
            scale = min((cell_w - _SEL_OUTLINE*2) / img_w,
                        (cell_h - _SEL_OUTLINE*2) / img_h)
            new_w = max(1, int(img_w * scale))
            new_h = max(1, int(img_h * scale))
            img   = img.resize((new_w, new_h), PILImage.LANCZOS)
            row_imgs.append(img)
        thumbs.append(row_imgs)
    return thumbs


def _cell_rect(row, col, grid_top, grid_left, cell_w, cell_h):
    x0 = grid_left + col * (cell_w + _SEL_GAP_X)
    y0 = grid_top  + row * (cell_h + _SEL_GAP_Y)
    return x0, y0, x0 + cell_w, y0 + cell_h


_SEL_HOVER_FRAMES = 50


def _get_index_tip_norm(hand_results, project=2.5):
    if not hand_results.multi_hand_landmarks:
        return None, None
    lm = hand_results.multi_hand_landmarks[0].landmark
    wx,  wy  = lm[0].x, lm[0].y
    mx,  my  = lm[5].x, lm[5].y
    dx = mx - wx
    dy = my - wy
    px = mx + project * dx
    py = my + project * dy
    return float(np.clip(px, 0.0, 1.0)), float(np.clip(py, 0.0, 1.0))

def overlay_bgra(base_bgr, overlay_bgra, x, y):
    h, w = overlay_bgra.shape[:2]

    x1 = min(x + w, base_bgr.shape[1])
    y1 = min(y + h, base_bgr.shape[0])

    if x >= x1 or y >= y1:
        return base_bgr

    overlay_crop = overlay_bgra[0:(y1 - y), 0:(x1 - x)]
    roi = base_bgr[y:y1, x:x1]

    bgr = overlay_crop[:, :, :3].astype(np.float32)
    alpha = overlay_crop[:, :, 3:4].astype(np.float32) / 255.0

    roi[:] = (alpha * bgr + (1 - alpha) * roi.astype(np.float32)).astype(np.uint8)
    return base_bgr


def _make_selection_frame(bg_base, t, out_w, out_h,
                           font_hdr, thumbnails,
                           grid_top, grid_left, cell_w, cell_h,
                           hover_fill,
                           tip_px=None):
    from PIL import Image as PILImage, ImageDraw

    H, W = out_h, out_w

    xs = np.linspace(0.0, 1.0, W, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, H, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    d     = 0.5 * (xv + yv)
    speed = 0.18
    phase = np.pi * (d - (t * speed) % 1.0)
    alpha_sh = (np.sin(phase) ** 2) * 0.28
    shimmer_bgr = np.array([180.0, 50.0, 170.0], dtype=np.float32)
    shimmer = (alpha_sh[:, :, None] * shimmer_bgr[None, None, :]).astype(np.float32)
    frame_bgr = np.clip(bg_base.astype(np.float32) + shimmer, 0, 255).astype(np.uint8)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    canvas = PILImage.fromarray(frame_rgb).convert("RGBA")
    draw   = ImageDraw.Draw(canvas)

    header = "Please choose the beauty standard you would like to emulate"
    try:
        bb = draw.textbbox((0, 0), header, font=font_hdr)
        tw = bb[2] - bb[0]
    except AttributeError:
        tw, _ = draw.textsize(header, font=font_hdr)
    hx = (W - tw) // 2
    hy = _SEL_MARGIN_X
    draw.text((hx + 2, hy + 2), header, font=font_hdr, fill=(0, 0, 0, 200))
    draw.text((hx,     hy    ), header, font=font_hdr, fill=(255, 255, 255, 255))

    shimmer_speed = 0.6
    shimmer_wave  = (t * shimmer_speed) % 1.0

    overlay = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
    odraw   = ImageDraw.Draw(overlay)

    for row in range(_SEL_ROWS):
        for col in range(_SEL_COLS):
            x0, y0, x1, y1 = _cell_rect(row, col, grid_top, grid_left,
                                          cell_w, cell_h)
            fill_frac = hover_fill.get((row, col), 0.0)

            cell_diag  = (col / max(_SEL_COLS - 1, 1) + row / max(_SEL_ROWS - 1, 1)) * 0.5
            phase      = np.pi * (cell_diag - shimmer_wave)
            shimmer_v  = float(np.sin(phase) ** 2)

            base_out_a  = 120
            shimmer_add = int(shimmer_v * 100)
            hover_add   = int(fill_frac * 35)
            out_a       = min(255, base_out_a + shimmer_add + hover_add)
            out_w_px    = _SEL_OUTLINE + int(fill_frac * 4)

            r_out = 255
            g_out = int(255 - (1.0 - shimmer_v) * 30 * (1.0 - fill_frac))
            b_out = int(255 - (1.0 - shimmer_v) * 60 * (1.0 - fill_frac))

            fill_a = int(8 + fill_frac * 247)

            odraw.rounded_rectangle(
                [x0, y0, x1, y1],
                radius=_SEL_CORNER_R,
                fill=(255, 255, 255, fill_a),
                outline=(r_out, g_out, b_out, out_a),
                width=out_w_px,
            )

            thumb = thumbnails[row][col]
            if thumb is not None:
                tw_i, th_i = thumb.size
                px = x0 + (cell_w - tw_i) // 2
                py = y0 + (cell_h - th_i) // 2
                canvas.paste(thumb,
                             (max(px, x0 + _SEL_OUTLINE),
                              max(py, y0 + _SEL_OUTLINE)),
                             mask=thumb)

    canvas = PILImage.alpha_composite(canvas, overlay)

    result_bgr = cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)
    if tip_px is not None:
        tx, ty = tip_px
        cv2.circle(result_bgr, (tx, ty), 18, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(result_bgr, (tx, ty), 18, (180, 180, 255),  2, cv2.LINE_AA)

    return result_bgr


def show_selection_screen(window_name, bg_base, cap):
    global current_beauty_standard

    from PIL import ImageFont as PILFont, Image as _PI3, ImageDraw as _ID3

    out_w, out_h = OUTPUT_W, OUTPUT_H
    usable_w = out_w - 2 * _SEL_MARGIN_X

    BOLD_PATHS = [
        "C:/Windows/Fonts/segoeuisb.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    header_text = "Please choose the beauty standard you would like to emulate"
    font_hdr = None
    for fpath in BOLD_PATHS:
        if not os.path.exists(fpath):
            continue
        for sz in range(52, 14, -1):
            try:
                f  = PILFont.truetype(fpath, sz)
                _d = _ID3.Draw(_PI3.new("RGB", (1, 1)))
                bb = _d.textbbox((0, 0), header_text, font=f)
                if (bb[2] - bb[0]) <= usable_w:
                    font_hdr = f
                    break
            except Exception:
                continue
        if font_hdr is not None:
            break
    if font_hdr is None:
        font_hdr = PILFont.load_default()

    _dd = _ID3.Draw(_PI3.new("RGB", (1, 1)))
    try:
        hdr_bb = _dd.textbbox((0, 0), header_text, font=font_hdr)
        hdr_h  = hdr_bb[3] - hdr_bb[1]
    except AttributeError:
        _, hdr_h = _dd.textsize(header_text, font=font_hdr)

    grid_top  = _SEL_MARGIN_X + hdr_h + _SEL_MARGIN_Y_TOP
    grid_left = _SEL_MARGIN_X
    grid_w    = out_w - 2 * _SEL_MARGIN_X
    grid_h    = out_h - _SEL_MARGIN_Y_BOT - grid_top
    cell_w    = (grid_w - _SEL_GAP_X * (_SEL_COLS - 1)) // _SEL_COLS
    cell_h    = (grid_h - _SEL_GAP_Y * (_SEL_ROWS - 1)) // _SEL_ROWS

    thumbnails = _load_bs_thumbnails(cell_w, cell_h)

    key_to_cell = {
        ord('1'): (0,0), ord('2'): (1,0), ord('3'): (2,0),
        ord('4'): (0,1), ord('5'): (1,1), ord('6'): (2,1),
    }

    hover_fill   = {(r, c): 0.0 for r in range(_SEL_ROWS) for c in range(_SEL_COLS)}
    hover_frames = {(r, c): 0   for r in range(_SEL_ROWS) for c in range(_SEL_COLS)}
    DRAIN_SPEED  = 8

    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands_sel:

        start  = time.time()
        result = None

        smooth_nx, smooth_ny = None, None
        SMOOTH_ALPHA = 0.35

        RENDER_EVERY   = 4
        infer_count    = 0
        cached_frame   = None

        while result is None:
            for _ in range(3):
                cap.grab()
            ok, fr = cap.read()
            if not ok:
                break
            fr  = rotate_frame(fr)
            if fr.shape[:2] != (out_h, out_w):
                fr = cv2.resize(fr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            fr  = cv2.flip(fr, 1)
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

            hand_res       = hands_sel.process(rgb)
            raw_nx, raw_ny = _get_index_tip_norm(hand_res)
            infer_count   += 1

            if raw_nx is not None:
                if smooth_nx is None:
                    smooth_nx, smooth_ny = raw_nx, raw_ny
                else:
                    smooth_nx = SMOOTH_ALPHA * raw_nx + (1 - SMOOTH_ALPHA) * smooth_nx
                    smooth_ny = SMOOTH_ALPHA * raw_ny + (1 - SMOOTH_ALPHA) * smooth_ny
            else:
                smooth_nx = smooth_ny = None

            tip_px = None
            if smooth_nx is not None:
                tip_px = (int(smooth_nx * out_w), int(smooth_ny * out_h))

            hovered_cell = None
            if tip_px is not None:
                tx, ty = tip_px
                for row in range(_SEL_ROWS):
                    for col in range(_SEL_COLS):
                        x0, y0, x1, y1 = _cell_rect(row, col, grid_top,
                                                      grid_left, cell_w, cell_h)
                        if x0 <= tx <= x1 and y0 <= ty <= y1:
                            hovered_cell = (row, col)
                            break
                    if hovered_cell:
                        break

            for cell in list(hover_fill.keys()):
                if cell == hovered_cell:
                    hover_frames[cell] = min(_SEL_HOVER_FRAMES,
                                             hover_frames[cell] + 1)
                    hover_fill[cell]   = hover_frames[cell] / _SEL_HOVER_FRAMES
                    if hover_frames[cell] >= _SEL_HOVER_FRAMES:
                        result = _SEL_NAMES[cell[0]][cell[1]]
                else:
                    hover_frames[cell] = max(0, hover_frames[cell] - DRAIN_SPEED)
                    hover_fill[cell]   = hover_frames[cell] / _SEL_HOVER_FRAMES

            if infer_count % RENDER_EVERY == 0 or cached_frame is None:
                t = time.time() - start
                cached_frame = _make_selection_frame(
                    bg_base, t, out_w, out_h,
                    font_hdr, thumbnails,
                    grid_top, grid_left, cell_w, cell_h,
                    hover_fill=hover_fill,
                    tip_px=tip_px,
                )

            cv2.imshow(window_name, cached_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in key_to_cell:
                result = _SEL_NAMES[key_to_cell[key][0]][key_to_cell[key][1]]
            if key in (ord('q'), 27):
                result = "__skip__"
                break

    if result == "__skip__" or result is None:
        current_beauty_standard = ""
        return None

    current_beauty_standard = result
    print(f"Beauty standard selected: {result}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# HOME SCREEN
# ══════════════════════════════════════════════════════════════════════════════

def show_home_screen(window_name, cap, pose):
    ABSENT_FRAMES = 5

    bg_img = cv2.imread(WELCOME_BG_PATH)
    if bg_img is None:
        bg_img = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)
        bg_img[:, :] = (60, 20, 10)
    else:
        bg_img = cv2.resize(bg_img, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)

    absent_count = 0
    armed        = False
    start        = time.time()

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=False,
        enable_segmentation=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as pose_home:

        while True:
            ok, fr = cap.read()
            if not ok:
                break
            fr  = rotate_frame(fr)
            if fr.shape[:2] != bg_img.shape[:2]:
                fr = cv2.resize(fr, (bg_img.shape[1], bg_img.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
            fr  = cv2.flip(fr, 1)
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

            res          = pose_home.process(rgb)
            person_here  = (res.pose_landmarks is not None)

            if person_here:
                absent_count = 0
                if armed:
                    print("Home screen: person detected — showing welcome screen")
                    show_welcome_screen(window_name, duration=WELCOME_SCREEN_DURATION)
                    return
            else:
                absent_count += 1
                if absent_count >= ABSENT_FRAMES:
                    armed = True

            t     = time.time() - start
            xs    = np.linspace(0.0, 1.0, OUTPUT_W, dtype=np.float32)
            ys    = np.linspace(0.0, 1.0, OUTPUT_H, dtype=np.float32)
            xv, yv = np.meshgrid(xs, ys)
            d     = 0.5 * (xv + yv)
            speed = 0.18
            phase = np.pi * (d - (t * speed) % 1.0)
            alpha = (np.sin(phase) ** 2) * 0.28
            shimmer_bgr = np.array([180.0, 50.0, 170.0], dtype=np.float32)
            shimmer = (alpha[:, :, None] * shimmer_bgr[None, None, :]).astype(np.float32)
            frame_out = np.clip(bg_img.astype(np.float32) + shimmer, 0, 255).astype(np.uint8)

            cv2.imshow(window_name, frame_out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_hand_brush_drag_arap_loop_skeleton(
    cap, segmenter, hands, pose, h, w,
    V_base, V_def, T,
    step, thresh, feather, show_mask, brush_radius,
    interaction_state, arap_cache,
    inference_thread=None,
    window_name="Hand brush drag on skeleton mesh",
):
    global SHOW_MESH_OUTLINE, current_beauty_standard
    print("Entered run_hand_brush_drag_arap_loop_skeleton")
    binding = interaction_state["binding"]

    _session_start     = time.time()
    _timeout_fired     = False
    _timeout_msg_start = 0.0
    _timeout_new_bs    = ""

    # ── Load beauty-standard reference image ─────────────────────────────
    _bs_overlay      = None
    _bs_overlay_x    = 0
    _bs_overlay_y    = 0
    _BS_ALPHA        = 0.35
    _BS_FIGURE_FRAC  = 0.82   # figure fills full image height

    if current_beauty_standard:
        _bs_path = os.path.join(_BS_IMAGE_DIR, f"{current_beauty_standard}.png")
        _bs_raw  = cv2.imread(_bs_path, cv2.IMREAD_UNCHANGED)
        if _bs_raw is None:
            print(f"WARNING: could not load BS image: {_bs_path}")
        else:
            _cam_h_px  = interaction_state.get("person_height_px", h * 0.75)
            _scale_cam_to_out = OUTPUT_H / h
            _person_out_px    = _cam_h_px * _scale_cam_to_out
            _src_h            = _bs_raw.shape[0]
            _target_img_h     = int(_person_out_px / _BS_FIGURE_FRAC)
            _target_img_w     = int(_bs_raw.shape[1] * _target_img_h / _src_h)
            _bs_scaled        = cv2.resize(_bs_raw,
                                           (_target_img_w, _target_img_h),
                                           interpolation=cv2.INTER_AREA)
            _BS_BOTTOM_MARGIN = 60
            _bs_overlay_x = (OUTPUT_W - _target_img_w) // 2
            _bs_overlay_y = OUTPUT_H - _target_img_h - _BS_BOTTOM_MARGIN
            _bs_overlay   = _bs_scaled
            print(f"BS overlay: {_target_img_w}x{_target_img_h} "
                  f"at ({_bs_overlay_x},{_bs_overlay_y}), "
                  f"person_height={_person_out_px:.0f}px")

    # ── Finish button + auto-return state ───────────────────────────────
    _restart_frames     = 0
    interaction_mode = "drag"

    _DRAG_UI_DURATION = 150
    _drag_ui_timer = _DRAG_UI_DURATION
    _mode_click_cooldown = 0
    _no_detect_frames   = 0
    _NO_DETECT_LIMIT    = 10
    # Finish pinch-click tracking
    _finish_prev_pinch_dist = 1.0
    _finish_pinch_click_cooldown = 0

    _FINISH_PINCH_OPEN_DIST = 0.05     # fingers considered open
    _FINISH_PINCH_CLOSED_DIST = 0.02  # fingers considered clicked/pinched
    _FINISH_PINCH_DROP_MIN = 0.025     # required distance drop
    _FINISH_CLICK_COOLDOWN = 10        # prevents repeated clicks

    LEG_OVERLAP_TRIGGER_FRAC = 0.015
    LEG_FRONT_SCORE_DEADBAND = 8.0
    ARM_OVERLAP_TRIGGER_FRAC = 0.020
    ARM_FRONT_SCORE_DEADBAND = 6.0

    # ── Finish button images (load once) ─────────────────────────────
    finish_btn_normal = cv2.imread("buttons/finish.png", cv2.IMREAD_UNCHANGED)
    finish_btn_hover  = cv2.imread("buttons/finish_hover.png", cv2.IMREAD_UNCHANGED)
    finish_btn_click  = cv2.imread("buttons/finish_clicked.png", cv2.IMREAD_UNCHANGED)

    finish_btn_normal = cv2.resize(finish_btn_normal, (_RESTART_W, _RESTART_H))
    finish_btn_hover  = cv2.resize(finish_btn_hover,  (_RESTART_W, _RESTART_H))
    finish_btn_click  = cv2.resize(finish_btn_click,  (_RESTART_W, _RESTART_H))

    brush_btn_normal = cv2.imread("buttons/brush.png", cv2.IMREAD_UNCHANGED)
    brush_btn_hover  = cv2.imread("buttons/brush_hover.png", cv2.IMREAD_UNCHANGED)
    brush_btn_click  = cv2.imread("buttons/brush_clicked.png", cv2.IMREAD_UNCHANGED)

    drag_btn_normal = cv2.imread("buttons/drag.png", cv2.IMREAD_UNCHANGED)
    drag_btn_hover  = cv2.imread("buttons/drag_hover.png", cv2.IMREAD_UNCHANGED)
    drag_btn_click  = cv2.imread("buttons/drag_clicked.png", cv2.IMREAD_UNCHANGED)

    brush_btn_normal = cv2.resize(brush_btn_normal, (_MODE_BTN_W, _MODE_BTN_H))
    brush_btn_hover  = cv2.resize(brush_btn_hover,  (_MODE_BTN_W, _MODE_BTN_H))
    brush_btn_click  = cv2.resize(brush_btn_click,  (_MODE_BTN_W, _MODE_BTN_H))

    drag_btn_normal = cv2.resize(drag_btn_normal, (_MODE_BTN_W, _MODE_BTN_H))
    drag_btn_hover  = cv2.resize(drag_btn_hover,  (_MODE_BTN_W, _MODE_BTN_H))
    drag_btn_click  = cv2.resize(drag_btn_click,  (_MODE_BTN_W, _MODE_BTN_H))

    pinch_icon = cv2.imread("UI_gestures/pinch.png", cv2.IMREAD_UNCHANGED)
    open_palm_icon = cv2.imread("UI_gestures/open_palm.png", cv2.IMREAD_UNCHANGED)
    fist_icon = cv2.imread("UI_gestures/fist.png", cv2.IMREAD_UNCHANGED)
    fist_drag_icon = cv2.imread("UI_gestures/fist_drag.png", cv2.IMREAD_UNCHANGED)

    _PINCH_ICON_H = 42
    if pinch_icon is not None:
        scale = _PINCH_ICON_H / pinch_icon.shape[0]
        pinch_icon = cv2.resize(
            pinch_icon,
            (int(pinch_icon.shape[1] * scale), _PINCH_ICON_H),
            interpolation=cv2.INTER_AREA
        )

    from PIL import ImageFont as _FF2
    _FBOLD2 = [
        'C:/Windows/Fonts/segoeuisb.ttf',
        'C:/Windows/Fonts/segoeuib.ttf',
        'C:/Windows/Fonts/calibrib.ttf',
        'C:/Windows/Fonts/arialbd.ttf',
    ]
    _finish_font = None
    for _fp2 in _FBOLD2:
        if os.path.exists(_fp2):
            try: _finish_font = _FF2.truetype(_fp2, 36); break
            except: pass
    if _finish_font is None: _finish_font = _FF2.load_default()
    _rbx0 = OUTPUT_W - _RESTART_MARGIN - _RESTART_W
    _rbx1 = OUTPUT_W - _RESTART_MARGIN
    _rby0 = _RESTART_MARGIN + 80
    _rby1 = _rby0 + _RESTART_H

    # ── Brush / Drag button positions ────────────────────────────────
    brush_x0 = _MODE_BTN_MARGIN
    brush_y0 = _rby0
    brush_x1 = brush_x0 + _MODE_BTN_W
    brush_y1 = brush_y0 + _MODE_BTN_H

    drag_x0 = brush_x1 + _MODE_BTN_GAP
    drag_y0 = _rby0
    drag_x1 = drag_x0 + _MODE_BTN_W
    drag_y1 = drag_y0 + _MODE_BTN_H

    while True:
        brush_radius = interaction_state.get('brush_radius', brush_radius)

        ok, frame_raw = cap.read()
        if not ok:
            print("Main loop: cap.read() failed, breaking"); break
        frame_raw = rotate_frame(frame_raw)
        if frame_raw.shape[:2] != (h, w):
            frame_raw = cv2.resize(frame_raw, (w, h), interpolation=cv2.INTER_LINEAR)
        frame_raw = cv2.flip(frame_raw, 1)

        if inference_thread is not None:
            inference_thread.submit_frame(frame_raw)
            result = inference_thread.latest_result()
        else:
            result = None

        if result is not None:
            seg_mask, rgb, cur_pts, hand_state = result
        else:
            seg_mask = np.zeros((h, w), dtype=np.float32)
            rgb      = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
            cur_pts  = None
            hand_state = {"center":None,"is_open":False,"is_fist":False,
                          "over_body":False,"detected":False,"handedness":None}
            
        _now = time.time()
        _elapsed_session = _now - _session_start

        if not _timeout_fired and _elapsed_session >= SESSION_DURATION_SECONDS:
            _timeout_new_bs    = _pick_new_beauty_standard(current_beauty_standard)
            _timeout_fired     = True
            _timeout_msg_start = _now
            print(f"Session timeout: rotating BS '{current_beauty_standard}' -> '{_timeout_new_bs}'")

        if _timeout_fired:
            if (_now - _timeout_msg_start) >= TIMEOUT_MESSAGE_DURATION:
                current_beauty_standard = _timeout_new_bs
                # reload the BS overlay with the new standard
                if current_beauty_standard:
                    _bs_path = os.path.join(_BS_IMAGE_DIR, f"{current_beauty_standard}.png")
                    _bs_raw  = cv2.imread(_bs_path, cv2.IMREAD_UNCHANGED)
                    if _bs_raw is not None:
                        _cam_h_px      = interaction_state.get("person_height_px", h * 0.75)
                        _person_out_px = _cam_h_px * (OUTPUT_H / h)
                        _target_img_h  = int(_person_out_px / _BS_FIGURE_FRAC)
                        _target_img_w  = int(_bs_raw.shape[1] * _target_img_h / _bs_raw.shape[0])
                        _bs_overlay    = cv2.resize(_bs_raw, (_target_img_w, _target_img_h), interpolation=cv2.INTER_AREA)
                        _bs_overlay_x  = (OUTPUT_W - _target_img_w) // 2
                        _bs_overlay_y  = OUTPUT_H - _target_img_h - 60
                _session_start = _now
                _timeout_fired = False
                print(f"BS switched to '{current_beauty_standard}', timer reset.")

        _hand_also_gone = not hand_state.get('detected', False)
        _person_visible = (cur_pts is not None) or (not _hand_also_gone)
        if _person_visible:
            _no_detect_frames = 0
        else:
            _no_detect_frames += 1
            if _no_detect_frames >= _NO_DETECT_LIMIT:
                print(f"No person/hand detected for {_NO_DETECT_LIMIT} frames — returning to welcome screen")
                return "restart"

        V_track = None; frames = None
        stable_pts = smooth_pose_points(cur_pts, interaction_state, alpha=0.88, max_jump_px=120.0, hold_frames=1)

        if stable_pts is not None:
            frames = build_segment_frames(stable_pts)
            if frames is not None:
                if frames_moved_enough(interaction_state["cached_frames"], frames, threshold_px=1.5):
                    frame_arrays = build_frame_arrays(frames, binding["vertex_segment"], len(V_base))
                    interaction_state["cached_frame_arrays"] = frame_arrays
                    interaction_state["cached_frames"]       = frames
                else:
                    frame_arrays = interaction_state["cached_frame_arrays"]
                V_track  = reconstruct_tracked_mesh_vectorized(binding, frame_arrays)
                V_def[:] = reconstruct_deformed_mesh_vectorized(binding, frame_arrays)

        active = binding["tri_active"]

        hand_center         = hand_state["center"]
        hand_is_open        = hand_state["is_open"]
        hand_is_fist        = hand_state["is_fist"]
        hand_over_body      = hand_state["over_body"]
        hand_detected       = hand_state["detected"]
        hand_pinch_dist     = hand_state.get("pinch_dist", 1.0)
        hand_handedness_raw = hand_state.get("handedness")
        if hand_handedness_raw == "Left":     hand_handedness = "Right"
        elif hand_handedness_raw == "Right":  hand_handedness = "Left"
        else:                                  hand_handedness = None

        _restart_tip_out = None
        if hand_detected and hand_center is not None:
            _rx = int(hand_center[0] * OUTPUT_W / w)
            _ry = int(hand_center[1] * OUTPUT_H / h)
            _restart_tip_out = (_rx, _ry)
            if not hasattr(run_hand_brush_drag_arap_loop_skeleton, '_dbg'):
                run_hand_brush_drag_arap_loop_skeleton._dbg = 0
            run_hand_brush_drag_arap_loop_skeleton._dbg += 1
            if run_hand_brush_drag_arap_loop_skeleton._dbg % 90 == 0:
                print(f'[BTN] hand_out={_restart_tip_out} zone=[{_rbx0-200}-{_rbx1+200}, {_rby0-200}-{_rby1+200}]')
            if _rx < 0 or _rx > OUTPUT_W or _ry < 0 or _ry > OUTPUT_H:
                pass

        active_hand  = interaction_state.get("active_hand", None)
        hand_allowed = (
            not hand_detected
            or active_hand is None
            or hand_handedness == active_hand
            or hand_handedness is None
        )

        hand_is_peace = hand_state.get("is_peace", False)
        if hand_detected:
            peace_fired = _peace_detector.update_raw(hand_is_peace)
            if peace_fired:
                cur = interaction_state.get("active_hand", None)
                if cur is None:
                    nxt = hand_handedness
                else:
                    nxt = None
                interaction_state["active_hand"] = nxt
                label = nxt if nxt is not None else "Either hand"
                interaction_state["gesture_feedback_msg"]    = f"Active hand: {label}"
                interaction_state["gesture_feedback_frames"] = 90
                print(f"Peace sign: active hand -> {nxt}")
        else:
            _peace_detector.reset()

        if interaction_mode == "brush" and hand_detected:
            pinch_delta, is_pinching = _pinch_tracker.update_from_dist(hand_pinch_dist)

            if is_pinching:
                old_r = interaction_state.get("brush_radius", brush_radius)
                new_r = int(np.clip(old_r + pinch_delta, BRUSH_RADIUS_MIN, BRUSH_RADIUS_MAX))
                if new_r != old_r:
                    interaction_state["brush_radius"] = new_r
                    brush_radius = new_r

            if _pinch_tracker.is_locked:
                interaction_mode = "drag"
                _drag_ui_timer = _DRAG_UI_DURATION
                _pinch_tracker.reset()
        else:
            _pinch_tracker.reset()

        new_preview_vertices  = np.zeros(len(V_def), dtype=bool)
        new_preview_triangles = np.zeros(len(T),     dtype=bool)

        if hand_detected and hand_allowed:
            if (not interaction_state["dragging"]) and hand_is_open and hand_over_body:
                new_preview_vertices, new_preview_triangles = TMh.compute_brush_selection(
                    V=V_def, T=T, active=active, center=hand_center, radius=brush_radius)
                new_preview_vertices, new_preview_triangles = filter_selection_disallow_same_side_arm(
                    new_preview_vertices, new_preview_triangles, T, binding, hand_handedness)

            if ((not interaction_state["dragging"]) and interaction_state["hand_was_open"]
                    and hand_is_fist and np.any(interaction_state["preview_vertices"])):
                print(">>> DRAG STARTED")
                interaction_state["dragging"]    = True
                interaction_state["drag_vertices"] = interaction_state["preview_vertices"].copy()
                interaction_state["drag_triangles"] = np.any(interaction_state["drag_vertices"][T], axis=1)
                interaction_state["prev_hand_center"] = hand_center

            elif (interaction_state["dragging"] and hand_is_fist
                  and hand_center is not None and interaction_state["prev_hand_center"] is not None):
                dx = hand_center[0] - interaction_state["prev_hand_center"][0]
                dy = hand_center[1] - interaction_state["prev_hand_center"][1]
                interaction_state["prev_hand_center"] = hand_center
                if abs(dx) >= 1 or abs(dy) >= 1:
                    print("dx dy:", dx, dy)
                    V_new = TMh.apply_arap_drag_step(
                        V_track=V_track, V_def=V_def, T=T, active_triangles=active,
                        drag_vertices=interaction_state["drag_vertices"],
                        delta_xy=np.array([dx, dy], dtype=np.float32),
                        arap_cache=arap_cache, region_rings=6, n_iters=5, falloff_power=1.6)
                    if frames is not None and interaction_state.get("cached_frame_arrays") is not None:
                        update_local_offsets_vectorized(binding, interaction_state["cached_frame_arrays"], V_new)
                    V_rebuilt = reconstruct_deformed_mesh_vectorized(binding, interaction_state["cached_frame_arrays"])
                    if V_rebuilt is not None: V_def[:] = V_rebuilt

            elif interaction_state["dragging"] and not hand_is_fist:
                interaction_state["dragging"] = False
                interaction_state["drag_vertices"][:] = False
                interaction_state["drag_triangles"][:] = False
                interaction_state["prev_hand_center"] = None

            if not interaction_state["dragging"]:
                interaction_state["preview_vertices"]  = new_preview_vertices
                interaction_state["preview_triangles"] = new_preview_triangles
            interaction_state["hand_was_open"] = hand_is_open

        else:
            if interaction_state["dragging"]:
                interaction_state["dragging"] = False
                interaction_state["drag_vertices"][:] = False
                interaction_state["drag_triangles"][:] = False
                interaction_state["prev_hand_center"] = None
            interaction_state["preview_vertices"][:]  = False
            interaction_state["preview_triangles"][:] = False
            interaction_state["hand_was_open"] = False

        layered_layers=None; render_order_used=FIXED_RENDER_ORDER
        yaw_amount=0.0; yaw_sign=0.0; yaw_state_txt="frontal"
        yaw_debug={"sign_value":0.0,"sign_value_smooth":0.0,"width_ratio":1.0}
        leg_overlap_frac=0.0; leg_front_score=0.0; leg_override_used=False
        left_arm_overlap_frac=0.0; right_arm_overlap_frac=0.0
        left_arm_score=0.0; right_arm_score=0.0; arm_override_used=False
        bg_plate = interaction_state.get("bg_plate")

        if USE_ANIMATED_BACKGROUND:
            bg_plate = make_gradient_background(
                h, w,
                t=time.time() * ANIM_BG_SPEED,
            )

        if V_track is not None and bg_plate is not None:
            if SHOW_LAYERED_RENDER and ("tri_render_group" in binding):
                if USE_DYNAMIC_YAW_RENDER_ORDER and stable_pts is not None and "body_metrics" in interaction_state:
                    render_order_used, yaw_amount, yaw_sign, yaw_state_txt, yaw_debug = \
                        compute_render_order_from_yaw(stable_pts, interaction_state["body_metrics"], interaction_state)
                else:
                    render_order_used = FIXED_RENDER_ORDER
                _min_area = interaction_state.get('warp_min_area', 4.0)
                layered_layers = build_layered_body_layers_fast(frame_raw, V_track, V_def, T, binding,
                                                                  _min_area, 3.0, LAYER_MASK_DILATE_KSIZE, LAYER_MASK_BLUR_KSIZE)
                if stable_pts is not None and layered_layers is not None:
                    render_order_used, leg_overlap_frac, leg_front_score, leg_override_used = \
                        compute_render_order_with_leg_override(render_order_used, layered_layers, stable_pts, yaw_state_txt)
                    render_order_used, left_arm_overlap_frac, right_arm_overlap_frac, left_arm_score, right_arm_score, arm_override_used = \
                        compute_render_order_with_arm_override(render_order_used, layered_layers, stable_pts, yaw_state_txt)
                mesh_mask_u8 = build_active_mesh_mask_fast(h, w, V_def, T, active)
                base_hole_filled, _ = composite_mesh_with_background_holefill(
                    frame_raw, bg_plate, None, mesh_mask_u8, seg_mask, 0.50, 7, 5, False)
                warped_frame = composite_prebuilt_layers(base_hole_filled, layered_layers, render_order_used)
            else:
                mesh_mask_u8 = build_active_mesh_mask_fast(h, w, V_def, T, active)
                base_hole_filled, _ = composite_mesh_with_background_holefill(
                    frame_raw, bg_plate, None, mesh_mask_u8, seg_mask, 0.50, 7, 5, False)
                _min_area = interaction_state.get('warp_min_area', 4.0)
                warped_frame = warp_mesh_piecewise(frame_raw, V_track, V_def, T, active,
                                                    base_hole_filled.copy(), _min_area, 3.0)
        else:
            warped_frame = frame_raw.copy()

        affected_vertices  = interaction_state["drag_vertices"]  if interaction_state["dragging"] else interaction_state["preview_vertices"]
        affected_triangles = (interaction_state["drag_triangles"] & active) if interaction_state["dragging"] else interaction_state["preview_triangles"]
        vis = warped_frame.copy()

        if SHOW_RENDER_GROUP_DEBUG and "tri_render_group" in binding:
            vis = draw_triangle_render_group_overlay(vis, V_def, T, active, binding["tri_render_group"],
                                                      RENDER_GROUP_DEBUG_ALPHA, 1)
        if SHOW_MESH_OUTLINE:
            vis = TMh._draw_triangle_overlay(vis, V_def, T, active, affected_triangles,
                                              (0,0,0), (0,0,0), 0.12, 0.42, 1)
            if np.any(affected_vertices):
                for x, y in np.round(V_def[affected_vertices]).astype(np.int32):
                    cv2.circle(vis, (x, y), 3, (0,255,0), -1, cv2.LINE_AA)
            if hand_center is not None and not interaction_state["dragging"]:
                cx, cy = hand_center
                brush_color = (0,255,0) if (hand_is_open and hand_over_body) else (180,180,180)
                cv2.circle(vis, (cx,cy), brush_radius, brush_color, 2, cv2.LINE_AA)
                cv2.circle(vis, (cx,cy), 4, brush_color, -1, cv2.LINE_AA)
        else:
            if np.any(affected_triangles):
                vis = draw_filled_triangle_highlight(vis, V_def, T, affected_triangles,
                    fill_color=(255,150,246),
                    fill_alpha=0.65 if interaction_state["dragging"] else 0.14,
                    edge_color=(255,150,246), edge_thickness=1, edge_alpha=0.95)

        vis_display = cv2.resize(vis, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)

        if interaction_state.get("gesture_feedback_frames", 0) > 0:
            interaction_state["gesture_feedback_frames"] -= 1
            msg = interaction_state.get("gesture_feedback_msg", "")
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
            tx = (OUTPUT_W - tw) // 2
            ty = OUTPUT_H - 80
            cv2.putText(vis_display, msg, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)

        if hand_detected and hand_is_peace:
            prog = _peace_detector.progress()
            bar_w = int(OUTPUT_W * 0.5 * prog)
            bar_x = (OUTPUT_W - int(OUTPUT_W * 0.5)) // 2
            bar_y = OUTPUT_H - 130
            cv2.rectangle(vis_display,
                          (bar_x, bar_y), (bar_x + bar_w, bar_y + 18),
                          (0, 255, 255), -1)

        _hand_near_button = (
            _restart_tip_out is not None and
            (_rbx0 - 200) <= _restart_tip_out[0] <= (_rbx1 + 200) and
            (_rby0 - 200) <= _restart_tip_out[1] <= (_rby1 + 200)
        )
        _show_brush_circle = (
            hand_detected and hand_center is not None and
            (_pinch_tracker.is_active or _pinch_tracker.is_locked) and
            not _hand_near_button
        )
        if _show_brush_circle:
            scx = int(hand_center[0] * OUTPUT_W / w)
            scy = int(hand_center[1] * OUTPUT_H / h)
            scale_factor = OUTPUT_H / h
            display_r    = max(1, int(brush_radius * scale_factor))
            _br_overlay  = vis_display.copy()
            if _pinch_tracker.is_locked:
                cv2.circle(_br_overlay, (scx, scy), display_r,
                           (160, 160, 160), 1, cv2.LINE_AA)
                cv2.addWeighted(_br_overlay, 0.35, vis_display, 0.65, 0, vis_display)
            else:
                cv2.circle(_br_overlay, (scx, scy), display_r,
                           (255, 255, 255), 2, cv2.LINE_AA)
                cv2.addWeighted(_br_overlay, 0.70, vis_display, 0.30, 0, vis_display)
                _prog = _pinch_tracker.settling_progress
                if _prog > 0.0:
                    _angle = int(360 * _prog)
                    cv2.ellipse(vis_display, (scx, scy), (display_r, display_r),
                                -90, 0, _angle, (0, 255, 255), 3, cv2.LINE_AA)

        # ── Beauty-standard reference overlay ───────────────────────────
        if _bs_overlay is not None and not _timeout_fired:
            ox, oy   = _bs_overlay_x, _bs_overlay_y
            ow, oh_  = _bs_overlay.shape[1], _bs_overlay.shape[0]
            x0 = max(ox, 0);             y0 = max(oy, 0)
            x1 = min(ox + ow, OUTPUT_W); y1 = min(oy + oh_, OUTPUT_H)
            sx0 = x0 - ox;  sy0 = y0 - oy
            sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)
            if x1 > x0 and y1 > y0:
                roi    = vis_display[y0:y1, x0:x1]
                patch  = _bs_overlay[sy0:sy1, sx0:sx1]
                if patch.shape[2] == 4:
                    a      = patch[:,:,3:4].astype(np.float32) / 255.0
                    a     *= _BS_ALPHA
                    bgr    = patch[:,:,:3].astype(np.float32)
                else:
                    a      = np.full((y1-y0, x1-x0, 1), _BS_ALPHA, dtype=np.float32)
                    bgr    = patch.astype(np.float32)
                vis_display[y0:y1, x0:x1] = np.clip(
                    a * bgr + (1.0 - a) * roi.astype(np.float32), 0, 255
                ).astype(np.uint8)
        if _timeout_fired:
            scrim = vis_display.copy()
            scrim[:] = (0, 0, 0)
            vis_display = cv2.addWeighted(scrim, 0.45, vis_display, 0.55, 0)
            vis_display = _draw_timeout_message(vis_display, _timeout_new_bs)

        # ── Finish button ─────────────────────────────────────────────────
        # ── Finish button: point + pinch to click ──────────────────────────
        # ── Finish button: PNG states ────────────────────────────────────

        _BTN_PAD = 200

        _hovering_restart = (
            _restart_tip_out is not None and
            (_rbx0 - _BTN_PAD) <= _restart_tip_out[0] <= (_rbx1 + _BTN_PAD) and
            (_rby0 - _BTN_PAD) <= _restart_tip_out[1] <= (_rby1 + _BTN_PAD)
        )

        pinch_dist = hand_state.get("pinch_dist", 1.0)

        # ── Brush / Drag hover detection ────────────────────────────────
        _MODE_BTN_PAD = 40

        _hovering_brush = (
            _restart_tip_out is not None and
            (brush_x0 - _MODE_BTN_PAD) <= _restart_tip_out[0] <= (brush_x1 + _MODE_BTN_PAD) and
            (brush_y0 - _MODE_BTN_PAD) <= _restart_tip_out[1] <= (brush_y1 + _MODE_BTN_PAD)
        )

        _hovering_drag = (
            _restart_tip_out is not None and
            (drag_x0 - _MODE_BTN_PAD) <= _restart_tip_out[0] <= (drag_x1 + _MODE_BTN_PAD) and
            (drag_y0 - _MODE_BTN_PAD) <= _restart_tip_out[1] <= (drag_y1 + _MODE_BTN_PAD)
        )

        if _mode_click_cooldown > 0:
            _mode_click_cooldown -= 1

        mode_pinch_click = (
            hand_state.get("detected", False)
            and (_hovering_brush or _hovering_drag)
            and _mode_click_cooldown <= 0
            and _finish_prev_pinch_dist > _FINISH_PINCH_OPEN_DIST
            and pinch_dist < _FINISH_PINCH_CLOSED_DIST
            and (_finish_prev_pinch_dist - pinch_dist) > _FINISH_PINCH_DROP_MIN
            and not _timeout_fired
        )

        if mode_pinch_click:
            if _hovering_brush:
                interaction_mode = "brush"
            elif _hovering_drag:
                interaction_mode = "drag"
                _drag_ui_timer = _DRAG_UI_DURATION

            _mode_click_cooldown = _FINISH_CLICK_COOLDOWN

        pinch_click = (
            hand_state.get("detected", False)
            and _hovering_restart
            and _finish_pinch_click_cooldown <= 0
            and _finish_prev_pinch_dist > _FINISH_PINCH_OPEN_DIST
            and pinch_dist < _FINISH_PINCH_CLOSED_DIST
            and (_finish_prev_pinch_dist - pinch_dist) > _FINISH_PINCH_DROP_MIN
            and not _timeout_fired
        )

        if _finish_pinch_click_cooldown > 0:
            _finish_pinch_click_cooldown -= 1

        if pinch_click:
            _finish_pinch_click_cooldown = _FINISH_CLICK_COOLDOWN

        _finish_prev_pinch_dist = pinch_dist


        # ── Choose button state ──────────────────────────────────────────
        if not _timeout_fired:
            if pinch_click:
                btn_img = finish_btn_click
            elif _hovering_restart:
                btn_img = finish_btn_hover
            else:
                btn_img = finish_btn_normal


            # ── Draw button ──────────────────────────────────────────────────
            vis_display = overlay_bgra(vis_display, btn_img, _rbx0, _rby0)

            # ── Draw brush / drag buttons ───────────────────────────────────
            if interaction_mode == "brush":
                brush_img = brush_btn_click
            elif _hovering_brush:
                brush_img = brush_btn_hover
            else:
                brush_img = brush_btn_normal

            if interaction_mode == "drag":
                drag_img = drag_btn_click
            elif _hovering_drag:
                drag_img = drag_btn_hover
            else:
                drag_img = drag_btn_normal

            vis_display = overlay_bgra(vis_display, brush_img, brush_x0, brush_y0)
            vis_display = overlay_bgra(vis_display, drag_img, drag_x0, drag_y0)

            if _hovering_brush:
                vis_display = draw_pinch_hint_under_button(
                    vis_display, pinch_icon, brush_x0, brush_y1, _MODE_BTN_W
                )

            if _hovering_drag:
                vis_display = draw_pinch_hint_under_button(
                    vis_display, pinch_icon, drag_x0, drag_y1, _MODE_BTN_W
                )

            if interaction_mode == "brush":
                # ── MUCH bigger pinch icon ───────────────────────────────────
                icon_scale = 2.4   # ↑ bigger than before (was ~1.6)

                if pinch_icon is not None:
                    big_icon = cv2.resize(
                        pinch_icon,
                        (int(pinch_icon.shape[1] * icon_scale),
                        int(pinch_icon.shape[0] * icon_scale)),
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    big_icon = None

                # ── Position (left edge, slightly lower for breathing room) ──
                base_x = 30
                base_y = brush_y1 + 90   # push down a bit

                if big_icon is not None:
                    vis_display = overlay_bgra(vis_display, big_icon, base_x, base_y)

                # ── Larger, more spaced text ─────────────────────────────────
                lines = ["Pinch", "Unpinch", "Hold"]

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.05     # ↑ bigger text
                thickness = 3         # ↑ thicker for readability
                line_gap = 75         # ↑ much more spacing

                text_x = base_x
                text_y = base_y + (big_icon.shape[0] if big_icon is not None else 0) + 75

                for i, line in enumerate(lines):
                    y = text_y + i * line_gap

                    # shadow
                    cv2.putText(
                        vis_display,
                        line,
                        (text_x + 3, y + 3),
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness + 2,
                        cv2.LINE_AA
                    )

                    # main text
                    cv2.putText(
                        vis_display,
                        line,
                        (text_x, y),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA
                    )

            if interaction_mode == "drag" and _drag_ui_timer > 0:
                # ── Fade factor ─────────────────────────────────────────────
                alpha = (_drag_ui_timer / _DRAG_UI_DURATION) ** 1.5

                drag_steps = [
                    (open_palm_icon, "Hover over your body"),
                    (fist_icon, "Slowly make a fist"),
                    (fist_drag_icon, "Slowly drag away"),
                    (open_palm_icon, "Let go"),
                ]

                icon_h = 95
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.9
                thickness = 3

                base_x = 30
                base_y = brush_y1 + 90
                row_gap = 95
                icon_text_gap = 22

                for i, (icon, text) in enumerate(drag_steps):
                    y = base_y + i * row_gap

                    if icon is not None:
                        scale = icon_h / icon.shape[0]
                        icon_resized = cv2.resize(
                            icon,
                            (int(icon.shape[1] * scale), icon_h),
                            interpolation=cv2.INTER_AREA
                        )

                        # ── Apply fade to icon ───────────────────────────────
                        if icon_resized.shape[2] == 4:
                            icon_resized = icon_resized.copy()
                            icon_resized[:, :, 3] = (
                                icon_resized[:, :, 3].astype(np.float32) * alpha
                            ).astype(np.uint8)

                        vis_display = overlay_bgra(vis_display, icon_resized, base_x, y)
                        text_x = base_x + icon_resized.shape[1] + icon_text_gap
                    else:
                        text_x = base_x

                    text_y = y + icon_h // 2 + 12

                    # ── Text colors with fade ───────────────────────────────
                    text_color = (
                        int(255 * alpha),
                        int(255 * alpha),
                        int(255 * alpha)
                    )

                    shadow_color = (
                        int(0 * alpha),
                        int(0 * alpha),
                        int(0 * alpha)
                    )

                    # shadow
                    cv2.putText(
                        vis_display,
                        text,
                        (text_x + 3, text_y + 3),
                        font,
                        font_scale,
                        shadow_color,
                        thickness + 2,
                        cv2.LINE_AA
                    )

                    # main text
                    cv2.putText(
                        vis_display,
                        text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        text_color,
                        thickness,
                        cv2.LINE_AA
                    )

            if _hovering_restart:
                hint_text = "Pinch to click"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2

                (tw, th), baseline = cv2.getTextSize(hint_text, font, font_scale, thickness)

                gap = 10
                icon_gap = 8

                icon_w = pinch_icon.shape[1] if pinch_icon is not None else 0
                icon_h = pinch_icon.shape[0] if pinch_icon is not None else 0

                total_w = icon_w + icon_gap + tw

                group_x = _rbx0 + (_RESTART_W - total_w) // 2
                group_y = _rby1 + gap

                icon_x = group_x
                icon_y = group_y

                text_x = group_x + icon_w + icon_gap
                text_y = group_y + (icon_h + th) // 2

                if pinch_icon is not None:
                    vis_display = overlay_bgra(vis_display, pinch_icon, icon_x, icon_y)

                cv2.putText(
                    vis_display,
                    hint_text,
                    (text_x + 2, text_y + 2),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness + 1,
                    cv2.LINE_AA
                )

                cv2.putText(
                    vis_display,
                    hint_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )


            # ── Optional pointer visual ──────────────────────────────────────
            if _restart_tip_out is not None and _hand_near_button:
                cv2.circle(vis_display, _restart_tip_out, 14,
                        (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(vis_display, _restart_tip_out, 14,
                        (180, 180, 255), 2, cv2.LINE_AA)


        # ── Trigger action ───────────────────────────────────────────────
        if pinch_click:
            print("Finish triggered — returning to welcome screen")
            return "restart"
        
        if _drag_ui_timer > 0:
            _drag_ui_timer -= 1

        cv2.imshow(window_name, vis_display)

        if show_mask:
            cv2.imshow("Segmentation mask",
                       cv2.resize((seg_mask*255).astype(np.uint8), (OUTPUT_W,OUTPUT_H), interpolation=cv2.INTER_NEAREST))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('m'): SHOW_MESH_OUTLINE = not SHOW_MESH_OUTLINE
        elif key == ord('r'):
            binding["vertex_local_uv_offset"][:] = 0.0
            interaction_state["preview_vertices"][:]  = False
            interaction_state["preview_triangles"][:] = False
            interaction_state["drag_vertices"][:]  = False
            interaction_state["drag_triangles"][:] = False
            interaction_state["dragging"] = False
            interaction_state["prev_hand_center"] = None
            interaction_state["hand_was_open"] = False
        elif key in (ord('+'), ord('=')):
            interaction_state['warp_min_area'] = min(
                interaction_state.get('warp_min_area', 4.0) + 2.0, 30.0)
            print(f"warp_min_area -> {interaction_state['warp_min_area']:.1f}")
        elif key in (ord('-'), ord('_')):
            interaction_state['warp_min_area'] = max(
                interaction_state.get('warp_min_area', 4.0) - 2.0, 1.0)
            print(f"warp_min_area -> {interaction_state['warp_min_area']:.1f}")

    print("Exiting run_hand_brush_drag_arap_loop_skeleton")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def test_hand_brush_drag_arap_live_skeleton(step=0, thresh=0.5, feather=0, show_mask=False):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    if not cap.isOpened():
        print("Error: could not open camera."); return
    print("Camera found")
    ok, frame = cap.read()
    if not ok:
        print("Error: could not read initial frame."); cap.release(); return
    frame = rotate_frame(frame)
    h, w = frame.shape[:2]

    V_base, T, nx, ny = TMh.build_grid_mesh(w, h, step=step)
    V_def = V_base.copy().astype(np.float32)
    E = TMh.build_unique_edges(T)
    neighbors = TMh.build_vertex_neighbors(len(V_base), E)
    arap_cache = {"E": E, "neighbors": neighbors}
    brush_radius = max(45, int(min(w, h) * 0.07))

    interaction_state = {
        "preview_vertices":  np.zeros(len(V_def), dtype=bool),
        "preview_triangles": np.zeros(len(T),     dtype=bool),
        "drag_vertices":     np.zeros(len(V_def), dtype=bool),
        "drag_triangles":    np.zeros(len(T),     dtype=bool),
        "dragging": False, "prev_hand_center": None, "hand_was_open": False,
        "pose_prev_pts": None, "pose_last_good_pts": None, "pose_missing_count": 0,
        "binding": None, "yaw_side_state": "frontal", "yaw_sign_value_smooth": 0.0,
        "bg_plate": None,
        "cached_frame_arrays": None, "cached_frames": None,
        "warp_min_area": 4.0,
        "active_hand": None,
        "gesture_feedback_frames": 0,
        "gesture_feedback_msg": "",
        "brush_radius": brush_radius,
    }

    window_name = "Hand brush drag on skeleton mesh"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, PRIMARY_MONITOR_WIDTH, 0)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # ── CHANGE 2: load initialization pose image once, outside the loop ───
    _init_pose_img = None
    if os.path.exists(_INIT_POSE_PATH):
        _init_pose_raw = cv2.imread(_INIT_POSE_PATH, cv2.IMREAD_UNCHANGED)
        if _init_pose_raw is not None:
            # Scale it the same way as BS images: match person height.
            # At this point we don't know person_height_px yet, so we scale
            # to fill OUTPUT_H * 0.80 as a reasonable default — it will look
            # the same every session since the pose image is fixed.
            _ip_src_h = _init_pose_raw.shape[0]
            _ip_target_h = int(OUTPUT_H * 0.90)
            _ip_target_w = int(_init_pose_raw.shape[1] * _ip_target_h / _ip_src_h)
            _init_pose_img = cv2.resize(_init_pose_raw,
                                        (_ip_target_w, _ip_target_h),
                                        interpolation=cv2.INTER_AREA)
            print(f"Initialization pose image loaded: {_ip_target_w}x{_ip_target_h}")
        else:
            print(f"WARNING: could not read init pose image at {_INIT_POSE_PATH}")
    else:
        print(f"WARNING: init pose image not found at {_INIT_POSE_PATH}")

    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter, \
         mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
         mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=False,
                      enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

      while True:  # ── Outer restart loop ──────────────────────────────
        _peace_detector.reset()
        _pinch_tracker.reset()

        show_home_screen(window_name, cap, pose)

        _sel_bg = cv2.imread(WELCOME_BG_PATH)
        if _sel_bg is None:
            _sel_bg = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)
            _sel_bg[:, :] = (60, 20, 10)
        else:
            _sel_bg = cv2.resize(_sel_bg, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)

        selected_bs = show_selection_screen(window_name, _sel_bg, cap)
        if selected_bs is not None:
            print(f"Selected beauty standard: {selected_bs}")
            interaction_state["selected_beauty_standard"] = selected_bs
        else:
            print("Beauty standard selection skipped.")
            interaction_state["selected_beauty_standard"] = None

        inference_thread = PoseInferenceThread(pose=pose, hands=hands, segmenter=segmenter,
                                                feather=feather, thresh=thresh, w=w, h=h)

        _BG_DIR  = "background_captures"
        _BG_PATH = os.path.join(_BG_DIR, "background.png")

        if CAPTURE_BACKGROUND_BEFORE_INIT:
            show_countdown(cap, h, w, 5, "Capturing background", window_name)
            bg_plate = capture_background_plate(cap, h, w, n_frames=20, window_name=window_name)
            if bg_plate is None:
                print("Could not capture background plate.")
                inference_thread.stop(); cap.release(); cv2.destroyAllWindows(); return
            os.makedirs(_BG_DIR, exist_ok=True)
            cv2.imwrite(_BG_PATH, bg_plate)
            print(f"Background plate saved to {_BG_PATH}")
            interaction_state["bg_plate"] = bg_plate
        else:
            if os.path.exists(_BG_PATH):
                bg_plate = cv2.imread(_BG_PATH)
                if bg_plate is not None:
                    if bg_plate.shape[:2] != (h, w):
                        bg_plate = cv2.resize(bg_plate, (w, h), interpolation=cv2.INTER_LINEAR)
                    interaction_state["bg_plate"] = bg_plate
                    print(f"Background plate loaded from {_BG_PATH}")
                else:
                    print(f"WARNING: could not read {_BG_PATH} — hole-fill disabled.")
                    interaction_state["bg_plate"] = None
            else:
                print(f"WARNING: no background file at {_BG_PATH} — hole-fill disabled.")
                print("Set CAPTURE_BACKGROUND_BEFORE_INIT=True once to create it.")
                interaction_state["bg_plate"] = None

        show_countdown(cap, h, w, 5, "Please match the pose displayed on the screen.\n\nInitialization starting", window_name, bs_img=_init_pose_img)
        print("Entering initialization loop")

        initialized=False; max_init_frames=300; full_pose_streak=0; required_streak=20
        mask_accum=None; mask_count=0; best_stable_pts=None

        # ── CHANGE 3: instruction line shown above countdown during init ──
        _INIT_INSTRUCTION = "Hold still."

        for k in range(max_init_frames):
            ok, fr = cap.read()
            if not ok:
                print(f"init frame {k}: camera read failed"); break
            fr = rotate_frame(fr)
            if fr.shape[:2] != (h, w): fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_LINEAR)
            fr = cv2.flip(fr, 1)
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            cur_pts = extract_pose_points(rgb, pose, w, h, min_vis=0.25)
            stable_pts = smooth_pose_points(cur_pts, interaction_state, alpha=0.65, max_jump_px=45.0, hold_frames=2)
            full_pose_ok = has_required_landmarks(stable_pts)

            if full_pose_ok:
                full_pose_streak += 1
                seg_mask_init, _ = PTh.get_segmentation_mask(fr, segmenter, feather=feather)
                if mask_accum is None: mask_accum = np.zeros_like(seg_mask_init, dtype=np.float32)
                mask_accum += seg_mask_init.astype(np.float32); mask_count += 1; best_stable_pts = stable_pts
            else:
                full_pose_streak=0; mask_accum=None; mask_count=0; best_stable_pts=None

            init_vis = fr.copy()
            # ── CHANGE 1: no frame counter in status msg ──────────────────
            status_msg = f"Waiting for full-body pose..."

            if full_pose_streak >= required_streak and mask_count > 0 and best_stable_pts is not None:
                agg_mask = finalize_init_mask(mask_accum, mask_count, avg_thresh=0.28,
                                               dilate_ksize=13, close_ksize=11)
                pose_capsule_mask = rasterize_pose_capsules(agg_mask.shape, best_stable_pts)
                init_mask_final   = np.maximum(agg_mask, pose_capsule_mask)
                init_mask_final   = trim_mask_arms_combined(init_mask_final, best_stable_pts,
                                                             min_component_area=250)

                print("Building adaptive body mesh...")
                V_base_adaptive, T_adaptive, active_adaptive = build_adaptive_body_mesh(
                    w=w, h=h,
                    body_mask=init_mask_final,
                    interior_step=max(step * 2, 50),
                    contour_step=10,
                    contour_inset=3,
                    min_contour_pts=60,
                )

                E_new = TMh.build_unique_edges(T_adaptive)
                nbrs_new = TMh.build_vertex_neighbors(len(V_base_adaptive), E_new)
                arap_cache = {"E": E_new, "neighbors": nbrs_new}

                V_base = V_base_adaptive.copy().astype(np.float32)
                T      = T_adaptive
                V_def  = V_base.copy().astype(np.float32)

                interaction_state["preview_vertices"]  = np.zeros(len(V_def), dtype=bool)
                interaction_state["preview_triangles"] = np.zeros(len(T),     dtype=bool)
                interaction_state["drag_vertices"]     = np.zeros(len(V_def), dtype=bool)
                interaction_state["drag_triangles"]    = np.zeros(len(T),     dtype=bool)

                binding = bind_mesh_to_skeleton(
                    V_base=V_base, T=T,
                    ref_pts=best_stable_pts,
                    init_seg_mask=init_mask_final,
                    mask_thresh=0.5,
                )

                print(f"init frame {k}: binding is {'ok' if binding is not None else 'None'}")
                if binding is not None:
                    tri_render_group = build_triangle_render_groups(binding, T)
                    binding["tri_render_group"]  = tri_render_group
                    binding["group_tri_indices"] = build_render_group_triangle_index_cache(binding)
                    print_render_group_triangle_stats(binding)
                    V_def  = V_base.copy().astype(np.float32)
                    interaction_state["binding"]      = binding
                    interaction_state["body_metrics"] = compute_body_metrics(best_stable_pts)
                    _nose  = best_stable_pts.get("nose")
                    _lankl = best_stable_pts.get("left_ankle")
                    _rankl = best_stable_pts.get("right_ankle")
                    if _nose is not None and (_lankl is not None or _rankl is not None):
                        _ankl = _lankl if _lankl is not None else _rankl
                        if _lankl is not None and _rankl is not None:
                            _ankl = 0.5 * (_lankl + _rankl)
                        interaction_state["person_height_px"] = float(abs(_ankl[1] - _nose[1]))
                    else:
                        interaction_state["person_height_px"] = float(h) * 0.75
                    initialized = True
                    print("Initialization succeeded")
                    status_msg = "Pose locked. Starting..."

            if stable_pts is not None:
                for name, p in stable_pts.items():
                    if p is not None: cv2.circle(init_vis, (int(p[0]),int(p[1])), 4, (0,255,0), -1, cv2.LINE_AA)

            _init_display = cv2.resize(init_vis, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)

            # ── CHANGE 2: composite the init pose image (same as BS overlay) ─
            _init_display = _composite_init_pose_overlay(_init_display, _init_pose_img, _INIT_POSE_ALPHA)

            # ── CHANGE 3: two-line text (instruction above, status below) ───
            if initialized:
                # Success — single line is fine
                _init_display = _draw_countdown_text(_init_display, status_msg, OUTPUT_W, OUTPUT_H)
            else:
                _init_display = _draw_two_line_countdown_text(
                    _init_display,
                    _INIT_INSTRUCTION,
                    status_msg,
                    OUTPUT_W, OUTPUT_H,
                )

            cv2.imshow(window_name, _init_display)
            key = cv2.waitKey(1) & 0xFF
            if initialized: break
            if key == ord('q'):
                print("User quit during initialization"); break

        print(f"Init loop done. initialized={initialized}")
        if not initialized:
            print("Could not initialize.")
            inference_thread.stop(); cap.release(); cv2.destroyAllWindows(); return

        print("About to enter main skeleton loop")
        _loop_result = run_hand_brush_drag_arap_loop_skeleton(
            cap=cap, segmenter=segmenter, hands=hands, pose=pose,
            h=h, w=w, V_base=V_base, V_def=V_def, T=T,
            step=step, thresh=thresh, feather=feather, show_mask=show_mask,
            brush_radius=brush_radius, interaction_state=interaction_state,
            arap_cache=arap_cache, inference_thread=inference_thread,
            window_name=window_name,
        )
        print(f"Main skeleton loop returned: {_loop_result}")
        inference_thread.stop()

        if _loop_result != "restart":
            break

        print("Restarting session...")
        brush_radius = max(45, int(min(w, h) * 0.07))
        interaction_state.update({
            "preview_vertices":  np.zeros(len(V_def), dtype=bool),
            "preview_triangles": np.zeros(len(T),     dtype=bool),
            "drag_vertices":     np.zeros(len(V_def), dtype=bool),
            "drag_triangles":    np.zeros(len(T),     dtype=bool),
            "dragging": False, "prev_hand_center": None, "hand_was_open": False,
            "pose_prev_pts": None, "pose_last_good_pts": None,
            "pose_missing_count": 0,
            "binding": None, "yaw_side_state": "frontal",
            "yaw_sign_value_smooth": 0.0, "bg_plate": None,
            "cached_frame_arrays": None, "cached_frames": None,
            "warp_min_area": 4.0, "active_hand": None,
            "gesture_feedback_frames": 0, "gesture_feedback_msg": "",
            "brush_radius": brush_radius,
        })
        V_base, T, _, _ = TMh.build_grid_mesh(w, h, step=step)
        V_def = V_base.copy().astype(np.float32)
        E = TMh.build_unique_edges(T)
        arap_cache = {"E": E, "neighbors": TMh.build_vertex_neighbors(len(V_base), E)}

    cap.release()
    cv2.destroyAllWindows()


test_hand_brush_drag_arap_live_skeleton(step=25, thresh=0.5, feather=0, show_mask=False)