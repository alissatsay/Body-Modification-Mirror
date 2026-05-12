import os
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
from scipy.spatial import Delaunay

mp_pose = mp.solutions.pose

import Triangle_Mesh_helpers as TMh
import Pose_Tracking_helpers as PTh

# Gesture tracker objects — created once, live for the whole session
_peace_detector = PTh.PeaceSignHoldDetector()
_pinch_tracker  = PTh.PinchGestureTracker()
import Loop_helpers as Lh
import Display_helpres as Dh

ROTATE_DEG = 90
ROTATE_DIR = "ccw"

OUTPUT_W = 1080
OUTPUT_H = 1920

PRIMARY_MONITOR_WIDTH = 1920

SEG_EVERY_N = 2

# Set to True to capture a clean background plate before initialization.
# Set to False to skip background capture and go straight to pose init.
# Background capture improves hole-fill quality but requires the person
# to step out of frame for ~5 seconds before the session starts.
CAPTURE_BACKGROUND_BEFORE_INIT = True

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
BRUSH_RADIUS_MIN = 20
BRUSH_RADIUS_MAX = 300
# Active hand: 'Left', 'Right', or None (= accept either)
# Swirl CCW → toggle to other hand; swirl CW → same
# (We store the mirrored label that matches what arrives from MediaPipe
#  after the left/right swap done in the main loop.)

LEG_OVERLAP_TRIGGER_FRAC = 0.015
LEG_FRONT_SCORE_DEADBAND = 8.0
ARM_OVERLAP_TRIGGER_FRAC = 0.020
ARM_FRONT_SCORE_DEADBAND = 6.0


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
    """
    Build a body-conforming triangle mesh using Delaunay triangulation.

    Strategy:
      1. Sample points densely along the body contour  — drives boundary quality
      2. Sample points on a coarse grid inside the body — drives interior coverage
      3. Add frame corner/edge points                   — keeps mesh well-conditioned
      4. Delaunay triangulate all points together
      5. Classify triangles as active (centroid inside body) or inactive

    Parameters
    ----------
    w, h            : frame width and height
    body_mask       : float32 (H, W) in [0,1], from finalize_init_mask
    interior_step   : grid spacing inside body (px). Use 50-80 for speed — interior
                      triangles don't need to be small; the warp is smooth there.
    contour_step    : sample every Nth contour pixel. 8-12 gives good boundary quality.
    contour_inset   : pull contour points slightly inward (px) so centroids land inside
    min_contour_pts : floor on contour sample count regardless of contour_step

    Returns
    -------
    V      : (N, 2) float32 vertex positions
    T      : (M, 3) int32  triangle indices
    active : (M,)   bool   True where triangle centroid is inside body
    """
    H, W = h, w
    mask_u8 = (body_mask >= 0.5).astype(np.uint8)

    # ── 1. Contour points ────────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contour_pts = []
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        pts_raw = main_contour[:, 0, :].astype(np.float32)   # (N, 2) as (x, y)

        n_contour = len(pts_raw)
        # pick step so we get at least min_contour_pts samples
        step_px = max(1, min(contour_step, n_contour // max(min_contour_pts, 1)))
        sampled = pts_raw[::step_px]

        # Inset toward centroid so the points land just inside the mask
        if contour_inset > 0 and len(sampled) >= 3:
            cx = float(np.mean(sampled[:, 0]))
            cy = float(np.mean(sampled[:, 1]))
            d = sampled - np.array([cx, cy], dtype=np.float32)
            norms = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
            sampled = sampled - contour_inset * (d / norms)
            # Clamp to frame
            sampled[:, 0] = np.clip(sampled[:, 0], 0, W - 1)
            sampled[:, 1] = np.clip(sampled[:, 1], 0, H - 1)

        contour_pts = sampled.tolist()

    # ── 2. Interior grid points (inside mask only) ───────────────────────────
    interior_pts = []
    for y in range(0, H, interior_step):
        for x in range(0, W, interior_step):
            if mask_u8[y, x] > 0:
                interior_pts.append([float(x), float(y)])

    # ── 3. Frame boundary points ─────────────────────────────────────────────
    # Keeps the triangulation from having huge degenerate triangles at the edges
    frame_pts = []
    for x in range(0, W + 1, interior_step):
        xc = float(min(x, W - 1))
        frame_pts.append([xc, 0.0])
        frame_pts.append([xc, float(H - 1)])
    for y in range(interior_step, H, interior_step):
        frame_pts.append([0.0, float(y)])
        frame_pts.append([float(W - 1), float(y)])
    # Always include the four corners
    frame_pts += [[0.0, 0.0], [float(W-1), 0.0],
                  [0.0, float(H-1)], [float(W-1), float(H-1)]]

    # ── 4. Combine & deduplicate within 2 px ────────────────────────────────
    all_pts = np.array(contour_pts + interior_pts + frame_pts, dtype=np.float32)

    if len(all_pts) < 3:
        # Fallback: uniform grid
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

    # ── 5. Delaunay triangulation ────────────────────────────────────────────
    tri = Delaunay(all_pts)
    T   = tri.simplices.astype(np.int32)
    V   = all_pts.astype(np.float32)

    # ── 6. Classify triangles by centroid ───────────────────────────────────
    centroids = (V[T[:, 0]] + V[T[:, 1]] + V[T[:, 2]]) / 3.0
    cx = np.clip(np.round(centroids[:, 0]).astype(np.int32), 0, W - 1)
    cy = np.clip(np.round(centroids[:, 1]).astype(np.int32), 0, H - 1)
    active = mask_u8[cy, cx] > 0

    print(f"build_adaptive_body_mesh: {len(V)} vertices, {len(T)} triangles, "
          f"{int(active.sum())} active ({len(contour_pts)} contour pts, "
          f"{len(interior_pts)} interior pts)")

    return V, T, active


# ══════════════════════════════════════════════════════════════════════════════
# FIX 4 — Background inference thread
# ══════════════════════════════════════════════════════════════════════════════

class PoseInferenceThread:
    """
    Runs MediaPipe segmentation + pose + hands inference on a background thread.
    The render loop reads .latest_result() without blocking.
    """

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
# FIX 1 — Vectorized mesh reconstruction helpers
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


# ══════════════════════════════════════════════════════════════════════════════
# FIX 2 — Frame array cache helper
# ══════════════════════════════════════════════════════════════════════════════

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
# Unchanged helpers
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
        vis = fr.copy()
        cv2.putText(vis, f"Capturing empty background... {k+1}/{n_frames}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, cv2.resize(vis, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if acc is None or got == 0:
        return None
    return np.clip(acc / float(got), 0, 255).astype(np.uint8)


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
        elapsed = time.time() - start_time
        remaining = int(np.ceil(seconds - elapsed))
        if remaining <= 0:
            break
        vis = fr.copy()
        text = f"{message} in {remaining}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.putText(vis, text, ((w - tw) // 2, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
        vis_display = cv2.resize(vis, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)
        if bs_img is not None:
            oe = 0.0 if overlay_timer_start is None else time.time() - overlay_timer_start
            vis_display = _draw_bs_timer_overlay(vis_display, bs_img, bs_size, bs_margin, oe, overlay_timer_duration)
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
    if state == "left_front":   render_order = LEFT_SIDE_FRONT_RENDER_ORDER;  yaw_state_txt = "left side front"
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
    """Original — kept for compatibility."""
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
    """Original — kept for compatibility."""
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
    """Original — kept for compatibility."""
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
    import os
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
    global SHOW_MESH_OUTLINE
    print("Entered run_hand_brush_drag_arap_loop_skeleton")
    binding = interaction_state["binding"]
    _BS_SIZE=240; _BS_MARGIN=18; _TIMER_DURATION=180.0; _timer_start=time.time()
    _bs_img_raw, _bs_img_path = load_beauty_standard_overlay(size=_BS_SIZE)
    if _bs_img_raw is None:
        print(f"WARNING: could not load beauty standard image: {_bs_img_path}")

    while True:
        # Read mutable brush radius from state (updated by pinch gesture)
        brush_radius = interaction_state.get('brush_radius', brush_radius)

        ok, frame_raw = cap.read()
        if not ok:
            print("Main loop: cap.read() failed, breaking"); break
        frame_raw = rotate_frame(frame_raw)
        if frame_raw.shape[:2] != (h, w):
            frame_raw = cv2.resize(frame_raw, (w, h), interpolation=cv2.INTER_LINEAR)
        frame_raw = cv2.flip(frame_raw, 1)

        # Fix 4: background thread
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

        # Fix 1+2: vectorized mesh with caching
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
        # Mirror handedness label to match on-screen left/right
        if hand_handedness_raw == "Left":     hand_handedness = "Right"
        elif hand_handedness_raw == "Right":  hand_handedness = "Left"
        else:                                  hand_handedness = None

        # ── Active hand filter ───────────────────────────────────────────
        active_hand  = interaction_state.get("active_hand", None)
        hand_allowed = (
            not hand_detected
            or active_hand is None
            or hand_handedness == active_hand
            or hand_handedness is None
        )

        # ── Peace-sign gesture: cycle active hand ─────────────────────────
        # Hold a peace / V sign for ~0.7 s to cycle:
        #   None -> this hand -> None -> ...
        hand_is_peace = hand_state.get("is_peace", False)
        if hand_detected:
            peace_fired = _peace_detector.update_raw(hand_is_peace)
            if peace_fired:
                cur = interaction_state.get("active_hand", None)
                if cur is None:
                    nxt = hand_handedness   # lock to the hand making the sign
                else:
                    nxt = None             # any second sign unlocks
                interaction_state["active_hand"] = nxt
                label = nxt if nxt is not None else "Either hand"
                interaction_state["gesture_feedback_msg"]    = f"Active hand: {label}"
                interaction_state["gesture_feedback_frames"] = 90
                print(f"Peace sign: active hand -> {nxt}")
        else:
            _peace_detector.reset()

        # ── Pinch gesture: resize brush ───────────────────────────────────
        if hand_detected:
            pinch_delta, is_pinching = _pinch_tracker.update_from_dist(hand_pinch_dist)
            if is_pinching:
                old_r = interaction_state.get("brush_radius", brush_radius)
                new_r = int(np.clip(old_r + pinch_delta, BRUSH_RADIUS_MIN, BRUSH_RADIUS_MAX))
                if new_r != old_r:
                    interaction_state["brush_radius"] = new_r
                    brush_radius = new_r
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

        # Build output
        layered_layers=None; render_order_used=FIXED_RENDER_ORDER
        yaw_amount=0.0; yaw_sign=0.0; yaw_state_txt="frontal"
        yaw_debug={"sign_value":0.0,"sign_value_smooth":0.0,"width_ratio":1.0}
        leg_overlap_frac=0.0; leg_front_score=0.0; leg_override_used=False
        left_arm_overlap_frac=0.0; right_arm_overlap_frac=0.0
        left_arm_score=0.0; right_arm_score=0.0; arm_override_used=False
        bg_plate = interaction_state.get("bg_plate")

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

        # Visualization
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
        if _bs_img_raw is not None:
            vis_display = _draw_bs_timer_overlay(vis_display, _bs_img_raw, _BS_SIZE, _BS_MARGIN,
                                                   time.time()-_timer_start, _TIMER_DURATION)
        # ── Gesture feedback text ────────────────────────────────────────
        if interaction_state.get("gesture_feedback_frames", 0) > 0:
            interaction_state["gesture_feedback_frames"] -= 1
            msg = interaction_state.get("gesture_feedback_msg", "")
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
            tx = (OUTPUT_W - tw) // 2
            ty = OUTPUT_H - 80
            cv2.putText(vis_display, msg, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)

        # ── Peace sign hold progress bar ──────────────────────────────────
        if hand_detected and hand_is_peace:
            prog = _peace_detector.progress()
            bar_w = int(OUTPUT_W * 0.5 * prog)
            bar_x = (OUTPUT_W - int(OUTPUT_W * 0.5)) // 2
            bar_y = OUTPUT_H - 130
            cv2.rectangle(vis_display,
                          (bar_x, bar_y), (bar_x + bar_w, bar_y + 18),
                          (0, 255, 255), -1)

        # ── Brush radius circle + settling feedback ───────────────────────
        # Visible while pinch gesture is active (moving/settling) or locked.
        _show_brush_circle = (
            hand_detected and hand_center is not None and
            (_pinch_tracker.is_active or _pinch_tracker.is_locked)
        )
        if _show_brush_circle:
            scx = int(hand_center[0] * OUTPUT_W / w)
            scy = int(hand_center[1] * OUTPUT_H / h)
            scale_factor = OUTPUT_H / h
            display_r    = max(1, int(brush_radius * scale_factor))
            _br_overlay  = vis_display.copy()
            if _pinch_tracker.is_locked:
                # Locked: dim grey circle, confirms size is committed
                cv2.circle(_br_overlay, (scx, scy), display_r,
                           (160, 160, 160), 1, cv2.LINE_AA)
                cv2.addWeighted(_br_overlay, 0.35, vis_display, 0.65, 0, vis_display)
            else:
                # Active (moving or settling): bright white circle
                cv2.circle(_br_overlay, (scx, scy), display_r,
                           (255, 255, 255), 2, cv2.LINE_AA)
                cv2.addWeighted(_br_overlay, 0.70, vis_display, 0.30, 0, vis_display)
                # Settling progress arc — cyan arc that fills as fingers go still
                _prog = _pinch_tracker.settling_progress
                if _prog > 0.0:
                    _angle = int(360 * _prog)
                    cv2.ellipse(vis_display, (scx, scy), (display_r, display_r),
                                -90, 0, _angle, (0, 255, 255), 3, cv2.LINE_AA)

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

        # Runtime warp-quality controls (no mesh rebuild needed)
        # + / =  raise min_area threshold -> skip more tiny triangles -> faster
        # - / _  lower min_area threshold -> warp more triangles    -> sharper
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
    if not cap.isOpened():
        print("Error: could not open camera."); return
    print("Camera found")
    ok, frame = cap.read()
    if not ok:
        print("Error: could not read initial frame."); cap.release(); return
    frame = rotate_frame(frame)
    h, w = frame.shape[:2]

    # Initial placeholder mesh (replaced after body mask is ready)
    V_base, T, nx, ny = TMh.build_grid_mesh(w, h, step=step)
    V_def = V_base.copy().astype(np.float32)
    E = TMh.build_unique_edges(T)
    neighbors = TMh.build_vertex_neighbors(len(V_base), E)
    arap_cache = {"E": E, "neighbors": neighbors}
    brush_radius = max(45, int(min(w, h) * 0.07))
    # Store in interaction_state so gesture handlers can update it

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
        "warp_min_area": 4.0,  # +/- keys raise/lower at runtime
        # Gesture state
        "active_hand": None,         # None = either hand, "Left"/"Right" = locked
        "gesture_feedback_frames": 0,# how many frames to show gesture feedback text
        "gesture_feedback_msg": "",  # text to flash on screen
        "brush_radius": brush_radius,  # mutable, changed by pinch gesture
    }

    window_name = "Hand brush drag on skeleton mesh"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, PRIMARY_MONITOR_WIDTH, 0)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter, \
         mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
         mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=False,
                      enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

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
            # Save to disk (create folder if needed, overwrite if exists)
            os.makedirs(_BG_DIR, exist_ok=True)
            cv2.imwrite(_BG_PATH, bg_plate)
            print(f"Background plate saved to {_BG_PATH}")
            interaction_state["bg_plate"] = bg_plate
        else:
            # Load previously saved background from disk
            if os.path.exists(_BG_PATH):
                bg_plate = cv2.imread(_BG_PATH)
                if bg_plate is not None:
                    # Resize to match current camera resolution if needed
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

        show_countdown(cap, h, w, 5, "Initialization starting", window_name)
        print("Entering initialization loop")

        initialized=False; max_init_frames=300; full_pose_streak=0; required_streak=20
        mask_accum=None; mask_count=0; best_stable_pts=None

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
            msg   = f"Waiting for full-body pose... streak={full_pose_streak}/{required_streak}"
            color = (0, 255, 255)

            if full_pose_streak >= required_streak and mask_count > 0 and best_stable_pts is not None:
                agg_mask = finalize_init_mask(mask_accum, mask_count, avg_thresh=0.28,
                                               dilate_ksize=13, close_ksize=11)
                pose_capsule_mask = rasterize_pose_capsules(agg_mask.shape, best_stable_pts)
                init_mask_final   = np.maximum(agg_mask, pose_capsule_mask)
                init_mask_final   = trim_mask_arms_combined(init_mask_final, best_stable_pts,
                                                             min_component_area=250)

                # ── Build adaptive Delaunay mesh from the body silhouette ──────
                print("Building adaptive body mesh...")
                # interior_step: coarse interior = fewer triangles = fast warp
                # contour_step:  fine boundary = good silhouette, independently
                V_base_adaptive, T_adaptive, active_adaptive = build_adaptive_body_mesh(
                    w=w, h=h,
                    body_mask=init_mask_final,
                    interior_step=max(step * 2, 50),  # coarse interior — fast
                    contour_step=10,                   # ~10px spacing on boundary
                    contour_inset=3,
                    min_contour_pts=60,
                )

                # Rebuild ARAP cache for the new mesh topology
                E_new = TMh.build_unique_edges(T_adaptive)
                nbrs_new = TMh.build_vertex_neighbors(len(V_base_adaptive), E_new)
                arap_cache = {"E": E_new, "neighbors": nbrs_new}

                # Update working mesh variables
                V_base = V_base_adaptive.copy().astype(np.float32)
                T      = T_adaptive
                V_def  = V_base.copy().astype(np.float32)

                # Resize interaction_state boolean arrays to match new mesh size
                interaction_state["preview_vertices"]  = np.zeros(len(V_def), dtype=bool)
                interaction_state["preview_triangles"] = np.zeros(len(T),     dtype=bool)
                interaction_state["drag_vertices"]     = np.zeros(len(V_def), dtype=bool)
                interaction_state["drag_triangles"]    = np.zeros(len(T),     dtype=bool)
                # ─────────────────────────────────────────────────────────────

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
                    initialized = True
                    print("Initialization succeeded")
                    msg = "Pose locked. Starting..."; color = (0, 255, 0)

            cv2.putText(init_vis, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            if stable_pts is not None:
                for name, p in stable_pts.items():
                    if p is not None: cv2.circle(init_vis, (int(p[0]),int(p[1])), 4, (0,255,0), -1, cv2.LINE_AA)
            cv2.imshow(window_name, cv2.resize(init_vis, (OUTPUT_W,OUTPUT_H), interpolation=cv2.INTER_LINEAR))
            key = cv2.waitKey(1) & 0xFF
            if initialized: break
            if key == ord('q'):
                print("User quit during initialization"); break

        print(f"Init loop done. initialized={initialized}")
        if not initialized:
            print("Could not initialize.")
            inference_thread.stop(); cap.release(); cv2.destroyAllWindows(); return

        print("About to enter main skeleton loop")
        run_hand_brush_drag_arap_loop_skeleton(
            cap=cap, segmenter=segmenter, hands=hands, pose=pose,
            h=h, w=w, V_base=V_base, V_def=V_def, T=T,
            step=step, thresh=thresh, feather=feather, show_mask=show_mask,
            brush_radius=brush_radius, interaction_state=interaction_state,
            arap_cache=arap_cache, inference_thread=inference_thread,
            window_name=window_name,
        )
        print("Main skeleton loop returned")
        inference_thread.stop()

    cap.release()
    cv2.destroyAllWindows()


test_hand_brush_drag_arap_live_skeleton(step=25, thresh=0.5, feather=0, show_mask=False)