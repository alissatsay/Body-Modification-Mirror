import cv2
import numpy as np
import mediapipe as mp
import time

mp_pose = mp.solutions.pose

import Triangle_Mesh_helpers as TMh
import Pose_Tracking_helpers as PTh
import Loop_helpers as Lh
import Display_helpres as Dh

ROTATE_DEG = 90
ROTATE_DIR = "ccw"

OUTPUT_W = 1080
OUTPUT_H = 1920

PRIMARY_MONITOR_WIDTH = 1920

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
    "torso": (255, 220, 0),      # cyan-yellow-ish
    "head": (255, 0, 255),       # magenta
    "left_arm": (0, 255, 0),     # green
    "right_arm": (0, 180, 255),  # orange
    "left_leg": (255, 0, 0),     # blue
    "right_leg": (0, 0, 255),    # red
}

SHOW_RENDER_GROUP_DEBUG = False
RENDER_GROUP_DEBUG_ALPHA = 0.28

SHOW_MESH_OUTLINE = True

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

# Hysteresis:
# stronger threshold to enter side-front mode,
# lower threshold to return to frontal.
YAW_ENTER_THRESHOLD = 0.22
YAW_EXIT_THRESHOLD = 0.12

# Sign smoothing / persistence
YAW_SIGN_SMOOTH_ALPHA = 0.30
YAW_SIGN_DEADBAND = 0.08

FRONTAL_RENDER_ORDER = [
    "left_leg",
    "right_leg",
    "torso",
    "head",
    "left_arm",
    "right_arm",
]

LEFT_SIDE_FRONT_RENDER_ORDER = [
    "right_leg",
    "left_leg",
    "torso",
    "head",
    "right_arm",
    "left_arm",
]

RIGHT_SIDE_FRONT_RENDER_ORDER = [
    "left_leg",
    "right_leg",
    "torso",
    "head",
    "left_arm",
    "right_arm",
]

LEG_OVERLAP_TRIGGER_FRAC = 0.015
LEG_FRONT_SCORE_DEADBAND = 8.0

ARM_OVERLAP_TRIGGER_FRAC = 0.020
ARM_FRONT_SCORE_DEADBAND = 6.0

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

    return cv2.warpAffine(
        frame_bgr,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

def capture_background_plate(cap, h, w, n_frames=20, window_name="Background capture"):
    """
    Capture a clean background before the user enters the scene.
    Averages several frames for a more stable plate.
    """
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

        if acc is None:
            acc = fr.astype(np.float32)
        else:
            acc += fr.astype(np.float32)
        got += 1

        vis = fr.copy()
        cv2.putText(
            vis,
            f"Capturing empty background... {k+1}/{n_frames}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        init_display = cv2.resize(vis, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(window_name, init_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if acc is None or got == 0:
        return None

    bg = np.clip(acc / float(got), 0, 255).astype(np.uint8)
    return bg

def show_countdown(
    cap,
    h,
    w,
    seconds,
    message,
    window_name,
    bs_img=None,
    bs_size=120,
    bs_margin=18,
    overlay_timer_start=None,
    overlay_timer_duration=180.0,
):
    """
    Display a live countdown overlay on camera feed.

    Optional beauty-standard overlay parameters are passed in explicitly so this
    function does not depend on local variables from another function.
    """
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

        x = (w - tw) // 2
        y = (h // 2)

        cv2.putText(
            vis,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )

        vis_display = cv2.resize(vis, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)

        if bs_img is not None:
            if overlay_timer_start is None:
                overlay_elapsed = 0.0
            else:
                overlay_elapsed = time.time() - overlay_timer_start

            vis_display = _draw_bs_timer_overlay(
                canvas=vis_display,
                bs_img=bs_img,
                size=bs_size,
                margin=bs_margin,
                elapsed=overlay_elapsed,
                duration=overlay_timer_duration,
            )

        cv2.imshow(window_name, vis_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

def build_active_mesh_mask_fast(h, w, V_def, T, active):
    """
    Binary mask of where the deformed mesh currently covers the image.
    """
    tri_indices = np.flatnonzero(active).astype(np.int32)
    return rasterize_triangle_mask_from_indices(h, w, V_def, T, tri_indices)

def composite_mesh_with_background_holefill(
    frame_raw,
    bg_plate,
    warped_mesh_bgr,
    mesh_mask_u8,
    seg_mask,
    body_thresh=0.50,
    dilate_ksize=7,
    blur_ksize=5,
    apply_mesh=True,
):
    """
    Keep live frame where valid, replace only newly revealed body holes with bg_plate,
    and optionally place warped mesh on top.

    hole = current body mask - deformed mesh mask
    """
    out = frame_raw.copy()

    body_mask = (seg_mask >= body_thresh).astype(np.uint8) * 255

    if dilate_ksize is not None and dilate_ksize > 1:
        k = int(dilate_ksize)
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), np.uint8)
        body_mask = cv2.dilate(body_mask, kernel, iterations=1)

    hole_mask = cv2.bitwise_and(body_mask, cv2.bitwise_not(mesh_mask_u8))

    if blur_ksize is not None and blur_ksize > 1:
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        hole_mask = cv2.GaussianBlur(hole_mask, (k, k), 0)

    if hole_mask.ndim == 2:
        hole_alpha = hole_mask.astype(np.float32) / 255.0
        out = alpha_composite_bgr(out, bg_plate, hole_alpha)

    if apply_mesh:
        out = composite_bgr_hard_mask(out, warped_mesh_bgr, mesh_mask_u8)

    return out, hole_mask

def filter_selection_disallow_same_side_arm(
    selection_vertices,
    selection_triangles,
    T,
    binding,
    handedness_label,
):
    """
    Prevent a hand from modifying its own-side arm/palm mesh.

    Important:
    - start from the already-computed selection_triangles from brush selection
      (which are already restricted to active/body triangles)
    - remove any selected triangle that touches blocked same-side arm vertices
    - rebuild selected vertices only from the surviving selected triangles
    """
    if handedness_label not in ("Left", "Right"):
        return selection_vertices, selection_triangles

    seg_ids = binding["vertex_segment"]

    if handedness_label == "Left":
        blocked_seg_ids = {
            SEGMENT_INDEX["left_upper_arm"],
            SEGMENT_INDEX["left_lower_arm"],
            SEGMENT_INDEX["left_palm"],
        }
    else:
        blocked_seg_ids = {
            SEGMENT_INDEX["right_upper_arm"],
            SEGMENT_INDEX["right_lower_arm"],
            SEGMENT_INDEX["right_palm"],
        }

    blocked_vertices = np.isin(seg_ids, list(blocked_seg_ids))

    # Only consider triangles that were already selected by the brush logic.
    tri_touches_blocked = np.any(blocked_vertices[T], axis=1)
    filtered_triangles = selection_triangles & (~tri_touches_blocked)

    # Rebuild selected vertices ONLY from surviving selected triangles.
    filtered_vertices = np.zeros_like(selection_vertices)
    if np.any(filtered_triangles):
        filtered_vertices[np.unique(T[filtered_triangles].ravel())] = True

    return filtered_vertices, filtered_triangles

def compute_arm_front_score(cur_pts, arm_name, yaw_state_txt):
    """
    Positive score -> this arm more likely front of torso
    Negative score -> this arm more likely behind torso

    For side poses, we use the current side state as the dominant cue.
    For frontal poses, we use a mild center-crossing cue.
    """
    ls = cur_pts.get("left_shoulder")
    rs = cur_pts.get("right_shoulder")
    lh = cur_pts.get("left_hip")
    rh = cur_pts.get("right_hip")

    if any(p is None for p in [ls, rs, lh, rh]):
        return 0.0

    torso_mid_x = 0.25 * (ls[0] + rs[0] + lh[0] + rh[0])

    if arm_name == "left_arm":
        e = cur_pts.get("left_elbow")
        w = cur_pts.get("left_wrist")
    else:
        e = cur_pts.get("right_elbow")
        w = cur_pts.get("right_wrist")

    if e is None or w is None:
        return 0.0

    # Side-state cue dominates for side poses
    if yaw_state_txt == "left side front":
        side_bias = 10.0 if arm_name == "left_arm" else -10.0
    elif yaw_state_txt == "right side front":
        side_bias = 10.0 if arm_name == "right_arm" else -10.0
    else:
        side_bias = 0.0

    # Mild center-crossing cue for frontal/ambiguous poses
    # Arm reaching across torso center tends to read as front.
    center_cross = -abs(e[0] - torso_mid_x) - abs(w[0] - torso_mid_x)

    # Re-scale so "less far from center" gives higher score
    # This keeps the cue small relative to side_bias.
    center_cross *= 0.15

    return float(side_bias + center_cross)


def override_arm_order(base_order, left_arm_front=None, right_arm_front=None):
    """
    Rebuild the order so that:
    - legs stay in their current relative order
    - explicit back arms go before torso/head
    - explicit/front/unknown arms go after torso/head
    - relative left/right arm ordering follows base_order
    """
    leg_block = [x for x in base_order if x in ("left_leg", "right_leg")]
    arm_block = [x for x in base_order if x in ("left_arm", "right_arm")]

    back_arms = []
    front_arms = []

    for arm_name in arm_block:
        if arm_name == "left_arm":
            front_flag = left_arm_front
        else:
            front_flag = right_arm_front

        if front_flag is False:
            back_arms.append(arm_name)
        else:
            # True or None both stay in front block by default
            front_arms.append(arm_name)

    return leg_block + back_arms + ["torso", "head"] + front_arms


def compute_render_order_with_arm_override(
    base_order,
    layers,
    stable_pts,
    yaw_state_txt,
):
    """
    Milestone 5A:
    If an arm overlaps the torso enough, decide whether that arm should be
    in front of or behind the torso/head block.

    Returns:
        new_order,
        left_overlap_frac,
        right_overlap_frac,
        left_score,
        right_score,
        arm_override_used
    """
    if layers is None:
        return base_order, 0.0, 0.0, 0.0, 0.0, False

    torso_mask = layers["torso"]["mask_u8"]
    left_arm_mask = layers["left_arm"]["mask_u8"]
    right_arm_mask = layers["right_arm"]["mask_u8"]

    left_overlap_frac = compute_mask_overlap_fraction(left_arm_mask, torso_mask)
    right_overlap_frac = compute_mask_overlap_fraction(right_arm_mask, torso_mask)

    left_score = 0.0
    right_score = 0.0
    left_front = None
    right_front = None
    arm_override_used = False

    if left_overlap_frac >= ARM_OVERLAP_TRIGGER_FRAC:
        left_score = compute_arm_front_score(stable_pts, "left_arm", yaw_state_txt)
        arm_override_used = True
        if left_score > ARM_FRONT_SCORE_DEADBAND:
            left_front = True
        elif left_score < -ARM_FRONT_SCORE_DEADBAND:
            left_front = False

    if right_overlap_frac >= ARM_OVERLAP_TRIGGER_FRAC:
        right_score = compute_arm_front_score(stable_pts, "right_arm", yaw_state_txt)
        arm_override_used = True
        if right_score > ARM_FRONT_SCORE_DEADBAND:
            right_front = True
        elif right_score < -ARM_FRONT_SCORE_DEADBAND:
            right_front = False

    new_order = override_arm_order(
        base_order=base_order,
        left_arm_front=left_front,
        right_arm_front=right_front,
    )

    return new_order, left_overlap_frac, right_overlap_frac, left_score, right_score, arm_override_used

def composite_prebuilt_layers(base_bgr, layers, render_order):
    """
    Composite already-rendered layers in the requested order.
    """
    out = base_bgr.copy()

    for group_name in render_order:
        layer = layers[group_name]
        if USE_FLOAT_ALPHA_COMPOSITING:
            alpha = layer["mask_u8"].astype(np.float32) / 255.0
            out = alpha_composite_bgr(out, layer["color"], alpha)
        else:
            out = composite_bgr_hard_mask(out, layer["color"], layer["mask_u8"])

    return out

def compute_mask_overlap_fraction(mask_a_u8, mask_b_u8):
    """
    Overlap fraction relative to the smaller mask area.
    Returns 0 if either mask is empty.
    """
    a = mask_a_u8 > 0
    b = mask_b_u8 > 0

    area_a = int(np.count_nonzero(a))
    area_b = int(np.count_nonzero(b))
    if area_a == 0 or area_b == 0:
        return 0.0

    inter = int(np.count_nonzero(a & b))
    denom = float(max(1, min(area_a, area_b)))
    return inter / denom


def compute_leg_front_score(cur_pts, yaw_state_txt):
    """
    Positive score -> left leg more likely front
    Negative score -> right leg more likely front

    Uses leg landmarks only.
    """
    lk = cur_pts.get("left_knee")
    rk = cur_pts.get("right_knee")
    la = cur_pts.get("left_ankle")
    ra = cur_pts.get("right_ankle")
    lh = cur_pts.get("left_hip")
    rh = cur_pts.get("right_hip")

    if any(p is None for p in [lk, rk, la, ra, lh, rh]):
        return 0.0

    hip_mid_x = 0.5 * (lh[0] + rh[0])

    # In the mirrored display, “more to the visible side” depends on side state.
    # We use the side state only to define the sign convention for the leg-crossing cue.
    if yaw_state_txt == "left side front":
        # More leftward in image tends to indicate front-left crossing
        left_cross = (hip_mid_x - lk[0]) + (hip_mid_x - la[0])
        right_cross = (hip_mid_x - rk[0]) + (hip_mid_x - ra[0])
        score = left_cross - right_cross
    elif yaw_state_txt == "right side front":
        # More rightward in image tends to indicate front-right crossing
        left_cross = (lk[0] - hip_mid_x) + (la[0] - hip_mid_x)
        right_cross = (rk[0] - hip_mid_x) + (ra[0] - hip_mid_x)
        score = left_cross - right_cross
    else:
        # Frontal fallback: whichever knee/ankle pair crosses the centerline more
        left_cross = abs(lk[0] - hip_mid_x) + abs(la[0] - hip_mid_x)
        right_cross = abs(rk[0] - hip_mid_x) + abs(ra[0] - hip_mid_x)
        score = left_cross - right_cross

    return float(score)


def override_leg_order(base_order, left_leg_front):
    """
    Replace just the relative order of left_leg/right_leg inside a full render order.
    """
    order = [x for x in base_order if x not in ("left_leg", "right_leg")]
    if left_leg_front:
        return ["right_leg", "left_leg"] + order
    else:
        return ["left_leg", "right_leg"] + order


def compute_render_order_with_leg_override(
    base_order,
    layers,
    stable_pts,
    yaw_state_txt,
):
    """
    Milestone 4:
    If left/right leg layers overlap enough, decide which leg is front using
    leg-specific cues and override only the leg order.
    """
    if layers is None:
        return base_order, 0.0, 0.0, False

    left_leg_mask = layers["left_leg"]["mask_u8"]
    right_leg_mask = layers["right_leg"]["mask_u8"]

    overlap_frac = compute_mask_overlap_fraction(left_leg_mask, right_leg_mask)

    if overlap_frac < LEG_OVERLAP_TRIGGER_FRAC:
        return base_order, overlap_frac, 0.0, False

    score = compute_leg_front_score(stable_pts, yaw_state_txt)

    if score > LEG_FRONT_SCORE_DEADBAND:
        return override_leg_order(base_order, left_leg_front=True), overlap_frac, score, True
    elif score < -LEG_FRONT_SCORE_DEADBAND:
        return override_leg_order(base_order, left_leg_front=False), overlap_frac, score, True
    else:
        return base_order, overlap_frac, score, True

def compute_yaw_sign_components(cur_pts):
    """
    Compute multiple left/right cues for determining which side is more frontal.

    Returns:
        sign_value: continuous signed score
        debug: dict of component values
    """
    ls = cur_pts.get("left_shoulder")
    rs = cur_pts.get("right_shoulder")
    lh = cur_pts.get("left_hip")
    rh = cur_pts.get("right_hip")
    nose = cur_pts.get("nose")

    if ls is None or rs is None or lh is None or rh is None:
        return 0.0, {
            "nose_term": 0.0,
            "torso_term": 0.0,
            "shoulder_term": 0.0,
        }

    shoulder_mid = 0.5 * (ls + rs)
    hip_mid = 0.5 * (lh + rh)
    torso_mid = 0.5 * (shoulder_mid + hip_mid)

    shoulder_vec = rs - ls
    shoulder_x, shoulder_len = safe_normalize(shoulder_vec)

    # 1) Nose cue: same idea as before, but normalized.
    nose_term = 0.0
    if nose is not None and shoulder_len > 1e-6:
        nose_offset = np.dot(nose - shoulder_mid, shoulder_x)
        nose_term = float(nose_offset / max(0.35 * shoulder_len, 1.0))

    # 2) Torso asymmetry cue:
    # compare how far each hip lies from the torso center along shoulder axis.
    left_proj = np.dot(lh - torso_mid, shoulder_x)
    right_proj = np.dot(rh - torso_mid, shoulder_x)

    # If the visible/front side dominates the silhouette, this asymmetry tends to drift.
    torso_term = float((right_proj + left_proj) / max(0.35 * shoulder_len, 1.0))

    # 3) Shoulder asymmetry cue relative to torso center
    ls_proj = np.dot(ls - torso_mid, shoulder_x)
    rs_proj = np.dot(rs - torso_mid, shoulder_x)
    shoulder_term = float((rs_proj + ls_proj) / max(0.35 * shoulder_len, 1.0))

    # Weighted combination.
    sign_value = 1.0 * nose_term + 0.8 * torso_term + 0.5 * shoulder_term

    return float(sign_value), {
        "nose_term": float(nose_term),
        "torso_term": float(torso_term),
        "shoulder_term": float(shoulder_term),
    }

def compute_render_order_from_yaw(cur_pts, ref_metrics, interaction_state):
    """
    More robust Phase 3:
    - uses hysteresis
    - smooths sign_value over time
    - remembers previous side state

    Returns:
        render_order, yaw_amount, yaw_sign, yaw_state_txt, yaw_debug
    """
    yaw_amount, yaw_sign_raw, yaw_debug = estimate_body_yaw(cur_pts, ref_metrics)
    sign_value_raw = float(yaw_debug["sign_value"])

    prev_s = float(interaction_state.get("yaw_sign_value_smooth", 0.0))
    sign_value_smooth = (
        (1.0 - YAW_SIGN_SMOOTH_ALPHA) * prev_s +
        YAW_SIGN_SMOOTH_ALPHA * sign_value_raw
    )
    interaction_state["yaw_sign_value_smooth"] = float(sign_value_smooth)

    # Deadband to suppress tiny sign flips.
    if sign_value_smooth > YAW_SIGN_DEADBAND:
        yaw_sign = 1.0
    elif sign_value_smooth < -YAW_SIGN_DEADBAND:
        yaw_sign = -1.0
    else:
        yaw_sign = 0.0

    prev_state = interaction_state.get("yaw_side_state", "frontal")
    state = prev_state

    if prev_state == "frontal":
        if yaw_amount >= YAW_ENTER_THRESHOLD:
            if yaw_sign > 0:
                state = "left_front"
            elif yaw_sign < 0:
                state = "right_front"
            else:
                state = "frontal"

    elif prev_state == "left_front":
        if yaw_amount <= YAW_EXIT_THRESHOLD:
            state = "frontal"
        elif yaw_sign < 0 and yaw_amount >= YAW_ENTER_THRESHOLD:
            state = "right_front"

    elif prev_state == "right_front":
        if yaw_amount <= YAW_EXIT_THRESHOLD:
            state = "frontal"
        elif yaw_sign > 0 and yaw_amount >= YAW_ENTER_THRESHOLD:
            state = "left_front"

    interaction_state["yaw_side_state"] = state

    if state == "left_front":
        render_order = LEFT_SIDE_FRONT_RENDER_ORDER
        yaw_state_txt = "left side front"
    elif state == "right_front":
        render_order = RIGHT_SIDE_FRONT_RENDER_ORDER
        yaw_state_txt = "right side front"
    else:
        render_order = FRONTAL_RENDER_ORDER
        yaw_state_txt = "frontal"

    yaw_debug["sign_value_smooth"] = float(sign_value_smooth)

    return render_order, yaw_amount, yaw_sign, yaw_state_txt, yaw_debug

def build_render_group_triangle_index_cache(binding):
    """
    Precompute triangle index arrays for each render group once at initialization.
    This avoids scanning all triangles for each group on every frame.
    """
    tri_active = binding["tri_active"]
    tri_render_group = binding["tri_render_group"]

    group_tri_indices = {}
    for group_name in RENDER_GROUP_NAMES:
        gid = RENDER_GROUP_INDEX[group_name]
        idx = np.flatnonzero(tri_active & (tri_render_group == gid)).astype(np.int32)
        group_tri_indices[group_name] = idx

    return group_tri_indices


def rasterize_triangle_mask_from_indices(h, w, V_dst, T, tri_indices):
    """
    Faster mask build using only the already-selected triangle indices.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    if tri_indices is None or len(tri_indices) == 0:
        return mask

    for k in tri_indices:
        tri = V_dst[T[k]].astype(np.float32)

        if not np.isfinite(tri).all():
            continue
        if triangle_area2(tri) < 1.0:
            continue
        if not _triangle_inside_image(tri, w, h):
            continue

        tri_i = np.round(tri).astype(np.int32)
        cv2.fillConvexPoly(mask, tri_i, 255, lineType=cv2.LINE_AA)

    return mask


def warp_mesh_piecewise_to_blank_indices(src_img, V_src, V_dst, T, tri_indices,
                                         min_area=1.0, min_bbox=2.0):
    """
    Faster version of warp_mesh_piecewise_to_blank using only a precomputed index list.
    """
    h, w = src_img.shape[:2]
    dst_img = np.zeros_like(src_img)

    if tri_indices is None or len(tri_indices) == 0:
        return dst_img

    for k in tri_indices:
        tri_idx = T[k]

        t_src = V_src[tri_idx].astype(np.float32)
        t_dst = V_dst[tri_idx].astype(np.float32)

        if not np.isfinite(t_src).all() or not np.isfinite(t_dst).all():
            continue

        if triangle_area2(t_src) < min_area or triangle_area2(t_dst) < min_area:
            continue

        src_bw, src_bh = _triangle_bbox_size(t_src)
        dst_bw, dst_bh = _triangle_bbox_size(t_dst)
        if src_bw < min_bbox or src_bh < min_bbox or dst_bw < min_bbox or dst_bh < min_bbox:
            continue

        if not _triangle_inside_image(t_src, w, h):
            continue

        try:
            TMh.warp_triangle(src_img, dst_img, t_src, t_dst)
        except cv2.error:
            continue

    return dst_img


def render_group_layer_fast(src_img, V_src, V_dst, T, tri_indices,
                            min_area=4.0, min_bbox=3.0,
                            mask_dilate_ksize=0, mask_blur_ksize=0):
    """
    Faster per-group layer renderer using:
    - precomputed triangle index list
    - optional hard mask
    """
    h, w = src_img.shape[:2]

    if tri_indices is None or len(tri_indices) == 0:
        return np.zeros_like(src_img), np.zeros((h, w), dtype=np.uint8)

    layer_color = warp_mesh_piecewise_to_blank_indices(
        src_img=src_img,
        V_src=V_src,
        V_dst=V_dst,
        T=T,
        tri_indices=tri_indices,
        min_area=min_area,
        min_bbox=min_bbox,
    )

    layer_mask_u8 = rasterize_triangle_mask_from_indices(h, w, V_dst, T, tri_indices)

    if USE_SOFT_LAYER_MASKS:
        layer_mask_u8 = soften_layer_mask(
            layer_mask_u8,
            dilate_ksize=mask_dilate_ksize,
            blur_ksize=mask_blur_ksize,
        )

    return layer_color, layer_mask_u8


def composite_bgr_hard_mask(base_bgr, over_bgr, mask_u8):
    """
    Much faster compositing for Phase 2:
    just copy pixels where mask is on.
    """
    out = base_bgr.copy()
    m = mask_u8 > 0
    out[m] = over_bgr[m]
    return out


def build_layered_body_layers_fast(frame_raw, V_track, V_def, T, binding,
                                   min_area=4.0, min_bbox=3.0,
                                   mask_dilate_ksize=0, mask_blur_ksize=0):
    """
    Build all per-group layers once, without compositing them yet.
    """
    layers = {}

    group_tri_indices = binding.get("group_tri_indices", None)
    if group_tri_indices is None:
        group_tri_indices = build_render_group_triangle_index_cache(binding)
        binding["group_tri_indices"] = group_tri_indices

    for group_name in RENDER_GROUP_NAMES:
        tri_indices = group_tri_indices[group_name]
        color, mask_u8 = render_group_layer_fast(
            src_img=frame_raw,
            V_src=V_track,
            V_dst=V_def,
            T=T,
            tri_indices=tri_indices,
            min_area=min_area,
            min_bbox=min_bbox,
            mask_dilate_ksize=mask_dilate_ksize,
            mask_blur_ksize=mask_blur_ksize,
        )
        layers[group_name] = {
            "color": color,
            "mask_u8": mask_u8,
        }

    return layers


def render_layered_body_fixed_order_fast(frame_raw, V_track, V_def, T, binding,
                                         render_order=None,
                                         min_area=4.0, min_bbox=3.0,
                                         mask_dilate_ksize=0, mask_blur_ksize=0):
    """
    Compatibility wrapper:
    builds layers once, then composites them in the requested order.
    """
    if render_order is None:
        render_order = FIXED_RENDER_ORDER

    layers = build_layered_body_layers_fast(
        frame_raw=frame_raw,
        V_track=V_track,
        V_def=V_def,
        T=T,
        binding=binding,
        min_area=min_area,
        min_bbox=min_bbox,
        mask_dilate_ksize=mask_dilate_ksize,
        mask_blur_ksize=mask_blur_ksize,
    )

    out = composite_prebuilt_layers(frame_raw, layers, render_order)
    return out, layers

def get_tri_mask_for_render_group(binding, group_name):
    """
    Return a boolean mask over triangles:
    active AND belonging to the requested coarse render group.
    """
    gid = RENDER_GROUP_INDEX[group_name]
    tri_active = binding["tri_active"]
    tri_render_group = binding["tri_render_group"]
    return tri_active & (tri_render_group == gid)


def rasterize_triangle_mask(h, w, V_dst, T, tri_mask):
    """
    Build a binary destination mask for the selected destination triangles.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    for k, tri_idx in enumerate(T):
        if not tri_mask[k]:
            continue

        tri = V_dst[tri_idx].astype(np.float32)

        if not np.isfinite(tri).all():
            continue
        if triangle_area2(tri) < 1.0:
            continue
        if not _triangle_inside_image(tri, w, h):
            continue

        tri_i = np.round(tri).astype(np.int32)
        cv2.fillConvexPoly(mask, tri_i, 255, lineType=cv2.LINE_AA)

    return mask


def soften_layer_mask(mask_u8, dilate_ksize=5, blur_ksize=5):
    """
    Slightly enlarge and soften a binary mask for cleaner compositing.
    """
    out = mask_u8.copy()

    if dilate_ksize is not None and dilate_ksize > 1:
        k = int(dilate_ksize)
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), np.uint8)
        out = cv2.dilate(out, kernel, iterations=1)

    if blur_ksize is not None and blur_ksize > 1:
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        out = cv2.GaussianBlur(out, (k, k), 0)

    return out


def warp_mesh_piecewise_to_blank(src_img, V_src, V_dst, T, active_mask=None,
                                 min_area=1.0, min_bbox=2.0):
    """
    Same idea as warp_mesh_piecewise, but render only the warped triangles into
    a blank image instead of starting from the full source frame.
    """
    h, w = src_img.shape[:2]
    dst_img = np.zeros_like(src_img)

    if active_mask is None:
        active_mask = np.ones(len(T), dtype=bool)

    for k, tri_idx in enumerate(T):
        if not active_mask[k]:
            continue

        t_src = V_src[tri_idx].astype(np.float32)
        t_dst = V_dst[tri_idx].astype(np.float32)

        if not np.isfinite(t_src).all() or not np.isfinite(t_dst).all():
            continue

        if triangle_area2(t_src) < min_area or triangle_area2(t_dst) < min_area:
            continue

        src_bw, src_bh = _triangle_bbox_size(t_src)
        dst_bw, dst_bh = _triangle_bbox_size(t_dst)
        if src_bw < min_bbox or src_bh < min_bbox or dst_bw < min_bbox or dst_bh < min_bbox:
            continue

        if not _triangle_inside_image(t_src, w, h):
            continue

        try:
            TMh.warp_triangle(src_img, dst_img, t_src, t_dst)
        except cv2.error:
            continue

    return dst_img


def render_group_layer(src_img, V_src, V_dst, T, tri_mask,
                       min_area=4.0, min_bbox=3.0,
                       mask_dilate_ksize=5, mask_blur_ksize=5):
    """
    Render one coarse body-part layer.

    Returns:
        layer_color: warped BGR image on black background
        layer_alpha: float alpha mask in [0,1], shape (h,w)
    """
    h, w = src_img.shape[:2]

    if tri_mask is None or not np.any(tri_mask):
        return np.zeros_like(src_img), np.zeros((h, w), dtype=np.float32)

    layer_color = warp_mesh_piecewise_to_blank(
        src_img=src_img,
        V_src=V_src,
        V_dst=V_dst,
        T=T,
        active_mask=tri_mask,
        min_area=min_area,
        min_bbox=min_bbox,
    )

    layer_mask_u8 = rasterize_triangle_mask(h, w, V_dst, T, tri_mask)
    layer_mask_u8 = soften_layer_mask(
        layer_mask_u8,
        dilate_ksize=mask_dilate_ksize,
        blur_ksize=mask_blur_ksize,
    )
    layer_alpha = layer_mask_u8.astype(np.float32) / 255.0

    return layer_color, layer_alpha


def alpha_composite_bgr(base_bgr, over_bgr, alpha):
    """
    Alpha composite 'over_bgr' onto 'base_bgr' using alpha in [0,1].
    alpha shape: (h,w) float
    """
    if alpha.ndim == 2:
        alpha3 = alpha[:, :, None]
    else:
        alpha3 = alpha

    out = (
        alpha3 * over_bgr.astype(np.float32) +
        (1.0 - alpha3) * base_bgr.astype(np.float32)
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def render_layered_body_fixed_order(frame_raw, V_track, V_def, T, binding,
                                    render_order=None,
                                    min_area=4.0, min_bbox=3.0,
                                    mask_dilate_ksize=5, mask_blur_ksize=5):
    """
    Phase 2 renderer:
    - build one layer per render group
    - composite them in a fixed order

    Returns:
        composited_bgr
        layers_dict: group_name -> {"color": ..., "alpha": ...}
    """
    if render_order is None:
        render_order = FIXED_RENDER_ORDER

    h, w = frame_raw.shape[:2]
    out = frame_raw.copy()
    layers = {}

    for group_name in RENDER_GROUP_NAMES:
        tri_mask = get_tri_mask_for_render_group(binding, group_name)
        color, alpha = render_group_layer(
            src_img=frame_raw,
            V_src=V_track,
            V_dst=V_def,
            T=T,
            tri_mask=tri_mask,
            min_area=min_area,
            min_bbox=min_bbox,
            mask_dilate_ksize=mask_dilate_ksize,
            mask_blur_ksize=mask_blur_ksize,
        )
        layers[group_name] = {
            "color": color,
            "alpha": alpha,
        }

    for group_name in render_order:
        layer = layers[group_name]
        out = alpha_composite_bgr(out, layer["color"], layer["alpha"])

    return out, layers

def fine_segment_to_render_group(seg_idx):
    """
    Map the existing fine body segment id to a coarse render-group id.
    Returns -1 for invalid/unassigned segments.
    """
    if seg_idx is None or seg_idx < 0 or seg_idx >= len(SEGMENT_NAMES):
        return -1

    seg_name = SEGMENT_NAMES[int(seg_idx)]

    if seg_name == "torso":
        return RENDER_GROUP_INDEX["torso"]
    if seg_name == "head":
        return RENDER_GROUP_INDEX["head"]

    if seg_name in ("left_upper_arm", "left_lower_arm", "left_palm"):
        return RENDER_GROUP_INDEX["left_arm"]
    if seg_name in ("right_upper_arm", "right_lower_arm", "right_palm"):
        return RENDER_GROUP_INDEX["right_arm"]

    if seg_name in ("left_thigh", "left_calf"):
        return RENDER_GROUP_INDEX["left_leg"]
    if seg_name in ("right_thigh", "right_calf"):
        return RENDER_GROUP_INDEX["right_leg"]

    return -1


def build_triangle_render_groups(binding, T):
    """
    Assign each triangle to one coarse render group by majority vote over the
    triangle's 3 vertex segment labels.

    Output:
        tri_render_group: (num_triangles,) int32
            -1 means unassigned/inactive
             0..N-1 are coarse render groups
    """
    vertex_segment = binding["vertex_segment"]
    tri_active = binding["tri_active"]

    tri_render_group = -np.ones(len(T), dtype=np.int32)

    for k, tri in enumerate(T):
        if not tri_active[k]:
            continue

        fine_ids = vertex_segment[tri]
        if np.any(fine_ids < 0):
            continue

        coarse_ids = [fine_segment_to_render_group(int(s)) for s in fine_ids]
        if np.any(np.array(coarse_ids) < 0):
            continue

        vals, counts = np.unique(np.array(coarse_ids, dtype=np.int32), return_counts=True)
        winner = int(vals[np.argmax(counts)])
        tri_render_group[k] = winner

    return tri_render_group


def draw_triangle_render_group_overlay(
    frame,
    V,
    T,
    tri_active,
    tri_render_group,
    alpha=0.28,
    line_thickness=1,
):
    """
    Debug overlay: color active triangles by coarse render group.
    """
    out = frame.copy()
    overlay = frame.copy()

    for k, tri_idx in enumerate(T):
        if not tri_active[k]:
            continue

        group_id = int(tri_render_group[k])
        if group_id < 0 or group_id >= len(RENDER_GROUP_NAMES):
            continue

        group_name = RENDER_GROUP_NAMES[group_id]
        color = RENDER_GROUP_COLORS[group_name]

        tri = np.round(V[tri_idx]).astype(np.int32)
        cv2.fillConvexPoly(overlay, tri, color, lineType=cv2.LINE_AA)
        cv2.polylines(
            overlay,
            [tri.reshape(-1, 1, 2)],
            isClosed=True,
            color=color,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )

    out = cv2.addWeighted(overlay, float(alpha), out, 1.0 - float(alpha), 0.0)
    return out

def draw_filled_triangle_highlight(
    frame,
    V,
    T,
    tri_mask,
    fill_color=(0, 255, 0),
    fill_alpha=0.16,
    edge_color=(0, 255, 0),
    edge_thickness=2,
    edge_alpha=0.95,
):
    """
    Show only the affected triangles:
    - subtle filled highlight
    - stronger outline in the same color

    This is intended for the "mesh hidden" mode.
    """
    out = frame.copy()

    if tri_mask is None or not np.any(tri_mask):
        return out

    # 1) fill pass
    fill_overlay = np.zeros_like(frame)
    tri_ids = np.flatnonzero(tri_mask)
    for k in tri_ids:
        tri = np.round(V[T[k]]).astype(np.int32)
        cv2.fillConvexPoly(fill_overlay, tri, fill_color, lineType=cv2.LINE_AA)

    out = cv2.addWeighted(fill_overlay, float(fill_alpha), out, 1.0, 0.0)

    # 2) edge pass (stronger / more intense)
    edge_overlay = np.zeros_like(frame)
    for k in tri_ids:
        tri = np.round(V[T[k]]).astype(np.int32)
        cv2.polylines(
            edge_overlay,
            [tri.reshape(-1, 1, 2)],
            isClosed=True,
            color=edge_color,
            thickness=edge_thickness,
            lineType=cv2.LINE_AA,
        )

    out = cv2.addWeighted(edge_overlay, float(edge_alpha), out, 1.0, 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def print_render_group_triangle_stats(binding):
    """
    Simple console summary so you can verify every coarse group got triangles.
    """
    tri_active = binding["tri_active"]
    tri_render_group = binding["tri_render_group"]

    print("\n--- Triangle render-group stats ---")
    print(f"active triangles total: {int(np.sum(tri_active))}/{len(tri_active)}")

    for group_name in RENDER_GROUP_NAMES:
        gid = RENDER_GROUP_INDEX[group_name]
        count = int(np.sum(tri_active & (tri_render_group == gid)))
        print(f"{group_name:>10s}: {count}")

    unassigned = int(np.sum(tri_active & (tri_render_group < 0)))
    print(f"{'unassigned':>10s}: {unassigned}")
    print("-----------------------------------\n")

def estimate_body_yaw(cur_pts, ref_metrics):
    ls = cur_pts.get("left_shoulder")
    rs = cur_pts.get("right_shoulder")
    lh = cur_pts.get("left_hip")
    rh = cur_pts.get("right_hip")

    if ls is None or rs is None or lh is None or rh is None:
        return 0.0, 0.0, {"width_ratio": 1.0, "sign_value": 0.0}

    cur_shoulder_w = float(np.linalg.norm(rs - ls))
    cur_hip_w = float(np.linalg.norm(rh - lh))

    ref_shoulder_w = max(ref_metrics["shoulder_width"], 1.0)
    ref_hip_w = max(ref_metrics["hip_width"], 1.0)

    shoulder_ratio = np.clip(cur_shoulder_w / ref_shoulder_w, 0.0, 1.2)
    hip_ratio = np.clip(cur_hip_w / ref_hip_w, 0.0, 1.2)
    width_ratio = 0.5 * (shoulder_ratio + hip_ratio)

    # 1 -> frontal, smaller -> more sideways
    yaw_amount = np.clip((1.0 - width_ratio) / 0.55, 0.0, 1.0)

    sign_value, sign_debug = compute_yaw_sign_components(cur_pts)

    if sign_value > 0.0:
        yaw_sign = 1.0
    elif sign_value < 0.0:
        yaw_sign = -1.0
    else:
        yaw_sign = 0.0

    debug = {
        "width_ratio": float(width_ratio),
        "sign_value": float(sign_value),
        **sign_debug,
    }
    return float(yaw_amount), float(yaw_sign), debug

def triangle_area2(tri):
    """
    Twice the signed triangle area magnitude.
    tri: (3,2)
    """
    a, b, c = tri
    return abs(
        (b[0] - a[0]) * (c[1] - a[1]) -
        (b[1] - a[1]) * (c[0] - a[0])
    )


def _triangle_bbox_size(tri):
    """
    Returns width, height of the float triangle bbox.
    """
    x0 = float(np.min(tri[:, 0]))
    x1 = float(np.max(tri[:, 0]))
    y0 = float(np.min(tri[:, 1]))
    y1 = float(np.max(tri[:, 1]))
    return (x1 - x0), (y1 - y0)


def _triangle_inside_image(tri, w, h, pad=2.0):
    """
    Loose bounds check with small padding allowance.
    """
    x0 = np.min(tri[:, 0])
    x1 = np.max(tri[:, 0])
    y0 = np.min(tri[:, 1])
    y1 = np.max(tri[:, 1])

    return not (
        x1 < -pad or y1 < -pad or
        x0 > (w - 1 + pad) or y0 > (h - 1 + pad)
    )


def warp_mesh_piecewise(src_img, V_src, V_dst, T, active_mask=None, dst_img=None,
                        min_area=1.0, min_bbox=2.0):
    """
    Piecewise affine mesh warp from V_src -> V_dst.
    Skips triangles that are degenerate, tiny, or off-image.
    """
    if dst_img is None:
        dst_img = src_img.copy()

    if active_mask is None:
        active_mask = np.ones(len(T), dtype=bool)

    h, w = src_img.shape[:2]

    warped_count = 0
    skipped_count = 0

    for k, tri_idx in enumerate(T):
        if not active_mask[k]:
            continue

        t_src = V_src[tri_idx].astype(np.float32)
        t_dst = V_dst[tri_idx].astype(np.float32)

        # Basic sanity
        if not np.isfinite(t_src).all() or not np.isfinite(t_dst).all():
            skipped_count += 1
            continue

        # Area checks
        if triangle_area2(t_src) < min_area or triangle_area2(t_dst) < min_area:
            skipped_count += 1
            continue

        # Bounding-box size checks (important for avoiding empty crops)
        src_bw, src_bh = _triangle_bbox_size(t_src)
        dst_bw, dst_bh = _triangle_bbox_size(t_dst)
        if src_bw < min_bbox or src_bh < min_bbox or dst_bw < min_bbox or dst_bh < min_bbox:
            skipped_count += 1
            continue

        # Skip triangles completely off image
        if not _triangle_inside_image(t_src, w, h):
            skipped_count += 1
            continue

        try:
            TMh.warp_triangle(src_img, dst_img, t_src, t_dst)
            warped_count += 1
        except cv2.error:
            skipped_count += 1
            continue

    # optional debug
    # print(f"warp_mesh_piecewise: warped={warped_count}, skipped={skipped_count}")

    return dst_img

def reconstruct_tracked_mesh_from_skeleton(binding, cur_pts, V_base):
    """
    Reconstruct the current body-attached mesh WITHOUT deformation offsets.
    This is the source mesh for texture sampling on the current frame.
    """
    frames = build_segment_frames(cur_pts)
    if frames is None:
        return None, None

    V_track = V_base.copy().astype(np.float32)
    seg_ids = binding["vertex_segment"]
    rest_uv = binding["vertex_local_uv_rest"]

    for i in range(len(V_track)):
        seg_idx = seg_ids[i]
        if seg_idx < 0:
            continue
        seg_name = SEGMENT_NAMES[seg_idx]
        uv = rest_uv[i]
        V_track[i] = world_from_local_in_frame(uv, frames[seg_name])

    return V_track, frames


def reconstruct_deformed_mesh_from_skeleton(binding, cur_pts, V_base):
    """
    Reconstruct the current body-attached mesh WITH deformation offsets.
    This is the destination mesh for warping / display.
    """
    frames = build_segment_frames(cur_pts)
    if frames is None:
        return None, None

    V_def = V_base.copy().astype(np.float32)
    seg_ids = binding["vertex_segment"]
    rest_uv = binding["vertex_local_uv_rest"]
    off_uv = binding["vertex_local_uv_offset"]

    for i in range(len(V_def)):
        seg_idx = seg_ids[i]
        if seg_idx < 0:
            continue
        seg_name = SEGMENT_NAMES[seg_idx]
        uv = rest_uv[i] + off_uv[i]
        V_def[i] = world_from_local_in_frame(uv, frames[seg_name])

    return V_def, frames



def has_required_landmarks(pts, required_names=REQUIRED_INIT_LANDMARKS):
    if pts is None:
        return False
    return all(pts.get(name) is not None for name in required_names)

def _copy_pt(p):
    return None if p is None else p.astype(np.float32).copy()


def _midpoint(a, b):
    if a is None or b is None:
        return None
    return 0.5 * (a + b)


def _segment_length(a, b, fallback):
    if a is None or b is None:
        return float(fallback)
    return float(max(np.linalg.norm(b - a), 1.0))


def _extend_point(origin, through, length, fallback_dir=None):
    """
    Return origin + direction * length, where direction is based on through-origin.
    If that direction is unavailable, fallback_dir is used.
    """
    if origin is not None and through is not None:
        d = through - origin
        xhat, _ = safe_normalize(d)
        return (origin + xhat * length).astype(np.float32)

    if origin is not None and fallback_dir is not None:
        xhat, _ = safe_normalize(fallback_dir)
        return (origin + xhat * length).astype(np.float32)

    return None

def extract_pose_points(rgb, pose, w, h, min_vis=0.45):
    res = pose.process(rgb)
    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark
    pts = {}

    for name, idx in POSE_IDS.items():
        p = lm[idx]
        if p.visibility is not None and p.visibility < min_vis:
            pts[name] = None
            continue
        x = float(np.clip(p.x * w, 0, w - 1))
        y = float(np.clip(p.y * h, 0, h - 1))
        pts[name] = np.array([x, y], dtype=np.float32)

    required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    if not all(pts.get(k) is not None for k in required):
        return None
    return pts

def sample_mask_at_points(mask, pts_xy, thresh=0.5):
    """
    mask: float mask in [0,1], shape (h,w)
    pts_xy: (N,2) float image coordinates
    returns bool array (N,)
    """
    h, w = mask.shape[:2]
    x = np.clip(np.round(pts_xy[:, 0]).astype(np.int32), 0, w - 1)
    y = np.clip(np.round(pts_xy[:, 1]).astype(np.int32), 0, h - 1)
    return mask[y, x] >= thresh


def triangle_centroids(V, T):
    return (V[T[:, 0]] + V[T[:, 1]] + V[T[:, 2]]) / 3.0


def expand_mask(mask, ksize=9):
    """
    Slight dilation to avoid chopping off boundary triangles.
    """
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate((mask > 0.5).astype(np.uint8), kernel, iterations=1).astype(np.float32)



def build_vertex_neighbors_from_triangles(num_vertices, T):
    neighbors = [set() for _ in range(num_vertices)]
    for tri in T:
        a, b, c = map(int, tri)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return neighbors


def extract_largest_mask_contour(mask, thresh=0.5):
    m = (mask >= thresh).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if contour is None or len(contour) < 3:
        return None
    return contour[:, 0, :].astype(np.float32)


def shape_mesh_boundary_to_mask(V_base, T, init_mask, mask_thresh=0.5, snap_dist_px=24.0, smooth_iters=1):
    """
    Snap only boundary vertices of the grid mesh toward the initialization silhouette
    contour, while keeping interior vertices unchanged.

    Boundary detection is based on inside/outside disagreement across mesh edges.
    Vertices are only snapped if they are already inside the body mask and are close
    enough to the contour, which keeps the shaping local and stable.
    """
    if init_mask is None:
        return V_base.copy().astype(np.float32)

    V_new = V_base.copy().astype(np.float32)
    inside = sample_mask_at_points(init_mask, V_base, thresh=mask_thresh)
    if not np.any(inside):
        return V_new

    contour_pts = extract_largest_mask_contour(init_mask, thresh=mask_thresh)
    if contour_pts is None or len(contour_pts) == 0:
        return V_new

    neighbors = build_vertex_neighbors_from_triangles(len(V_base), T)
    boundary = np.zeros(len(V_base), dtype=bool)
    for i in range(len(V_base)):
        if not inside[i]:
            continue
        for j in neighbors[i]:
            if not inside[j]:
                boundary[i] = True
                break

    boundary_ids = np.flatnonzero(boundary)
    if len(boundary_ids) == 0:
        return V_new

    snap_dist_px = float(max(snap_dist_px, 1.0))
    snap_dist2 = snap_dist_px * snap_dist_px

    for i in boundary_ids:
        p = V_new[i]
        d = contour_pts - p[None, :]
        d2 = np.sum(d * d, axis=1)
        j = int(np.argmin(d2))
        if d2[j] <= snap_dist2:
            V_new[i] = contour_pts[j]

    for _ in range(max(0, int(smooth_iters))):
        prev = V_new.copy()
        for i in boundary_ids:
            nbrs = [j for j in neighbors[i] if inside[j]]
            if not nbrs:
                continue
            avg = np.mean(prev[nbrs], axis=0)
            V_new[i] = 0.7 * prev[i] + 0.3 * avg

    return V_new.astype(np.float32)

def trim_mask_distal_to_elbows(mask, pts, keep_forearm_frac=0.15):
    """
    Remove most of the wrist/hand region from the initialization silhouette.
    Keeps the arm only slightly beyond the elbow for continuity.

    keep_forearm_frac:
        0.0  -> cut exactly at elbow
        0.15 -> keep a little bit past elbow
        0.30 -> keep more forearm
    """
    out = mask.copy()
    h, w = out.shape[:2]

    def cut_one_side(elbow_name, wrist_name):
        elbow = pts.get(elbow_name)
        wrist = pts.get(wrist_name)
        if elbow is None or wrist is None:
            return

        ew = wrist - elbow
        seg_len = np.linalg.norm(ew)
        if seg_len < 1e-6:
            return

        direction = ew / seg_len

        # keep only a small amount beyond elbow
        cut_point = elbow + keep_forearm_frac * seg_len * direction

        # normal to arm axis
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)

        # Build a very large half-plane polygon covering the distal side
        arm_half_width = max(30.0, 0.30 * seg_len)
        far = 4000.0

        p1 = cut_point + arm_half_width * normal
        p2 = cut_point - arm_half_width * normal
        p3 = p2 + far * direction
        p4 = p1 + far * direction

        poly = np.array([p1, p2, p3, p4], dtype=np.int32)
        cv2.fillConvexPoly(out, poly, 0)

    cut_one_side("left_elbow", "left_wrist")
    cut_one_side("right_elbow", "right_wrist")

    return out

def remove_small_components(mask, min_area=250):
    """
    Remove tiny disconnected blobs from a binary float mask.
    """
    m = (mask > 0.5).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    out = np.zeros_like(m)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == label] = 1

    return out.astype(np.float32)


def trim_mask_arms_combined(
    mask,
    pts,
    shoulder_keep_arm_frac=1.0,
    shoulder_cap_scale=0.24,
    elbow_keep_forearm_frac=1.0,
    min_component_area=250,
):
    """
    No arm trimming: keep the full arms/hands in the initialization silhouette.
    We only remove tiny disconnected junk.
    """
    out = mask.copy()
    out = remove_small_components(out, min_area=min_component_area)
    return out

def point_segment_distance(pt, a, b):
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-8:
        return np.linalg.norm(pt - a), 0.0, a.copy()
    t = np.dot(pt - a, ab) / denom
    t = float(np.clip(t, 0.0, 1.0))
    proj = a + t * ab
    dist = np.linalg.norm(pt - proj)
    return dist, t, proj

def compute_body_metrics(pts):
    shoulder_mid = _midpoint(pts.get("left_shoulder"), pts.get("right_shoulder"))
    hip_mid = _midpoint(pts.get("left_hip"), pts.get("right_hip"))

    torso_len = _segment_length(shoulder_mid, hip_mid, 120.0)
    shoulder_width = _segment_length(pts.get("left_shoulder"), pts.get("right_shoulder"), 60.0)
    hip_width = _segment_length(pts.get("left_hip"), pts.get("right_hip"), 50.0)

    return {
        "torso_len": torso_len,
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "left_upper_arm_len": _segment_length(pts.get("left_shoulder"), pts.get("left_elbow"), 0.55 * torso_len),
        "left_lower_arm_len": _segment_length(pts.get("left_elbow"), pts.get("left_wrist"), 0.55 * torso_len),
        "right_upper_arm_len": _segment_length(pts.get("right_shoulder"), pts.get("right_elbow"), 0.55 * torso_len),
        "right_lower_arm_len": _segment_length(pts.get("right_elbow"), pts.get("right_wrist"), 0.55 * torso_len),
        "left_thigh_len": _segment_length(pts.get("left_hip"), pts.get("left_knee"), 0.75 * torso_len),
        "left_calf_len": _segment_length(pts.get("left_knee"), pts.get("left_ankle"), 0.75 * torso_len),
        "right_thigh_len": _segment_length(pts.get("right_hip"), pts.get("right_knee"), 0.75 * torso_len),
        "right_calf_len": _segment_length(pts.get("right_knee"), pts.get("right_ankle"), 0.75 * torso_len),
    }


def clamp_point_jump(prev_pt, cur_pt, max_jump_px=25.0):
    if prev_pt is None or cur_pt is None:
        return cur_pt
    d = cur_pt - prev_pt
    dist = np.linalg.norm(d)
    if dist <= max_jump_px or dist < 1e-6:
        return cur_pt
    return prev_pt + d * (max_jump_px / dist)

def trim_mask_distal_to_shoulders(mask, pts, keep_arm_frac=0.06, shoulder_cap_scale=0.24):
    """
    Remove everything distal to the shoulders, while preserving a small shoulder cap.

    keep_arm_frac:
        fraction of the shoulder->elbow segment to keep before cutting.
        0.03 - 0.08 is a good range.

    shoulder_cap_scale:
        radius of the preserved shoulder cap relative to shoulder width.
    """
    out = mask.copy()

    ls = pts.get("left_shoulder")
    rs = pts.get("right_shoulder")
    if ls is None or rs is None:
        return out

    shoulder_w = np.linalg.norm(rs - ls)
    shoulder_cap_r = max(18.0, shoulder_cap_scale * shoulder_w)

    def cut_one_side(shoulder_name, elbow_name):
        shoulder = pts.get(shoulder_name)
        elbow = pts.get(elbow_name)
        if shoulder is None or elbow is None:
            return

        se = elbow - shoulder
        seg_len = np.linalg.norm(se)
        if seg_len < 1e-6:
            return

        direction = se / seg_len
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)

        # keep a tiny amount past the shoulder, then remove everything beyond
        cut_point = shoulder + keep_arm_frac * seg_len * direction

        # much wider than before so the whole arm is actually cleared
        half_width = max(60.0, 0.60 * shoulder_w)
        far = 5000.0

        # Large distal rectangle
        p1 = cut_point + half_width * normal
        p2 = cut_point - half_width * normal
        p3 = p2 + far * direction
        p4 = p1 + far * direction
        poly = np.array([p1, p2, p3, p4], dtype=np.int32)

        cv2.fillConvexPoly(out, poly, 0)

        # Restore a circular shoulder cap
        c = tuple(np.round(shoulder).astype(np.int32))
        cv2.circle(out, c, int(round(shoulder_cap_r)), 1, -1, lineType=cv2.LINE_AA)

    cut_one_side("left_shoulder", "left_elbow")
    cut_one_side("right_shoulder", "right_elbow")

    return out



def rasterize_pose_capsules(mask_shape, pts):
    """
    Make a binary mask from thick body segments so thin regions are not lost if
    segmentation misses them. Arms are protected all the way to the wrists.
    """
    h, w = mask_shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)

    segments = [
        ("left_shoulder", "left_elbow", 28),
        ("left_elbow", "left_wrist", 24),
        ("right_shoulder", "right_elbow", 28),
        ("right_elbow", "right_wrist", 24),
        ("left_hip", "left_knee", 30),
        ("left_knee", "left_ankle", 26),
        ("right_hip", "right_knee", 30),
        ("right_knee", "right_ankle", 26),
    ]

    for a_name, b_name, thickness in segments:
        a = pts.get(a_name)
        b = pts.get(b_name)
        if a is None or b is None:
            continue
        pa = tuple(np.round(a).astype(np.int32))
        pb = tuple(np.round(b).astype(np.int32))
        cv2.line(out, pa, pb, 255, thickness=thickness, lineType=cv2.LINE_AA)

    req = ["left_shoulder", "right_shoulder", "right_hip", "left_hip"]
    if all(pts.get(k) is not None for k in req):
        quad = np.array([
            pts["left_shoulder"],
            pts["right_shoulder"],
            pts["right_hip"],
            pts["left_hip"],
        ], dtype=np.int32)
        cv2.fillConvexPoly(out, quad, 255)

    if pts.get("nose") is not None and pts.get("left_shoulder") is not None and pts.get("right_shoulder") is not None:
        shoulder_w = np.linalg.norm(pts["right_shoulder"] - pts["left_shoulder"])
        r = max(20, int(0.32 * shoulder_w))
        c = tuple(np.round(pts["nose"]).astype(np.int32))
        cv2.circle(out, c, r, 255, -1, lineType=cv2.LINE_AA)

    return (out > 0).astype(np.float32)

def make_torso_frame(pts):
    ls = pts["left_shoulder"]
    rs = pts["right_shoulder"]
    lh = pts["left_hip"]
    rh = pts["right_hip"]

    shoulder_mid = 0.5 * (ls + rs)
    hip_mid = 0.5 * (lh + rh)

    y_axis = hip_mid - shoulder_mid
    yhat, torso_len = safe_normalize(y_axis)

    shoulder_width = np.linalg.norm(rs - ls)
    if shoulder_width < 1e-6:
        shoulder_width = 40.0

    xhat = np.array([yhat[1], -yhat[0]], dtype=np.float32)
    origin = shoulder_mid.astype(np.float32)

    return {
        "origin": origin,
        "xhat": xhat,
        "yhat": yhat,
        "length": float(max(torso_len, 20.0)),
        "width": float(max(shoulder_width, 20.0)),
    }

def build_segment_frames(pts):
    if pts is None:
        return None

    required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    if not all(pts.get(k) is not None for k in required):
        return None

    frames = {}

    # torso frame
    frames["torso"] = make_torso_frame(pts)

    shoulder_mid = 0.5 * (pts["left_shoulder"] + pts["right_shoulder"])
    hip_mid = 0.5 * (pts["left_hip"] + pts["right_hip"])
    torso_axis = hip_mid - shoulder_mid

    # ----- head -----
    if pts.get("nose") is not None:
        frames["head"] = make_frame_from_points(
            shoulder_mid,
            pts["nose"],
            fallback_origin=shoulder_mid,
            fallback_dir=np.array([0.0, -40.0], dtype=np.float32),
        )
    else:
        frames["head"] = make_frame_from_points(
            None,
            None,
            fallback_origin=shoulder_mid,
            fallback_dir=np.array([0.0, -40.0], dtype=np.float32),
        )

    # ----- left upper arm -----
    left_upper_dir = (
        pts["left_elbow"] - pts["left_shoulder"]
        if pts.get("left_shoulder") is not None and pts.get("left_elbow") is not None
        else np.array([-1.0, 0.0], dtype=np.float32) * max(30.0, np.linalg.norm(torso_axis) * 0.35)
    )
    frames["left_upper_arm"] = make_frame_from_points(
        pts.get("left_shoulder"),
        pts.get("left_elbow"),
        fallback_origin=pts["left_shoulder"],
        fallback_dir=left_upper_dir,
    )

    # ----- left lower arm -----
    left_lower_dir = None
    if pts.get("left_elbow") is not None and pts.get("left_wrist") is not None:
        left_lower_dir = pts["left_wrist"] - pts["left_elbow"]
    elif pts.get("left_elbow") is not None and pts.get("left_shoulder") is not None:
        left_lower_dir = pts["left_elbow"] - pts["left_shoulder"]
    else:
        left_lower_dir = left_upper_dir

    left_lower_origin = (
        pts["left_elbow"] if pts.get("left_elbow") is not None
        else pts["left_shoulder"]
    )
    frames["left_lower_arm"] = make_frame_from_points(
        pts.get("left_elbow"),
        pts.get("left_wrist"),
        fallback_origin=left_lower_origin,
        fallback_dir=left_lower_dir,
    )

    # ----- left palm -----
    left_palm_origin = pts.get("left_wrist")
    if pts.get("left_elbow") is not None and pts.get("left_wrist") is not None:
        left_palm_dir = pts["left_wrist"] - pts["left_elbow"]
    else:
        left_palm_dir = left_lower_dir

    left_palm_len = max(18.0, 0.75 * np.linalg.norm(left_palm_dir))
    if np.linalg.norm(left_palm_dir) < 1e-6 or left_palm_origin is None:
        left_palm_tip = None
    else:
        left_palm_tip = left_palm_origin + (left_palm_dir / np.linalg.norm(left_palm_dir)) * left_palm_len

    frames["left_palm"] = make_frame_from_points(
        left_palm_origin,
        left_palm_tip,
        fallback_origin=left_palm_origin,
        fallback_dir=left_palm_dir,
    )

    # ----- right upper arm -----
    right_upper_dir = (
        pts["right_elbow"] - pts["right_shoulder"]
        if pts.get("right_shoulder") is not None and pts.get("right_elbow") is not None
        else np.array([1.0, 0.0], dtype=np.float32) * max(30.0, np.linalg.norm(torso_axis) * 0.35)
    )
    frames["right_upper_arm"] = make_frame_from_points(
        pts.get("right_shoulder"),
        pts.get("right_elbow"),
        fallback_origin=pts["right_shoulder"],
        fallback_dir=right_upper_dir,
    )

    # ----- right lower arm -----
    right_lower_dir = None
    if pts.get("right_elbow") is not None and pts.get("right_wrist") is not None:
        right_lower_dir = pts["right_wrist"] - pts["right_elbow"]
    elif pts.get("right_elbow") is not None and pts.get("right_shoulder") is not None:
        right_lower_dir = pts["right_elbow"] - pts["right_shoulder"]
    else:
        right_lower_dir = right_upper_dir

    right_lower_origin = (
        pts["right_elbow"] if pts.get("right_elbow") is not None
        else pts["right_shoulder"]
    )
    frames["right_lower_arm"] = make_frame_from_points(
        pts.get("right_elbow"),
        pts.get("right_wrist"),
        fallback_origin=right_lower_origin,
        fallback_dir=right_lower_dir,
    )

    # ----- right palm -----
    right_palm_origin = pts.get("right_wrist")
    if pts.get("right_elbow") is not None and pts.get("right_wrist") is not None:
        right_palm_dir = pts["right_wrist"] - pts["right_elbow"]
    else:
        right_palm_dir = right_lower_dir

    right_palm_len = max(18.0, 0.75 * np.linalg.norm(right_palm_dir))
    if np.linalg.norm(right_palm_dir) < 1e-6 or right_palm_origin is None:
        right_palm_tip = None
    else:
        right_palm_tip = right_palm_origin + (right_palm_dir / np.linalg.norm(right_palm_dir)) * right_palm_len

    frames["right_palm"] = make_frame_from_points(
        right_palm_origin,
        right_palm_tip,
        fallback_origin=right_palm_origin,
        fallback_dir=right_palm_dir,
    )

    # ----- left thigh -----
    left_thigh_dir = (
        pts["left_knee"] - pts["left_hip"]
        if pts.get("left_hip") is not None and pts.get("left_knee") is not None
        else torso_axis
    )
    frames["left_thigh"] = make_frame_from_points(
        pts.get("left_hip"),
        pts.get("left_knee"),
        fallback_origin=pts["left_hip"],
        fallback_dir=left_thigh_dir,
    )

    # ----- left calf -----
    if pts.get("left_knee") is not None and pts.get("left_ankle") is not None:
        left_calf_dir = pts["left_ankle"] - pts["left_knee"]
    else:
        left_calf_dir = left_thigh_dir

    left_calf_origin = (
        pts["left_knee"] if pts.get("left_knee") is not None
        else pts["left_hip"]
    )
    frames["left_calf"] = make_frame_from_points(
        pts.get("left_knee"),
        pts.get("left_ankle"),
        fallback_origin=left_calf_origin,
        fallback_dir=left_calf_dir,
    )

    # ----- right thigh -----
    right_thigh_dir = (
        pts["right_knee"] - pts["right_hip"]
        if pts.get("right_hip") is not None and pts.get("right_knee") is not None
        else torso_axis
    )
    frames["right_thigh"] = make_frame_from_points(
        pts.get("right_hip"),
        pts.get("right_knee"),
        fallback_origin=pts["right_hip"],
        fallback_dir=right_thigh_dir,
    )

    # ----- right calf -----
    if pts.get("right_knee") is not None and pts.get("right_ankle") is not None:
        right_calf_dir = pts["right_ankle"] - pts["right_knee"]
    else:
        right_calf_dir = right_thigh_dir

    right_calf_origin = (
        pts["right_knee"] if pts.get("right_knee") is not None
        else pts["right_hip"]
    )
    frames["right_calf"] = make_frame_from_points(
        pts.get("right_knee"),
        pts.get("right_ankle"),
        fallback_origin=right_calf_origin,
        fallback_dir=right_calf_dir,
    )

    return frames

def point_in_quad(pt, quad):
    return cv2.pointPolygonTest(quad.astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0


def point_segment_distance(pt, a, b):
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-8:
        return np.linalg.norm(pt - a), 0.0, a.copy()
    t = np.dot(pt - a, ab) / denom
    t = float(np.clip(t, 0.0, 1.0))
    proj = a + t * ab
    dist = np.linalg.norm(pt - proj)
    return dist, t, proj

def make_frame_from_points(p0, p1, fallback_origin=None, fallback_dir=None, min_len=10.0):
    """
    Build a local frame from two points.
    If one or both points are missing, fall back to a reasonable default.
    """
    if p0 is not None and p1 is not None:
        axis = p1 - p0
        xhat, length = safe_normalize(axis)
        length = max(length, min_len)
        yhat = np.array([-xhat[1], xhat[0]], dtype=np.float32)
        return {
            "origin": p0.astype(np.float32),
            "xhat": xhat,
            "yhat": yhat,
            "length": float(length),
        }

    if fallback_origin is None:
        fallback_origin = np.array([0.0, 0.0], dtype=np.float32)
    else:
        fallback_origin = fallback_origin.astype(np.float32)

    if fallback_dir is None:
        fallback_dir = np.array([1.0, 0.0], dtype=np.float32)
    else:
        fallback_dir = fallback_dir.astype(np.float32)

    xhat, length = safe_normalize(fallback_dir)
    length = max(length, min_len)
    yhat = np.array([-xhat[1], xhat[0]], dtype=np.float32)

    return {
        "origin": fallback_origin,
        "xhat": xhat,
        "yhat": yhat,
        "length": float(length),
    }

def safe_normalize(v, eps=1e-6):
    n = np.linalg.norm(v)
    if n < eps:
        return np.array([1.0, 0.0], dtype=np.float32), 1.0
    return (v / n).astype(np.float32), float(n)

def torso_quad_from_pts(pts, expand_x=0.22, expand_y_top=0.10, expand_y_bottom=0.10):
    ls = pts["left_shoulder"]
    rs = pts["right_shoulder"]
    lh = pts["left_hip"]
    rh = pts["right_hip"]

    shoulder_mid = 0.5 * (ls + rs)
    hip_mid = 0.5 * (lh + rh)

    y = hip_mid - shoulder_mid
    yhat, torso_h = safe_normalize(y)
    xhat = np.array([yhat[1], -yhat[0]], dtype=np.float32)

    shoulder_w = np.linalg.norm(rs - ls)
    hip_w = np.linalg.norm(rh - lh)
    half_w = 0.5 * max(shoulder_w, hip_w)
    half_w *= (1.0 + expand_x)

    top = shoulder_mid - expand_y_top * torso_h * yhat
    bot = hip_mid + expand_y_bottom * torso_h * yhat

    quad = np.stack([
        top - half_w * xhat,
        top + half_w * xhat,
        bot + half_w * xhat,
        bot - half_w * xhat,
    ], axis=0)
    return quad.astype(np.float32)


def localize_point_in_frame(pt, frame):
    d = pt - frame["origin"]
    u = np.dot(d, frame["xhat"]) / frame["length"]
    v = np.dot(d, frame["yhat"]) / frame["length"]
    return np.array([u, v], dtype=np.float32)


def world_from_local_in_frame(uv, frame):
    return (
        frame["origin"]
        + uv[0] * frame["length"] * frame["xhat"]
        + uv[1] * frame["length"] * frame["yhat"]
    ).astype(np.float32)



def bind_mesh_to_skeleton(V_base, T, ref_pts, init_seg_mask, mask_thresh=0.5):
    """
    Build a fixed mesh from the initialization silhouette, then bind its vertices
    to torso/head/upper-arms/lower-arms/legs.

    Important detail:
    segment assignment uses HARD acceptance gates, so a vertex is only assigned
    to an arm/leg if it is actually close to that limb. This prevents distant
    leg vertices from accidentally binding to an arm (or vice versa).
    """
    ref_frames = build_segment_frames(ref_pts)
    if ref_frames is None:
        return None

    if init_seg_mask is None:
        return None

    init_mask = expand_mask(init_seg_mask, ksize=9)
    vertex_in_mask = sample_mask_at_points(init_mask, V_base, thresh=mask_thresh)

    vertex_segment = -np.ones(len(V_base), dtype=np.int32)
    vertex_local_uv = np.zeros((len(V_base), 2), dtype=np.float32)

    shoulder_mid = 0.5 * (ref_pts["left_shoulder"] + ref_pts["right_shoulder"])
    hip_mid = 0.5 * (ref_pts["left_hip"] + ref_pts["right_hip"])
    shoulder_w = np.linalg.norm(ref_pts["right_shoulder"] - ref_pts["left_shoulder"])
    shoulder_y = min(ref_pts["left_shoulder"][1], ref_pts["right_shoulder"][1])
    hip_y = max(ref_pts["left_hip"][1], ref_pts["right_hip"][1])

    torso_quad = torso_quad_from_pts(
        ref_pts,
        expand_x=0.28,
        expand_y_top=0.22,
        expand_y_bottom=0.12,
    )

    if ref_pts.get("nose") is not None:
        head_center = ref_pts["nose"] + np.array([0.0, -0.18 * shoulder_w], dtype=np.float32)
    else:
        head_center = shoulder_mid + np.array([0.0, -0.75 * shoulder_w], dtype=np.float32)

    head_radius = max(36.0, 0.70 * shoulder_w)

    torso_left = min(ref_pts["left_shoulder"][0], ref_pts["left_hip"][0]) - 0.22 * shoulder_w
    torso_right = max(ref_pts["right_shoulder"][0], ref_pts["right_hip"][0]) + 0.22 * shoulder_w

    left_shoulder = ref_pts.get("left_shoulder")
    right_shoulder = ref_pts.get("right_shoulder")
    shoulder_cap_r = max(24.0, 0.30 * shoulder_w)

    arm_capsules = []

    # left upper / lower arm
    if ref_pts.get("left_shoulder") is not None and ref_pts.get("left_elbow") is not None:
        arm_capsules.append(("left_upper_arm", ref_pts["left_shoulder"], ref_pts["left_elbow"], 0.26))
    if ref_pts.get("left_elbow") is not None and ref_pts.get("left_wrist") is not None:
        arm_capsules.append(("left_lower_arm", ref_pts["left_elbow"], ref_pts["left_wrist"], 0.22))

    # left palm (extend a short distance beyond wrist in forearm direction)
    if ref_pts.get("left_elbow") is not None and ref_pts.get("left_wrist") is not None:
        lw_dir = ref_pts["left_wrist"] - ref_pts["left_elbow"]
        lw_len = np.linalg.norm(lw_dir)
        if lw_len > 1e-6:
            lw_hat = lw_dir / lw_len
            left_palm_tip = ref_pts["left_wrist"] + 0.75 * lw_len * lw_hat
            arm_capsules.append(("left_palm", ref_pts["left_wrist"], left_palm_tip, 0.42))

    # right upper / lower arm
    if ref_pts.get("right_shoulder") is not None and ref_pts.get("right_elbow") is not None:
        arm_capsules.append(("right_upper_arm", ref_pts["right_shoulder"], ref_pts["right_elbow"], 0.26))
    if ref_pts.get("right_elbow") is not None and ref_pts.get("right_wrist") is not None:
        arm_capsules.append(("right_lower_arm", ref_pts["right_elbow"], ref_pts["right_wrist"], 0.22))

    # right palm
    if ref_pts.get("right_elbow") is not None and ref_pts.get("right_wrist") is not None:
        rw_dir = ref_pts["right_wrist"] - ref_pts["right_elbow"]
        rw_len = np.linalg.norm(rw_dir)
        if rw_len > 1e-6:
            rw_hat = rw_dir / rw_len
            right_palm_tip = ref_pts["right_wrist"] + 0.75 * rw_len * rw_hat
            arm_capsules.append(("right_palm", ref_pts["right_wrist"], right_palm_tip, 0.42))

    leg_capsules = []
    if ref_pts.get("left_hip") is not None and ref_pts.get("left_knee") is not None:
        leg_capsules.append(("left_thigh", ref_pts["left_hip"], ref_pts["left_knee"], 0.28))
    if ref_pts.get("left_knee") is not None and ref_pts.get("left_ankle") is not None:
        leg_capsules.append(("left_calf", ref_pts["left_knee"], ref_pts["left_ankle"], 0.24))
    if ref_pts.get("right_hip") is not None and ref_pts.get("right_knee") is not None:
        leg_capsules.append(("right_thigh", ref_pts["right_hip"], ref_pts["right_knee"], 0.28))
    if ref_pts.get("right_knee") is not None and ref_pts.get("right_ankle") is not None:
        leg_capsules.append(("right_calf", ref_pts["right_knee"], ref_pts["right_ankle"], 0.24))

    def assign_vertex(i, seg):
        vertex_segment[i] = SEGMENT_INDEX[seg]
        vertex_local_uv[i] = localize_point_in_frame(V_base[i], ref_frames[seg])

    def best_capsule_match(p, capsule_defs, min_radius_px, accept_scale, allowed_sides=None, y_min=None, y_max=None):
        best_seg = None
        best_dist = np.inf

        for seg, a, b, radius_scale in capsule_defs:
            seg_len = np.linalg.norm(b - a)
            radius = max(min_radius_px, radius_scale * seg_len)

            if allowed_sides is not None:
                if "left" in seg and "left" not in allowed_sides:
                    continue
                if "right" in seg and "right" not in allowed_sides:
                    continue

            if y_min is not None and p[1] < y_min:
                continue
            if y_max is not None and p[1] > y_max:
                continue

            dist, _, _ = point_segment_distance(p, a, b)
            if dist <= accept_scale * radius and dist < best_dist:
                best_dist = dist
                best_seg = seg

        return best_seg

    mid_x = shoulder_mid[0]
    arm_y_max = hip_y + 0.10 * shoulder_w
    leg_y_min = shoulder_y + 0.35 * shoulder_w

    for i, p in enumerate(V_base):
        if not vertex_in_mask[i]:
            continue

        assigned = False

        # 1) torso
        if point_in_quad(p, torso_quad):
            assign_vertex(i, "torso")
            assigned = True

        # 2) shoulder caps belong to upper arms, not torso
        if not assigned:
            in_left_cap = left_shoulder is not None and np.linalg.norm(p - left_shoulder) <= shoulder_cap_r
            in_right_cap = right_shoulder is not None and np.linalg.norm(p - right_shoulder) <= shoulder_cap_r

            if in_left_cap and not in_right_cap:
                assign_vertex(i, "left_upper_arm")
                assigned = True
            elif in_right_cap and not in_left_cap:
                assign_vertex(i, "right_upper_arm")
                assigned = True
            elif in_left_cap and in_right_cap:
                # rare center/overlap case: pick nearer shoulder
                dL = np.linalg.norm(p - left_shoulder)
                dR = np.linalg.norm(p - right_shoulder)
                assign_vertex(i, "left_upper_arm" if dL <= dR else "right_upper_arm")
                assigned = True

        # 3) head
        if (not assigned) and (np.linalg.norm(p - head_center) <= head_radius):
            assign_vertex(i, "head")
            assigned = True

        # 4) neck band stays torso
        if not assigned:
            neck_band_top = shoulder_y - 0.24 * shoulder_w
            neck_band_bottom = shoulder_y + 0.22 * shoulder_w
            in_neck_band = (
                neck_band_top <= p[1] <= neck_band_bottom and
                torso_left <= p[0] <= torso_right
            )
            if in_neck_band:
                assign_vertex(i, "torso")
                assigned = True

        # 5) legs first, with lower-body gate
        if not assigned and len(leg_capsules) > 0:
            best_seg = best_capsule_match(
                p,
                leg_capsules,
                min_radius_px=12.0,
                accept_scale=1.15,
                allowed_sides=None,
                y_min=leg_y_min,
            )
            if best_seg is not None:
                assign_vertex(i, best_seg)
                assigned = True

        # 6) palms first, so the hand region becomes its own segment
        if not assigned and len(arm_capsules) > 0:
            palm_capsules = [c for c in arm_capsules if c[0] in ("left_palm", "right_palm")]
            if len(palm_capsules) > 0:
                best_seg = best_capsule_match(
                    p,
                    palm_capsules,
                    min_radius_px=18.0,
                    accept_scale=1.60,
                    allowed_sides=None,
                    y_max=arm_y_max + 0.35 * shoulder_w,
                )
                if best_seg is not None:
                    assign_vertex(i, best_seg)
                    assigned = True

        # 7) remaining arm regions
        if not assigned and len(arm_capsules) > 0:
            non_palm_arm_capsules = [c for c in arm_capsules if c[0] not in ("left_palm", "right_palm")]
            if len(non_palm_arm_capsules) > 0:
                best_seg = best_capsule_match(
                    p,
                    non_palm_arm_capsules,
                    min_radius_px=10.0,
                    accept_scale=1.25,
                    allowed_sides=None,
                    y_max=arm_y_max + 0.15 * shoulder_w,
                )
                if best_seg is not None:
                    assign_vertex(i, best_seg)
                    assigned = True

    tri_vertices_in_mask = np.all(vertex_in_mask[T], axis=1)
    tri_centroids = triangle_centroids(V_base, T)
    tri_centroid_in_mask = sample_mask_at_points(init_mask, tri_centroids, thresh=mask_thresh)

    tri_assigned = np.all(vertex_segment[T] >= 0, axis=1)
    tri_active = tri_vertices_in_mask & tri_centroid_in_mask & tri_assigned

    used_vertices = np.zeros(len(V_base), dtype=bool)
    if np.any(tri_active):
        used_vertices[np.unique(T[tri_active].reshape(-1))] = True

    vertex_segment[~used_vertices] = -1

    return {
        "ref_pts": ref_pts,
        "ref_frames": ref_frames,
        "vertex_segment": vertex_segment,
        "vertex_local_uv_rest": vertex_local_uv.copy(),
        "vertex_local_uv_offset": np.zeros_like(vertex_local_uv),
        "tri_active": tri_active,
        "vertex_in_mask": vertex_in_mask,
    }

def reconstruct_mesh_from_skeleton(binding, cur_pts, V_base):
    frames = build_segment_frames(cur_pts)
    if frames is None:
        return None, None

    V_track = V_base.copy().astype(np.float32)
    seg_ids = binding["vertex_segment"]
    rest_uv = binding["vertex_local_uv_rest"]
    off_uv = binding["vertex_local_uv_offset"]

    for i in range(len(V_track)):
        seg_idx = seg_ids[i]
        if seg_idx < 0:
            continue
        seg_name = SEGMENT_NAMES[seg_idx]
        uv = rest_uv[i] + off_uv[i]
        V_track[i] = world_from_local_in_frame(uv, frames[seg_name])

    return V_track, frames


def update_local_offsets_from_world(binding, frames, V_new):
    seg_ids = binding["vertex_segment"]
    rest_uv = binding["vertex_local_uv_rest"]
    off_uv = binding["vertex_local_uv_offset"]

    for i in range(len(V_new)):
        seg_idx = seg_ids[i]
        if seg_idx < 0:
            continue
        seg_name = SEGMENT_NAMES[seg_idx]
        uv_now = localize_point_in_frame(V_new[i], frames[seg_name])
        off_uv[i] = uv_now - rest_uv[i]

def smooth_pose_points(cur_pts, state, alpha=0.35, max_jump_px=25.0, hold_frames=6):
    if cur_pts is None:
        state["pose_missing_count"] += 1
        if state["pose_missing_count"] <= hold_frames:
            return state["pose_last_good_pts"]
        return None

    state["pose_missing_count"] = 0
    prev_pts = state.get("pose_prev_pts", None)

    smoothed = {}
    for name in TRACKED_POSE_NAMES:
        cur = cur_pts.get(name, None)
        prev = None if prev_pts is None else prev_pts.get(name, None)

        if cur is None and prev is not None:
            smoothed[name] = prev.copy()
            continue
        if cur is None:
            smoothed[name] = None
            continue

        cur = clamp_point_jump(prev, cur, max_jump_px=max_jump_px)

        if prev is None:
            smoothed[name] = cur.copy()
        else:
            smoothed[name] = (alpha * cur + (1.0 - alpha) * prev).astype(np.float32)

    state["pose_prev_pts"] = {k: (None if v is None else v.copy()) for k, v in smoothed.items()}
    state["pose_last_good_pts"] = {k: (None if v is None else v.copy()) for k, v in smoothed.items()}
    return smoothed

def finalize_init_mask(mask_accum, count, avg_thresh=0.30, dilate_ksize=11, close_ksize=11):
    """
    Build a robust init silhouette from multiple segmentation masks.
    """
    if count <= 0:
        return None

    avg_mask = mask_accum / float(count)
    mask_bin = (avg_mask >= avg_thresh).astype(np.uint8)

    if close_ksize > 0:
        k = close_ksize if close_ksize % 2 == 1 else close_ksize + 1
        kernel = np.ones((k, k), np.uint8)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)

    if dilate_ksize > 0:
        k = dilate_ksize if dilate_ksize % 2 == 1 else dilate_ksize + 1
        kernel = np.ones((k, k), np.uint8)
        mask_bin = cv2.dilate(mask_bin, kernel, iterations=1)

    return mask_bin.astype(np.float32)

def _resolve_beauty_standard_image_path(filename="beauty_standard.png"):
    """
    Resolve the beauty-standard image path robustly relative to this script first,
    then fall back to the current working directory.
    """
    import os

    candidates = []

    if "__file__" in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(script_dir, "beauty_standard_images", filename))

    candidates.append(os.path.join(os.getcwd(), "beauty_standard_images", filename))

    for p in candidates:
        if os.path.exists(p):
            return p

    return candidates[0] if candidates else os.path.join("beauty_standard_images", filename)


def load_beauty_standard_overlay(size=120, filename="beauty_standard.png"):
    """
    Load and resize the beauty-standard reference image.
    Returns (image_or_none, resolved_path).
    """
    path = _resolve_beauty_standard_image_path(filename)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        return None, path

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img, path


def _draw_bs_timer_overlay(canvas, bs_img, size, margin, elapsed, duration):
    """
    Paste bs_img in the top-right corner of canvas and draw a clockwise
    rectangular border timer that grows around the image over `duration` seconds.
    canvas : uint8 BGR, shape (H, W, 3)
    """
    H, W = canvas.shape[:2]

    x0 = W - margin - size
    y0 = margin

    # ---- paste image ----
    if bs_img is not None:
        roi = canvas[y0:y0+size, x0:x0+size]
        if bs_img.shape[2] == 4:
            alpha = bs_img[:, :, 3:4].astype(np.float32) / 255.0
            rgb   = bs_img[:, :, :3].astype(np.float32)
            blended = (alpha * rgb + (1.0 - alpha) * roi.astype(np.float32))
            canvas[y0:y0+size, x0:x0+size] = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            canvas[y0:y0+size, x0:x0+size] = bs_img

    # ---- draw rectangular progress border ----
    frac  = float(np.clip(elapsed / max(duration, 1.0), 0.0, 1.0))
    color = (203, 120, 255)   # blue-orange in BGR
    thick = 8
    pad   = thick // 2 + 2  # offset so stroke sits just outside the image

    # Corner coords of the border rectangle (outside the image by `pad` px)
    rx0 = x0 - pad
    ry0 = y0 - pad
    rx1 = x0 + size + pad
    ry1 = y0 + size + pad

    # Perimeter split into 4 segments, starting from top-left corner, going clockwise:
    #   top edge (left→right), right edge (top→bottom),
    #   bottom edge (right→left), left edge (bottom→top)
    perimeter = 4 * (size + 2 * pad)
    draw_len  = frac * perimeter

    segments = [
        # (start_pt, end_pt, length)
        ((rx0, ry0), (rx1, ry0), rx1 - rx0),   # top
        ((rx1, ry0), (rx1, ry1), ry1 - ry0),   # right
        ((rx1, ry1), (rx0, ry1), rx1 - rx0),   # bottom
        ((rx0, ry1), (rx0, ry0), ry1 - ry0),   # left
    ]

    remaining = draw_len
    for (sx, sy), (ex, ey), seg_len in segments:
        if remaining <= 0:
            break
        t = min(remaining, seg_len) / seg_len
        ex_draw = int(round(sx + t * (ex - sx)))
        ey_draw = int(round(sy + t * (ey - sy)))
        cv2.line(canvas, (int(sx), int(sy)), (ex_draw, ey_draw),
                 color, thick, lineType=cv2.LINE_AA)
        remaining -= seg_len

    return canvas

def run_hand_brush_drag_arap_loop_skeleton(
    cap,
    segmenter,
    hands,
    pose,
    h,
    w,
    V_base,
    V_def,
    T,
    step,
    thresh,
    feather,
    show_mask,
    brush_radius,
    interaction_state,
    arap_cache,
    window_name="Hand brush drag on skeleton mesh",
):
    import os

    global SHOW_MESH_OUTLINE
    print("Entered run_hand_brush_drag_arap_loop_skeleton")
    binding = interaction_state["binding"]

    # ---- Beauty-standard overlay setup ----
    _BS_SIZE = 240          # square size on the OUTPUT canvas (pixels)          # square size on the OUTPUT canvas (pixels)
    _BS_MARGIN = 18         # distance from top-right corner
    _TIMER_DURATION = 180.0 # seconds (3 minutes)
    _timer_start = time.time()

    _bs_img_raw, _bs_img_path = load_beauty_standard_overlay(size=_BS_SIZE)
    if _bs_img_raw is None:
        print(f"WARNING: could not load beauty standard image: {_bs_img_path}")

    # Save exactly what is displayed to the user
    save_dir = os.path.join("saved_frames", "arap_")
    os.makedirs(save_dir, exist_ok=True)
    frame_counter = 0

    while True:
        ok, frame_raw = cap.read()
        if not ok:
            print("Main loop: cap.read() failed, breaking")
            break

        frame_raw = rotate_frame(frame_raw)

        if frame_raw.shape[:2] != (h, w):
            frame_raw = cv2.resize(frame_raw, (w, h), interpolation=cv2.INTER_LINEAR)

        frame_raw = cv2.flip(frame_raw, 1)

        seg_mask, rgb = PTh.get_segmentation_mask(frame_raw, segmenter, feather=feather)

        cur_pts = extract_pose_points(rgb, pose, w, h, min_vis=0.45)
        stable_pts = smooth_pose_points(
            cur_pts,
            interaction_state,
            alpha=0.88,
            max_jump_px=120.0,
            hold_frames=1,
        )

        V_track = None
        frames = None

        # 2. current undeformed triangle coordinates
        if stable_pts is not None:
            V_track, frames = reconstruct_tracked_mesh_from_skeleton(binding, stable_pts, V_base)

            # 3. current deformed triangle coordinates
            V_current_def, _ = reconstruct_deformed_mesh_from_skeleton(binding, stable_pts, V_base)
            if V_current_def is not None:
                V_def[:] = V_current_def

        active = binding["tri_active"]

        # -------- HAND LOGIC --------
        hand_state = PTh.get_hand_state(rgb, hands, seg_mask, thresh, w, h)
        hand_center = hand_state["center"]
        hand_is_open = hand_state["is_open"]
        hand_is_fist = hand_state["is_fist"]
        hand_over_body = hand_state["over_body"]
        hand_detected = hand_state["detected"]
        hand_handedness_raw = hand_state.get("handedness", None)

        # Because the frame is mirrored before Hands processing / interaction,
        # swap the label for selection-blocking logic.
        if hand_handedness_raw == "Left":
            hand_handedness = "Right"
        elif hand_handedness_raw == "Right":
            hand_handedness = "Left"
        else:
            hand_handedness = None

        new_preview_vertices = np.zeros(len(V_def), dtype=bool)
        new_preview_triangles = np.zeros(len(T), dtype=bool)

        if hand_detected:
            if (not interaction_state["dragging"]) and hand_is_open and hand_over_body:
                new_preview_vertices, new_preview_triangles = TMh.compute_brush_selection(
                    V=V_def,
                    T=T,
                    active=active,
                    center=hand_center,
                    radius=brush_radius,
                )

                new_preview_vertices, new_preview_triangles = filter_selection_disallow_same_side_arm(
                    selection_vertices=new_preview_vertices,
                    selection_triangles=new_preview_triangles,
                    T=T,
                    binding=binding,
                    handedness_label=hand_handedness,
                )

            if (
                (not interaction_state["dragging"])
                and interaction_state["hand_was_open"]
                and hand_is_fist
                and np.any(interaction_state["preview_vertices"])
            ):
                print(">>> DRAG STARTED")
                interaction_state["dragging"] = True
                interaction_state["drag_vertices"] = interaction_state["preview_vertices"].copy()
                interaction_state["drag_triangles"] = np.any(
                    interaction_state["drag_vertices"][T], axis=1
                )
                interaction_state["prev_hand_center"] = hand_center

            elif (
                interaction_state["dragging"]
                and hand_is_fist
                and hand_center is not None
                and interaction_state["prev_hand_center"] is not None
            ):
                dx = hand_center[0] - interaction_state["prev_hand_center"][0]
                dy = hand_center[1] - interaction_state["prev_hand_center"][1]
                interaction_state["prev_hand_center"] = hand_center

                if abs(dx) >= 1 or abs(dy) >= 1:
                    print("dx dy:", dx, dy)

                    V_new = TMh.apply_arap_drag_step(
                        V_track=V_track,
                        V_def=V_def,
                        T=T,
                        active_triangles=active,
                        drag_vertices=interaction_state["drag_vertices"],
                        delta_xy=np.array([dx, dy], dtype=np.float32),
                        arap_cache=arap_cache,
                        region_rings=6,
                        n_iters=5,
                        falloff_power=1.6,
                    )

                    if frames is not None:
                        update_local_offsets_from_world(binding, frames, V_new)

                    # Recompute the deformed mesh after offsets changed
                    V_rebuilt, _ = reconstruct_deformed_mesh_from_skeleton(binding, stable_pts, V_base)
                    if V_rebuilt is not None:
                        V_def[:] = V_rebuilt

            elif interaction_state["dragging"] and (not hand_is_fist):
                interaction_state["dragging"] = False
                interaction_state["drag_vertices"][:] = False
                interaction_state["drag_triangles"][:] = False
                interaction_state["prev_hand_center"] = None

            if not interaction_state["dragging"]:
                interaction_state["preview_vertices"] = new_preview_vertices
                interaction_state["preview_triangles"] = new_preview_triangles

            interaction_state["hand_was_open"] = hand_is_open

        else:
            if interaction_state["dragging"]:
                interaction_state["dragging"] = False
                interaction_state["drag_vertices"][:] = False
                interaction_state["drag_triangles"][:] = False
                interaction_state["prev_hand_center"] = None

            interaction_state["preview_vertices"][:] = False
            interaction_state["preview_triangles"][:] = False
            interaction_state["hand_was_open"] = False

        # -------- 4. BUILD OUTPUT TEXTURE MAP --------
        layered_layers = None
        render_order_used = FIXED_RENDER_ORDER
        yaw_amount = 0.0
        yaw_sign = 0.0
        yaw_state_txt = "frontal"
        yaw_debug = {"sign_value": 0.0, "sign_value_smooth": 0.0, "width_ratio": 1.0}

        leg_overlap_frac = 0.0
        leg_front_score = 0.0
        leg_override_used = False

        left_arm_overlap_frac = 0.0
        right_arm_overlap_frac = 0.0
        left_arm_score = 0.0
        right_arm_score = 0.0
        arm_override_used = False

        hole_mask_vis = None
        bg_plate = interaction_state.get("bg_plate", None)

        if V_track is not None and bg_plate is not None:
            if SHOW_LAYERED_RENDER and ("tri_render_group" in binding):
                if USE_DYNAMIC_YAW_RENDER_ORDER and (stable_pts is not None) and ("body_metrics" in interaction_state):
                    render_order_used, yaw_amount, yaw_sign, yaw_state_txt, yaw_debug = compute_render_order_from_yaw(
                        stable_pts,
                        interaction_state["body_metrics"],
                        interaction_state,
                    )
                else:
                    render_order_used = FIXED_RENDER_ORDER

                layered_layers = build_layered_body_layers_fast(
                    frame_raw=frame_raw,
                    V_track=V_track,
                    V_def=V_def,
                    T=T,
                    binding=binding,
                    min_area=4.0,
                    min_bbox=3.0,
                    mask_dilate_ksize=LAYER_MASK_DILATE_KSIZE,
                    mask_blur_ksize=LAYER_MASK_BLUR_KSIZE,
                )

                if stable_pts is not None and layered_layers is not None:
                    render_order_legged, leg_overlap_frac, leg_front_score, leg_override_used = compute_render_order_with_leg_override(
                        base_order=render_order_used,
                        layers=layered_layers,
                        stable_pts=stable_pts,
                        yaw_state_txt=yaw_state_txt,
                    )
                    render_order_used = render_order_legged

                if stable_pts is not None and layered_layers is not None:
                    render_order_used, left_arm_overlap_frac, right_arm_overlap_frac, left_arm_score, right_arm_score, arm_override_used = compute_render_order_with_arm_override(
                        base_order=render_order_used,
                        layers=layered_layers,
                        stable_pts=stable_pts,
                        yaw_state_txt=yaw_state_txt,
                    )

                # Build total mesh mask from all active triangles
                mesh_mask_u8 = build_active_mesh_mask_fast(h, w, V_def, T, active)

                # First fill only the holes with background
                base_hole_filled, hole_mask_vis = composite_mesh_with_background_holefill(
                    frame_raw=frame_raw,
                    bg_plate=bg_plate,
                    warped_mesh_bgr=None,
                    mesh_mask_u8=mesh_mask_u8,
                    seg_mask=seg_mask,
                    body_thresh=0.50,
                    dilate_ksize=7,
                    blur_ksize=5,
                    apply_mesh=False,
                )

                # Then composite the layered mesh onto that cleaned base
                warped_frame = composite_prebuilt_layers(
                    base_bgr=base_hole_filled,
                    layers=layered_layers,
                    render_order=render_order_used,
                )

            else:
                mesh_mask_u8 = build_active_mesh_mask_fast(h, w, V_def, T, active)

                # Step 1: build the hole-filled base, but do NOT place mesh yet
                base_hole_filled, hole_mask_vis = composite_mesh_with_background_holefill(
                    frame_raw=frame_raw,
                    bg_plate=bg_plate,
                    warped_mesh_bgr=None,
                    mesh_mask_u8=mesh_mask_u8,
                    seg_mask=seg_mask,
                    body_thresh=0.50,
                    dilate_ksize=7,
                    blur_ksize=5,
                    apply_mesh=False,
                )

                # Step 2: warp directly into that base
                warped_frame = warp_mesh_piecewise(
                    src_img=frame_raw,
                    V_src=V_track,
                    V_dst=V_def,
                    T=T,
                    active_mask=active,
                    dst_img=base_hole_filled.copy(),
                    min_area=4.0,
                    min_bbox=3.0,
                )
        else:
            warped_frame = frame_raw.copy()

        # -------- VISUALIZATION --------
        if interaction_state["dragging"]:
            affected_vertices = interaction_state["drag_vertices"]
            affected_triangles = interaction_state["drag_triangles"] & active
        else:
            affected_vertices = interaction_state["preview_vertices"]
            affected_triangles = interaction_state["preview_triangles"]

        vis = warped_frame.copy()

        if SHOW_RENDER_GROUP_DEBUG and ("tri_render_group" in binding):
            vis = draw_triangle_render_group_overlay(
                frame=vis,
                V=V_def,
                T=T,
                tri_active=active,
                tri_render_group=binding["tri_render_group"],
                alpha=RENDER_GROUP_DEBUG_ALPHA,
                line_thickness=1,
            )

        if SHOW_MESH_OUTLINE:
            vis = TMh._draw_triangle_overlay(
                frame=vis,
                V=V_def,
                T=T,
                active_mask=active,
                affected_mask=affected_triangles,
                pale_color=(0, 0, 0),
                bright_color=(0, 0, 0),
                pale_alpha=0.12,
                bright_alpha=0.42,
                line_thickness=1,
            )

            if np.any(affected_vertices):
                pts = np.round(V_def[affected_vertices]).astype(np.int32)
                for x, y in pts:
                    cv2.circle(vis, (x, y), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)

            if hand_center is not None and not interaction_state["dragging"]:
                cx, cy = hand_center
                brush_color = (0, 255, 0) if (hand_is_open and hand_over_body) else (180, 180, 180)
                cv2.circle(vis, (cx, cy), brush_radius, brush_color, 2, lineType=cv2.LINE_AA)
                cv2.circle(vis, (cx, cy), 4, brush_color, -1, lineType=cv2.LINE_AA)

        else:
            if np.any(affected_triangles):
                if interaction_state["dragging"]:
                    vis = draw_filled_triangle_highlight(
                        frame=vis,
                        V=V_def,
                        T=T,
                        tri_mask=affected_triangles,
                        fill_color=(255, 150, 246),
                        fill_alpha=0.56,
                        edge_color=(255, 150, 246),
                        edge_thickness=1,
                        edge_alpha=0.95,
                    )
                else:
                    vis = draw_filled_triangle_highlight(
                        frame=vis,
                        V=V_def,
                        T=T,
                        tri_mask=affected_triangles,
                        fill_color=(255, 150, 246),
                        fill_alpha=0.14,
                        edge_color=(255, 150, 246),
                        edge_thickness=1,
                        edge_alpha=0.95,
                    )

        # cv2.putText(
        #     vis,
        #     f"fixed active: {int(active.sum())}/{len(T)}",
        #     (12, 28),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (255, 255, 255),
        #     2,
        #     cv2.LINE_AA
        # )
        # mode_text = "layered render" if SHOW_LAYERED_RENDER else "single-sheet render"

        # cv2.putText(
        #     vis,
        #     f"skeleton-attached mesh | {mode_text} | m mesh on/off | q quit | r reset offsets",
        #     (12, 56),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.65,
        #     (255, 255, 255),
        #     2,
        #     cv2.LINE_AA
        # )

        # if SHOW_LAYERED_RENDER and USE_DYNAMIC_YAW_RENDER_ORDER:
        #     cv2.putText(
        #         vis,
        #         f"yaw: {yaw_amount:.2f} | sign: {yaw_sign:+.0f} | state: {yaw_state_txt}",
        #         (12, 84),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.55,
        #         (255, 255, 255),
        #         2,
        #         cv2.LINE_AA
        #     )

        #     cv2.putText(
        #         vis,
        #         f"sign_raw: {yaw_debug['sign_value']:+.2f} | sign_smooth: {yaw_debug['sign_value_smooth']:+.2f} | width_ratio: {yaw_debug['width_ratio']:.2f}",
        #         (12, 132),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.46,
        #         (255, 255, 255),
        #         1,
        #         cv2.LINE_AA
        #     )

        # if SHOW_RENDER_GROUP_DEBUG:
        #     cv2.putText(
        #         vis,
        #         "debug render groups: torso yellow | head magenta | L arm green | R arm orange | L leg blue | R leg red",
        #         (12, 108),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.48,
        #         (255, 255, 255),
        #         1,
        #         cv2.LINE_AA
        #     )

        # if SHOW_LAYERED_RENDER:
        #     cv2.putText(
        #         vis,
        #         f"leg overlap: {leg_overlap_frac:.3f} | leg score: {leg_front_score:+.1f} | leg override: {leg_override_used}",
        #         (12, 156),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.46,
        #         (255, 255, 255),
        #         1,
        #         cv2.LINE_AA
        #     )
        #     cv2.putText(
        #         vis,
        #         f"L arm ov: {left_arm_overlap_frac:.3f} | R arm ov: {right_arm_overlap_frac:.3f} | L score: {left_arm_score:+.1f} | R score: {right_arm_score:+.1f} | arm override: {arm_override_used}",
        #         (12, 180),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.46,
        #         (255, 255, 255),
        #         1,
        #         cv2.LINE_AA
        #     )

        vis_display = cv2.resize(vis, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)

        if _bs_img_raw is not None:
            _elapsed = time.time() - _timer_start
            vis_display = _draw_bs_timer_overlay(
                canvas=vis_display,
                bs_img=_bs_img_raw,
                size=_BS_SIZE,
                margin=_BS_MARGIN,
                elapsed=_elapsed,
                duration=_TIMER_DURATION,
            )

        # Save exactly what the user sees every 10 frames
        if frame_counter % 10 == 0:
            save_path = os.path.join(save_dir, f"frame_{frame_counter:06d}.png")
            cv2.imwrite(save_path, vis_display)

        frame_counter += 1

        cv2.imshow(window_name, vis_display)

        if show_mask:
            mask_vis = (seg_mask * 255).astype(np.uint8)
            mask_display = cv2.resize(mask_vis, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Segmentation mask", mask_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        elif key == ord('m'):
            SHOW_MESH_OUTLINE = not SHOW_MESH_OUTLINE

        elif key == ord('r'):
            binding["vertex_local_uv_offset"][:] = 0.0
            interaction_state["preview_vertices"][:] = False
            interaction_state["preview_triangles"][:] = False
            interaction_state["drag_vertices"][:] = False
            interaction_state["drag_triangles"][:] = False
            interaction_state["dragging"] = False
            interaction_state["prev_hand_center"] = None
            interaction_state["hand_was_open"] = False

    print("Exiting run_hand_brush_drag_arap_loop_skeleton")


def test_hand_brush_drag_arap_live_skeleton(step=0, thresh=0.5, feather=0, show_mask=False):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return
    print("Camera found")

    ok, frame = cap.read()
    if not ok:
        print("Error: could not read initial frame.")
        cap.release()
        return

    frame = rotate_frame(frame)
    h, w = frame.shape[:2]

    V_base, T, nx, ny = TMh.build_grid_mesh(w, h, step=step)
    V_def = V_base.copy().astype(np.float32)

    E = TMh.build_unique_edges(T)
    neighbors = TMh.build_vertex_neighbors(len(V_base), E)
    arap_cache = {"E": E, "neighbors": neighbors}

    brush_radius = max(45, int(min(w, h) * 0.07))

    interaction_state = {
        "preview_vertices": np.zeros(len(V_def), dtype=bool),
        "preview_triangles": np.zeros(len(T), dtype=bool),
        "drag_vertices": np.zeros(len(V_def), dtype=bool),
        "drag_triangles": np.zeros(len(T), dtype=bool),
        "dragging": False,
        "prev_hand_center": None,
        "hand_was_open": False,

        "pose_prev_pts": None,
        "pose_last_good_pts": None,
        "pose_missing_count": 0,

        "binding": None,
        "yaw_side_state": "frontal",
        "yaw_sign_value_smooth": 0.0,

        "bg_plate": None,
    }

    window_name = "Hand brush drag on skeleton mesh"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, PRIMARY_MONITOR_WIDTH, 0)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter, \
         mp.solutions.hands.Hands(
             static_image_mode=False,
             max_num_hands=1,
             model_complexity=1,
             min_detection_confidence=0.5,
             min_tracking_confidence=0.5,
         ) as hands, \
         mp_pose.Pose(
             static_image_mode=False,
             model_complexity=1,
             smooth_landmarks=False,
             enable_segmentation=False,
             min_detection_confidence=0.5,
             min_tracking_confidence=0.5,
         ) as pose:
        
        print("Preparing to capture background...")

        show_countdown(
            cap=cap,
            h=h,
            w=w,
            seconds=5,
            message="Capturing background",
            window_name=window_name,
        )

        bg_plate = capture_background_plate(
            cap=cap,
            h=h,
            w=w,
            n_frames=20,
            window_name=window_name,
        )

        if bg_plate is None:
            print("Could not capture background plate.")
            cap.release()
            cv2.destroyAllWindows()
            return

        interaction_state["bg_plate"] = bg_plate

        print("Preparing initialization...")

        show_countdown(
            cap=cap,
            h=h,
            w=w,
            seconds=5,
            message="Initialization starting",
            window_name=window_name,
        )

        print("Entering initialization loop")

        initialized = False
        max_init_frames = 300
        full_pose_streak = 0
        required_streak =20

        mask_accum = None
        mask_count = 0
        best_stable_pts = None

        for k in range(max_init_frames):
            ok, fr = cap.read()
            if not ok:
                print(f"init frame {k}: camera read failed")
                break

            fr = rotate_frame(fr)

            if fr.shape[:2] != (h, w):
                fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_LINEAR)

            fr = cv2.flip(fr, 1)
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

            cur_pts = extract_pose_points(rgb, pose, w, h, min_vis=0.25)
            stable_pts = smooth_pose_points(
                cur_pts,
                interaction_state,
                alpha=0.65,
                max_jump_px=45.0,
                hold_frames=2,
            )

            full_pose_ok = has_required_landmarks(stable_pts)

            if full_pose_ok:
                full_pose_streak += 1

                seg_mask_init, _ = PTh.get_segmentation_mask(fr, segmenter, feather=feather)
                if mask_accum is None:
                    mask_accum = np.zeros_like(seg_mask_init, dtype=np.float32)

                mask_accum += seg_mask_init.astype(np.float32)
                mask_count += 1
                best_stable_pts = stable_pts
            else:
                full_pose_streak = 0
                mask_accum = None
                mask_count = 0
                best_stable_pts = None

            init_vis = fr.copy()
            msg = f"Waiting for full-body pose... streak={full_pose_streak}/{required_streak}"
            color = (0, 255, 255)

            if full_pose_streak >= required_streak and mask_count > 0 and best_stable_pts is not None:
                agg_mask = finalize_init_mask(
                    mask_accum,
                    mask_count,
                    avg_thresh=0.28,
                    dilate_ksize=13,
                    close_ksize=11,
                )

                pose_capsule_mask = rasterize_pose_capsules(agg_mask.shape, best_stable_pts)
                init_mask_final = np.maximum(agg_mask, pose_capsule_mask)

                # Keep the full arms/hands in the fixed init mesh.
                init_mask_final = trim_mask_arms_combined(
                    init_mask_final,
                    best_stable_pts,
                    shoulder_keep_arm_frac=1.0,
                    shoulder_cap_scale=0.24,
                    elbow_keep_forearm_frac=1.0,
                    min_component_area=250,
                )

                V_base_shaped = shape_mesh_boundary_to_mask(
                    V_base,
                    T,
                    init_mask_final,
                    mask_thresh=0.5,
                    snap_dist_px=max(12.0, 1.15 * step),
                    smooth_iters=1,
                )
                V_def[:] = V_base_shaped

                binding = bind_mesh_to_skeleton(
                    V_base=V_base_shaped,
                    T=T,
                    ref_pts=best_stable_pts,
                    init_seg_mask=init_mask_final,
                    mask_thresh=0.5,
                )

                print(f"init frame {k}: binding is {'ok' if binding is not None else 'None'}")
                if binding is not None:
                    tri_render_group = build_triangle_render_groups(binding, T)
                    binding["tri_render_group"] = tri_render_group
                    binding["group_tri_indices"] = build_render_group_triangle_index_cache(binding)
                    print_render_group_triangle_stats(binding)

                    V_base = V_base_shaped.copy().astype(np.float32)
                    V_def = V_base.copy().astype(np.float32)
                    interaction_state["binding"] = binding
                    interaction_state["body_metrics"] = compute_body_metrics(best_stable_pts)
                    initialized = True
                    print("Initialization succeeded, binding created")
                    msg = "Pose locked. Starting..."
                    color = (0, 255, 0)

            cv2.putText(
                init_vis,
                msg,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA,
            )

            if stable_pts is not None:
                for name, p in stable_pts.items():
                    if p is not None:
                        x, y = int(p[0]), int(p[1])
                        cv2.circle(init_vis, (x, y), 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)

            init_display = cv2.resize(init_vis, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(window_name, init_display)
            key = cv2.waitKey(1) & 0xFF

            if initialized:
                break
            if key == ord('q'):
                print("User quit during initialization")
                break

        print(f"Init loop done. initialized={initialized}")

        if not initialized:
            print("Could not initialize full-body pose binding.")
            cap.release()
            cv2.destroyAllWindows()
            return

        print("About to enter main skeleton loop")

        run_hand_brush_drag_arap_loop_skeleton(
            cap=cap,
            segmenter=segmenter,
            hands=hands,
            pose=pose,
            h=h,
            w=w,
            V_base=V_base,
            V_def=V_def,
            T=T,
            step=step,
            thresh=thresh,
            feather=feather,
            show_mask=show_mask,
            brush_radius=brush_radius,
            interaction_state=interaction_state,
            arap_cache=arap_cache,
            window_name=window_name,
        )

        print("Main skeleton loop returned")

    cap.release()
    cv2.destroyAllWindows()


test_hand_brush_drag_arap_live_skeleton(step=25, thresh=0.5, feather=0, show_mask=False)