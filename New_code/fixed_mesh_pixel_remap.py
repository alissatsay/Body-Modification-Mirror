import cv2
import numpy as np
import mediapipe as mp

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
    "right_upper_arm",
    "right_lower_arm",
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

def estimate_body_yaw(cur_pts, ref_metrics):
    ls = cur_pts.get("left_shoulder")
    rs = cur_pts.get("right_shoulder")
    lh = cur_pts.get("left_hip")
    rh = cur_pts.get("right_hip")
    nose = cur_pts.get("nose")

    if ls is None or rs is None or lh is None or rh is None:
        return 0.0, 0.0

    cur_shoulder_w = float(np.linalg.norm(rs - ls))
    cur_hip_w = float(np.linalg.norm(rh - lh))

    ref_shoulder_w = max(ref_metrics["shoulder_width"], 1.0)
    ref_hip_w = max(ref_metrics["hip_width"], 1.0)

    shoulder_ratio = np.clip(cur_shoulder_w / ref_shoulder_w, 0.0, 1.2)
    hip_ratio = np.clip(cur_hip_w / ref_hip_w, 0.0, 1.2)

    width_ratio = 0.5 * (shoulder_ratio + hip_ratio)

    # width_ratio near 1 -> frontal
    # width_ratio much smaller -> sideways
    yaw_amount = np.clip((1.0 - width_ratio) / 0.55, 0.0, 1.0)

    yaw_sign = 0.0
    if nose is not None:
        shoulder_mid = 0.5 * (ls + rs)
        shoulder_vec = rs - ls
        shoulder_len = np.linalg.norm(shoulder_vec)
        if shoulder_len > 1e-6:
            shoulder_x = shoulder_vec / shoulder_len
            nose_offset = np.dot(nose - shoulder_mid, shoulder_x)
            yaw_sign = float(np.sign(nose_offset))

    return float(yaw_amount), float(yaw_sign)

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
    shoulder_keep_arm_frac=0.03,
    shoulder_cap_scale=0.24,
    elbow_keep_forearm_frac=0.0,
    min_component_area=250,
):
    """
    Combine shoulder-level and elbow-level trimming, then remove tiny leftover blobs.
    """
    out = mask.copy()

    # First remove everything distal to shoulders, but keep a small shoulder cap
    out = trim_mask_distal_to_shoulders(
        out,
        pts,
        keep_arm_frac=shoulder_keep_arm_frac,
        shoulder_cap_scale=shoulder_cap_scale,
    )

    # Then also remove anything distal to elbows, which helps kill wrist remnants
    out = trim_mask_distal_to_elbows(
        out,
        pts,
        keep_forearm_frac=elbow_keep_forearm_frac,
    )

    # Remove tiny disconnected leftovers like wrist blobs
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
    segmentation misses them. Arms are not protected here, only torso/head/legs.
    """
    h, w = mask_shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)

    segments = [
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

def safe_normalize(v, eps=1e-6):
    n = np.linalg.norm(v)
    if n < eps:
        return np.array([1.0, 0.0], dtype=np.float32), 1.0
    return (v / n).astype(np.float32), float(n)


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

def rasterize_pose_capsules(mask_shape, pts):
    """
    Make a binary mask from thick body segments so thin regions are not lost if
    segmentation misses them. Arms are protected only up to the elbows.
    """
    h, w = mask_shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)

    segments = [
        ("left_shoulder", "left_elbow", 26),
        ("right_shoulder", "right_elbow", 26),
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
    Build a fixed mesh whose geometry comes from the initialization silhouette,
    then bind its vertices to torso/head/legs for motion.
    Arms are excluded, but shoulder caps are preserved as part of the torso.
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
    shoulder_w = np.linalg.norm(ref_pts["right_shoulder"] - ref_pts["left_shoulder"])
    shoulder_y = min(ref_pts["left_shoulder"][1], ref_pts["right_shoulder"][1])

    # Wider + higher torso so the figure keeps its shoulder breadth
    torso_quad = torso_quad_from_pts(
        ref_pts,
        expand_x=0.38,
        expand_y_top=0.34,
        expand_y_bottom=0.12,
    )

    capsule_defs = []

    if ref_pts.get("left_hip") is not None and ref_pts.get("left_knee") is not None:
        capsule_defs.append(("left_thigh", ref_pts["left_hip"], ref_pts["left_knee"], 0.28))
    if ref_pts.get("left_knee") is not None and ref_pts.get("left_ankle") is not None:
        capsule_defs.append(("left_calf", ref_pts["left_knee"], ref_pts["left_ankle"], 0.24))

    if ref_pts.get("right_hip") is not None and ref_pts.get("right_knee") is not None:
        capsule_defs.append(("right_thigh", ref_pts["right_hip"], ref_pts["right_knee"], 0.28))
    if ref_pts.get("right_knee") is not None and ref_pts.get("right_ankle") is not None:
        capsule_defs.append(("right_calf", ref_pts["right_knee"], ref_pts["right_ankle"], 0.24))

    if ref_pts.get("nose") is not None:
        head_center = ref_pts["nose"] + np.array([0.0, -0.18 * shoulder_w], dtype=np.float32)
    else:
        head_center = shoulder_mid + np.array([0.0, -0.75 * shoulder_w], dtype=np.float32)

    head_radius = max(36.0, 0.70 * shoulder_w)

    torso_left = min(ref_pts["left_shoulder"][0], ref_pts["left_hip"][0]) - 0.22 * shoulder_w
    torso_right = max(ref_pts["right_shoulder"][0], ref_pts["right_hip"][0]) + 0.22 * shoulder_w

    left_shoulder = ref_pts.get("left_shoulder")
    right_shoulder = ref_pts.get("right_shoulder")
    shoulder_cap_r = max(20.0, 0.24 * shoulder_w)

    for i, p in enumerate(V_base):
        if not vertex_in_mask[i]:
            continue

        assigned = False

        # 1) torso first
        if point_in_quad(p, torso_quad):
            seg = "torso"
            vertex_segment[i] = SEGMENT_INDEX[seg]
            vertex_local_uv[i] = localize_point_in_frame(p, ref_frames[seg])
            assigned = True

        # 2) explicit shoulder-cap zones -> torso
        if not assigned:
            in_left_cap = left_shoulder is not None and np.linalg.norm(p - left_shoulder) <= shoulder_cap_r
            in_right_cap = right_shoulder is not None and np.linalg.norm(p - right_shoulder) <= shoulder_cap_r
            if in_left_cap or in_right_cap:
                seg = "torso"
                vertex_segment[i] = SEGMENT_INDEX[seg]
                vertex_local_uv[i] = localize_point_in_frame(p, ref_frames[seg])
                assigned = True

        # 3) head second
        if (not assigned) and (np.linalg.norm(p - head_center) <= head_radius):
            seg = "head"
            vertex_segment[i] = SEGMENT_INDEX[seg]
            vertex_local_uv[i] = localize_point_in_frame(p, ref_frames[seg])
            assigned = True

        # 4) protect neck / upper torso band
        if not assigned:
            neck_band_top = shoulder_y - 0.24 * shoulder_w
            neck_band_bottom = shoulder_y + 0.22 * shoulder_w
            in_neck_band = (
                neck_band_top <= p[1] <= neck_band_bottom and
                torso_left <= p[0] <= torso_right
            )
            if in_neck_band:
                seg = "torso"
                vertex_segment[i] = SEGMENT_INDEX[seg]
                vertex_local_uv[i] = localize_point_in_frame(p, ref_frames[seg])
                assigned = True

        # 5) legs only after torso/head protection
        if not assigned and len(capsule_defs) > 0:
            best_seg = None
            best_score = np.inf

            for seg, a, b, radius_scale in capsule_defs:
                seg_len = np.linalg.norm(b - a)
                radius = max(12.0, radius_scale * seg_len)
                dist, _, _ = point_segment_distance(p, a, b)

                if dist <= radius:
                    score = dist
                else:
                    score = radius + 2.0 * (dist - radius)

                if score < best_score:
                    best_score = score
                    best_seg = seg

            if best_seg is not None:
                vertex_segment[i] = SEGMENT_INDEX[best_seg]
                vertex_local_uv[i] = localize_point_in_frame(p, ref_frames[best_seg])

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
    print("Entered run_hand_brush_drag_arap_loop_skeleton")
    binding = interaction_state["binding"]

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
            alpha=0.65,
            max_jump_px=45.0,
            hold_frames=2,
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
                        V_track=V_def.copy(),
                        V_def=V_def,
                        T=T,
                        active_triangles=active,
                        drag_vertices=interaction_state["drag_vertices"],
                        delta_xy=np.array([dx, dy], dtype=np.float32),
                        arap_cache=arap_cache,
                        region_rings=4,
                        n_iters=3,
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
        if V_track is not None:
            # source texture = current raw frame
            # source triangles = current undeformed mesh
            # dest triangles   = current deformed mesh
            warped_frame = warp_mesh_piecewise(
                src_img=frame_raw,
                V_src=V_track,
                V_dst=V_def,
                T=T,
                active_mask=active,
                dst_img=frame_raw.copy(),
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

        vis = TMh._draw_triangle_overlay(
            frame=warped_frame,
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

        cv2.putText(
            vis,
            f"fixed active: {int(active.sum())}/{len(T)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            vis,
            "skeleton-attached mesh | q quit | r reset offsets",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        vis_display = cv2.resize(vis, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(window_name, vis_display)

        if show_mask:
            mask_vis = (seg_mask * 255).astype(np.uint8)
            mask_display = cv2.resize(mask_vis, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Segmentation mask", mask_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

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

    brush_radius = max(45, int(min(w, h) * 0.05))

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

                # Remove the arms from the fixed init mesh starting at the shoulders.
                init_mask_final = trim_mask_arms_combined(
                    init_mask_final,
                    best_stable_pts,
                    shoulder_keep_arm_frac=0.03,
                    shoulder_cap_scale=0.24,
                    elbow_keep_forearm_frac=0.0,
                    min_component_area=250,
                )

                binding = bind_mesh_to_skeleton(
                    V_base=V_base,
                    T=T,
                    ref_pts=best_stable_pts,
                    init_seg_mask=init_mask_final,
                    mask_thresh=0.5,
                )

                print(f"init frame {k}: binding is {'ok' if binding is not None else 'None'}")
                if binding is not None:
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


test_hand_brush_drag_arap_live_skeleton(step=15, thresh=0.5, feather=0, show_mask=False)