import cv2
import numpy as np
import mediapipe as mp
from collections import deque

mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


def _pose_band_from_landmarks(landmarks, w, h,
                              band_half_width_frac=0.22,
                              top_frac_of_torso=0.15,
                              bot_frac_of_torso=0.75):
    L_SH  = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    R_SH  = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    L_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
    R_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value

    ls = landmarks[L_SH]; rs = landmarks[R_SH]
    lh = landmarks[L_HIP]; rh = landmarks[R_HIP]

    vis_ok = (ls.visibility > 0.4 and rs.visibility > 0.4
              and lh.visibility > 0.4 and rh.visibility > 0.4)
    if not vis_ok:
        return None

    shoulder_mid_x = 0.5 * (ls.x + rs.x) * w
    shoulder_mid_y = 0.5 * (ls.y + rs.y) * h
    hip_mid_x      = 0.5 * (lh.x + rh.x) * w
    hip_mid_y      = 0.5 * (lh.y + rh.y) * h

    torso_len = abs(hip_mid_y - shoulder_mid_y)
    if torso_len < 20:
        return None

    torso_mid_y = 0.5 * (shoulder_mid_y + hip_mid_y)
    y0 = int(torso_mid_y + top_frac_of_torso * torso_len)
    y1 = int(torso_mid_y + bot_frac_of_torso * torso_len)

    half_w = band_half_width_frac * w
    cx     = hip_mid_x
    x0 = int(cx - half_w); x1 = int(cx + half_w)

    x0 = max(0, min(w - 1, x0)); x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0)); y1 = max(0, min(h - 1, y1))
    if y1 <= y0 or x1 <= x0:
        return None

    return (x0, y0, x1, y1), (hip_mid_x, hip_mid_y), (shoulder_mid_x, shoulder_mid_y), torso_mid_y


def _is_open_palm(hand_landmarks, handedness_label=None, require_extended=4):
    lm = hand_landmarks.landmark
    wrist = np.array([lm[0].x, lm[0].y], dtype=np.float32)

    finger_defs = [
        (5, 6, 8),    # index
        (9, 10, 12),  # middle
        (13, 14, 16), # ring
        (17, 18, 20), # pinky
    ]

    extended = 0
    for mcp_idx, pip_idx, tip_idx in finger_defs:
        mcp = np.array([lm[mcp_idx].x, lm[mcp_idx].y], dtype=np.float32)
        pip = np.array([lm[pip_idx].x, lm[pip_idx].y], dtype=np.float32)
        tip = np.array([lm[tip_idx].x, lm[tip_idx].y], dtype=np.float32)

        d_tip_wrist = np.linalg.norm(tip - wrist)
        d_pip_wrist = np.linalg.norm(pip - wrist)
        d_tip_mcp   = np.linalg.norm(tip - mcp)
        d_pip_mcp   = np.linalg.norm(pip - mcp)

        if (d_tip_wrist > 1.15 * d_pip_wrist) and (d_tip_mcp > 1.10 * d_pip_mcp):
            extended += 1

    thumb_cmc = np.array([lm[1].x, lm[1].y], dtype=np.float32)
    thumb_ip  = np.array([lm[3].x, lm[3].y], dtype=np.float32)
    thumb_tip = np.array([lm[4].x, lm[4].y], dtype=np.float32)

    d_tip_cmc      = np.linalg.norm(thumb_tip - thumb_cmc)
    d_ip_cmc       = np.linalg.norm(thumb_ip  - thumb_cmc)
    thumb_extended = d_tip_cmc > 1.10 * d_ip_cmc

    num_extended = extended + int(thumb_extended)
    is_open      = extended >= require_extended

    return is_open, num_extended


# ==============================================================================
# HAND-SWITCH GESTURE: Peace / V-sign held for N frames
# ==============================================================================
#
# The user holds up a peace sign (index + middle extended, ring + pinky + thumb
# folded).  After HOLD_FRAMES consecutive frames of detecting this pose the
# gesture fires and the active hand cycles:
#     None  ->  whichever hand made the sign  ->  None  -> ...
#
# This is a static hold, not a motion, so it is robust to hand jitter and does
# not accidentally fire during normal open-palm interaction (which requires 4
# fingers extended).

def _is_peace_sign(hand_landmarks):
    """
    Returns True when the hand shows a peace / V sign:
      • Index finger extended
      • Middle finger extended
      • Ring finger folded
      • Pinky finger folded
      • Thumb folded (or tucked — we are lenient here)

    Orientation-invariant: uses wrist-distance ratios, not absolute direction.
    """
    lm    = hand_landmarks.landmark
    wrist = np.array([lm[0].x, lm[0].y], dtype=np.float32)

    def dist(i, j):
        return np.linalg.norm(
            np.array([lm[i].x, lm[i].y], dtype=np.float32) -
            np.array([lm[j].x, lm[j].y], dtype=np.float32)
        )

    # Each entry: (pip_idx, tip_idx, should_be_extended)
    checks = [
        (6,  8,  True),   # index
        (10, 12, True),   # middle
        (14, 16, False),  # ring
        (18, 20, False),  # pinky
    ]

    for pip_i, tip_i, want_extended in checks:
        d_tip  = dist(0, tip_i)   # wrist to tip
        d_pip  = dist(0, pip_i)   # wrist to pip
        is_ext = d_tip > 1.15 * d_pip
        if is_ext != want_extended:
            return False

    # Thumb: should be folded — tip not much farther than ip from cmc
    d_thumb_tip = dist(1, 4)
    d_thumb_ip  = dist(1, 3)
    thumb_folded = d_thumb_tip <= 1.25 * d_thumb_ip
    if not thumb_folded:
        return False

    return True


class PeaceSignHoldDetector:
    """
    Fires once after the peace sign has been held for HOLD_FRAMES consecutive
    frames.  After firing there is a COOLDOWN_FRAMES lockout to prevent
    double-triggers.

    Usage:
        detector = PeaceSignHoldDetector()
        fired = detector.update(hand_landmarks)   # True once per hold
    """

    HOLD_FRAMES     = 20   # ~0.67 s at 30 fps — long enough to be intentional
    COOLDOWN_FRAMES = 45   # ~1.5 s cooldown after firing

    def __init__(self):
        self._streak   = 0
        self._cooldown = 0

    def update(self, hand_landmarks):
        """Returns True the frame the gesture fires, False otherwise."""
        return self.update_raw(_is_peace_sign(hand_landmarks))

    def update_raw(self, is_peace: bool):
        """
        Same as update() but accepts a pre-computed bool.
        Used by the main loop which reads is_peace from hand_state.
        """
        if self._cooldown > 0:
            self._cooldown -= 1
            self._streak = 0
            return False

        if is_peace:
            self._streak += 1
        else:
            self._streak = 0

        if self._streak >= self.HOLD_FRAMES:
            self._streak   = 0
            self._cooldown = self.COOLDOWN_FRAMES
            return True

        return False

    def progress(self):
        """0.0-1.0 fraction of hold completed. Useful for a progress indicator."""
        return min(1.0, self._streak / max(1, self.HOLD_FRAMES))

    def reset(self):
        self._streak   = 0
        self._cooldown = 0


# ==============================================================================
# PINCH GESTURE: thumb + index distance controls brush radius
# ==============================================================================

def _pinch_distance(hand_landmarks):
    """
    Returns the normalised distance between the index fingertip (8) and the
    thumb tip (4).  Small = pinched, large = spread.
    """
    lm    = hand_landmarks.landmark
    thumb = np.array([lm[4].x, lm[4].y], dtype=np.float32)
    index = np.array([lm[8].x, lm[8].y], dtype=np.float32)
    return float(np.linalg.norm(thumb - index))


class PinchGestureTracker:
    """
    Tracks thumb-index pinch distance and locks brush size when fingers go still.

    States
    ------
    IDLE      : spread fingers, waiting. Enters MOVING when dist < PINCH_ENTER_THRESHOLD.
    MOVING    : fingers pinched and moving; brush size updates live as preview.
                Enters SETTLING when movement < STILL_TOLERANCE for one frame.
                Returns to IDLE if fingers spread past SPREAD_THRESHOLD.
    SETTLING  : fingers nearly still; cyan arc fills on screen.
                Enters LOCKED after STILL_FRAMES consecutive still frames.
                Returns to MOVING if fingers move again.
                Returns to IDLE if fingers spread past SPREAD_THRESHOLD.
    LOCKED    : size committed, no deltas. Dim grey circle shown.
                Returns to IDLE when fingers spread past SPREAD_THRESHOLD.

    Usage:
        delta, is_active = tracker.update_from_dist(pinch_dist)
        # delta != 0 only in MOVING. is_active True in MOVING + SETTLING.
    """

    PINCH_ENTER_THRESHOLD = 0.08   # dist below which MOVING begins
    SPREAD_THRESHOLD      = 0.18   # dist above which gesture is cancelled / LOCKED exits
    STILL_TOLERANCE       = 0.003  # max dist change per frame considered "still"
    STILL_FRAMES          = 12     # frames of stillness required to commit
    BRUSH_SCALE_FACTOR    = 500.0  # px change per normalised dist unit

    _STATE_IDLE     = "idle"
    _STATE_MOVING   = "moving"
    _STATE_SETTLING = "settling"
    _STATE_LOCKED   = "locked"

    def __init__(self):
        self._state       = self._STATE_IDLE
        self._prev_dist   = None
        self._still_count = 0

    # ── public properties ──────────────────────────────────────────────────────

    @property
    def is_active(self):
        """True while gesture is in progress (MOVING or SETTLING)."""
        return self._state in (self._STATE_MOVING, self._STATE_SETTLING)

    @property
    def is_locked(self):
        """True once size has been committed and not yet released."""
        return self._state == self._STATE_LOCKED

    @property
    def settling_progress(self):
        """0.0-1.0: how far through the still-hold we are. 0 outside SETTLING."""
        if self._state != self._STATE_SETTLING:
            return 0.0
        return min(1.0, self._still_count / max(1, self.STILL_FRAMES))

    # ── update API ─────────────────────────────────────────────────────────────

    def update(self, hand_landmarks):
        return self.update_from_dist(_pinch_distance(hand_landmarks))

    def update_from_dist(self, dist: float):
        """
        Call once per frame.  dist = normalised thumb-index distance.
        Returns (brush_delta_px: float, is_active: bool).
        """
        if self._state == self._STATE_IDLE:
            if dist < self.PINCH_ENTER_THRESHOLD:
                self._state       = self._STATE_MOVING
                self._prev_dist   = dist
                self._still_count = 0
            return 0.0, False

        if self._state == self._STATE_MOVING:
            if dist > self.SPREAD_THRESHOLD:
                # Spread out — cancel
                self._state     = self._STATE_IDLE
                self._prev_dist = None
                return 0.0, False

            # Compute movement BEFORE updating _prev_dist
            movement = abs(dist - self._prev_dist)
            delta    = (dist - self._prev_dist) * self.BRUSH_SCALE_FACTOR
            self._prev_dist = dist

            if movement < self.STILL_TOLERANCE:
                # Just went still — enter SETTLING
                self._state       = self._STATE_SETTLING
                self._still_count = 1
            return float(delta), True

        if self._state == self._STATE_SETTLING:
            if dist > self.SPREAD_THRESHOLD:
                # Spread out — cancel
                self._state       = self._STATE_IDLE
                self._prev_dist   = None
                self._still_count = 0
                return 0.0, False

            movement = abs(dist - self._prev_dist) if self._prev_dist is not None else 0.0
            self._prev_dist = dist

            if movement >= self.STILL_TOLERANCE:
                # Moving again — back to MOVING
                self._state       = self._STATE_MOVING
                self._still_count = 0
                # Emit delta for this frame so the size responds immediately
                delta = (movement if dist > (self._prev_dist or dist) else -movement)                         * self.BRUSH_SCALE_FACTOR
                return float(delta), True

            # Still quiet — advance counter
            self._still_count += 1
            if self._still_count >= self.STILL_FRAMES:
                # Committed — lock
                self._state       = self._STATE_LOCKED
                self._still_count = 0
                self._prev_dist   = None
            return 0.0, True   # no delta but is_active=True so UI stays visible

        if self._state == self._STATE_LOCKED:
            if dist > self.SPREAD_THRESHOLD:
                self._state = self._STATE_IDLE
            return 0.0, False

        return 0.0, False

    def reset(self):
        self._state       = self._STATE_IDLE
        self._prev_dist   = None
        self._still_count = 0


# ==============================================================================
# Existing helpers (unchanged)
# ==============================================================================

def _hand_center_px(hand_landmarks, w, h):
    lm  = hand_landmarks.landmark
    ids = [0, 5, 9, 13, 17]
    xs  = [lm[i].x * w for i in ids]
    ys  = [lm[i].y * h for i in ids]
    return int(np.mean(xs)), int(np.mean(ys))


def get_segmentation_mask(frame_bgr, segmenter, feather=0):
    rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    seg      = segmenter.process(rgb)
    seg_mask = seg.segmentation_mask.astype(np.float32)

    if feather and feather > 0:
        k = int(feather)
        if k % 2 == 0:
            k += 1
        seg_mask = cv2.GaussianBlur(seg_mask, (k, k), 0)

    return seg_mask, rgb


def get_hand_state(rgb, hands, seg_mask, thresh, w, h):
    """
    Returns a dict describing the current hand state.

    Fields:
      detected, center, is_open, is_fist, over_body,
      extended_count, handedness           (original)
      is_peace   : bool   peace/V sign gesture
      pinch_dist : float  normalised thumb-index distance (always present)
    """
    hand_results = hands.process(rgb)

    state = {
        "detected":       False,
        "center":         None,
        "is_open":        False,
        "is_fist":        False,
        "over_body":      False,
        "extended_count": 0,
        "handedness":     None,
        "is_peace":       False,
        "pinch_dist":     1.0,
        # Normalised wrist + index MCP for pointer projection
        "wrist_norm":     (0.5, 0.5),
        "index_mcp_norm": (0.5, 0.5),
    }

    if not hand_results.multi_hand_landmarks:
        return state

    hand_landmarks = hand_results.multi_hand_landmarks[0]

    handedness_label = None
    if hand_results.multi_handedness:
        handedness_label = hand_results.multi_handedness[0].classification[0].label

    is_open, n_extended = _is_open_palm(
        hand_landmarks,
        handedness_label=handedness_label,
        require_extended=4,
    )

    cx, cy = _hand_center_px(hand_landmarks, w, h)

    over_body = False
    if 0 <= cx < w and 0 <= cy < h:
        over_body = seg_mask[cy, cx] >= thresh

    is_fist   = (not is_open) and (n_extended <= 1)
    is_peace  = _is_peace_sign(hand_landmarks)
    pinch_dist = _pinch_distance(hand_landmarks)

    lm = hand_landmarks.landmark
    state.update({
        "detected":       True,
        "center":         (cx, cy),
        "is_open":        is_open,
        "is_fist":        is_fist,
        "over_body":      over_body,
        "extended_count": n_extended,
        "handedness":     handedness_label,
        "is_peace":       is_peace,
        "pinch_dist":     pinch_dist,
        "wrist_norm":     (float(lm[0].x), float(lm[0].y)),
        "index_mcp_norm": (float(lm[5].x), float(lm[5].y)),
    })
    return state


def get_torso_box_from_pose(pose_results, w, h, visibility_thresh=0.5):
    if pose_results is None or pose_results.pose_landmarks is None:
        return None

    lms = pose_results.pose_landmarks.landmark
    ids = {
        "ls": mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        "rs": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        "lh": mp_pose.PoseLandmark.LEFT_HIP.value,
        "rh": mp_pose.PoseLandmark.RIGHT_HIP.value,
    }

    pts = {}
    for name, idx in ids.items():
        lm = lms[idx]
        if lm.visibility < visibility_thresh:
            return None
        pts[name] = np.array([lm.x * w, lm.y * h], dtype=np.float32)

    shoulder_center = 0.5 * (pts["ls"] + pts["rs"])
    hip_center      = 0.5 * (pts["lh"] + pts["rh"])
    torso_center    = 0.5 * (shoulder_center + hip_center)
    shoulder_width  = np.linalg.norm(pts["rs"] - pts["ls"])
    hip_width       = np.linalg.norm(pts["rh"] - pts["lh"])
    torso_width     = max(1.0, 0.5 * (shoulder_width + hip_width))
    torso_height    = max(1.0, np.linalg.norm(hip_center - shoulder_center))

    return {
        "center":          torso_center,
        "shoulder_center": shoulder_center,
        "hip_center":      hip_center,
        "width":           float(torso_width),
        "height":          float(torso_height),
        "ls": pts["ls"], "rs": pts["rs"],
        "lh": pts["lh"], "rh": pts["rh"],
    }


def get_body_box_from_pose(pose_results, w, h, visibility_thresh=0.5):
    if pose_results is None or pose_results.pose_landmarks is None:
        return None

    lms = pose_results.pose_landmarks.landmark
    landmark_ids = [
        mp_pose.PoseLandmark.NOSE.value,
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    ]

    pts = []
    for idx in landmark_ids:
        lm = lms[idx]
        if lm.visibility >= visibility_thresh:
            pts.append([lm.x * w, lm.y * h])

    if len(pts) < 4:
        return None

    pts     = np.array(pts, dtype=np.float32)
    x0, y0  = pts.min(axis=0)
    x1, y1  = pts.max(axis=0)
    cx      = 0.5 * (x0 + x1)
    cy      = 0.5 * (y0 + y1)

    return {
        "center": np.array([cx, cy], dtype=np.float32),
        "width":  float(max(1.0, x1 - x0)),
        "height": float(max(1.0, y1 - y0)),
        "x0": int(x0), "y0": int(y0),
        "x1": int(x1), "y1": int(y1),
        "pts": pts,
    }


def body_bbox_from_mask(seg_mask, thresh=0.5):
    ys, xs = np.where(seg_mask > thresh)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    return {
        "cx": float(cx), "cy": float(cy),
        "w":  float(max(1.0, x1 - x0)),
        "h":  float(max(1.0, y1 - y0)),
        "x0": int(x0), "x1": int(x1),
        "y0": int(y0), "y1": int(y1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Migrated from run_installation.py (pose/segment name tables + pose-geometry helpers)
# ══════════════════════════════════════════════════════════════════════════════

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
REQUIRED_INIT_LANDMARKS = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

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

def expand_mask(mask, ksize=9):
    k = max(1, int(ksize)); k = k if k%2==1 else k+1
    return cv2.dilate((mask>0.5).astype(np.uint8), np.ones((k,k),np.uint8), iterations=1).astype(np.float32)

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

