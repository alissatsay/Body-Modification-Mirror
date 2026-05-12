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