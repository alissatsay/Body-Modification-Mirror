import cv2
import numpy as np
import mediapipe as mp

mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_pose = mp.solutions.pose


def _pose_band_from_landmarks(landmarks, w, h,
                              band_half_width_frac=0.22,
                              top_frac_of_torso=0.15,
                              bot_frac_of_torso=0.75):
    """
    Compute (x0,y0,x1,y1) bbox for hip/abdomen deformation using pose landmarks.

    Strategy:
      - compute shoulder_mid and hip_mid
      - torso_len = |hip_mid_y - shoulder_mid_y|
      - define band vertical range around torso using fractions of torso_len
      - center X at hip_mid_x (more stable for hip deformation)
    """
    # landmark indices
    L_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    R_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    L_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
    R_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value

    ls = landmarks[L_SH]
    rs = landmarks[R_SH]
    lh = landmarks[L_HIP]
    rh = landmarks[R_HIP]

    # require decent visibility (optional but helps)
    vis_ok = (ls.visibility > 0.4 and rs.visibility > 0.4 and lh.visibility > 0.4 and rh.visibility > 0.4)
    if not vis_ok:
        return None

    shoulder_mid_x = 0.5 * (ls.x + rs.x) * w
    shoulder_mid_y = 0.5 * (ls.y + rs.y) * h

    hip_mid_x = 0.5 * (lh.x + rh.x) * w
    hip_mid_y = 0.5 * (lh.y + rh.y) * h

    torso_len = abs(hip_mid_y - shoulder_mid_y)
    if torso_len < 20:  # too small / unstable
        return None

    # Vertical middle of the person (torso-mid). You asked for "vertical middle":
    # this is the midpoint between shoulders and hips.
    torso_mid_y = 0.5 * (shoulder_mid_y + hip_mid_y)

    # Build band relative to torso size
    y0 = int(torso_mid_y + top_frac_of_torso * torso_len)
    y1 = int(torso_mid_y + bot_frac_of_torso * torso_len)

    # Width relative to image; optionally you can tie to hip width too, later.
    half_w = band_half_width_frac * w
    cx = hip_mid_x
    x0 = int(cx - half_w)
    x1 = int(cx + half_w)

    # clamp
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    if y1 <= y0 or x1 <= x0:
        return None

    return (x0, y0, x1, y1), (hip_mid_x, hip_mid_y), (shoulder_mid_x, shoulder_mid_y), torso_mid_y

def _is_open_palm(hand_landmarks, handedness_label=None, require_extended=4):
    """
    Simple open-palm test:
    - index/middle/ring/pinky count as extended if tip is above pip in image coords
    - thumb is checked with a simple x-direction heuristic
    Returns (is_open, num_extended)
    """
    lm = hand_landmarks.landmark

    # Finger landmark indices
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    extended = 0

    # For the 4 fingers: in image coordinates, smaller y means "higher"
    for tip_idx, pip_idx in zip(tips, pips):
        if lm[tip_idx].y < lm[pip_idx].y:
            extended += 1

    # Optional thumb check, not required for open palm by default
    thumb_extended = False
    if handedness_label == "Right":
        thumb_extended = lm[4].x < lm[3].x
    elif handedness_label == "Left":
        thumb_extended = lm[4].x > lm[3].x

    is_open = extended >= require_extended
    return is_open, extended + int(thumb_extended)

def _hand_center_px(hand_landmarks, w, h):
    """
    Robust-ish palm center:
    average of wrist + MCP joints of index/middle/ring/pinky.
    """
    lm = hand_landmarks.landmark
    ids = [0, 5, 9, 13, 17]  # wrist + finger MCPs
    xs = [lm[i].x * w for i in ids]
    ys = [lm[i].y * h for i in ids]
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
    return cx, cy

def get_segmentation_mask(frame_bgr, segmenter, feather=0):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    seg = segmenter.process(rgb)
    seg_mask = seg.segmentation_mask.astype(np.float32)

    if feather and feather > 0:
        k = int(feather)
        if k % 2 == 0:
            k += 1
        seg_mask = cv2.GaussianBlur(seg_mask, (k, k), 0)

    return seg_mask, rgb

def get_hand_state(rgb, hands, seg_mask, thresh, w, h):
    hand_results = hands.process(rgb)

    state = {
        "detected": False,
        "center": None,
        "is_open": False,
        "is_fist": False,
        "over_body": False,
        "extended_count": 0,
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
        require_extended=4
    )

    cx, cy = _hand_center_px(hand_landmarks, w, h)

    over_body = False
    if 0 <= cx < w and 0 <= cy < h:
        over_body = seg_mask[cy, cx] >= thresh

    is_fist = (not is_open) and (n_extended <= 1)

    state.update({
        "detected": True,
        "center": (cx, cy),
        "is_open": is_open,
        "is_fist": is_fist,
        "over_body": over_body,
        "extended_count": n_extended,
    })
    return state