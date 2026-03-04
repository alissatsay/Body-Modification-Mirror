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