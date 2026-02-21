import cv2
import numpy as np
import mediapipe as mp
import os

mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation


NUM_PERS = 1
PERSON_IMAGE_PATH = "images_for_warping/pers" + str(NUM_PERS) + ".jpg"
BACKGROUND_IMAGE_PATH = "images_for_warping/pers" + str(NUM_PERS) + "bg.jpg"
OUTPUT_PATH = "images_for_warping/pers" + str(NUM_PERS) + "W.png"

U_GAIN = 0.30        # Warp strength
SIGMA_Y = 0.30       # Vertical spread of warp
SEG_THRESH = 0.5     # Segmentation threshold
FEATHER_PX = 5       # Feathering radius
MIRROR = False       # Selfie-style mirror



def build_warp_maps(width, height, uCenterX, uPeakY, uGain, sigma_y=0.2):
    x_norm = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y_norm = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xv_norm, yv_norm = np.meshgrid(x_norm, y_norm)

    dy = (yv_norm - uPeakY) / max(sigma_y, 1e-6)
    vertical_profile = np.exp(-(dy ** 2))

    scale = 1.0 + uGain * vertical_profile
    dx = xv_norm - uCenterX
    srcx_norm = uCenterX + dx / scale

    map_x = (srcx_norm * (width - 1)).astype(np.float32)
    map_y = (yv_norm * (height - 1)).astype(np.float32)
    return map_x, map_y


def warp_frame(frame_bgr, map_x, map_y):
    return cv2.remap(
        frame_bgr, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )



def get_hip_center_and_peakY_from_pose(results):
    if not results.pose_landmarks:
        return None, None

    lm = results.pose_landmarks.landmark
    left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

    uCenterX = 0.5 * (left_hip.x + right_hip.x)
    uPeakY = 0.5 * (left_hip.y + right_hip.y) - 0.1

    uCenterX = np.clip(uCenterX, 0.0, 1.0)
    uPeakY = np.clip(uPeakY, 0.0, 1.0)

    return uCenterX, uPeakY



def composite_person_over_bg(person_bgr, seg_mask, bg_bgr):
    h, w = person_bgr.shape[:2]

    if bg_bgr.shape[:2] != (h, w):
        bg_bgr = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    bg_f32 = bg_bgr.astype(np.float32)

    person_mask = (seg_mask >= SEG_THRESH).astype(np.float32)

    if FEATHER_PX > 0:
        k = max(1, int(FEATHER_PX))
        if k % 2 == 0:
            k += 1
        person_mask = cv2.GaussianBlur(person_mask, (k, k), 0)

    mask_3 = np.dstack([person_mask] * 3)

    person_f32 = person_bgr.astype(np.float32)
    out = mask_3 * person_f32 + (1.0 - mask_3) * bg_f32

    return out.astype(np.uint8)



def main():

    person_bgr = cv2.imread(PERSON_IMAGE_PATH)
    bg_bgr = cv2.imread(BACKGROUND_IMAGE_PATH)

    if person_bgr is None:
        raise FileNotFoundError(f"Could not read {PERSON_IMAGE_PATH}")
    if bg_bgr is None:
        raise FileNotFoundError(f"Could not read {BACKGROUND_IMAGE_PATH}")

    # Resize person to match background
    bg_h, bg_w = bg_bgr.shape[:2]
    person_bgr = cv2.resize(person_bgr, (bg_w, bg_h))

    h, w = person_bgr.shape[:2]

    fallback_centerX = 0.5
    fallback_peakY = 0.55

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose, mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:

        pose_input = cv2.flip(person_bgr, 1) if MIRROR else person_bgr

        rgb_for_pose = cv2.cvtColor(pose_input, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_for_pose)

        uCenterX, uPeakY = get_hip_center_and_peakY_from_pose(pose_results)

        if uCenterX is None:
            uCenterX = fallback_centerX
            uPeakY = fallback_peakY

        map_x, map_y = build_warp_maps(w, h, uCenterX, uPeakY, U_GAIN, SIGMA_Y)
        warped = warp_frame(pose_input, map_x, map_y)

        rgb_for_seg = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        seg_results = segmenter.process(rgb_for_seg)
        seg_mask = seg_results.segmentation_mask

        final = composite_person_over_bg(warped, seg_mask, bg_bgr)

    cv2.imwrite(OUTPUT_PATH, final)
    cv2.imshow("Warped Composite", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Saved to: {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
