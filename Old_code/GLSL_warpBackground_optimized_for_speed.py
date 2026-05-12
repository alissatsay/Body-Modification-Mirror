import cv2
import numpy as np
import mediapipe as mp
import threading
import queue

mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation


def build_warp_maps(width, height, uCenterX, uPeakY, uGain, sigma_y=0.2):
    x_norm = np.linspace(0.0, 1.0, width,  dtype=np.float32)
    y_norm = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xv_norm, yv_norm = np.meshgrid(x_norm, y_norm)
    dy = (yv_norm - uPeakY) / max(sigma_y, 1e-6)
    vertical_profile = np.exp(-(dy ** 2))
    scale = 1.0 + uGain * vertical_profile
    dx = xv_norm - uCenterX
    srcx_norm = uCenterX + dx / scale
    map_x = (srcx_norm * (width  - 1)).astype(np.float32)
    map_y = (yv_norm   * (height - 1)).astype(np.float32)
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
    left_hip  = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
    uCenterX = max(0.0, min(1.0, 0.5 * (left_hip.x + right_hip.x)))
    uPeakY   = max(0.0, min(1.0, 0.5 * (left_hip.y + right_hip.y) - 0.1))
    return uCenterX, uPeakY


def composite_person_over_bg(person_bgr, seg_mask, bg_bgr=None, thresh=0.5, feather_px=5):
    h, w = person_bgr.shape[:2]
    if bg_bgr is None:
        bg_f32 = np.ones_like(person_bgr, dtype=np.float32) * 255.0
    else:
        if bg_bgr.shape[:2] != (h, w):
            bg_bgr = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        bg_f32 = bg_bgr.astype(np.float32)
    person_mask = (seg_mask >= thresh).astype(np.float32)
    if feather_px > 0:
        k = max(1, int(feather_px))
        if k % 2 == 0:
            k += 1
        person_mask = cv2.GaussianBlur(person_mask, (k, k), 0)
    mask_3 = np.dstack([person_mask] * 3)
    out = mask_3 * person_bgr.astype(np.float32) + (1.0 - mask_3) * bg_f32
    return out.astype(np.uint8)


def capture_thread(cap, frame_queue):
    """Continuously grab frames into a queue so the main thread never waits on I/O."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)


def main():
    # ── tunables ──────────────────────────────────────────────────────────────
    uGain             = 0.30
    sigma_y           = 0.30
    fallback_centerX  = 0.5
    fallback_peakY    = 0.55
    POSE_EVERY_N      = 3      # run pose every N frames  (3 → ~3× cheaper)
    SEG_EVERY_N       = 2      # run segmentation every N frames
    SEG_SCALE         = 0.5    # downscale factor for segmentation inference
    # ─────────────────────────────────────────────────────────────────────────

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return
    # Hint: request a lower resolution if 1080p isn't needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    ret, frame = cap.read()
    if not ret:
        print("Error: couldn't read initial frame.")
        cap.release()
        return
    height, width = frame.shape[:2]
    seg_w = int(width  * SEG_SCALE)
    seg_h = int(height * SEG_SCALE)

    print("Please move out of the frame. Capturing background in 3 seconds...")
    cv2.waitKey(3000)
    ret, captured_bg = cap.read()
    if ret:
        captured_bg = cv2.flip(captured_bg, 1)
        print("Background captured.")
    else:
        captured_bg = None
        print("Warning: background capture failed. Falling back to white.")

    # Kick off a dedicated capture thread so the GPU/ML work never stalls on I/O
    frame_queue = queue.Queue(maxsize=2)
    t = threading.Thread(target=capture_thread, args=(cap, frame_queue), daemon=True)
    t.start()

    # Cached state
    map_x, map_y       = build_warp_maps(width, height,
                                          fallback_centerX, fallback_peakY,
                                          uGain, sigma_y)
    last_seg_mask       = np.zeros((height, width), dtype=np.float32)
    last_uCenterX       = fallback_centerX
    last_uPeakY         = fallback_peakY
    frame_idx           = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,          # ← 0 is the lite model; much faster
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose, mp_selfie_segmentation.SelfieSegmentation(
        model_selection=0            # ← 0 is faster than 1
    ) as segmenter:

        while True:
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Pose (every POSE_EVERY_N frames) ──────────────────────────────
            if frame_idx % POSE_EVERY_N == 0:
                pose_results = pose.process(rgb)
                cx, py = get_hip_center_and_peakY_from_pose(pose_results)
                if cx is not None:
                    # Only rebuild warp maps when the pose actually moved enough
                    if (abs(cx - last_uCenterX) > 0.01 or
                            abs(py - last_uPeakY) > 0.01):
                        map_x, map_y = build_warp_maps(width, height,
                                                        cx, py, uGain, sigma_y)
                        last_uCenterX, last_uPeakY = cx, py

            # ── Warp + mirror ─────────────────────────────────────────────────
            warped   = warp_frame(frame, map_x, map_y)
            mirrored = cv2.flip(warped, 1)

            # ── Segmentation (every SEG_EVERY_N frames, on downscaled input) ──
            if frame_idx % SEG_EVERY_N == 0:
                small_rgb = cv2.resize(
                    cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB),
                    (seg_w, seg_h),
                    interpolation=cv2.INTER_LINEAR
                )
                seg_results  = segmenter.process(small_rgb)
                # Upscale mask back to full resolution
                last_seg_mask = cv2.resize(
                    seg_results.segmentation_mask,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR
                )

            # ── Composite ─────────────────────────────────────────────────────
            final_frame = composite_person_over_bg(
                mirrored, last_seg_mask,
                bg_bgr=captured_bg, thresh=0.5, feather_px=5
            )

            cv2.imshow("Warped Mirror over Captured Background", final_frame)
            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()