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


# Maps digit keys 0-6 to their uGain values
GAIN_PRESETS = {
    ord('0'): 0.00,
    ord('1'): 0.10,
    ord('2'): 0.20,
    ord('3'): 0.30,
    ord('4'): 0.40,
    ord('5'): 0.50,
    ord('6'): 0.60,
}


def main():
    # ── tunables ──────────────────────────────────────────────────────────────
    uGain             = 0.30
    uPeakY_manual     = None   # None = let pose drive it; set by u/d keys
    sigma_y           = 0.30
    fallback_centerX  = 0.5
    fallback_peakY    = 0.55
    POSE_EVERY_N      = 3      # run pose every N frames  (3 → ~3× cheaper)
    SEG_EVERY_N       = 2      # run segmentation every N frames
    SEG_SCALE         = 0.5    # downscale factor for segmentation inference

    GAIN_STEP         = 0.05   # increment for +/- keys
    PEAK_STEP         = 0.05   # increment for u/d keys
    # ─────────────────────────────────────────────────────────────────────────

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return
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

    print("\nControls:")
    print("  +  /  -   : increase / decrease uGain by 0.05")
    print("  0 … 6     : set uGain directly (0.00, 0.10, 0.20, … 0.60)")
    print("  u  /  d   : increase / decrease uPeak by 0.05 (overrides pose tracking)")
    print("  q         : quit\n")

    frame_queue = queue.Queue(maxsize=2)
    t = threading.Thread(target=capture_thread, args=(cap, frame_queue), daemon=True)
    t.start()

    map_x, map_y       = build_warp_maps(width, height,
                                          fallback_centerX, fallback_peakY,
                                          uGain, sigma_y)
    last_seg_mask       = np.zeros((height, width), dtype=np.float32)
    last_uCenterX       = fallback_centerX
    last_uPeakY         = fallback_peakY
    maps_dirty          = False   # flag: rebuild maps before next warp
    frame_idx           = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose, mp_selfie_segmentation.SelfieSegmentation(
        model_selection=0
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
                    # uPeakY: use manual override if set, else pose-driven
                    effective_peakY = uPeakY_manual if uPeakY_manual is not None else py
                    if (abs(cx - last_uCenterX) > 0.01 or
                            abs(effective_peakY - last_uPeakY) > 0.01 or
                            maps_dirty):
                        map_x, map_y = build_warp_maps(width, height,
                                                        cx, effective_peakY,
                                                        uGain, sigma_y)
                        last_uCenterX = cx
                        last_uPeakY   = effective_peakY
                        maps_dirty    = False
                elif maps_dirty:
                    # No pose detected but params changed — rebuild with last known center
                    effective_peakY = uPeakY_manual if uPeakY_manual is not None else last_uPeakY
                    map_x, map_y = build_warp_maps(width, height,
                                                    last_uCenterX, effective_peakY,
                                                    uGain, sigma_y)
                    last_uPeakY = effective_peakY
                    maps_dirty  = False

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

            # ── HUD overlay ───────────────────────────────────────────────────
            peak_label = f"{last_uPeakY:.2f}" + (" [manual]" if uPeakY_manual is not None else " [pose]")
            # cv2.putText(final_frame, f"uGain: {uGain:.2f}  (+/- or 0-6)",
            #             (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # cv2.putText(final_frame, f"uPeak: {peak_label}  (u/d)",
            #             (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Warped Mirror over Captured Background", final_frame)
            frame_idx += 1

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('+') or key == ord('='):   # = shares key with + (no shift needed)
                uGain = round(min(uGain + GAIN_STEP, 2.0), 4)
                maps_dirty = True
                print(f"uGain → {uGain:.2f}")

            elif key == ord('-'):
                uGain = round(max(uGain - GAIN_STEP, 0.0), 4)
                maps_dirty = True
                print(f"uGain → {uGain:.2f}")

            elif key in GAIN_PRESETS:
                uGain = GAIN_PRESETS[key]
                maps_dirty = True
                print(f"uGain → {uGain:.2f}  [preset {chr(key)}]")

            elif key == ord('u'):
                base = uPeakY_manual if uPeakY_manual is not None else last_uPeakY
                uPeakY_manual = round(max(base - PEAK_STEP, 0.0), 4)  # up = smaller Y
                maps_dirty = True
                print(f"uPeakY → {uPeakY_manual:.2f} [manual]")

            elif key == ord('d'):
                base = uPeakY_manual if uPeakY_manual is not None else last_uPeakY
                uPeakY_manual = round(min(base + PEAK_STEP, 1.0), 4)  # down = larger Y
                maps_dirty = True
                print(f"uPeakY → {uPeakY_manual:.2f} [manual]")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()