import os
import cv2
import numpy as np
import mediapipe as mp
import threading

mp_pose = mp.solutions.pose

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "helpers"))

import triangle_mesh as TMh
import pose_tracking as PTh
import interaction_loop as Lh
import display as Dh
import screens as Sc

# Gesture tracker objects — created once, live for the whole session
_peace_detector = PTh.PeaceSignHoldDetector()
_pinch_tracker  = PTh.PinchGestureTracker()

ROTATE_DEG = 270
ROTATE_DIR = "ccw"

OUTPUT_W = 1080
OUTPUT_H = 1920

# helpers/interaction_loop.py and helpers/screens.py need these same values
# (they read/write things sized to the output canvas); this is simpler and
# less error-prone than having each module recompute them independently,
# since they're plain shared numbers rather than derived paths.
Lh.OUTPUT_W = OUTPUT_W
Lh.OUTPUT_H = OUTPUT_H
Sc.OUTPUT_W = OUTPUT_W
Sc.OUTPUT_H = OUTPUT_H

PRIMARY_MONITOR_WIDTH = 1920

SEG_EVERY_N = 2

# Path to the welcome background image (relative to script or cwd).
WELCOME_BG_PATH = os.path.join(_ASSETS_DIR, "welcome_background.png")

# Set to True to capture a clean background plate before initialization.
CAPTURE_BACKGROUND_BEFORE_INIT = False

# ── Initialization pose image ─────────────────────────────────────────────
_INIT_POSE_PATH = os.path.join(_ASSETS_DIR, "UI_gestures", "initialization_pose.png")
_INIT_POSE_ALPHA = 0.35   # same opacity as BS overlay

class PoseInferenceThread:
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
            cur_pts    = PTh.extract_pose_points(rgb, self.pose, self.w, self.h, min_vis=0.45)
            hand_state = PTh.get_hand_state(rgb, self.hands, seg_mask, self.thresh, self.w, self.h)
            with self._lock:
                self._result = (seg_mask, rgb, cur_pts, hand_state)


def main(step=25, thresh=0.5, feather=0, show_mask=False):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    if not cap.isOpened():
        print("Error: could not open camera."); return
    print("Camera found")
    ok, frame = cap.read()
    if not ok:
        print("Error: could not read initial frame."); cap.release(); return
    frame = Dh.rotate_frame_for_output(frame, deg=ROTATE_DEG, direction=ROTATE_DIR)
    h, w = frame.shape[:2]

    V_base, T, nx, ny = TMh.build_grid_mesh(w, h, step=step)
    V_def = V_base.copy().astype(np.float32)
    E = TMh.build_unique_edges(T)
    neighbors = TMh.build_vertex_neighbors(len(V_base), E)
    arap_cache = {"E": E, "neighbors": neighbors}
    brush_radius = max(45, int(min(w, h) * 0.07))

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
        "warp_min_area": 4.0,
        "active_hand": None,
        "gesture_feedback_frames": 0,
        "gesture_feedback_msg": "",
        "brush_radius": brush_radius,
    }

    window_name = "Hand brush drag on skeleton mesh"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, PRIMARY_MONITOR_WIDTH, 0)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # ── CHANGE 2: load initialization pose image once, outside the loop ───
    _init_pose_img = None
    if os.path.exists(_INIT_POSE_PATH):
        _init_pose_raw = cv2.imread(_INIT_POSE_PATH, cv2.IMREAD_UNCHANGED)
        if _init_pose_raw is not None:
            # Scale it the same way as BS images: match person height.
            # At this point we don't know person_height_px yet, so we scale
            # to fill OUTPUT_H * 0.80 as a reasonable default — it will look
            # the same every session since the pose image is fixed.
            _ip_src_h = _init_pose_raw.shape[0]
            _ip_target_h = int(OUTPUT_H * 0.90)
            _ip_target_w = int(_init_pose_raw.shape[1] * _ip_target_h / _ip_src_h)
            _init_pose_img = cv2.resize(_init_pose_raw,
                                        (_ip_target_w, _ip_target_h),
                                        interpolation=cv2.INTER_AREA)
            print(f"Initialization pose image loaded: {_ip_target_w}x{_ip_target_h}")
        else:
            print(f"WARNING: could not read init pose image at {_INIT_POSE_PATH}")
    else:
        print(f"WARNING: init pose image not found at {_INIT_POSE_PATH}")

    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter, \
         mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
         mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=False,
                      enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

      while True:  # ── Outer restart loop ──────────────────────────────
        _peace_detector.reset()
        _pinch_tracker.reset()

        Sc.show_home_screen(window_name, cap, pose)

        _sel_bg = cv2.imread(WELCOME_BG_PATH)
        if _sel_bg is None:
            _sel_bg = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)
            _sel_bg[:, :] = (60, 20, 10)
        else:
            _sel_bg = cv2.resize(_sel_bg, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)

        selected_bs = Sc.show_selection_screen(window_name, _sel_bg, cap)
        if selected_bs is not None:
            print(f"Selected beauty standard: {selected_bs}")
            interaction_state["selected_beauty_standard"] = selected_bs
        else:
            print("Beauty standard selection skipped.")
            interaction_state["selected_beauty_standard"] = None

        inference_thread = PoseInferenceThread(pose=pose, hands=hands, segmenter=segmenter,
                                                feather=feather, thresh=thresh, w=w, h=h)

        _BG_DIR  = os.path.join(_ASSETS_DIR, "background_captures")
        _BG_PATH = os.path.join(_BG_DIR, "background.png")

        if CAPTURE_BACKGROUND_BEFORE_INIT:
            Sc.show_countdown(cap, h, w, 5, "Capturing background", window_name)
            bg_plate = Sc.capture_background_plate(cap, h, w, n_frames=20, window_name=window_name)
            if bg_plate is None:
                print("Could not capture background plate.")
                inference_thread.stop(); cap.release(); cv2.destroyAllWindows(); return
            os.makedirs(_BG_DIR, exist_ok=True)
            cv2.imwrite(_BG_PATH, bg_plate)
            print(f"Background plate saved to {_BG_PATH}")
            interaction_state["bg_plate"] = bg_plate
        else:
            if os.path.exists(_BG_PATH):
                bg_plate = cv2.imread(_BG_PATH)
                if bg_plate is not None:
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

        Sc.show_countdown(cap, h, w, 5, "Please match the pose displayed on the screen.\n\nInitialization starting", window_name, bs_img=_init_pose_img)
        print("Entering initialization loop")

        initialized=False; max_init_frames=300; full_pose_streak=0; required_streak=20
        mask_accum=None; mask_count=0; best_stable_pts=None
        _no_detect_frames_init = 0

        # ── CHANGE 3: instruction line shown above countdown during init ──
        _INIT_INSTRUCTION = "Hold still."

        for k in range(max_init_frames):
            ok, fr = cap.read()
            if not ok:
                print(f"init frame {k}: camera read failed"); break
            fr = Dh.rotate_frame_for_output(fr, deg=ROTATE_DEG, direction=ROTATE_DIR)
            if fr.shape[:2] != (h, w): fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_LINEAR)
            fr = cv2.flip(fr, 1)
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            cur_pts = PTh.extract_pose_points(rgb, pose, w, h, min_vis=0.25)
            if cur_pts is None:
                _no_detect_frames_init += 1
                if _no_detect_frames_init >= 10:
                    print("No person detected for 10 frames during init — restarting")
                    break
            else:
                _no_detect_frames_init = 0
            stable_pts = PTh.smooth_pose_points(cur_pts, interaction_state, alpha=0.65, max_jump_px=45.0, hold_frames=2)
            full_pose_ok = PTh.has_required_landmarks(stable_pts)

            if full_pose_ok:
                full_pose_streak += 1
                seg_mask_init, _ = PTh.get_segmentation_mask(fr, segmenter, feather=feather)
                if mask_accum is None: mask_accum = np.zeros_like(seg_mask_init, dtype=np.float32)
                mask_accum += seg_mask_init.astype(np.float32); mask_count += 1; best_stable_pts = stable_pts
            else:
                full_pose_streak=0; mask_accum=None; mask_count=0; best_stable_pts=None

            init_vis = fr.copy()
            # ── CHANGE 1: no frame counter in status msg ──────────────────
            status_msg = f"Waiting for full-body pose..."

            if full_pose_streak >= required_streak and mask_count > 0 and best_stable_pts is not None:
                agg_mask = PTh.finalize_init_mask(mask_accum, mask_count, avg_thresh=0.28,
                                               dilate_ksize=13, close_ksize=11)
                pose_capsule_mask = PTh.rasterize_pose_capsules(agg_mask.shape, best_stable_pts)
                init_mask_final   = np.maximum(agg_mask, pose_capsule_mask)
                init_mask_final   = PTh.trim_mask_arms_combined(init_mask_final, best_stable_pts,
                                                             min_component_area=250)

                print("Building adaptive body mesh...")
                V_base_adaptive, T_adaptive, active_adaptive = TMh.build_adaptive_body_mesh(
                    w=w, h=h,
                    body_mask=init_mask_final,
                    interior_step=max(step * 2, 50),
                    contour_step=10,
                    contour_inset=3,
                    min_contour_pts=60,
                )

                E_new = TMh.build_unique_edges(T_adaptive)
                nbrs_new = TMh.build_vertex_neighbors(len(V_base_adaptive), E_new)
                arap_cache = {"E": E_new, "neighbors": nbrs_new}

                V_base = V_base_adaptive.copy().astype(np.float32)
                T      = T_adaptive
                V_def  = V_base.copy().astype(np.float32)

                interaction_state["preview_vertices"]  = np.zeros(len(V_def), dtype=bool)
                interaction_state["preview_triangles"] = np.zeros(len(T),     dtype=bool)
                interaction_state["drag_vertices"]     = np.zeros(len(V_def), dtype=bool)
                interaction_state["drag_triangles"]    = np.zeros(len(T),     dtype=bool)

                binding = TMh.bind_mesh_to_skeleton(
                    V_base=V_base, T=T,
                    ref_pts=best_stable_pts,
                    init_seg_mask=init_mask_final,
                    mask_thresh=0.5,
                )

                print(f"init frame {k}: binding is {'ok' if binding is not None else 'None'}")
                if binding is not None:
                    tri_render_group = TMh.build_triangle_render_groups(binding, T)
                    binding["tri_render_group"]  = tri_render_group
                    binding["group_tri_indices"] = TMh.build_render_group_triangle_index_cache(binding)
                    TMh.print_render_group_triangle_stats(binding)
                    V_def  = V_base.copy().astype(np.float32)
                    interaction_state["binding"]      = binding
                    interaction_state["body_metrics"] = PTh.compute_body_metrics(best_stable_pts)
                    _nose  = best_stable_pts.get("nose")
                    _lankl = best_stable_pts.get("left_ankle")
                    _rankl = best_stable_pts.get("right_ankle")
                    if _nose is not None and (_lankl is not None or _rankl is not None):
                        _ankl = _lankl if _lankl is not None else _rankl
                        if _lankl is not None and _rankl is not None:
                            _ankl = 0.5 * (_lankl + _rankl)
                        interaction_state["person_height_px"] = float(abs(_ankl[1] - _nose[1]))
                    else:
                        interaction_state["person_height_px"] = float(h) * 0.75
                    initialized = True
                    print("Initialization succeeded")
                    status_msg = "Pose locked. Starting..."

            if stable_pts is not None:
                for name, p in stable_pts.items():
                    if p is not None: cv2.circle(init_vis, (int(p[0]),int(p[1])), 4, (0,255,0), -1, cv2.LINE_AA)

            _init_display = cv2.resize(init_vis, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)

            # ── CHANGE 2: composite the init pose image (same as BS overlay) ─
            _init_display = Sc._composite_init_pose_overlay(_init_display, _init_pose_img, _INIT_POSE_ALPHA)

            # ── CHANGE 3: two-line text (instruction above, status below) ───
            if initialized:
                # Success — single line is fine
                _init_display = Sc._draw_countdown_text(_init_display, status_msg, OUTPUT_W, OUTPUT_H)
            else:
                _init_display = Sc._draw_two_line_countdown_text(
                    _init_display,
                    _INIT_INSTRUCTION,
                    status_msg,
                    OUTPUT_W, OUTPUT_H,
                )

            cv2.imshow(window_name, _init_display)
            key = cv2.waitKey(1) & 0xFF
            if initialized: break
            if key == ord('q'):
                print("User quit during initialization"); break

        print(f"Init loop done. initialized={initialized}")
        if not initialized:
            print("Could not initialize — restarting to home screen.")
            inference_thread.stop()
            continue

        print("About to enter main skeleton loop")
        _loop_result = Lh.run_hand_brush_drag_arap_loop_skeleton(
            cap=cap, segmenter=segmenter, hands=hands, pose=pose,
            h=h, w=w, V_base=V_base, V_def=V_def, T=T,
            step=step, thresh=thresh, feather=feather, show_mask=show_mask,
            brush_radius=brush_radius, interaction_state=interaction_state,
            arap_cache=arap_cache, inference_thread=inference_thread,
            window_name=window_name,
            _peace_detector=_peace_detector, _pinch_tracker=_pinch_tracker,
        )
        print(f"Main skeleton loop returned: {_loop_result}")
        inference_thread.stop()

        if _loop_result != "restart":
            break

        print("Restarting session...")
        brush_radius = max(45, int(min(w, h) * 0.07))
        interaction_state.update({
            "preview_vertices":  np.zeros(len(V_def), dtype=bool),
            "preview_triangles": np.zeros(len(T),     dtype=bool),
            "drag_vertices":     np.zeros(len(V_def), dtype=bool),
            "drag_triangles":    np.zeros(len(T),     dtype=bool),
            "dragging": False, "prev_hand_center": None, "hand_was_open": False,
            "pose_prev_pts": None, "pose_last_good_pts": None,
            "pose_missing_count": 0,
            "binding": None, "yaw_side_state": "frontal",
            "yaw_sign_value_smooth": 0.0, "bg_plate": None,
            "cached_frame_arrays": None, "cached_frames": None,
            "warp_min_area": 4.0, "active_hand": None,
            "gesture_feedback_frames": 0, "gesture_feedback_msg": "",
            "brush_radius": brush_radius,
        })
        V_base, T, _, _ = TMh.build_grid_mesh(w, h, step=step)
        V_def = V_base.copy().astype(np.float32)
        E = TMh.build_unique_edges(T)
        arap_cache = {"E": E, "neighbors": TMh.build_vertex_neighbors(len(V_base), E)}

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
