## Test module for triangle mesh

import cv2
import numpy as np
import Triangle_Mesh_helpers as TMh
import Pose_Tracking_helpers as PTh
import Loop_helpers as Lh
import Display_helpres as Dh

import mediapipe as mp
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_hands = mp.solutions.hands
#import GLSL_HT_UI

mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_pose = mp.solutions.pose

def test_warp_triangle():
    # create source image
    h, w = 500, 500
    src_img = np.zeros((h, w, 3), dtype=np.uint8)

    # draw grid for visual reference
    for i in range(0, w, 25):
        cv2.line(src_img, (i, 0), (i, h), (60, 60, 60), 1)
    for j in range(0, h, 25):
        cv2.line(src_img, (0, j), (w, j), (60, 60, 60), 1)

    # colored circle helps visualize distortion
    cv2.circle(src_img, (250, 250), 80, (0, 200, 255), -1)

    # destination canvas
    dst_img = src_img.copy()

    # Define source triangle
    t_src = np.float32([
        [150, 150],
        [350, 150],
        [250, 350]
    ])

    # Define destination triangle
    t_dst = np.float32([
        [120, 180],
        [380, 130],
        [260, 420]
    ])

    # draw triangles for reference
    cv2.polylines(src_img, [np.int32(t_src)], True, (0,255,0), 2)
    cv2.polylines(dst_img, [np.int32(t_dst)], True, (0,0,255), 2)

    # Apply warp
    TMh.warp_triangle(src_img, dst_img, t_src, t_dst)

    # Show result
    combined = np.hstack([src_img, dst_img])

    cv2.imshow("LEFT: source | RIGHT: warped result", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def interactive_triangle_test():

    # STATE CONTAINER
    state = {
        "selected_vertex": -1,
        "radius_select": 15,
        "t_dst": None
    }

    # Mouse callback
    def mouse_callback(event, x, y, flags, param):

        s = param
        t_dst = s["t_dst"]

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, v in enumerate(t_dst):
                if np.linalg.norm(v - [x, y]) < s["radius_select"]:
                    s["selected_vertex"] = i

        elif event == cv2.EVENT_MOUSEMOVE:
            if s["selected_vertex"] != -1:
                t_dst[s["selected_vertex"]] = [x, y]

        elif event == cv2.EVENT_LBUTTONUP:
            s["selected_vertex"] = -1

    # Create synthetic image
    h, w = 600, 600
    src_img = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(0, w, 30):
        cv2.line(src_img, (i, 0), (i, h), (60,60,60), 1)
    for j in range(0, h, 30):
        cv2.line(src_img, (0, j), (w, j), (60,60,60), 1)

    cv2.circle(src_img, (300,300), 120, (0,200,255), -1)

    t_src = np.float32([
        [200,200],
        [400,200],
        [300,420]
    ])

    state["t_dst"] = t_src.copy()

    cv2.namedWindow("Triangle Warp")
    cv2.setMouseCallback(
        "Triangle Warp",
        mouse_callback,
        state
    )

    # Main loop
    while True:

        dst_img = src_img.copy()

        TMh.warp_triangle(
            src_img,
            dst_img,
            t_src,
            state["t_dst"]
        )

        cv2.polylines(dst_img,
                      [np.int32(t_src)],
                      True,(0,255,0),2)

        cv2.polylines(dst_img,
                      [np.int32(state["t_dst"])],
                      True,(0,0,255),2)

        for p in state["t_dst"]:
            cv2.circle(dst_img,
                       tuple(p.astype(int)),
                       6,(0,0,255),-1)

        cv2.imshow("Triangle Warp", dst_img)

        if cv2.waitKey(16) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def make_draw_mesh_test():
    # Create a test image
    h, w = 480, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Build mesh
    V, T, nx, ny = TMh.build_grid_mesh(w, h, step=40)

    vis = TMh.draw_mesh_with_vertices(img, V, T)
    cv2.imshow("mesh+verts", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_active_triangles(frame_bgr, V, T, active, color=(0, 0, 0), thickness=1):
    """
    Draw only triangles where active[k] == True.
    """
    out = frame_bgr.copy()
    active_idxs = np.flatnonzero(active)
    for k in active_idxs:
        tri = T[k]
        pts = V[tri].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    return out


def test_active_triangles_live(step=40, thresh=0.5, feather=0, show_mask=True):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return
    else:
        print("Camera found")

    # Read one frame to get shape and build mesh once
    ok, frame = cap.read()
    if not ok:
        print("Error: could not read initial frame.")
        cap.release()
        return
    #frame = GLSL_HT_UI.rotate_frame(frame)

    h, w = frame.shape[:2]
    V, T, nx, ny = TMh.build_grid_mesh(w, h, step=step)

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Keep frame size consistent with the mesh
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            seg = segmenter.process(rgb)
            seg_mask = seg.segmentation_mask  # float32 [0,1], shape (h,w)

            # Optional smoothing of mask to reduce flicker
            if feather and feather > 0:
                k = int(feather)
                if k % 2 == 0:
                    k += 1
                seg_mask = cv2.GaussianBlur(seg_mask, (k, k), 0)

            active = TMh.active_triangles_from_mask(V, T, seg_mask, thresh=thresh)

            vis = draw_active_triangles(frame, V, T, active, color=(0, 0, 0), thickness=1)

            # Debug overlay text
            cv2.putText(
                vis,
                f"active: {int(active.sum())}/{len(T)}  step={step}  thresh={thresh:.2f}  feather={feather}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                vis,
                "q quit | [ ] thresh | - + step | f toggle feather",
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Active triangles on live video", vis)

            if show_mask:
                mask_vis = (seg_mask * 255).astype(np.uint8)
                cv2.imshow("Segmentation mask", mask_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Adjust threshold
            if key == ord(']'):
                thresh = min(0.95, thresh + 0.05)
            elif key == ord('['):
                thresh = max(0.05, thresh - 0.05)

            # Adjust mesh density
            elif key in (ord('+'), ord('=')):
                step = max(10, step - 5)  # smaller step => denser mesh
                V, T, nx, ny = TMh.build_grid_mesh(w, h, step=step)
            elif key in (ord('-'), ord('_')):
                step = min(150, step + 5)  # larger step => coarser mesh
                V, T, nx, ny = TMh.build_grid_mesh(w, h, step=step)

            # Toggle feather
            elif key == ord('f'):
                feather = 0 if feather else 9  # switch between none and blur-kernel=9

    cap.release()
    cv2.destroyAllWindows()

def test_hand_brush_triangles_live(step=40, thresh=0.5, feather=0, show_mask=True):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return
    else:
        print("Camera found")

    # Read one frame to fix mesh dimensions
    ok, frame = cap.read()
    if not ok:
        print("Error: could not read initial frame.")
        cap.release()
        return

    h, w = frame.shape[:2]
    V, T, nx, ny = TMh.build_grid_mesh(w, h, step=step)

    # Brush radius chosen automatically from frame size
    brush_radius = max(45, int(min(w, h) * 0.08))

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter, \
         mp_hands.Hands(
             static_image_mode=False,
             max_num_hands=1,
             model_complexity=1,
             min_detection_confidence=0.5,
             min_tracking_confidence=0.5,
         ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

            # Mirror for more natural interaction
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- Person segmentation ---
            seg = segmenter.process(rgb)
            seg_mask = seg.segmentation_mask.astype(np.float32)  # (h, w) in [0,1]

            if feather and feather > 0:
                k = int(feather)
                if k % 2 == 0:
                    k += 1
                seg_mask = cv2.GaussianBlur(seg_mask, (k, k), 0)

            # Active body triangles
            active = TMh.active_triangles_from_mask(V, T, seg_mask, thresh=thresh)

            # --- Hand detection ---
            hand_results = hands.process(rgb)

            hand_center = None
            hand_is_open = False
            hand_over_body = False
            affected_vertices = np.zeros(len(V), dtype=bool)
            affected_triangles = np.zeros(len(T), dtype=bool)

            if hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]

                handedness_label = None
                if hand_results.multi_handedness:
                    handedness_label = hand_results.multi_handedness[0].classification[0].label

                hand_is_open, _ = PTh._is_open_palm(
                    hand_landmarks,
                    handedness_label=handedness_label,
                    require_extended=4
                )

                cx, cy = PTh._hand_center_px(hand_landmarks, w, h)
                hand_center = (cx, cy)

                # Only consider the brush "active" if the palm center is over the segmented body
                if 0 <= cx < w and 0 <= cy < h:
                    hand_over_body = seg_mask[cy, cx] >= thresh

                # Find vertices inside brush radius
                if hand_is_open and hand_over_body:
                    d2 = (V[:, 0] - cx) ** 2 + (V[:, 1] - cy) ** 2
                    affected_vertices = d2 <= (brush_radius ** 2)

                    # A triangle is affected if:
                    # - it is active
                    # - and at least one of its vertices lies in the brush radius
                    affected_triangles = active & np.any(affected_vertices[T], axis=1)

            # --- Draw overlays ---
            vis = TMh._draw_triangle_overlay(
                frame=frame,
                V=V,
                T=T,
                active_mask=active,
                affected_mask=affected_triangles,
                pale_color=(0, 0, 0),
                bright_color=(0, 0, 0),
                pale_alpha=0.12,
                bright_alpha=0.42,
                line_thickness=1,
            )

            # Draw affected vertices
            if np.any(affected_vertices):
                pts = np.round(V[affected_vertices]).astype(np.int32)
                for x, y in pts:
                    cv2.circle(vis, (x, y), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)

            # Draw brush circle
            if hand_center is not None:
                cx, cy = hand_center
                brush_color = (0, 255, 0) if (hand_is_open and hand_over_body) else (180, 180, 180)
                cv2.circle(vis, (cx, cy), brush_radius, brush_color, 2, lineType=cv2.LINE_AA)
                cv2.circle(vis, (cx, cy), 4, brush_color, -1, lineType=cv2.LINE_AA)

                status = "OPEN PALM" if hand_is_open else "hand detected"
                if hand_is_open and not hand_over_body:
                    status += " (not over body)"
                cv2.putText(
                    vis,
                    status,
                    (cx + 12, max(20, cy - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    brush_color,
                    2,
                    cv2.LINE_AA
                )

            # Debug text
            cv2.putText(
                vis,
                f"active: {int(active.sum())}/{len(T)}   affected: {int(affected_triangles.sum())}   step={step}   thresh={thresh:.2f}   feather={feather}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                vis,
                "q quit | [ ] thresh | - + step | f toggle feather",
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Hand brush over active triangles", vis)

            if show_mask:
                mask_vis = (seg_mask * 255).astype(np.uint8)
                cv2.imshow("Segmentation mask", mask_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Adjust threshold
            if key == ord(']'):
                thresh = min(0.95, thresh + 0.05)
            elif key == ord('['):
                thresh = max(0.05, thresh - 0.05)

            # Adjust mesh density
            elif key in (ord('+'), ord('=')):
                step = max(10, step - 5)   # smaller step => denser mesh
                V, T, nx, ny = TMh.build_grid_mesh(w, h, step=step)
            elif key in (ord('-'), ord('_')):
                step = min(150, step + 5)  # larger step => coarser mesh
                V, T, nx, ny = TMh.build_grid_mesh(w, h, step=step)

            # Toggle feather
            elif key == ord('f'):
                feather = 0 if feather else 9

    cap.release()
    cv2.destroyAllWindows()


def body_bbox_from_mask(seg_mask, thresh=0.5):
    ys, xs = np.where(seg_mask > thresh)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)

    return {
        "cx": float(cx),
        "cy": float(cy),
        "w": float(bw),
        "h": float(bh),
        "x0": int(x0),
        "x1": int(x1),
        "y0": int(y0),
        "y1": int(y1),
    }


def make_tracked_mesh(V_base, ref_box, cur_box):
    """
    Build a version of V_base that follows the person's current position/scale.
    """
    sx = cur_box["w"] / max(ref_box["w"], 1e-6)
    sy = cur_box["h"] / max(ref_box["h"], 1e-6)

    V_track = V_base.copy().astype(np.float32)
    V_track[:, 0] = (V_base[:, 0] - ref_box["cx"]) * sx + cur_box["cx"]
    V_track[:, 1] = (V_base[:, 1] - ref_box["cy"]) * sy + cur_box["cy"]

    return V_track, sx, sy


def test_hand_brush_drag_live(step=40, thresh=0.5, feather=0, show_mask=True):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return
    else:
        print("Camera found")

    ok, frame = cap.read()
    if not ok:
        print("Error: could not read initial frame.")
        cap.release()
        return

    h, w = frame.shape[:2]

    V_base, T, nx, ny = TMh.build_grid_mesh(w, h, step=step)
    V_def = V_base.copy().astype(np.float32)

    # persistent deformation stored in BODY-LOCAL coordinates
    offset_local = np.zeros_like(V_base, dtype=np.float32)

    brush_radius = max(45, int(min(w, h) * 0.08))

    interaction_state = {
        "preview_vertices": np.zeros(len(V_def), dtype=bool),
        "preview_triangles": np.zeros(len(T), dtype=bool),
        "drag_vertices": np.zeros(len(V_def), dtype=bool),
        "drag_triangles": np.zeros(len(T), dtype=bool),
        "dragging": False,
        "prev_hand_center": None,
        "hand_was_open": False,
        "ref_body_box": None,
    }

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter, \
         mp_hands.Hands(
             static_image_mode=False,
             max_num_hands=1,
             model_complexity=1,
             min_detection_confidence=0.5,
             min_tracking_confidence=0.5,
         ) as hands:

        run_hand_brush_drag_loop(
            cap=cap,
            segmenter=segmenter,
            hands=hands,
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
            offset_local=offset_local,
        )

    cap.release()
    cv2.destroyAllWindows()

def test_hand_brush_drag_live_pose(step=40, thresh=0.5, feather=0, show_mask=False):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return
    else:
        print("Camera found")

    ok, frame = cap.read()
    if not ok:
        print("Error: could not read initial frame.")
        cap.release()
        return

    h, w = frame.shape[:2]

    # FULL-frame mesh
    V_base, T, nx, ny = TMh.build_grid_mesh(w, h, step=step)
    V_def = V_base.copy().astype(np.float32)

    # persistent offsets in normalized body coordinates
    offset_local = np.zeros_like(V_base, dtype=np.float32)

    brush_radius = max(45, int(min(w, h) * 0.08))

    interaction_state = {
        "preview_vertices": np.zeros(len(V_def), dtype=bool),
        "preview_triangles": np.zeros(len(T), dtype=bool),
        "drag_vertices": np.zeros(len(V_def), dtype=bool),
        "drag_triangles": np.zeros(len(T), dtype=bool),
        "dragging": False,
        "prev_hand_center": None,
        "hand_was_open": False,
        "ref_body_box": None,
    }

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter, \
         mp_hands.Hands(
             static_image_mode=False,
             max_num_hands=1,
             model_complexity=1,
             min_detection_confidence=0.5,
             min_tracking_confidence=0.5,
         ) as hands, \
         mp_pose.Pose(
             static_image_mode=False,
             model_complexity=1,
             smooth_landmarks=True,
             enable_segmentation=False,
             min_detection_confidence=0.5,
             min_tracking_confidence=0.5,
         ) as pose:

        Lh.run_hand_brush_drag_loop_pose(
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
            offset_local=offset_local,
        )

    cap.release()
    cv2.destroyAllWindows()

def test_hand_brush_drag_live(step=40, thresh=0.5, feather=0, show_mask=False):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return
    else:
        print("Camera found")

    ok, frame = cap.read()
    if not ok:
        print("Error: could not read initial frame.")
        cap.release()
        return

    h, w = frame.shape[:2]

    V_base, T, nx, ny = TMh.build_grid_mesh(w, h, step=step)
    V_def = V_base.copy().astype(np.float32)

    # persistent deformation stored in BODY-LOCAL coordinates
    offset_local = np.zeros_like(V_base, dtype=np.float32)

    brush_radius = max(45, int(min(w, h) * 0.08))

    interaction_state = {
        "preview_vertices": np.zeros(len(V_def), dtype=bool),
        "preview_triangles": np.zeros(len(T), dtype=bool),
        "drag_vertices": np.zeros(len(V_def), dtype=bool),
        "drag_triangles": np.zeros(len(T), dtype=bool),
        "dragging": False,
        "prev_hand_center": None,
        "hand_was_open": False,
        "ref_body_box": None,
    }

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter, \
         mp_hands.Hands(
             static_image_mode=False,
             max_num_hands=1,
             model_complexity=1,
             min_detection_confidence=0.5,
             min_tracking_confidence=0.5,
         ) as hands:

        Lh.run_hand_brush_drag_loop(
            cap=cap,
            segmenter=segmenter,
            hands=hands,
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
            offset_local=offset_local,
        )

    cap.release()
    cv2.destroyAllWindows()

def test_hand_brush_drag_arap_live_rotated(
    step=40,
    thresh=0.5,
    feather=0,
    show_mask=False,
    rotate_deg=90,
    rotate_dir="ccw",
    primary_monitor_width=1920,
    window_name="Hand brush drag on mesh",
):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return
    else:
        print("Camera found")

    ok, frame = cap.read()
    if not ok:
        print("Error: could not read initial frame.")
        cap.release()
        return

    h, w = frame.shape[:2]

    V_base, T, nx, ny = TMh.build_grid_mesh(w, h, step=step)
    V_def = V_base.copy().astype(np.float32)

    offset_local = np.zeros_like(V_base, dtype=np.float32)

    E = TMh.build_unique_edges(T)
    neighbors = TMh.build_vertex_neighbors(len(V_base), E)
    arap_cache = {
        "E": E,
        "neighbors": neighbors,
    }

    brush_radius = max(45, int(min(w, h) * 0.08))

    interaction_state = {
        "preview_vertices": np.zeros(len(V_def), dtype=bool),
        "preview_triangles": np.zeros(len(T), dtype=bool),
        "drag_vertices": np.zeros(len(V_def), dtype=bool),
        "drag_triangles": np.zeros(len(T), dtype=bool),
        "dragging": False,
        "prev_hand_center": None,
        "hand_was_open": False,
        "ref_body_box": None,
    }

    Dh.setup_fullscreen_second_monitor_window(
        window_name=window_name,
        primary_monitor_width=primary_monitor_width,
        y=0,
    )

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter, \
         mp_hands.Hands(
             static_image_mode=False,
             max_num_hands=1,
             model_complexity=1,
             min_detection_confidence=0.5,
             min_tracking_confidence=0.5,
         ) as hands:

        Lh.run_hand_brush_drag_arap_loop_rotated(
            cap=cap,
            segmenter=segmenter,
            hands=hands,
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
            offset_local=offset_local,
            arap_cache=arap_cache,
            rotate_deg=rotate_deg,
            rotate_dir=rotate_dir,
            window_name=window_name,
        )

    cap.release()
    cv2.destroyAllWindows()


def test_hand_brush_drag_arap_live(step=40, thresh=0.5, feather=0, show_mask=False):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return
    else:
        print("Camera found")

    ok, frame = cap.read()
    if not ok:
        print("Error: could not read initial frame.")
        cap.release()
        return

    h, w = frame.shape[:2]

    V_base, T, nx, ny = TMh.build_grid_mesh(w, h, step=step)
    V_def = V_base.copy().astype(np.float32)

    # persistent deformation stored in BODY-LOCAL coordinates
    offset_local = np.zeros_like(V_base, dtype=np.float32)

    E = TMh.build_unique_edges(T)
    neighbors = TMh.build_vertex_neighbors(len(V_base), E)
    arap_cache = {
        "E": E,
        "neighbors": neighbors,
    }

    brush_radius = max(45, int(min(w, h) * 0.2))

    interaction_state = {
        "preview_vertices": np.zeros(len(V_def), dtype=bool),
        "preview_triangles": np.zeros(len(T), dtype=bool),
        "drag_vertices": np.zeros(len(V_def), dtype=bool),
        "drag_triangles": np.zeros(len(T), dtype=bool),
        "dragging": False,
        "prev_hand_center": None,
        "hand_was_open": False,
        "ref_body_box": None,
    }

    # simple window (main monitor, no movement)
    window_name = "Hand brush drag on mesh"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter, \
         mp_hands.Hands(
             static_image_mode=False,
             max_num_hands=1,
             model_complexity=1,
             min_detection_confidence=0.5,
             min_tracking_confidence=0.5,
         ) as hands:

        Lh.run_hand_brush_drag_arap_loop(
            cap=cap,
            segmenter=segmenter,
            hands=hands,
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
            offset_local=offset_local,
            arap_cache=arap_cache,
            window_name=window_name,  # make sure loop uses this
        )

    cap.release()
    cv2.destroyAllWindows()




def run_hand_brush_drag_loop(
    cap,
    segmenter,
    hands,
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
    offset_local,
):
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)

        # --- segmentation ---
        seg_mask, rgb = PTh.get_segmentation_mask(frame, segmenter, feather=feather)

        # --- body tracking ---
        cur_body_box = body_bbox_from_mask(seg_mask, thresh=0.5)

        if cur_body_box is not None:
            if interaction_state["ref_body_box"] is None:
                interaction_state["ref_body_box"] = cur_body_box

            ref_body_box = interaction_state["ref_body_box"]
            V_track, sx, sy = make_tracked_mesh(V_base, ref_body_box, cur_body_box)

            # reconstruct visible mesh from tracked neutral mesh + stored local offsets
            V_def[:, 0] = V_track[:, 0] + offset_local[:, 0] * sx
            V_def[:, 1] = V_track[:, 1] + offset_local[:, 1] * sy
        else:
            # if body temporarily lost, keep last mesh
            sx, sy = 1.0, 1.0
            V_track = V_def.copy()

        # active triangles based on CURRENT deformed mesh
        active = TMh.active_triangles_from_mask(V_def, T, seg_mask, thresh=thresh)

        # --- hand state ---
        hand_state = PTh.get_hand_state(rgb, hands, seg_mask, thresh, w, h)

        hand_center = hand_state["center"]
        hand_is_open = hand_state["is_open"]
        hand_is_fist = hand_state["is_fist"]
        hand_over_body = hand_state["over_body"]
        hand_detected = hand_state["detected"]

        # fresh preview for this frame
        new_preview_vertices = np.zeros(len(V_def), dtype=bool)
        new_preview_triangles = np.zeros(len(T), dtype=bool)

        if hand_detected:
            # preview selection from CURRENT tracked/deformed mesh
            if (not interaction_state["dragging"]) and hand_is_open and hand_over_body:
                new_preview_vertices, new_preview_triangles = TMh.compute_brush_selection(
                    V=V_def,
                    T=T,
                    active=active,
                    center=hand_center,
                    radius=brush_radius,
                )

            # open palm -> fist starts dragging using last preview selection
            if (
                (not interaction_state["dragging"])
                and interaction_state["hand_was_open"]
                and hand_is_fist
                and np.any(interaction_state["preview_vertices"])
            ):
                interaction_state["dragging"] = True
                interaction_state["drag_vertices"] = interaction_state["preview_vertices"].copy()
                interaction_state["drag_triangles"] = np.any(
                    interaction_state["drag_vertices"][T], axis=1
                )
                interaction_state["prev_hand_center"] = hand_center

            # while fist is held, update BODY-LOCAL offsets
            elif (
                interaction_state["dragging"]
                and hand_is_fist
                and hand_center is not None
                and interaction_state["prev_hand_center"] is not None
            ):
                dx = hand_center[0] - interaction_state["prev_hand_center"][0]
                dy = hand_center[1] - interaction_state["prev_hand_center"][1]

                # store drag in local/body coordinates, not raw screen coordinates
                offset_local[interaction_state["drag_vertices"], 0] += dx / max(sx, 1e-6)
                offset_local[interaction_state["drag_vertices"], 1] += dy / max(sy, 1e-6)

                interaction_state["prev_hand_center"] = hand_center

                # rebuild current visible mesh immediately after local offset update
                if interaction_state["ref_body_box"] is not None and cur_body_box is not None:
                    V_track, sx, sy = make_tracked_mesh(
                        V_base,
                        interaction_state["ref_body_box"],
                        cur_body_box
                    )
                    V_def[:, 0] = V_track[:, 0] + offset_local[:, 0] * sx
                    V_def[:, 1] = V_track[:, 1] + offset_local[:, 1] * sy

            # release fist -> stop dragging
            elif interaction_state["dragging"] and (not hand_is_fist):
                interaction_state["dragging"] = False
                interaction_state["drag_vertices"][:] = False
                interaction_state["drag_triangles"][:] = False
                interaction_state["prev_hand_center"] = None

            # update preview only when not dragging
            if not interaction_state["dragging"]:
                interaction_state["preview_vertices"] = new_preview_vertices
                interaction_state["preview_triangles"] = new_preview_triangles

            interaction_state["hand_was_open"] = hand_is_open

        else:
            # if hand is lost, stop dragging safely
            if interaction_state["dragging"]:
                interaction_state["dragging"] = False
                interaction_state["drag_vertices"][:] = False
                interaction_state["drag_triangles"][:] = False
                interaction_state["prev_hand_center"] = None

            interaction_state["preview_vertices"][:] = False
            interaction_state["preview_triangles"][:] = False
            interaction_state["hand_was_open"] = False

        # choose current highlighted set
        if interaction_state["dragging"]:
            affected_vertices = interaction_state["drag_vertices"]
            affected_triangles = interaction_state["drag_triangles"] & active
        else:
            affected_vertices = interaction_state["preview_vertices"]
            affected_triangles = interaction_state["preview_triangles"]

        # --- draw ---
        vis = TMh._draw_triangle_overlay(
            frame=frame,
            V=V_def,
            T=T,
            active_mask=active,
            affected_mask=affected_triangles,
            pale_color=(0, 255, 0),
            bright_color=(0, 255, 0),
            pale_alpha=0.12,
            bright_alpha=0.42,
            line_thickness=1,
        )

        # draw selected vertices
        if np.any(affected_vertices):
            pts = np.round(V_def[affected_vertices]).astype(np.int32)
            for x, y in pts:
                cv2.circle(vis, (x, y), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)

        # draw brush circle only when not dragging
        if hand_center is not None and not interaction_state["dragging"]:
            cx, cy = hand_center
            brush_color = (0, 255, 0) if (hand_is_open and hand_over_body) else (180, 180, 180)
            cv2.circle(vis, (cx, cy), brush_radius, brush_color, 2, lineType=cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 4, brush_color, -1, lineType=cv2.LINE_AA)

        # draw body bbox for debugging
        if cur_body_box is not None:
            cv2.rectangle(
                vis,
                (cur_body_box["x0"], cur_body_box["y0"]),
                (cur_body_box["x1"], cur_body_box["y1"]),
                (255, 180, 0),
                2,
                lineType=cv2.LINE_AA
            )

        # hand status text
        if hand_center is not None:
            cx, cy = hand_center
            if interaction_state["dragging"]:
                status = "DRAGGING"
                color = (0, 255, 0)
            elif hand_is_open and hand_over_body:
                status = "OPEN PALM"
                color = (0, 255, 0)
            elif hand_is_open:
                status = "OPEN PALM (not over body)"
                color = (180, 180, 180)
            elif hand_is_fist:
                status = "FIST"
                color = (0, 220, 255)
            else:
                status = "hand detected"
                color = (180, 180, 180)

            cv2.putText(
                vis,
                status,
                (cx + 12, max(20, cy - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

        # debug text
        cv2.putText(
            vis,
            f"active: {int(active.sum())}/{len(T)}   selected: {int(affected_triangles.sum())}   step={step}   thresh={thresh:.2f}   feather={feather}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            vis,
            "q quit | [ ] thresh | - + step | f toggle feather | r reset mesh",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Hand brush drag on mesh", vis)

        if show_mask:
            mask_vis = (seg_mask * 255).astype(np.uint8)
            cv2.imshow("Segmentation mask", mask_vis)

        # --- keyboard ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        elif key == ord(']'):
            thresh = min(0.95, thresh + 0.05)

        elif key == ord('['):
            thresh = max(0.05, thresh - 0.05)

        elif key in (ord('+'), ord('=')):
            step = max(10, step - 5)
            V_base, T, nx, ny = TMh.build_grid_mesh(w, h, step=step)
            V_def = V_base.copy().astype(np.float32)
            offset_local = np.zeros_like(V_base, dtype=np.float32)

            interaction_state["preview_vertices"] = np.zeros(len(V_def), dtype=bool)
            interaction_state["preview_triangles"] = np.zeros(len(T), dtype=bool)
            interaction_state["drag_vertices"] = np.zeros(len(V_def), dtype=bool)
            interaction_state["drag_triangles"] = np.zeros(len(T), dtype=bool)
            interaction_state["dragging"] = False
            interaction_state["prev_hand_center"] = None
            interaction_state["hand_was_open"] = False
            interaction_state["ref_body_box"] = None

        elif key in (ord('-'), ord('_')):
            step = min(150, step + 5)
            V_base, T, nx, ny = TMh.build_grid_mesh(w, h, step=step)
            V_def = V_base.copy().astype(np.float32)
            offset_local = np.zeros_like(V_base, dtype=np.float32)

            interaction_state["preview_vertices"] = np.zeros(len(V_def), dtype=bool)
            interaction_state["preview_triangles"] = np.zeros(len(T), dtype=bool)
            interaction_state["drag_vertices"] = np.zeros(len(V_def), dtype=bool)
            interaction_state["drag_triangles"] = np.zeros(len(T), dtype=bool)
            interaction_state["dragging"] = False
            interaction_state["prev_hand_center"] = None
            interaction_state["hand_was_open"] = False
            interaction_state["ref_body_box"] = None

        elif key == ord('f'):
            feather = 0 if feather else 9

        elif key == ord('r'):
            V_def = V_base.copy().astype(np.float32)
            offset_local = np.zeros_like(V_base, dtype=np.float32)

            interaction_state["preview_vertices"][:] = False
            interaction_state["preview_triangles"][:] = False
            interaction_state["drag_vertices"][:] = False
            interaction_state["drag_triangles"][:] = False
            interaction_state["dragging"] = False
            interaction_state["prev_hand_center"] = None
            interaction_state["hand_was_open"] = False
            interaction_state["ref_body_box"] = None



#Stest_warp_triangle()
# interactive_triangle_test()
# make_draw_mesh_test()
# test_active_triangles_live()
# test_hand_brush_triangles_live(step=40, thresh=0.5, feather=0, show_mask=True)
# test_hand_brush_drag_live(step=40, thresh=0.5, feather=0, show_mask=True)
# test_hand_brush_drag_live_pose(step=40, thresh=0.5, feather=0, show_mask=False)
# test_hand_brush_drag_live(step=40, thresh=0.5, feather=0, show_mask=False)
test_hand_brush_drag_arap_live(step=40, thresh=0.5, feather=0, show_mask=False)
# test_hand_brush_drag_arap_live_rotated(
#     step=40,
#     thresh=0.5,
#     feather=0,
#     show_mask=False,
#     rotate_deg=90,
#     rotate_dir="cw",
#     primary_monitor_width=1920,
#     window_name="Hand brush drag on mesh",
# )
