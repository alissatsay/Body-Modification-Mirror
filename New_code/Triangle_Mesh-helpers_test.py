## Test module for triangle mesh

import cv2
import numpy as np
import Triangle_Mesh_helpers
import Pose_Tracking_helpers as PTh

import mediapipe as mp
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
    Triangle_Mesh_helpers.warp_triangle(src_img, dst_img, t_src, t_dst)

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

        Triangle_Mesh_helpers.warp_triangle(
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
    V, T, nx, ny = Triangle_Mesh_helpers.build_grid_mesh(w, h, step=40)

    vis = Triangle_Mesh_helpers.draw_mesh_with_vertices(img, V, T)
    cv2.imshow("mesh+verts", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_active_triangles(frame_bgr, V, T, active, color=(0, 255, 0), thickness=1):
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
    V, T, nx, ny = Triangle_Mesh_helpers.build_grid_mesh(w, h, step=step)

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

            active = Triangle_Mesh_helpers.active_triangles_from_mask(V, T, seg_mask, thresh=thresh)

            vis = draw_active_triangles(frame, V, T, active, color=(0, 255, 0), thickness=1)

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
                V, T, nx, ny = Triangle_Mesh_helpers.build_grid_mesh(w, h, step=step)
            elif key in (ord('-'), ord('_')):
                step = min(150, step + 5)  # larger step => coarser mesh
                V, T, nx, ny = Triangle_Mesh_helpers.build_grid_mesh(w, h, step=step)

            # Toggle feather
            elif key == ord('f'):
                feather = 0 if feather else 9  # switch between none and blur-kernel=9

    cap.release()
    cv2.destroyAllWindows()



def test_mesh_warp_hips_live(step=40, thresh=0.5, feather=9, show_mask=True):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return
    print("Camera found")

    ok, frame = cap.read()
    if not ok:
        print("Error: could not read initial frame.")
        cap.release()
        return

    h, w = frame.shape[:2]
    V, T, nx, ny = Triangle_Mesh_helpers.build_grid_mesh(w, h, step=step)

    # Defaults (used if pose fails)
    band_top = 0.45
    band_bot = 0.85
    band_half_width = 0.22
    gain = 22.0

    # Pose-tied tuning knobs
    top_frac_of_torso = 0.10   # band starts a bit below torso midpoint
    bot_frac_of_torso = 0.85   # down toward upper thighs
    band_half_width_frac = 0.22

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter, \
         mp_pose.Pose(model_complexity=1, enable_segmentation=False,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Segmentation
            seg = segmenter.process(rgb)
            seg_mask = seg.segmentation_mask  # float32 [0,1], shape (h,w)
            if feather and feather > 0:
                k = int(feather)
                if k % 2 == 0:
                    k += 1
                seg_mask = cv2.GaussianBlur(seg_mask, (k, k), 0)

            # Pose (for person-centric band)
            pose_res = pose.process(rgb)
            bbox = None
            hip_mid = shoulder_mid = torso_mid_y = None
            if pose_res.pose_landmarks:
                out = PTh._pose_band_from_landmarks(
                    pose_res.pose_landmarks.landmark, w, h,
                    band_half_width_frac=band_half_width_frac,
                    top_frac_of_torso=top_frac_of_torso,
                    bot_frac_of_torso=bot_frac_of_torso
                )
                if out is not None:
                    bbox, hip_mid, shoulder_mid, torso_mid_y = out

            # Fallback if pose is missing/unreliable
            if bbox is None:
                cx = w * 0.5
                x0 = int(cx - band_half_width * w)
                x1 = int(cx + band_half_width * w)
                y0 = int(band_top * h)
                y1 = int(band_bot * h)
                bbox = (x0, y0, x1, y1)

            # Active triangles: centroid inside body
            active = Triangle_Mesh_helpers.active_triangles_from_mask(V, T, seg_mask, thresh=thresh)

            # Deform vertices in the band
            V_dst = Triangle_Mesh_helpers.deform_hips_abdomen(V, bbox=bbox, gain=gain)

            # Freeze vertices outside mask
            inside = Triangle_Mesh_helpers.vertex_inside_mask(V, seg_mask, thresh=thresh)
            V_dst[~inside] = V[~inside]

            # Warp mesh
            warped = Triangle_Mesh_helpers.warp_mesh(frame, V, T, V_dst, active)

            # Composite body region
            mask = (seg_mask >= thresh).astype(np.float32)
            mask3 = np.dstack([mask, mask, mask])
            out = (frame * (1.0 - mask3) + warped * mask3).astype(np.uint8)

            # Debug overlays
            vis = out.copy()
            vis = draw_active_triangles(vis, V_dst, T, active, color=(0, 255, 0), thickness=1)

            x0, y0, x1, y1 = bbox
            # cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 255), 2)

            # if hip_mid is not None:
            #     cv2.circle(vis, (int(hip_mid[0]), int(hip_mid[1])), 5, (255, 255, 255), -1)
            # if shoulder_mid is not None:
            #     cv2.circle(vis, (int(shoulder_mid[0]), int(shoulder_mid[1])), 5, (255, 255, 255), -1)
            # if torso_mid_y is not None:
            #     cv2.line(vis, (0, int(torso_mid_y)), (w - 1, int(torso_mid_y)), (255, 255, 255), 1)

            cv2.putText(
                vis,
                f"active: {int(active.sum())}/{len(T)}  step={step}  thresh={thresh:.2f}  gain={gain:.1f}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA
            )
            cv2.putText(
                vis,
                "q quit | [ ] thresh | - + mesh | g/G gain | f feather | i/k topFrac | o/l botFrac | w/W widthFrac",
                (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA
            )

            cv2.imshow("Mesh warp (hips/abdomen, pose-tied)", vis)
            if show_mask:
                cv2.imshow("Segmentation mask", (seg_mask * 255).astype(np.uint8))

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
                step = max(10, step - 5)
                V, T, nx, ny = Triangle_Mesh_helpers.build_grid_mesh(w, h, step=step)
            elif key in (ord('-'), ord('_')):
                step = min(150, step + 5)
                V, T, nx, ny = Triangle_Mesh_helpers.build_grid_mesh(w, h, step=step)

            # Toggle feather
            elif key == ord('f'):
                feather = 0 if feather else 9

            # Adjust gain
            elif key == ord('g'):
                gain = max(0.0, gain - 2.0)
            elif key == ord('G'):
                gain = min(80.0, gain + 2.0)

            # Pose band tuning (relative to torso)
            elif key == ord('i'):
                top_frac_of_torso = max(-0.5, top_frac_of_torso - 0.05)
            elif key == ord('k'):
                top_frac_of_torso = min(bot_frac_of_torso - 0.05, top_frac_of_torso + 0.05)
            elif key == ord('o'):
                bot_frac_of_torso = max(top_frac_of_torso + 0.05, bot_frac_of_torso - 0.05)
            elif key == ord('l'):
                bot_frac_of_torso = min(1.5, bot_frac_of_torso + 0.05)

            # Band width
            elif key == ord('w'):
                band_half_width_frac = max(0.05, band_half_width_frac - 0.02)
            elif key == ord('W'):
                band_half_width_frac = min(0.48, band_half_width_frac + 0.02)

    cap.release()
    cv2.destroyAllWindows()


#Stest_warp_triangle()
interactive_triangle_test()
# make_draw_mesh_test()
# test_active_triangles_live()
# test_mesh_warp_hips_live(step=40, thresh=0.5, feather=9, show_mask=True)
