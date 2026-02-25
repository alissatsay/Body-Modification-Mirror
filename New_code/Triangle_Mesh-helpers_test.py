## Test module for triangle mesh

import cv2
import numpy as np
import Triangle_Mesh_helpers

import mediapipe as mp
import GLSL_HT_UI

mp_selfie_segmentation = mp.solutions.selfie_segmentation

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

    h, w = frame.shape[:2]
    V, T, nx, ny = Triangle_Mesh_helpers.build_grid_mesh(w, h, step=step)

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = rotate_frame(frame)

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


#Stest_warp_triangle()
# interactive_triangle_test()
# make_draw_mesh_test()
test_active_triangles_live()
