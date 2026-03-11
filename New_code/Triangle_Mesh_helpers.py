import cv2
import numpy as np
import mediapipe as mp
import time

import Pose_Tracking_helpers as PTh

state = {
    "dragging": False,
    "prev_hand_center": None,
    "hand_was_open": False,
    "preview_vertices": ...,
    "preview_triangles": ...,
    "drag_vertices": ...,
    "drag_triangles": ...,
}

def warp_triangle(src_img, dst_img, t_src, t_dst):
    """
    src_img: original BGR image
    dst_img: destination canvas to write into (BGR)
    t_src, t_dst: (3,2) shape array, float32 triangle vertices in (x,y)
    """
    t_src = np.float32(t_src)
    t_dst = np.float32(t_dst)

    # get the smallest axis-aligned rectangles that bound our input triangles
    # output: (x,y,w,h) tuple for each rectangle, where (x,y) is the top left corner
    r1 = cv2.boundingRect(t_src)
    r2 = cv2.boundingRect(t_dst)

    # Offset triangles to their bounding boxes
    t1_rect = t_src - np.array([r1[0], r1[1]], dtype=np.float32)
    t2_rect = t_dst - np.array([r2[0], r2[1]], dtype=np.float32)

    # Crop source patch
    src_rect = src_img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]

    # Affine transform
    M = cv2.getAffineTransform(t1_rect, t2_rect)
    warped_rect = cv2.warpAffine(
        src_rect, M, (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # Mask for the triangle in destination rect
    mask = np.zeros((r2[3], r2[2]), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), 1.0, 16)

    # Composite into dst_img
    dst_roi = dst_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    if dst_roi.shape[:2] != warped_rect.shape[:2]:
        return  # safety

    # alpha blend triangle region
    mask3 = np.dstack([mask, mask, mask])
    dst_roi[:] = dst_roi * (1.0 - mask3) + warped_rect * mask3



def build_grid_mesh(w, h, step=30):
    xs = np.arange(0, w, step, dtype=np.float32)
    ys = np.arange(0, h, step, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    V = np.stack([xv.ravel(), yv.ravel()], axis=1)  # (N,2)

    nx = len(xs)
    ny = len(ys)

    def vid(i, j):
        return j * nx + i

    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])

    T = np.array(tris, dtype=np.int32)
    return V, T, nx, ny

def draw_mesh(img_bgr, V, T, color=(0, 255, 0), thickness=1):
    """
    img_bgr: (H,W,3) uint8
    V: (N,2) float32 vertices (x,y)
    T: (M,3) int32 triangle indices
    """
    out = img_bgr.copy()
    for tri in T:
        pts = V[tri].astype(np.int32).reshape(-1, 1, 2)  # (3,1,2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    return out

def draw_vertices(img_bgr, V, color=(0, 0, 255), r=2):
    out = img_bgr.copy()
    for x, y in V:
        cv2.circle(out, (int(x), int(y)), r, color, -1)
    return out

def draw_mesh_with_vertices(img_bgr, V, T):
    out = draw_mesh(img_bgr, V, T, color=(0, 255, 0), thickness=1)
    out = draw_vertices(out, V, color=(0, 0, 255), r=2)
    return out

def active_triangles_from_mask(V, T, seg_mask, thresh=0.5):
    H, W = seg_mask.shape
    tri_pts = V[T]                      # (M,3,2)
    centroids = tri_pts.mean(axis=1)    # (M,2)
    cx = np.clip(centroids[:,0].astype(np.int32), 0, W-1)
    cy = np.clip(centroids[:,1].astype(np.int32), 0, H-1)
    active = seg_mask[cy, cx] >= thresh
    return active

def deform_vertices_gaussian(V, cx, cy, gain=25.0, sigma=120.0):
    V2 = V.copy()
    dx = V[:,0] - cx
    dy = V[:,1] - cy
    w = np.exp(-(dx*dx + dy*dy) / (2*sigma*sigma)).astype(np.float32)
    # push outwards from center (left goes more left, right goes more right)
    V2[:,0] += gain * w * np.sign(dx + 1e-6)
    return V2

def warp_mesh(src_bgr, V, T, V_dst, active):
    dst = src_bgr.copy()
    for i, tri in enumerate(T):
        if not active[i]:
            continue
        t_src = V[tri]
        t_dst = V_dst[tri]
        warp_triangle(src_bgr, dst, t_src, t_dst)
    return dst

def vertex_inside_mask(V, seg_mask, thresh=0.5):
    """Returns boolean array (N,) telling whether each vertex is inside the segmentation mask."""
    H, W = seg_mask.shape
    x = np.clip(V[:, 0].astype(np.int32), 0, W - 1)
    y = np.clip(V[:, 1].astype(np.int32), 0, H - 1)
    return seg_mask[y, x] >= thresh

def deform_hips_abdomen(V, bbox, gain=40.0, sigma_x=140.0, sigma_y=110.0):
    """
    Hip/abdomen-only deformation: push x outward, but only within a vertical band.
    bbox: (x0,y0,x1,y1) region where deformation is allowed
    """
    x0, y0, x1, y1 = bbox
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    V2 = V.copy()
    dx = V[:, 0] - cx
    dy = V[:, 1] - cy

    # region gating: only vertices inside bbox get weighted
    in_box = (V[:, 0] >= x0) & (V[:, 0] <= x1) & (V[:, 1] >= y0) & (V[:, 1] <= y1)

    # elliptical gaussian falloff
    w = np.exp(-(dx*dx)/(2*sigma_x*sigma_x) - (dy*dy)/(2*sigma_y*sigma_y)).astype(np.float32)

    # widen (left moves left, right moves right) within box
    V2[in_box, 0] += gain * w[in_box] * np.sign(dx[in_box] + 1e-6)
    return V2

def draw_active_triangles(img_bgr, V, T, active, color=(0,255,0), thickness=1):
    out = img_bgr.copy()
    for i, tri in enumerate(T):
        if not active[i]:
            continue
        pts = V[tri].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, color, thickness)
    return out

def _draw_triangle_overlay(
    frame,
    V,
    T,
    active_mask,
    affected_mask=None,
    pale_color=(0, 255, 0),
    bright_color=(0, 255, 0),
    pale_alpha=0.12,
    bright_alpha=0.45,
    line_thickness=1,
):
    """
    Draws:
    - all active triangles in pale green
    - affected triangles in brighter green on top
    """
    vis = frame.copy()
    fill = frame.copy()

    # Pale fill for all active triangles
    active_ids = np.where(active_mask)[0]
    for tid in active_ids:
        tri = np.round(V[T[tid]]).astype(np.int32)
        cv2.fillConvexPoly(fill, tri, pale_color)

    vis = cv2.addWeighted(fill, pale_alpha, vis, 1.0 - pale_alpha, 0)

    # Pale outlines
    for tid in active_ids:
        tri = np.round(V[T[tid]]).astype(np.int32)
        cv2.polylines(vis, [tri], isClosed=True, color=pale_color, thickness=line_thickness, lineType=cv2.LINE_AA)

    # Bright fill + outline for affected triangles
    if affected_mask is not None and np.any(affected_mask):
        fill2 = vis.copy()
        affected_ids = np.where(affected_mask)[0]

        for tid in affected_ids:
            tri = np.round(V[T[tid]]).astype(np.int32)
            cv2.fillConvexPoly(fill2, tri, bright_color)

        vis = cv2.addWeighted(fill2, bright_alpha, vis, 1.0 - bright_alpha, 0)

        for tid in affected_ids:
            tri = np.round(V[T[tid]]).astype(np.int32)
            cv2.polylines(vis, [tri], isClosed=True, color=bright_color, thickness=max(2, line_thickness), lineType=cv2.LINE_AA)

    return vis

def compute_brush_selection(V, T, active, center, radius):
    if center is None:
        return np.zeros(len(V), dtype=bool), np.zeros(len(T), dtype=bool)

    cx, cy = center
    d2 = (V[:, 0] - cx) ** 2 + (V[:, 1] - cy) ** 2
    selected_vertices = d2 <= (radius ** 2)
    selected_triangles = active & np.any(selected_vertices[T], axis=1)
    return selected_vertices, selected_triangles

def update_drag_state(V_def, T, active, hand_state, state, brush_radius):
    new_preview_vertices = np.zeros(len(V_def), dtype=bool)
    new_preview_triangles = np.zeros(len(T), dtype=bool)

    if hand_state["detected"]:
        hand_center = hand_state["center"]
        hand_is_open = hand_state["is_open"]
        hand_is_fist = hand_state["is_fist"]
        hand_over_body = hand_state["over_body"]

        if (not state["dragging"]) and hand_is_open and hand_over_body:
            new_preview_vertices, new_preview_triangles = compute_brush_selection(
                V_def, T, active, hand_center, brush_radius
            )

        if (not state["dragging"]) and state["hand_was_open"] and hand_is_fist and np.any(state["preview_vertices"]):
            state["dragging"] = True
            state["drag_vertices"] = state["preview_vertices"].copy()
            state["drag_triangles"] = np.any(state["drag_vertices"][T], axis=1)
            state["prev_hand_center"] = hand_center

        elif state["dragging"] and hand_is_fist and hand_center is not None and state["prev_hand_center"] is not None:
            dx = hand_center[0] - state["prev_hand_center"][0]
            dy = hand_center[1] - state["prev_hand_center"][1]

            V_def[state["drag_vertices"], 0] += dx
            V_def[state["drag_vertices"], 1] += dy

            state["prev_hand_center"] = hand_center

        elif state["dragging"] and (not hand_is_fist):
            state["dragging"] = False
            state["drag_vertices"][:] = False
            state["drag_triangles"][:] = False
            state["prev_hand_center"] = None

        if not state["dragging"]:
            state["preview_vertices"] = new_preview_vertices
            state["preview_triangles"] = new_preview_triangles

        state["hand_was_open"] = hand_is_open

    else:
        if state["dragging"]:
            state["dragging"] = False
            state["drag_vertices"][:] = False
            state["drag_triangles"][:] = False
            state["prev_hand_center"] = None

        state["preview_vertices"][:] = False
        state["preview_triangles"][:] = False
        state["hand_was_open"] = False

    return V_def, state

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

        # active triangles based on CURRENT deformed mesh
        active = active_triangles_from_mask(V_def, T, seg_mask, thresh=thresh)

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
            # open palm previews selection if not currently dragging
            if (not interaction_state["dragging"]) and hand_is_open and hand_over_body:
                new_preview_vertices, new_preview_triangles = compute_brush_selection(
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

            # while fist is held, move locked vertices by hand delta
            elif (
                interaction_state["dragging"]
                and hand_is_fist
                and hand_center is not None
                and interaction_state["prev_hand_center"] is not None
            ):
                dx = hand_center[0] - interaction_state["prev_hand_center"][0]
                dy = hand_center[1] - interaction_state["prev_hand_center"][1]

                V_def[interaction_state["drag_vertices"], 0] += dx
                V_def[interaction_state["drag_vertices"], 1] += dy

                interaction_state["prev_hand_center"] = hand_center

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
        vis = _draw_triangle_overlay(
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
            V_base, T, nx, ny = build_grid_mesh(w, h, step=step)
            V_def = V_base.copy().astype(np.float32)

            interaction_state["preview_vertices"] = np.zeros(len(V_def), dtype=bool)
            interaction_state["preview_triangles"] = np.zeros(len(T), dtype=bool)
            interaction_state["drag_vertices"] = np.zeros(len(V_def), dtype=bool)
            interaction_state["drag_triangles"] = np.zeros(len(T), dtype=bool)
            interaction_state["dragging"] = False
            interaction_state["prev_hand_center"] = None
            interaction_state["hand_was_open"] = False

        elif key in (ord('-'), ord('_')):
            step = min(150, step + 5)
            V_base, T, nx, ny = build_grid_mesh(w, h, step=step)
            V_def = V_base.copy().astype(np.float32)

            interaction_state["preview_vertices"] = np.zeros(len(V_def), dtype=bool)
            interaction_state["preview_triangles"] = np.zeros(len(T), dtype=bool)
            interaction_state["drag_vertices"] = np.zeros(len(V_def), dtype=bool)
            interaction_state["drag_triangles"] = np.zeros(len(T), dtype=bool)
            interaction_state["dragging"] = False
            interaction_state["prev_hand_center"] = None
            interaction_state["hand_was_open"] = False

        elif key == ord('f'):
            feather = 0 if feather else 9

        elif key == ord('r'):
            V_def = V_base.copy().astype(np.float32)
            interaction_state["preview_vertices"][:] = False
            interaction_state["preview_triangles"][:] = False
            interaction_state["drag_vertices"][:] = False
            interaction_state["drag_triangles"][:] = False
            interaction_state["dragging"] = False
            interaction_state["prev_hand_center"] = None
            interaction_state["hand_was_open"] = False