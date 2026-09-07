"""
Triangle-mesh geometry for run_installation.py: building the body mesh
(fixed grid or adaptive, bound to the tracked skeleton), warping/
reconstructing it every frame (vectorized "vectorized"/"from_skeleton"
variants, ARAP-based drag deformation), and rendering it back over the
background (piecewise warping, per-render-group layering with yaw-based
draw-order overrides, mask/overlay debug drawing). Also owns the
segment/render-group name tables and layering config that this geometry
is organized around.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay

from helpers import pose_tracking as PTh


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

def make_pose_tracked_mesh(V_base, ref_torso, cur_torso):
    """
    Track mesh by torso center + independent x/y scale.
    This is axis-aligned and deliberately simple/stable.
    """
    sx = cur_torso["width"] / max(ref_torso["width"], 1e-6)
    sy = cur_torso["height"] / max(ref_torso["height"], 1e-6)

    ref_c = ref_torso["center"]
    cur_c = cur_torso["center"]

    V_track = V_base.copy().astype(np.float32)
    V_track[:, 0] = (V_base[:, 0] - ref_c[0]) * sx + cur_c[0]
    V_track[:, 1] = (V_base[:, 1] - ref_c[1]) * sy + cur_c[1]

    return V_track, sx, sy


def draw_torso_debug(vis, torso):
    if torso is None:
        return vis

    pts = [
        torso["ls"],
        torso["rs"],
        torso["rh"],
        torso["lh"],
    ]
    pts = np.round(np.array(pts)).astype(np.int32)

    cv2.polylines(vis, [pts], isClosed=True, color=(255, 180, 0), thickness=2, lineType=cv2.LINE_AA)

    c = tuple(np.round(torso["center"]).astype(np.int32))
    sc = tuple(np.round(torso["shoulder_center"]).astype(np.int32))
    hc = tuple(np.round(torso["hip_center"]).astype(np.int32))

    cv2.circle(vis, c, 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(vis, sc, 4, (255, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(vis, hc, 4, (255, 0, 255), -1, lineType=cv2.LINE_AA)

    return vis

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

def make_body_tracked_mesh(V_base, ref_body, cur_body):
    """
    Move the FULL mesh using the current body box.
    The mesh remains full-frame; this just applies a global body-following transform.
    """
    sx = cur_body["width"] / max(ref_body["width"], 1e-6)
    sy = cur_body["height"] / max(ref_body["height"], 1e-6)

    ref_c = ref_body["center"]
    cur_c = cur_body["center"]

    V_track = V_base.copy().astype(np.float32)
    V_track[:, 0] = (V_base[:, 0] - ref_c[0]) * sx + cur_c[0]
    V_track[:, 1] = (V_base[:, 1] - ref_c[1]) * sy + cur_c[1]

    return V_track, sx, sy


def draw_body_debug(vis, body_box):
    if body_box is None:
        return vis

    cv2.rectangle(
        vis,
        (body_box["x0"], body_box["y0"]),
        (body_box["x1"], body_box["y1"]),
        (255, 180, 0),
        2,
        lineType=cv2.LINE_AA
    )

    c = tuple(np.round(body_box["center"]).astype(np.int32))
    cv2.circle(vis, c, 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    for p in body_box["pts"]:
        pt = tuple(np.round(p).astype(np.int32))
        cv2.circle(vis, pt, 3, (255, 0, 255), -1, lineType=cv2.LINE_AA)

    return vis

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

def build_unique_edges(T):
    """
    T: (m,3) triangle indices
    returns E: (k,2) unique undirected edges
    """
    edges = set()
    for tri in T:
        a, b, c = map(int, tri)
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return np.array(sorted(edges), dtype=np.int32)


def build_vertex_neighbors(nv, E):
    nbrs = [[] for _ in range(nv)]
    for i, j in E:
        nbrs[i].append(j)
        nbrs[j].append(i)
    return nbrs


def vertex_mask_from_active_triangles(T, active_triangles, nv):
    mask = np.zeros(nv, dtype=bool)
    if np.any(active_triangles):
        mask[np.unique(T[active_triangles].ravel())] = True
    return mask


def expand_vertex_region(seed_mask, neighbors, rings=4, valid_mask=None):
    """
    Expand a boolean vertex mask by graph rings.
    valid_mask restricts growth if provided.
    """
    region = seed_mask.copy()
    frontier = np.where(seed_mask)[0]

    for _ in range(rings):
        new_ids = []
        for v in frontier:
            for nb in neighbors[v]:
                if valid_mask is not None and not valid_mask[nb]:
                    continue
                if not region[nb]:
                    region[nb] = True
                    new_ids.append(nb)
        frontier = np.array(new_ids, dtype=np.int32)
        if len(frontier) == 0:
            break

    return region


def boundary_of_region(region_mask, neighbors):
    """
    Boundary = vertices inside region with at least one neighbor outside region.
    """
    boundary = np.zeros_like(region_mask)
    inside = np.where(region_mask)[0]

    for v in inside:
        for nb in neighbors[v]:
            if not region_mask[nb]:
                boundary[v] = True
                break

    return boundary

def compute_handle_falloff_weights(V_ref, handle_mask, neighbors, region_mask, power=1.6):
    """
    Build radial-like falloff weights inside the selected region.

    - handle vertices get weight 1.0
    - surrounding region vertices get weights decreasing with graph distance
    - outside region gets 0

    Returns
    -------
    weights : (n,) float32 in [0,1]
    """
    n = len(V_ref)
    weights = np.zeros(n, dtype=np.float32)

    handle_ids = np.where(handle_mask)[0]
    if len(handle_ids) == 0:
        return weights

    # Use handle centroid in the reference mesh as the center
    center = np.mean(V_ref[handle_ids], axis=0)

    region_ids = np.where(region_mask)[0]
    if len(region_ids) == 0:
        return weights

    d = np.linalg.norm(V_ref[region_ids] - center[None, :], axis=1)

    handle_d = np.linalg.norm(V_ref[handle_ids] - center[None, :], axis=1)
    inner_r = float(max(np.max(handle_d), 1.0))
    outer_r = float(max(np.max(d), inner_r + 1.0))

    denom = max(outer_r - inner_r, 1e-6)

    # 1 at/inside handle zone, then smooth falloff to 0 at outer boundary
    for idx, vid in enumerate(region_ids):
        dv = d[idx]
        if dv <= inner_r:
            w = 1.0
        else:
            t = np.clip((dv - inner_r) / denom, 0.0, 1.0)
            w = (1.0 - t) ** power
        weights[vid] = float(w)

    weights[handle_mask] = 1.0
    return weights

def apply_arap_drag_step_2(
    V_track,
    V_def,
    T,
    active_triangles,
    drag_vertices,
    arap_cache,
    delta_xy,
    region_rings=6,
    n_iters=5,
    falloff_power=1.6,
    ref_track_weight=0.10,
    boundary_track_weight=0.02,
):
    """
    One incremental ARAP drag step in screen coordinates.

    Compromise version:
    - rigidity reference is a blend of current deformed shape and tracked neutral shape
    - boundary is pinned mostly to current deformed shape, only slightly toward tracked shape
    - handle region uses falloff
    """
    nv = len(V_def)
    neighbors = arap_cache["neighbors"]

    active_vertex_mask = vertex_mask_from_active_triangles(T, active_triangles, nv)
    handle_mask = drag_vertices & active_vertex_mask

    if not np.any(handle_mask):
        return V_def.copy()

    region_mask = expand_vertex_region(
        seed_mask=handle_mask,
        neighbors=neighbors,
        rings=region_rings,
        valid_mask=active_vertex_mask,
    )

    boundary_mask = boundary_of_region(region_mask, neighbors)
    boundary_mask &= ~handle_mask

    X_init = V_def.copy()

    # Compromise rigidity reference:
    # mostly preserve current edited shape, but regularize toward live neutral tracked shape
    X_ref = (
        (1.0 - ref_track_weight) * X_init +
        ref_track_weight * V_track
    ).astype(np.float32)

    # Compromise boundary targets:
    # keep outer ring mostly where it already is, with only mild pull toward tracked neutral
    boundary_targets = (
        (1.0 - boundary_track_weight) * X_init +
        boundary_track_weight * V_track
    ).astype(np.float32)

    delta_xy = np.asarray(delta_xy, dtype=np.float32)

    weights = compute_handle_falloff_weights(
        V_ref=X_init,
        handle_mask=handle_mask,
        neighbors=neighbors,
        region_mask=region_mask,
        power=falloff_power,
    )

    handle_targets = X_init.copy()

    # Full motion for actual handle vertices
    handle_targets[handle_mask] += delta_xy

    # Soft pull for nearby interior vertices
    soft_mask = region_mask & (~handle_mask) & (~boundary_mask)
    if np.any(soft_mask):
        handle_targets[soft_mask] += weights[soft_mask, None] * delta_xy

    X_new = arap_deform_2d_2(
        X_ref=X_ref,
        X_init=X_init,
        neighbors=neighbors,
        region_mask=region_mask,
        handle_mask=handle_mask,
        handle_targets=handle_targets,
        boundary_mask=boundary_mask,
        boundary_targets=boundary_targets,
        n_iters=n_iters,
    )

    return X_new

def arap_deform_2d_2(
    X_ref,
    X_init,
    neighbors,
    region_mask,
    handle_mask,
    handle_targets,
    boundary_mask=None,
    boundary_targets=None,
    n_iters=3,
):
    """
    2D ARAP on a local region.

    X_ref:
        reference geometry used for rigidity preservation
    X_init:
        current deformed geometry used as initialization
    boundary_targets:
        explicit positions for the boundary pins
    """
    n = len(X_ref)
    X = X_init.copy()

    if boundary_mask is None:
        boundary_mask = np.zeros(n, dtype=bool)

    if boundary_targets is None:
        boundary_targets = X_init.copy()

    constrained = (handle_mask | boundary_mask) & region_mask
    free = region_mask & (~constrained)

    region_ids = np.where(region_mask)[0]
    free_ids = np.where(free)[0]

    if len(region_ids) == 0:
        return X

    X[handle_mask] = handle_targets[handle_mask]
    X[boundary_mask] = boundary_targets[boundary_mask]

    def w_ij(i, j):
        return 1.0

    if len(free_ids) == 0:
        return X

    id_map = -np.ones(n, dtype=np.int32)
    id_map[free_ids] = np.arange(len(free_ids))

    rows, cols, vals = [], [], []

    for i in free_ids:
        diag = 0.0
        for j in neighbors[i]:
            if not region_mask[j]:
                continue
            wij = w_ij(i, j)
            diag += wij
            if free[j]:
                rows.append(id_map[i])
                cols.append(id_map[j])
                vals.append(-wij)

        rows.append(id_map[i])
        cols.append(id_map[i])
        vals.append(diag)

    Lff = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(free_ids), len(free_ids))
    )

    for _ in range(n_iters):
        # ----- local step -----
        R = np.tile(np.eye(2, dtype=np.float32)[None, :, :], (n, 1, 1))

        for i in region_ids:
            S = np.zeros((2, 2), dtype=np.float64)
            pi = X_ref[i]
            xi = X[i]

            for j in neighbors[i]:
                if not region_mask[j]:
                    continue
                wij = w_ij(i, j)
                pj = X_ref[j]
                xj = X[j]

                p_ij = (pi - pj).reshape(2, 1)
                x_ij = (xi - xj).reshape(2, 1)
                S += wij * (x_ij @ p_ij.T)

            U, _, Vt = np.linalg.svd(S)
            Ri = U @ Vt
            if np.linalg.det(Ri) < 0:
                U[:, -1] *= -1
                Ri = U @ Vt

            R[i] = Ri.astype(np.float32)

        # ----- global step -----
        bx = np.zeros(len(free_ids), dtype=np.float64)
        by = np.zeros(len(free_ids), dtype=np.float64)

        for i in free_ids:
            rhs = np.zeros(2, dtype=np.float64)

            for j in neighbors[i]:
                if not region_mask[j]:
                    continue

                wij = w_ij(i, j)
                p_ij = (X_ref[i] - X_ref[j]).astype(np.float64)
                rhs += 0.5 * wij * ((R[i] + R[j]) @ p_ij)

                if constrained[j]:
                    rhs += wij * X[j]

            bx[id_map[i]] = rhs[0]
            by[id_map[i]] = rhs[1]

        x_sol = spsolve(Lff, bx)
        y_sol = spsolve(Lff, by)

        X[free_ids, 0] = x_sol
        X[free_ids, 1] = y_sol

        X[handle_mask] = handle_targets[handle_mask]
        X[boundary_mask] = boundary_targets[boundary_mask]

    return X

def clamp_vector_magnitudes(vectors, max_mag):
    """
    Clamp row-wise 2D vector magnitudes to max_mag.
    vectors: (N,2)
    """
    out = vectors.copy().astype(np.float32)
    mags = np.linalg.norm(out, axis=1, keepdims=True)
    safe = np.maximum(mags, 1e-6)
    scale = np.minimum(1.0, float(max_mag) / safe)
    out *= scale
    return out

def limit_edge_stretch_step(X_new, X_ref, neighbors, region_mask, max_stretch_ratio=1.18, n_passes=2):
    """
    Limit how much edges are allowed to stretch in THIS step relative to X_ref.

    max_stretch_ratio = 1.18 means:
        an edge in X_new may be at most 18% longer than it was in X_ref.

    This does not pull toward the original mesh; it only limits local per-step distortion.
    """
    X = X_new.copy().astype(np.float32)
    region_ids = np.where(region_mask)[0]

    for _ in range(max(1, int(n_passes))):
        for i in region_ids:
            for j in neighbors[i]:
                if j <= i or not region_mask[j]:
                    continue

                ref_vec = X_ref[j] - X_ref[i]
                ref_len = float(np.linalg.norm(ref_vec))
                if ref_len < 1e-6:
                    continue

                cur_vec = X[j] - X[i]
                cur_len = float(np.linalg.norm(cur_vec))
                if cur_len < 1e-6:
                    continue

                max_len = max_stretch_ratio * ref_len
                if cur_len > max_len:
                    mid = 0.5 * (X[i] + X[j])
                    dir_ij = cur_vec / cur_len
                    half = 0.5 * max_len * dir_ij
                    X[i] = mid - half
                    X[j] = mid + half

    return X

def apply_arap_drag_step(
    V_track,
    V_def,
    T,
    active_triangles,
    drag_vertices,
    arap_cache,
    delta_xy,
    region_rings=6,
    n_iters=5,
    falloff_power=1.6,
    handle_strength=0.55,
    max_handle_step_px=18.0,
    max_soft_step_px=10.0,
    max_stretch_ratio=1.18,
):
    """
    One incremental ARAP drag step in screen coordinates.

    More restrictive version:
    - current mesh remains the ARAP reference
    - handle targets are softened
    - per-step displacement is capped
    - post-solve edge stretch is limited

    This preserves cumulative edits while making each step less wild.
    """
    nv = len(V_def)
    neighbors = arap_cache["neighbors"]

    active_vertex_mask = vertex_mask_from_active_triangles(T, active_triangles, nv)
    handle_mask = drag_vertices & active_vertex_mask

    if not np.any(handle_mask):
        return V_def.copy()

    region_mask = expand_vertex_region(
        seed_mask=handle_mask,
        neighbors=neighbors,
        rings=region_rings,
        valid_mask=active_vertex_mask,
    )

    boundary_mask = boundary_of_region(region_mask, neighbors)
    boundary_mask &= ~handle_mask

    # Keep cumulative-edit behavior
    X_ref = V_def.copy()
    X_init = V_def.copy()
    boundary_targets = X_init.copy()

    delta_xy = np.asarray(delta_xy, dtype=np.float32).reshape(1, 2)

    weights = compute_handle_falloff_weights(
        V_ref=X_init,
        handle_mask=handle_mask,
        neighbors=neighbors,
        region_mask=region_mask,
        power=falloff_power,
    )

    handle_targets = X_init.copy()

    # ---------- hard handle region: softened + capped ----------
    if np.any(handle_mask):
        handle_delta = np.repeat(delta_xy, int(np.sum(handle_mask)), axis=0)
        handle_delta *= float(handle_strength)
        handle_delta = clamp_vector_magnitudes(handle_delta, max_handle_step_px)
        handle_targets[handle_mask] += handle_delta

    # ---------- surrounding region: falloff + softer cap ----------
    soft_mask = region_mask & (~handle_mask) & (~boundary_mask)
    if np.any(soft_mask):
        soft_delta = weights[soft_mask, None] * delta_xy
        soft_delta = clamp_vector_magnitudes(soft_delta, max_soft_step_px)
        handle_targets[soft_mask] += soft_delta

    X_new = arap_deform_2d(
        X_ref=X_ref,
        X_init=X_init,
        neighbors=neighbors,
        region_mask=region_mask,
        handle_mask=handle_mask,
        handle_targets=handle_targets,
        boundary_mask=boundary_mask,
        boundary_targets=boundary_targets,
        n_iters=n_iters,
    )

    # ---------- post-solve restriction ----------
    X_new = limit_edge_stretch_step(
        X_new=X_new,
        X_ref=X_ref,
        neighbors=neighbors,
        region_mask=region_mask,
        max_stretch_ratio=max_stretch_ratio,
        n_passes=2,
    )

    # Keep actual boundary pinned
    X_new[boundary_mask] = boundary_targets[boundary_mask]

    return X_new

def arap_deform_2d(
    X_ref,
    X_init,
    neighbors,
    region_mask,
    handle_mask,
    handle_targets,
    boundary_mask=None,
    boundary_targets=None,
    n_iters=3,
):
    """
    2D ARAP on a local region.

    Here:
    - X_ref should be the current mesh shape used as the rigidity reference
    - X_init should be the current mesh shape used as the initialization
    - boundary_targets should usually also come from the current mesh shape
    """
    n = len(X_ref)
    X = X_init.copy()

    if boundary_mask is None:
        boundary_mask = np.zeros(n, dtype=bool)

    if boundary_targets is None:
        boundary_targets = X_init.copy()

    constrained = (handle_mask | boundary_mask) & region_mask
    free = region_mask & (~constrained)

    region_ids = np.where(region_mask)[0]
    free_ids = np.where(free)[0]

    if len(region_ids) == 0:
        return X

    X[handle_mask] = handle_targets[handle_mask]
    X[boundary_mask] = boundary_targets[boundary_mask]

    def w_ij(i, j):
        edge_len = float(np.linalg.norm(X_ref[i] - X_ref[j]))
        edge_len = max(edge_len, 1e-3)

        # Inverse-length weighting, lightly clamped for stability
        w = 1.0 / edge_len

        # Clamp to avoid extreme values
        return float(np.clip(w, 0.02, 0.25))

    if len(free_ids) == 0:
        return X

    id_map = -np.ones(n, dtype=np.int32)
    id_map[free_ids] = np.arange(len(free_ids))

    rows, cols, vals = [], [], []

    for i in free_ids:
        diag = 0.0
        for j in neighbors[i]:
            if not region_mask[j]:
                continue
            wij = w_ij(i, j)
            diag += wij
            if free[j]:
                rows.append(id_map[i])
                cols.append(id_map[j])
                vals.append(-wij)

        rows.append(id_map[i])
        cols.append(id_map[i])
        vals.append(diag)

    Lff = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(free_ids), len(free_ids))
    )

    for _ in range(n_iters):
        R = np.tile(np.eye(2, dtype=np.float32)[None, :, :], (n, 1, 1))

        for i in region_ids:
            S = np.zeros((2, 2), dtype=np.float64)
            pi = X_ref[i]
            xi = X[i]

            for j in neighbors[i]:
                if not region_mask[j]:
                    continue
                wij = w_ij(i, j)
                pj = X_ref[j]
                xj = X[j]

                p_ij = (pi - pj).reshape(2, 1)
                x_ij = (xi - xj).reshape(2, 1)
                S += wij * (x_ij @ p_ij.T)

            U, _, Vt = np.linalg.svd(S)
            Ri = U @ Vt
            if np.linalg.det(Ri) < 0:
                U[:, -1] *= -1
                Ri = U @ Vt

            R[i] = Ri.astype(np.float32)

        bx = np.zeros(len(free_ids), dtype=np.float64)
        by = np.zeros(len(free_ids), dtype=np.float64)

        for i in free_ids:
            rhs = np.zeros(2, dtype=np.float64)

            for j in neighbors[i]:
                if not region_mask[j]:
                    continue

                wij = w_ij(i, j)
                p_ij = (X_ref[i] - X_ref[j]).astype(np.float64)
                rhs += 0.5 * wij * ((R[i] + R[j]) @ p_ij)

                if constrained[j]:
                    rhs += wij * X[j]

            bx[id_map[i]] = rhs[0]
            by[id_map[i]] = rhs[1]

        x_sol = spsolve(Lff, bx)
        y_sol = spsolve(Lff, by)

        X[free_ids, 0] = x_sol
        X[free_ids, 1] = y_sol

        X[handle_mask] = handle_targets[handle_mask]
        X[boundary_mask] = boundary_targets[boundary_mask]

    return X


# ══════════════════════════════════════════════════════════════════════════════
# Migrated from run_installation.py (render-group/mesh-geometry tables + mesh helpers)
# ══════════════════════════════════════════════════════════════════════════════

SEGMENT_NAMES = [
    "torso",
    "head",
    "left_upper_arm",
    "left_lower_arm",
    "left_palm",
    "right_upper_arm",
    "right_lower_arm",
    "right_palm",
    "left_thigh",
    "left_calf",
    "right_thigh",
    "right_calf",
]
SEGMENT_INDEX = {name: i for i, name in enumerate(SEGMENT_NAMES)}
RENDER_GROUP_NAMES = [
    "torso",
    "head",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
]
RENDER_GROUP_INDEX = {name: i for i, name in enumerate(RENDER_GROUP_NAMES)}
RENDER_GROUP_COLORS = {
    "torso": (255, 220, 0),
    "head": (255, 0, 255),
    "left_arm": (0, 255, 0),
    "right_arm": (0, 180, 255),
    "left_leg": (255, 0, 0),
    "right_leg": (0, 0, 255),
}
FIXED_RENDER_ORDER = [
    "left_leg",
    "right_leg",
    "torso",
    "head",
    "left_arm",
    "right_arm",
]
USE_SOFT_LAYER_MASKS = False
USE_FLOAT_ALPHA_COMPOSITING = False
YAW_ENTER_THRESHOLD = 0.22
YAW_EXIT_THRESHOLD = 0.12
YAW_SIGN_SMOOTH_ALPHA = 0.30
YAW_SIGN_DEADBAND = 0.08

# Pre-existing bug found while moving this code: these 4 were only ever
# assigned as *local* variables inside run_installation.py's old
# run_hand_brush_drag_arap_loop_skeleton (now helpers/interaction_loop.py),
# never at module level, even though compute_render_order_with_arm_override
# and compute_render_order_with_leg_override below reference them as bare
# globals — so calling either function would have raised a NameError before
# this move too, whenever SHOW_LAYERED_RENDER enabled that code path. Adding
# them here (with the exact values from that dead local assignment) so the
# functions are actually callable; flagged for Alissa to double-check.
ARM_OVERLAP_TRIGGER_FRAC = 0.020
ARM_FRONT_SCORE_DEADBAND = 6.0
LEG_OVERLAP_TRIGGER_FRAC = 0.015
LEG_FRONT_SCORE_DEADBAND = 8.0

FRONTAL_RENDER_ORDER = [
    "left_leg", "right_leg", "torso", "head", "left_arm", "right_arm",
]
LEFT_SIDE_FRONT_RENDER_ORDER = [
    "right_leg", "left_leg", "torso", "head", "right_arm", "left_arm",
]
RIGHT_SIDE_FRONT_RENDER_ORDER = [
    "left_leg", "right_leg", "torso", "head", "left_arm", "right_arm",
]

def build_adaptive_body_mesh(
    w, h,
    body_mask,
    interior_step=60,
    contour_step=8,
    contour_inset=3,
    min_contour_pts=80,
):
    H, W = h, w
    mask_u8 = (body_mask >= 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contour_pts = []
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        pts_raw = main_contour[:, 0, :].astype(np.float32)

        n_contour = len(pts_raw)
        step_px = max(1, min(contour_step, n_contour // max(min_contour_pts, 1)))
        sampled = pts_raw[::step_px]

        if contour_inset > 0 and len(sampled) >= 3:
            cx = float(np.mean(sampled[:, 0]))
            cy = float(np.mean(sampled[:, 1]))
            d = sampled - np.array([cx, cy], dtype=np.float32)
            norms = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
            sampled = sampled - contour_inset * (d / norms)
            sampled[:, 0] = np.clip(sampled[:, 0], 0, W - 1)
            sampled[:, 1] = np.clip(sampled[:, 1], 0, H - 1)

        contour_pts = sampled.tolist()

    interior_pts = []
    for y in range(0, H, interior_step):
        for x in range(0, W, interior_step):
            if mask_u8[y, x] > 0:
                interior_pts.append([float(x), float(y)])

    frame_pts = []
    for x in range(0, W + 1, interior_step):
        xc = float(min(x, W - 1))
        frame_pts.append([xc, 0.0])
        frame_pts.append([xc, float(H - 1)])
    for y in range(interior_step, H, interior_step):
        frame_pts.append([0.0, float(y)])
        frame_pts.append([float(W - 1), float(y)])
    frame_pts += [[0.0, 0.0], [float(W-1), 0.0],
                  [0.0, float(H-1)], [float(W-1), float(H-1)]]

    all_pts = np.array(contour_pts + interior_pts + frame_pts, dtype=np.float32)

    if len(all_pts) < 3:
        print("build_adaptive_body_mesh: not enough points, falling back to grid")
        V_fb, T_fb, _, _ = build_grid_mesh(w, h, step=interior_step)
        active_fb = np.ones(len(T_fb), dtype=bool)
        return V_fb.astype(np.float32), T_fb, active_fb

    rounded = np.round(all_pts / 2.0).astype(np.int32)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    all_pts = all_pts[unique_idx]

    if len(all_pts) < 3:
        V_fb, T_fb, _, _ = build_grid_mesh(w, h, step=interior_step)
        active_fb = np.ones(len(T_fb), dtype=bool)
        return V_fb.astype(np.float32), T_fb, active_fb

    tri = Delaunay(all_pts)
    T   = tri.simplices.astype(np.int32)
    V   = all_pts.astype(np.float32)

    centroids = (V[T[:, 0]] + V[T[:, 1]] + V[T[:, 2]]) / 3.0
    cx = np.clip(np.round(centroids[:, 0]).astype(np.int32), 0, W - 1)
    cy = np.clip(np.round(centroids[:, 1]).astype(np.int32), 0, H - 1)
    active = mask_u8[cy, cx] > 0

    print(f"build_adaptive_body_mesh: {len(V)} vertices, {len(T)} triangles, "
          f"{int(active.sum())} active ({len(contour_pts)} contour pts, "
          f"{len(interior_pts)} interior pts)")

    return V, T, active

def build_frame_arrays(frames, seg_ids, num_vertices):
    origins = np.zeros((num_vertices, 2), dtype=np.float32)
    xhats   = np.zeros((num_vertices, 2), dtype=np.float32)
    yhats   = np.zeros((num_vertices, 2), dtype=np.float32)
    lengths = np.zeros(num_vertices,      dtype=np.float32)
    for i in range(num_vertices):
        si = seg_ids[i]
        if si < 0:
            continue
        f = frames[SEGMENT_NAMES[si]]
        origins[i] = f["origin"]
        xhats[i]   = f["xhat"]
        yhats[i]   = f["yhat"]
        lengths[i] = f["length"]
    return origins, xhats, yhats, lengths

def reconstruct_tracked_mesh_vectorized(binding, frame_arrays):
    origins, xhats, yhats, lengths = frame_arrays
    rest_uv = binding["vertex_local_uv_rest"]
    seg_ids = binding["vertex_segment"]
    u = rest_uv[:, 0:1]
    v = rest_uv[:, 1:2]
    V = origins + u * lengths[:, None] * xhats \
                + v * lengths[:, None] * yhats
    V[seg_ids < 0] = 0.0
    return V.astype(np.float32)

def reconstruct_deformed_mesh_vectorized(binding, frame_arrays):
    origins, xhats, yhats, lengths = frame_arrays
    rest_uv = binding["vertex_local_uv_rest"]
    off_uv  = binding["vertex_local_uv_offset"]
    seg_ids = binding["vertex_segment"]
    uv = rest_uv + off_uv
    u  = uv[:, 0:1]
    v  = uv[:, 1:2]
    V = origins + u * lengths[:, None] * xhats \
                + v * lengths[:, None] * yhats
    V[seg_ids < 0] = 0.0
    return V.astype(np.float32)

def update_local_offsets_vectorized(binding, frame_arrays, V_new):
    origins, xhats, yhats, lengths = frame_arrays
    seg_ids = binding["vertex_segment"]
    rest_uv = binding["vertex_local_uv_rest"]
    valid   = seg_ids >= 0
    safe_len = np.where(lengths > 1e-6, lengths, 1.0)
    d = V_new - origins
    u = np.sum(d * xhats, axis=1) / safe_len
    v = np.sum(d * yhats, axis=1) / safe_len
    uv_now = np.stack([u, v], axis=1)
    binding["vertex_local_uv_offset"][valid] = (uv_now - rest_uv)[valid]

def frames_moved_enough(prev_frames, new_frames, threshold_px=1.5):
    if prev_frames is None:
        return True
    for seg in SEGMENT_NAMES:
        if seg not in prev_frames or seg not in new_frames:
            return True
        if np.linalg.norm(new_frames[seg]["origin"] - prev_frames[seg]["origin"]) > threshold_px:
            return True
    return False

def build_active_mesh_mask_fast(h, w, V_def, T, active):
    return rasterize_triangle_mask_from_indices(h, w, V_def, T, np.flatnonzero(active).astype(np.int32))

def composite_mesh_with_background_holefill(frame_raw, bg_plate, warped_mesh_bgr,
                                             mesh_mask_u8, seg_mask, body_thresh=0.50,
                                             dilate_ksize=7, blur_ksize=5, apply_mesh=True):
    out = frame_raw.copy()
    body_mask = (seg_mask >= body_thresh).astype(np.uint8) * 255
    if dilate_ksize is not None and dilate_ksize > 1:
        k = int(dilate_ksize); k = k if k % 2 == 1 else k + 1
        body_mask = cv2.dilate(body_mask, np.ones((k, k), np.uint8), iterations=1)
    hole_mask = cv2.bitwise_and(body_mask, cv2.bitwise_not(mesh_mask_u8))
    if blur_ksize is not None and blur_ksize > 1:
        k = int(blur_ksize); k = k if k % 2 == 1 else k + 1
        hole_mask = cv2.GaussianBlur(hole_mask, (k, k), 0)
    if hole_mask.ndim == 2:
        out = alpha_composite_bgr(out, bg_plate, hole_mask.astype(np.float32) / 255.0)
    if apply_mesh:
        out = composite_bgr_hard_mask(out, warped_mesh_bgr, mesh_mask_u8)
    return out, hole_mask

def override_arm_order(base_order, left_arm_front=None, right_arm_front=None):
    leg_block = [x for x in base_order if x in ("left_leg", "right_leg")]
    arm_block  = [x for x in base_order if x in ("left_arm", "right_arm")]
    back_arms, front_arms = [], []
    for arm_name in arm_block:
        flag = left_arm_front if arm_name == "left_arm" else right_arm_front
        (back_arms if flag is False else front_arms).append(arm_name)
    return leg_block + back_arms + ["torso", "head"] + front_arms

def compute_render_order_with_arm_override(base_order, layers, stable_pts, yaw_state_txt):
    if layers is None:
        return base_order, 0.0, 0.0, 0.0, 0.0, False
    torso_mask     = layers["torso"]["mask_u8"]
    left_arm_mask  = layers["left_arm"]["mask_u8"]
    right_arm_mask = layers["right_arm"]["mask_u8"]
    left_ov  = compute_mask_overlap_fraction(left_arm_mask, torso_mask)
    right_ov = compute_mask_overlap_fraction(right_arm_mask, torso_mask)
    ls, rs, lf, rf, used = 0.0, 0.0, None, None, False
    if left_ov >= ARM_OVERLAP_TRIGGER_FRAC:
        ls = PTh.compute_arm_front_score(stable_pts, "left_arm", yaw_state_txt); used = True
        if ls > ARM_FRONT_SCORE_DEADBAND: lf = True
        elif ls < -ARM_FRONT_SCORE_DEADBAND: lf = False
    if right_ov >= ARM_OVERLAP_TRIGGER_FRAC:
        rs = PTh.compute_arm_front_score(stable_pts, "right_arm", yaw_state_txt); used = True
        if rs > ARM_FRONT_SCORE_DEADBAND: rf = True
        elif rs < -ARM_FRONT_SCORE_DEADBAND: rf = False
    return override_arm_order(base_order, lf, rf), left_ov, right_ov, ls, rs, used

def composite_prebuilt_layers(base_bgr, layers, render_order):
    out = base_bgr.copy()
    for group_name in render_order:
        layer = layers[group_name]
        if USE_FLOAT_ALPHA_COMPOSITING:
            out = alpha_composite_bgr(out, layer["color"],
                                      layer["mask_u8"].astype(np.float32) / 255.0)
        else:
            out = composite_bgr_hard_mask(out, layer["color"], layer["mask_u8"])
    return out

def compute_mask_overlap_fraction(mask_a_u8, mask_b_u8):
    a = mask_a_u8 > 0; b = mask_b_u8 > 0
    area_a = int(np.count_nonzero(a)); area_b = int(np.count_nonzero(b))
    if area_a == 0 or area_b == 0:
        return 0.0
    return int(np.count_nonzero(a & b)) / float(max(1, min(area_a, area_b)))

def override_leg_order(base_order, left_leg_front):
    order = [x for x in base_order if x not in ("left_leg", "right_leg")]
    return (["right_leg", "left_leg"] if left_leg_front else ["left_leg", "right_leg"]) + order

def compute_render_order_with_leg_override(base_order, layers, stable_pts, yaw_state_txt):
    if layers is None:
        return base_order, 0.0, 0.0, False
    ov = compute_mask_overlap_fraction(layers["left_leg"]["mask_u8"], layers["right_leg"]["mask_u8"])
    if ov < LEG_OVERLAP_TRIGGER_FRAC:
        return base_order, ov, 0.0, False
    score = PTh.compute_leg_front_score(stable_pts, yaw_state_txt)
    if score > LEG_FRONT_SCORE_DEADBAND:
        return override_leg_order(base_order, True),  ov, score, True
    elif score < -LEG_FRONT_SCORE_DEADBAND:
        return override_leg_order(base_order, False), ov, score, True
    return base_order, ov, score, True

def compute_render_order_from_yaw(cur_pts, ref_metrics, interaction_state):
    yaw_amount, _, yaw_debug = PTh.estimate_body_yaw(cur_pts, ref_metrics)
    prev_s = float(interaction_state.get("yaw_sign_value_smooth", 0.0))
    sign_value_smooth = (1.0 - YAW_SIGN_SMOOTH_ALPHA) * prev_s + \
                         YAW_SIGN_SMOOTH_ALPHA * float(yaw_debug["sign_value"])
    interaction_state["yaw_sign_value_smooth"] = float(sign_value_smooth)
    if sign_value_smooth > YAW_SIGN_DEADBAND: yaw_sign = 1.0
    elif sign_value_smooth < -YAW_SIGN_DEADBAND: yaw_sign = -1.0
    else: yaw_sign = 0.0
    prev_state = interaction_state.get("yaw_side_state", "frontal")
    state = prev_state
    if prev_state == "frontal":
        if yaw_amount >= YAW_ENTER_THRESHOLD:
            state = "left_front" if yaw_sign > 0 else ("right_front" if yaw_sign < 0 else "frontal")
    elif prev_state == "left_front":
        if yaw_amount <= YAW_EXIT_THRESHOLD: state = "frontal"
        elif yaw_sign < 0 and yaw_amount >= YAW_ENTER_THRESHOLD: state = "right_front"
    elif prev_state == "right_front":
        if yaw_amount <= YAW_EXIT_THRESHOLD: state = "frontal"
        elif yaw_sign > 0 and yaw_amount >= YAW_ENTER_THRESHOLD: state = "left_front"
    interaction_state["yaw_side_state"] = state
    if state == "left_front":    render_order = LEFT_SIDE_FRONT_RENDER_ORDER;  yaw_state_txt = "left side front"
    elif state == "right_front": render_order = RIGHT_SIDE_FRONT_RENDER_ORDER; yaw_state_txt = "right side front"
    else:                        render_order = FRONTAL_RENDER_ORDER;           yaw_state_txt = "frontal"
    yaw_debug["sign_value_smooth"] = float(sign_value_smooth)
    return render_order, yaw_amount, yaw_sign, yaw_state_txt, yaw_debug

def build_render_group_triangle_index_cache(binding):
    tri_active = binding["tri_active"]; tri_render_group = binding["tri_render_group"]
    return {name: np.flatnonzero(tri_active & (tri_render_group == RENDER_GROUP_INDEX[name])).astype(np.int32)
            for name in RENDER_GROUP_NAMES}

def rasterize_triangle_mask_from_indices(h, w, V_dst, T, tri_indices):
    mask = np.zeros((h, w), dtype=np.uint8)
    if tri_indices is None or len(tri_indices) == 0:
        return mask
    for k in tri_indices:
        tri = V_dst[T[k]].astype(np.float32)
        if not np.isfinite(tri).all(): continue
        if triangle_area2(tri) < 1.0: continue
        if not _triangle_inside_image(tri, w, h): continue
        cv2.fillConvexPoly(mask, np.round(tri).astype(np.int32), 255, lineType=cv2.LINE_AA)
    return mask

def warp_mesh_piecewise_to_blank_indices(src_img, V_src, V_dst, T, tri_indices,
                                          min_area=1.0, min_bbox=2.0):
    h, w = src_img.shape[:2]
    dst_img = np.zeros_like(src_img)
    if tri_indices is None or len(tri_indices) == 0:
        return dst_img
    for k in tri_indices:
        tri_idx = T[k]
        t_src = V_src[tri_idx].astype(np.float32)
        t_dst = V_dst[tri_idx].astype(np.float32)
        if not np.isfinite(t_src).all() or not np.isfinite(t_dst).all(): continue
        if triangle_area2(t_src) < min_area or triangle_area2(t_dst) < min_area: continue
        src_bw, src_bh = _triangle_bbox_size(t_src)
        dst_bw, dst_bh = _triangle_bbox_size(t_dst)
        if src_bw < min_bbox or src_bh < min_bbox or dst_bw < min_bbox or dst_bh < min_bbox: continue
        if not _triangle_inside_image(t_src, w, h): continue
        try:
            warp_triangle(src_img, dst_img, t_src, t_dst)
        except cv2.error:
            continue
    return dst_img

def render_group_layer_fast(src_img, V_src, V_dst, T, tri_indices,
                             min_area=4.0, min_bbox=3.0,
                             mask_dilate_ksize=0, mask_blur_ksize=0):
    h, w = src_img.shape[:2]
    if tri_indices is None or len(tri_indices) == 0:
        return np.zeros_like(src_img), np.zeros((h, w), dtype=np.uint8)
    layer_color = warp_mesh_piecewise_to_blank_indices(src_img, V_src, V_dst, T, tri_indices, min_area, min_bbox)
    layer_mask_u8 = rasterize_triangle_mask_from_indices(h, w, V_dst, T, tri_indices)
    if USE_SOFT_LAYER_MASKS:
        layer_mask_u8 = soften_layer_mask(layer_mask_u8, mask_dilate_ksize, mask_blur_ksize)
    return layer_color, layer_mask_u8

def composite_bgr_hard_mask(base_bgr, over_bgr, mask_u8):
    out = base_bgr.copy(); out[mask_u8 > 0] = over_bgr[mask_u8 > 0]; return out

def build_layered_body_layers_fast(frame_raw, V_track, V_def, T, binding,
                                    min_area=4.0, min_bbox=3.0,
                                    mask_dilate_ksize=0, mask_blur_ksize=0):
    layers = {}
    group_tri_indices = binding.get("group_tri_indices") or build_render_group_triangle_index_cache(binding)
    binding["group_tri_indices"] = group_tri_indices
    for group_name in RENDER_GROUP_NAMES:
        color, mask_u8 = render_group_layer_fast(frame_raw, V_track, V_def, T,
                                                  group_tri_indices[group_name],
                                                  min_area, min_bbox,
                                                  mask_dilate_ksize, mask_blur_ksize)
        layers[group_name] = {"color": color, "mask_u8": mask_u8}
    return layers

def render_layered_body_fixed_order_fast(frame_raw, V_track, V_def, T, binding,
                                          render_order=None, min_area=4.0, min_bbox=3.0,
                                          mask_dilate_ksize=0, mask_blur_ksize=0):
    if render_order is None: render_order = FIXED_RENDER_ORDER
    layers = build_layered_body_layers_fast(frame_raw, V_track, V_def, T, binding,
                                             min_area, min_bbox, mask_dilate_ksize, mask_blur_ksize)
    return composite_prebuilt_layers(frame_raw, layers, render_order), layers

def get_tri_mask_for_render_group(binding, group_name):
    gid = RENDER_GROUP_INDEX[group_name]
    return binding["tri_active"] & (binding["tri_render_group"] == gid)

def rasterize_triangle_mask(h, w, V_dst, T, tri_mask):
    mask = np.zeros((h, w), dtype=np.uint8)
    for k, tri_idx in enumerate(T):
        if not tri_mask[k]: continue
        tri = V_dst[tri_idx].astype(np.float32)
        if not np.isfinite(tri).all(): continue
        if triangle_area2(tri) < 1.0: continue
        if not _triangle_inside_image(tri, w, h): continue
        cv2.fillConvexPoly(mask, np.round(tri).astype(np.int32), 255, lineType=cv2.LINE_AA)
    return mask

def soften_layer_mask(mask_u8, dilate_ksize=5, blur_ksize=5):
    out = mask_u8.copy()
    if dilate_ksize is not None and dilate_ksize > 1:
        k = int(dilate_ksize); k = k if k % 2 == 1 else k + 1
        out = cv2.dilate(out, np.ones((k, k), np.uint8), iterations=1)
    if blur_ksize is not None and blur_ksize > 1:
        k = int(blur_ksize); k = k if k % 2 == 1 else k + 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out

def warp_mesh_piecewise_to_blank(src_img, V_src, V_dst, T, active_mask=None,
                                  min_area=1.0, min_bbox=2.0):
    h, w = src_img.shape[:2]
    dst_img = np.zeros_like(src_img)
    if active_mask is None: active_mask = np.ones(len(T), dtype=bool)
    for k, tri_idx in enumerate(T):
        if not active_mask[k]: continue
        t_src = V_src[tri_idx].astype(np.float32); t_dst = V_dst[tri_idx].astype(np.float32)
        if not np.isfinite(t_src).all() or not np.isfinite(t_dst).all(): continue
        if triangle_area2(t_src) < min_area or triangle_area2(t_dst) < min_area: continue
        src_bw, src_bh = _triangle_bbox_size(t_src); dst_bw, dst_bh = _triangle_bbox_size(t_dst)
        if src_bw < min_bbox or src_bh < min_bbox or dst_bw < min_bbox or dst_bh < min_bbox: continue
        if not _triangle_inside_image(t_src, w, h): continue
        try: warp_triangle(src_img, dst_img, t_src, t_dst)
        except cv2.error: continue
    return dst_img

def render_group_layer(src_img, V_src, V_dst, T, tri_mask,
                       min_area=4.0, min_bbox=3.0,
                       mask_dilate_ksize=5, mask_blur_ksize=5):
    h, w = src_img.shape[:2]
    if tri_mask is None or not np.any(tri_mask):
        return np.zeros_like(src_img), np.zeros((h, w), dtype=np.float32)
    layer_color = warp_mesh_piecewise_to_blank(src_img, V_src, V_dst, T, tri_mask, min_area, min_bbox)
    layer_mask_u8 = soften_layer_mask(rasterize_triangle_mask(h, w, V_dst, T, tri_mask),
                                       mask_dilate_ksize, mask_blur_ksize)
    return layer_color, layer_mask_u8.astype(np.float32) / 255.0

def alpha_composite_bgr(base_bgr, over_bgr, alpha):
    alpha3 = alpha[:, :, None] if alpha.ndim == 2 else alpha
    return np.clip(alpha3 * over_bgr.astype(np.float32) +
                   (1.0 - alpha3) * base_bgr.astype(np.float32), 0, 255).astype(np.uint8)

def render_layered_body_fixed_order(frame_raw, V_track, V_def, T, binding,
                                     render_order=None, min_area=4.0, min_bbox=3.0,
                                     mask_dilate_ksize=5, mask_blur_ksize=5):
    if render_order is None: render_order = FIXED_RENDER_ORDER
    out = frame_raw.copy(); layers = {}
    for group_name in RENDER_GROUP_NAMES:
        tri_mask = get_tri_mask_for_render_group(binding, group_name)
        color, alpha = render_group_layer(frame_raw, V_track, V_def, T, tri_mask,
                                          min_area, min_bbox, mask_dilate_ksize, mask_blur_ksize)
        layers[group_name] = {"color": color, "alpha": alpha}
    for group_name in render_order:
        out = alpha_composite_bgr(out, layers[group_name]["color"], layers[group_name]["alpha"])
    return out, layers

def fine_segment_to_render_group(seg_idx):
    if seg_idx is None or seg_idx < 0 or seg_idx >= len(SEGMENT_NAMES): return -1
    seg_name = SEGMENT_NAMES[int(seg_idx)]
    if seg_name == "torso": return RENDER_GROUP_INDEX["torso"]
    if seg_name == "head":  return RENDER_GROUP_INDEX["head"]
    if seg_name in ("left_upper_arm",  "left_lower_arm",  "left_palm"):  return RENDER_GROUP_INDEX["left_arm"]
    if seg_name in ("right_upper_arm", "right_lower_arm", "right_palm"): return RENDER_GROUP_INDEX["right_arm"]
    if seg_name in ("left_thigh",  "left_calf"):  return RENDER_GROUP_INDEX["left_leg"]
    if seg_name in ("right_thigh", "right_calf"): return RENDER_GROUP_INDEX["right_leg"]
    return -1

def build_triangle_render_groups(binding, T):
    vertex_segment = binding["vertex_segment"]; tri_active = binding["tri_active"]
    tri_render_group = -np.ones(len(T), dtype=np.int32)
    for k, tri in enumerate(T):
        if not tri_active[k]: continue
        fine_ids = vertex_segment[tri]
        if np.any(fine_ids < 0): continue
        coarse_ids = [fine_segment_to_render_group(int(s)) for s in fine_ids]
        if np.any(np.array(coarse_ids) < 0): continue
        vals, counts = np.unique(np.array(coarse_ids, dtype=np.int32), return_counts=True)
        tri_render_group[k] = int(vals[np.argmax(counts)])
    return tri_render_group

def draw_triangle_render_group_overlay(frame, V, T, tri_active, tri_render_group,
                                        alpha=0.28, line_thickness=1):
    out = frame.copy(); overlay = frame.copy()
    for k, tri_idx in enumerate(T):
        if not tri_active[k]: continue
        group_id = int(tri_render_group[k])
        if group_id < 0 or group_id >= len(RENDER_GROUP_NAMES): continue
        color = RENDER_GROUP_COLORS[RENDER_GROUP_NAMES[group_id]]
        tri = np.round(V[tri_idx]).astype(np.int32)
        cv2.fillConvexPoly(overlay, tri, color, lineType=cv2.LINE_AA)
        cv2.polylines(overlay, [tri.reshape(-1, 1, 2)], True, color, line_thickness, cv2.LINE_AA)
    return cv2.addWeighted(overlay, float(alpha), out, 1.0 - float(alpha), 0.0)

def draw_filled_triangle_highlight(frame, V, T, tri_mask,
                                    fill_color=(0,255,0), fill_alpha=0.16,
                                    edge_color=(0,255,0), edge_thickness=2, edge_alpha=0.95):
    out = frame.copy()
    if tri_mask is None or not np.any(tri_mask): return out
    tri_ids = np.flatnonzero(tri_mask)
    fill_overlay = np.zeros_like(frame)
    for k in tri_ids:
        cv2.fillConvexPoly(fill_overlay, np.round(V[T[k]]).astype(np.int32), fill_color, cv2.LINE_AA)
    out = cv2.addWeighted(fill_overlay, float(fill_alpha), out, 1.0, 0.0)
    edge_overlay = np.zeros_like(frame)
    for k in tri_ids:
        cv2.polylines(edge_overlay, [np.round(V[T[k]]).astype(np.int32).reshape(-1,1,2)],
                      True, edge_color, edge_thickness, cv2.LINE_AA)
    out = cv2.addWeighted(edge_overlay, float(edge_alpha), out, 1.0, 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)

def print_render_group_triangle_stats(binding):
    tri_active = binding["tri_active"]; tri_render_group = binding["tri_render_group"]
    print("\n--- Triangle render-group stats ---")
    print(f"active triangles total: {int(np.sum(tri_active))}/{len(tri_active)}")
    for group_name in RENDER_GROUP_NAMES:
        gid = RENDER_GROUP_INDEX[group_name]
        print(f"{group_name:>10s}: {int(np.sum(tri_active & (tri_render_group == gid)))}")
    print(f"{'unassigned':>10s}: {int(np.sum(tri_active & (tri_render_group < 0)))}")
    print("-----------------------------------\n")

def triangle_area2(tri):
    a, b, c = tri
    return abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))

def _triangle_bbox_size(tri):
    return float(np.max(tri[:,0])-np.min(tri[:,0])), float(np.max(tri[:,1])-np.min(tri[:,1]))

def _triangle_inside_image(tri, w, h, pad=2.0):
    x0=np.min(tri[:,0]); x1=np.max(tri[:,0]); y0=np.min(tri[:,1]); y1=np.max(tri[:,1])
    return not (x1 < -pad or y1 < -pad or x0 > w-1+pad or y0 > h-1+pad)

def warp_mesh_piecewise(src_img, V_src, V_dst, T, active_mask=None, dst_img=None,
                         min_area=1.0, min_bbox=2.0):
    if dst_img is None: dst_img = src_img.copy()
    if active_mask is None: active_mask = np.ones(len(T), dtype=bool)
    h, w = src_img.shape[:2]
    for k, tri_idx in enumerate(T):
        if not active_mask[k]: continue
        t_src = V_src[tri_idx].astype(np.float32); t_dst = V_dst[tri_idx].astype(np.float32)
        if not np.isfinite(t_src).all() or not np.isfinite(t_dst).all(): continue
        if triangle_area2(t_src) < min_area or triangle_area2(t_dst) < min_area: continue
        src_bw, src_bh = _triangle_bbox_size(t_src); dst_bw, dst_bh = _triangle_bbox_size(t_dst)
        if src_bw < min_bbox or src_bh < min_bbox or dst_bw < min_bbox or dst_bh < min_bbox: continue
        if not _triangle_inside_image(t_src, w, h): continue
        try: warp_triangle(src_img, dst_img, t_src, t_dst)
        except cv2.error: continue
    return dst_img

def reconstruct_tracked_mesh_from_skeleton(binding, cur_pts, V_base):
    frames = PTh.build_segment_frames(cur_pts)
    if frames is None: return None, None
    V_track = V_base.copy().astype(np.float32)
    seg_ids = binding["vertex_segment"]; rest_uv = binding["vertex_local_uv_rest"]
    for i in range(len(V_track)):
        si = seg_ids[i]
        if si < 0: continue
        V_track[i] = PTh.world_from_local_in_frame(rest_uv[i], frames[SEGMENT_NAMES[si]])
    return V_track, frames

def reconstruct_deformed_mesh_from_skeleton(binding, cur_pts, V_base):
    frames = PTh.build_segment_frames(cur_pts)
    if frames is None: return None, None
    V_def = V_base.copy().astype(np.float32)
    seg_ids = binding["vertex_segment"]
    rest_uv = binding["vertex_local_uv_rest"]; off_uv = binding["vertex_local_uv_offset"]
    for i in range(len(V_def)):
        si = seg_ids[i]
        if si < 0: continue
        V_def[i] = PTh.world_from_local_in_frame(rest_uv[i] + off_uv[i], frames[SEGMENT_NAMES[si]])
    return V_def, frames

def triangle_centroids(V, T):
    return (V[T[:,0]] + V[T[:,1]] + V[T[:,2]]) / 3.0

def build_vertex_neighbors_from_triangles(num_vertices, T):
    neighbors = [set() for _ in range(num_vertices)]
    for tri in T:
        a, b, c = map(int, tri)
        neighbors[a].update((b,c)); neighbors[b].update((a,c)); neighbors[c].update((a,b))
    return neighbors

def extract_largest_mask_contour(mask, thresh=0.5):
    m = (mask >= thresh).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return None
    contour = max(contours, key=cv2.contourArea)
    return None if (contour is None or len(contour) < 3) else contour[:,0,:].astype(np.float32)

def shape_mesh_boundary_to_mask(V_base, T, init_mask, mask_thresh=0.5, snap_dist_px=24.0, smooth_iters=1):
    if init_mask is None: return V_base.copy().astype(np.float32)
    V_new = V_base.copy().astype(np.float32)
    inside = PTh.sample_mask_at_points(init_mask, V_base, thresh=mask_thresh)
    if not np.any(inside): return V_new
    contour_pts = extract_largest_mask_contour(init_mask, thresh=mask_thresh)
    if contour_pts is None or len(contour_pts) == 0: return V_new
    neighbors = build_vertex_neighbors_from_triangles(len(V_base), T)
    boundary = np.zeros(len(V_base), dtype=bool)
    for i in range(len(V_base)):
        if not inside[i]: continue
        for j in neighbors[i]:
            if not inside[j]: boundary[i] = True; break
    boundary_ids = np.flatnonzero(boundary)
    if len(boundary_ids) == 0: return V_new
    snap_dist2 = float(max(snap_dist_px, 1.0)) ** 2
    for i in boundary_ids:
        d2 = np.sum((contour_pts - V_new[i][None,:])**2, axis=1)
        j = int(np.argmin(d2))
        if d2[j] <= snap_dist2: V_new[i] = contour_pts[j]
    for _ in range(max(0, int(smooth_iters))):
        prev = V_new.copy()
        for i in boundary_ids:
            nbrs = [j for j in neighbors[i] if inside[j]]
            if nbrs: V_new[i] = 0.7*prev[i] + 0.3*np.mean(prev[nbrs], axis=0)
    return V_new.astype(np.float32)

def bind_mesh_to_skeleton(V_base, T, ref_pts, init_seg_mask, mask_thresh=0.5):
    ref_frames = PTh.build_segment_frames(ref_pts)
    if ref_frames is None or init_seg_mask is None: return None
    init_mask = PTh.expand_mask(init_seg_mask, ksize=9)
    vertex_in_mask = PTh.sample_mask_at_points(init_mask, V_base, thresh=mask_thresh)
    vertex_segment = -np.ones(len(V_base), dtype=np.int32)
    vertex_local_uv = np.zeros((len(V_base), 2), dtype=np.float32)

    shoulder_mid = 0.5*(ref_pts["left_shoulder"]+ref_pts["right_shoulder"])
    hip_mid      = 0.5*(ref_pts["left_hip"]+ref_pts["right_hip"])
    shoulder_w   = np.linalg.norm(ref_pts["right_shoulder"]-ref_pts["left_shoulder"])
    shoulder_y   = min(ref_pts["left_shoulder"][1], ref_pts["right_shoulder"][1])
    hip_y        = max(ref_pts["left_hip"][1],      ref_pts["right_hip"][1])
    torso_quad   = PTh.torso_quad_from_pts(ref_pts, expand_x=0.28, expand_y_top=0.22, expand_y_bottom=0.12)

    head_center = (ref_pts["nose"] + np.array([0.0,-0.18*shoulder_w],dtype=np.float32)
                   if ref_pts.get("nose") is not None
                   else shoulder_mid + np.array([0.0,-0.75*shoulder_w],dtype=np.float32))
    head_radius  = max(36.0, 0.70*shoulder_w)
    torso_left   = min(ref_pts["left_shoulder"][0],  ref_pts["left_hip"][0])  - 0.22*shoulder_w
    torso_right  = max(ref_pts["right_shoulder"][0], ref_pts["right_hip"][0]) + 0.22*shoulder_w
    left_shoulder  = ref_pts.get("left_shoulder")
    right_shoulder = ref_pts.get("right_shoulder")
    shoulder_cap_r = max(24.0, 0.30*shoulder_w)

    arm_capsules = []
    for s_key, e_key, w_key, ua, la, pa, ud, ld in [
        ("left_shoulder",  "left_elbow",  "left_wrist",
         "left_upper_arm",  "left_lower_arm",  "left_palm",  0.26, 0.22),
        ("right_shoulder", "right_elbow", "right_wrist",
         "right_upper_arm", "right_lower_arm", "right_palm", 0.26, 0.22),
    ]:
        s=ref_pts.get(s_key); e=ref_pts.get(e_key); w=ref_pts.get(w_key)
        if s is not None and e is not None: arm_capsules.append((ua, s, e, ud))
        if e is not None and w is not None:
            arm_capsules.append((la, e, w, ld))
            lw_dir = w - e; lw_len = np.linalg.norm(lw_dir)
            if lw_len > 1e-6:
                arm_capsules.append((pa, w, w + (lw_dir/lw_len)*0.75*lw_len, 0.42))

    leg_capsules = []
    for h_key, k_key, a_key, th, ca, thrad, carad in [
        ("left_hip",  "left_knee",  "left_ankle",  "left_thigh",  "left_calf",  0.28, 0.24),
        ("right_hip", "right_knee", "right_ankle", "right_thigh", "right_calf", 0.28, 0.24),
    ]:
        h=ref_pts.get(h_key); k=ref_pts.get(k_key); a=ref_pts.get(a_key)
        if h is not None and k is not None: leg_capsules.append((th, h, k, thrad))
        if k is not None and a is not None: leg_capsules.append((ca, k, a, carad))

    def assign_vertex(i, seg):
        vertex_segment[i] = SEGMENT_INDEX[seg]
        vertex_local_uv[i] = PTh.localize_point_in_frame(V_base[i], ref_frames[seg])

    def best_capsule_match(p, capsule_defs, min_radius_px, accept_scale,
                            allowed_sides=None, y_min=None, y_max=None):
        best_seg, best_dist = None, np.inf
        for seg, a, b, radius_scale in capsule_defs:
            seg_len = np.linalg.norm(b - a)
            radius  = max(min_radius_px, radius_scale * seg_len)
            if allowed_sides is not None:
                if "left"  in seg and "left"  not in allowed_sides: continue
                if "right" in seg and "right" not in allowed_sides: continue
            if y_min is not None and p[1] < y_min: continue
            if y_max is not None and p[1] > y_max: continue
            dist, _, _ = PTh.point_segment_distance(p, a, b)
            if dist <= accept_scale * radius and dist < best_dist:
                best_dist, best_seg = dist, seg
        return best_seg

    arm_y_max = hip_y + 0.10 * shoulder_w
    leg_y_min = shoulder_y + 0.35 * shoulder_w

    for i, p in enumerate(V_base):
        if not vertex_in_mask[i]: continue
        assigned = False
        if PTh.point_in_quad(p, torso_quad):
            assign_vertex(i, "torso"); assigned = True
        if not assigned:
            in_l = left_shoulder  is not None and np.linalg.norm(p-left_shoulder)  <= shoulder_cap_r
            in_r = right_shoulder is not None and np.linalg.norm(p-right_shoulder) <= shoulder_cap_r
            if in_l and not in_r:   assign_vertex(i,"left_upper_arm");  assigned=True
            elif in_r and not in_l: assign_vertex(i,"right_upper_arm"); assigned=True
            elif in_l and in_r:
                assign_vertex(i, "left_upper_arm" if np.linalg.norm(p-left_shoulder) <= np.linalg.norm(p-right_shoulder) else "right_upper_arm")
                assigned=True
        if not assigned and np.linalg.norm(p-head_center) <= head_radius:
            assign_vertex(i,"head"); assigned=True
        if not assigned:
            neck_top=shoulder_y-0.24*shoulder_w; neck_bot=shoulder_y+0.22*shoulder_w
            if neck_top<=p[1]<=neck_bot and torso_left<=p[0]<=torso_right:
                assign_vertex(i,"torso"); assigned=True
        if not assigned and leg_capsules:
            seg=best_capsule_match(p, leg_capsules, 12.0, 1.15, y_min=leg_y_min)
            if seg: assign_vertex(i,seg); assigned=True
        if not assigned and arm_capsules:
            palm_caps=[c for c in arm_capsules if c[0].endswith("palm")]
            if palm_caps:
                seg=best_capsule_match(p, palm_caps, 18.0, 1.60, y_max=arm_y_max+0.35*shoulder_w)
                if seg: assign_vertex(i,seg); assigned=True
        if not assigned and arm_capsules:
            non_palm=[c for c in arm_capsules if not c[0].endswith("palm")]
            if non_palm:
                seg=best_capsule_match(p, non_palm, 10.0, 1.25, y_max=arm_y_max+0.15*shoulder_w)
                if seg: assign_vertex(i,seg); assigned=True

    tri_vertices_in_mask = np.all(vertex_in_mask[T], axis=1)
    tri_centroid_in_mask = PTh.sample_mask_at_points(init_mask, triangle_centroids(V_base, T), thresh=mask_thresh)
    tri_assigned = np.all(vertex_segment[T] >= 0, axis=1)
    tri_active   = tri_vertices_in_mask & tri_centroid_in_mask & tri_assigned
    used_vertices = np.zeros(len(V_base), dtype=bool)
    if np.any(tri_active):
        used_vertices[np.unique(T[tri_active].reshape(-1))] = True
    vertex_segment[~used_vertices] = -1
    return {
        "ref_pts": ref_pts, "ref_frames": ref_frames,
        "vertex_segment": vertex_segment,
        "vertex_local_uv_rest": vertex_local_uv.copy(),
        "vertex_local_uv_offset": np.zeros_like(vertex_local_uv),
        "tri_active": tri_active, "vertex_in_mask": vertex_in_mask,
    }

def update_local_offsets_from_world(binding, frames, V_new):
    seg_ids=binding["vertex_segment"]; rest_uv=binding["vertex_local_uv_rest"]; off_uv=binding["vertex_local_uv_offset"]
    for i in range(len(V_new)):
        si=seg_ids[i]
        if si<0: continue
        off_uv[i] = PTh.localize_point_in_frame(V_new[i], frames[SEGMENT_NAMES[si]]) - rest_uv[i]

