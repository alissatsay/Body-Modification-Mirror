import cv2
import numpy as np
import mediapipe as mp
import time

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