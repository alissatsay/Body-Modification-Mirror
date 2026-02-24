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