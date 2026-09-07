import os
import cv2
import numpy as np
import mediapipe as mp
import time

from helpers import pose_tracking as PTh
from helpers import triangle_mesh as TMh
from helpers import display as Dh
from helpers import screens as Sc

def run_hand_brush_drag_loop_pose(
    cap,
    segmenter,
    hands,
    pose,
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

        frame = cv2.flip(frame, 1)

        # segmentation still used internally for body masking / active triangles
        seg_mask, rgb = PTh.get_segmentation_mask(frame, segmenter, feather=feather)

        # pose-based whole-body tracking anchor
        pose_results = pose.process(rgb)
        cur_body_box = PTh.get_body_box_from_pose(pose_results, w, h, visibility_thresh=0.5)

        if cur_body_box is not None:
            if interaction_state["ref_body_box"] is None:
                interaction_state["ref_body_box"] = cur_body_box

            ref_body_box = interaction_state["ref_body_box"]
            V_track, sx, sy = TMh.make_body_tracked_mesh(V_base, ref_body_box, cur_body_box)

            # Rebuild full mesh every frame
            V_def[:, 0] = V_track[:, 0] + offset_local[:, 0] * cur_body_box["width"]
            V_def[:, 1] = V_track[:, 1] + offset_local[:, 1] * cur_body_box["height"]
        else:
            sx, sy = 1.0, 1.0
            V_track = V_def.copy()

        # active triangles still selected from full visible mesh against the body mask
        active = TMh.active_triangles_from_mask(V_def, T, seg_mask, thresh=thresh)

        hand_state = PTh.get_hand_state(rgb, hands, seg_mask, thresh, w, h)

        hand_center = hand_state["center"]
        hand_is_open = hand_state["is_open"]
        hand_is_fist = hand_state["is_fist"]
        hand_over_body = hand_state["over_body"]
        hand_detected = hand_state["detected"]

        new_preview_vertices = np.zeros(len(V_def), dtype=bool)
        new_preview_triangles = np.zeros(len(T), dtype=bool)

        if hand_detected:
            if (not interaction_state["dragging"]) and hand_is_open and hand_over_body:
                new_preview_vertices, new_preview_triangles = TMh.compute_brush_selection(
                    V=V_def,
                    T=T,
                    active=active,
                    center=hand_center,
                    radius=brush_radius,
                )

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

            elif (
                interaction_state["dragging"]
                and hand_is_fist
                and hand_center is not None
                and interaction_state["prev_hand_center"] is not None
                and cur_body_box is not None
            ):
                dx = hand_center[0] - interaction_state["prev_hand_center"][0]
                dy = hand_center[1] - interaction_state["prev_hand_center"][1]

                # store edits in body-normalized coordinates
                offset_local[interaction_state["drag_vertices"], 0] += dx / max(cur_body_box["width"], 1e-6)
                offset_local[interaction_state["drag_vertices"], 1] += dy / max(cur_body_box["height"], 1e-6)

                interaction_state["prev_hand_center"] = hand_center

                # immediate rebuild
                if interaction_state["ref_body_box"] is not None:
                    ref_body_box = interaction_state["ref_body_box"]
                    V_track, sx, sy = TMh.make_body_tracked_mesh(V_base, ref_body_box, cur_body_box)
                    V_def[:, 0] = V_track[:, 0] + offset_local[:, 0] * cur_body_box["width"]
                    V_def[:, 1] = V_track[:, 1] + offset_local[:, 1] * cur_body_box["height"]

            elif interaction_state["dragging"] and (not hand_is_fist):
                interaction_state["dragging"] = False
                interaction_state["drag_vertices"][:] = False
                interaction_state["drag_triangles"][:] = False
                interaction_state["prev_hand_center"] = None

            if not interaction_state["dragging"]:
                interaction_state["preview_vertices"] = new_preview_vertices
                interaction_state["preview_triangles"] = new_preview_triangles

            interaction_state["hand_was_open"] = hand_is_open

        else:
            if interaction_state["dragging"]:
                interaction_state["dragging"] = False
                interaction_state["drag_vertices"][:] = False
                interaction_state["drag_triangles"][:] = False
                interaction_state["prev_hand_center"] = None

            interaction_state["preview_vertices"][:] = False
            interaction_state["preview_triangles"][:] = False
            interaction_state["hand_was_open"] = False

        if interaction_state["dragging"]:
            affected_vertices = interaction_state["drag_vertices"]
            affected_triangles = interaction_state["drag_triangles"] & active
        else:
            affected_vertices = interaction_state["preview_vertices"]
            affected_triangles = interaction_state["preview_triangles"]

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

        if np.any(affected_vertices):
            pts = np.round(V_def[affected_vertices]).astype(np.int32)
            for x, y in pts:
                cv2.circle(vis, (x, y), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)

        if hand_center is not None and not interaction_state["dragging"]:
            cx, cy = hand_center
            brush_color = (0, 255, 0) if (hand_is_open and hand_over_body) else (180, 180, 180)
            cv2.circle(vis, (cx, cy), brush_radius, brush_color, 2, lineType=cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 4, brush_color, -1, lineType=cv2.LINE_AA)

        vis = TMh.draw_body_debug(vis, cur_body_box)

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

        body_status = "tracked" if cur_body_box is not None else "lost"

        cv2.putText(
            vis,
            f"active: {int(active.sum())}/{len(T)}   selected: {int(affected_triangles.sum())}   body: {body_status}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            vis,
            f"step={step}   thresh={thresh:.2f}   feather={feather}",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            vis,
            "q quit | [ ] thresh | - + step | f toggle feather | r reset mesh",
            (12, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Hand brush drag on whole-body mesh", vis)

        if show_mask:
            mask_vis = (seg_mask * 255).astype(np.uint8)
            cv2.imshow("Segmentation mask", mask_vis)

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
        cur_body_box = PTh.body_bbox_from_mask(seg_mask, thresh=0.5)

        if cur_body_box is not None:
            if interaction_state["ref_body_box"] is None:
                interaction_state["ref_body_box"] = cur_body_box

            ref_body_box = interaction_state["ref_body_box"]
            V_track, sx, sy = TMh.make_tracked_mesh(V_base, ref_body_box, cur_body_box)

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
                    V_track, sx, sy = TMh.make_tracked_mesh(
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


def run_hand_brush_drag_arap_loop_rotated(
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
    arap_cache,
    rotate_deg=90,
    rotate_dir="ccw",
    window_name="Hand brush drag on mesh",
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
        cur_body_box = PTh.body_bbox_from_mask(seg_mask, thresh=0.5)

        if cur_body_box is not None:
            if interaction_state["ref_body_box"] is None:
                interaction_state["ref_body_box"] = cur_body_box

            ref_body_box = interaction_state["ref_body_box"]
            V_track, sx, sy = TMh.make_tracked_mesh(V_base, ref_body_box, cur_body_box)

            # reconstruct visible mesh from tracked neutral mesh + stored local offsets
            V_def[:, 0] = V_track[:, 0] + offset_local[:, 0] * sx
            V_def[:, 1] = V_track[:, 1] + offset_local[:, 1] * sy
        else:
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

            # open palm -> fist starts dragging
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

            # while fist is held, do ARAP instead of simple rigid translation of handles only
            elif (
                interaction_state["dragging"]
                and hand_is_fist
                and hand_center is not None
                and interaction_state["prev_hand_center"] is not None
            ):
                dx = hand_center[0] - interaction_state["prev_hand_center"][0]
                dy = hand_center[1] - interaction_state["prev_hand_center"][1]

                interaction_state["prev_hand_center"] = hand_center

                if interaction_state["ref_body_box"] is not None and cur_body_box is not None:
                    V_track, sx, sy = TMh.make_tracked_mesh(
                        V_base,
                        interaction_state["ref_body_box"],
                        cur_body_box
                    )

                    V_new = TMh.apply_arap_drag_step(
                        V_track=V_track,
                        V_def=V_def,
                        T=T,
                        active_triangles=active,
                        drag_vertices=interaction_state["drag_vertices"],
                        delta_xy=np.array([dx, dy], dtype=np.float32),
                        arap_cache=arap_cache,
                        region_rings=4,
                        n_iters=3,
                    )

                    # convert solved visible positions back into body-local offsets
                    offset_local[:, 0] = (V_new[:, 0] - V_track[:, 0]) / max(sx, 1e-6)
                    offset_local[:, 1] = (V_new[:, 1] - V_track[:, 1]) / max(sy, 1e-6)

                    # rebuild current visible mesh immediately after ARAP update
                    V_def[:, 0] = V_track[:, 0] + offset_local[:, 0] * sx
                    V_def[:, 1] = V_track[:, 1] + offset_local[:, 1] * sy

            # release fist -> stop dragging
            elif interaction_state["dragging"] and (not hand_is_fist):
                interaction_state["dragging"] = False
                interaction_state["drag_vertices"][:] = False
                interaction_state["drag_triangles"][:] = False
                interaction_state["prev_hand_center"] = None

            if not interaction_state["dragging"]:
                interaction_state["preview_vertices"] = new_preview_vertices
                interaction_state["preview_triangles"] = new_preview_triangles

            interaction_state["hand_was_open"] = hand_is_open

        else:
            if interaction_state["dragging"]:
                interaction_state["dragging"] = False
                interaction_state["drag_vertices"][:] = False
                interaction_state["drag_triangles"][:] = False
                interaction_state["prev_hand_center"] = None

            interaction_state["preview_vertices"][:] = False
            interaction_state["preview_triangles"][:] = False
            interaction_state["hand_was_open"] = False

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

        if np.any(affected_vertices):
            pts = np.round(V_def[affected_vertices]).astype(np.int32)
            for x, y in pts:
                cv2.circle(vis, (x, y), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)

        if hand_center is not None and not interaction_state["dragging"]:
            cx, cy = hand_center
            brush_color = (0, 255, 0) if (hand_is_open and hand_over_body) else (180, 180, 180)
            cv2.circle(vis, (cx, cy), brush_radius, brush_color, 2, lineType=cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 4, brush_color, -1, lineType=cv2.LINE_AA)

        if cur_body_box is not None:
            cv2.rectangle(
                vis,
                (cur_body_box["x0"], cur_body_box["y0"]),
                (cur_body_box["x1"], cur_body_box["y1"]),
                (255, 180, 0),
                2,
                lineType=cv2.LINE_AA
            )

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

        Dh.show_output_frame(
            window_name=window_name,
            frame_bgr=vis,
            rotate_deg=rotate_deg,
            rotate_dir=rotate_dir,
        )

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

            E = TMh.build_unique_edges(T)
            neighbors = TMh.build_vertex_neighbors(len(V_base), E)
            arap_cache = {"E": E, "neighbors": neighbors}

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

            E = TMh.build_unique_edges(T)
            neighbors = TMh.build_vertex_neighbors(len(V_base), E)
            arap_cache = {"E": E, "neighbors": neighbors}

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


def run_hand_brush_drag_arap_loop(
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
    arap_cache,
    rotate_deg=90,
    rotate_dir="ccw",
    window_name="Hand brush drag on mesh",
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
        cur_body_box = PTh.body_bbox_from_mask(seg_mask, thresh=0.5)

        if cur_body_box is not None:
            if interaction_state["ref_body_box"] is None:
                interaction_state["ref_body_box"] = cur_body_box

            ref_body_box = interaction_state["ref_body_box"]
            V_track, sx, sy = TMh.make_tracked_mesh(V_base, ref_body_box, cur_body_box)

            # reconstruct visible mesh from tracked neutral mesh + stored local offsets
            V_def[:, 0] = V_track[:, 0] + offset_local[:, 0] * sx
            V_def[:, 1] = V_track[:, 1] + offset_local[:, 1] * sy
        else:
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

            # open palm -> fist starts dragging
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

            # while fist is held, do ARAP instead of simple rigid translation of handles only
            elif (
                interaction_state["dragging"]
                and hand_is_fist
                and hand_center is not None
                and interaction_state["prev_hand_center"] is not None
            ):
                dx = hand_center[0] - interaction_state["prev_hand_center"][0]
                dy = hand_center[1] - interaction_state["prev_hand_center"][1]

                interaction_state["prev_hand_center"] = hand_center

                if interaction_state["ref_body_box"] is not None and cur_body_box is not None:
                    V_track, sx, sy = TMh.make_tracked_mesh(
                        V_base,
                        interaction_state["ref_body_box"],
                        cur_body_box
                    )

                    V_new = TMh.apply_arap_drag_step(
                        V_track=V_track,
                        V_def=V_def,
                        T=T,
                        active_triangles=active,
                        drag_vertices=interaction_state["drag_vertices"],
                        delta_xy=np.array([dx, dy], dtype=np.float32),
                        arap_cache=arap_cache,
                        region_rings=4,
                        n_iters=3,
                    )

                    # convert solved visible positions back into body-local offsets
                    offset_local[:, 0] = (V_new[:, 0] - V_track[:, 0]) / max(sx, 1e-6)
                    offset_local[:, 1] = (V_new[:, 1] - V_track[:, 1]) / max(sy, 1e-6)

                    # rebuild current visible mesh immediately after ARAP update
                    V_def[:, 0] = V_track[:, 0] + offset_local[:, 0] * sx
                    V_def[:, 1] = V_track[:, 1] + offset_local[:, 1] * sy

            # release fist -> stop dragging
            elif interaction_state["dragging"] and (not hand_is_fist):
                interaction_state["dragging"] = False
                interaction_state["drag_vertices"][:] = False
                interaction_state["drag_triangles"][:] = False
                interaction_state["prev_hand_center"] = None

            if not interaction_state["dragging"]:
                interaction_state["preview_vertices"] = new_preview_vertices
                interaction_state["preview_triangles"] = new_preview_triangles

            interaction_state["hand_was_open"] = hand_is_open

        else:
            if interaction_state["dragging"]:
                interaction_state["dragging"] = False
                interaction_state["drag_vertices"][:] = False
                interaction_state["drag_triangles"][:] = False
                interaction_state["prev_hand_center"] = None

            interaction_state["preview_vertices"][:] = False
            interaction_state["preview_triangles"][:] = False
            interaction_state["hand_was_open"] = False

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

        if np.any(affected_vertices):
            pts = np.round(V_def[affected_vertices]).astype(np.int32)
            for x, y in pts:
                cv2.circle(vis, (x, y), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)

        if hand_center is not None and not interaction_state["dragging"]:
            cx, cy = hand_center
            brush_color = (0, 255, 0) if (hand_is_open and hand_over_body) else (180, 180, 180)
            cv2.circle(vis, (cx, cy), brush_radius, brush_color, 2, lineType=cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 4, brush_color, -1, lineType=cv2.LINE_AA)

        if cur_body_box is not None:
            cv2.rectangle(
                vis,
                (cur_body_box["x0"], cur_body_box["y0"]),
                (cur_body_box["x1"], cur_body_box["y1"]),
                (255, 180, 0),
                2,
                lineType=cv2.LINE_AA
            )

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

        cv2.imshow(window_name, vis)

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

            E = TMh.build_unique_edges(T)
            neighbors = TMh.build_vertex_neighbors(len(V_base), E)
            arap_cache = {"E": E, "neighbors": neighbors}

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

            E = TMh.build_unique_edges(T)
            neighbors = TMh.build_vertex_neighbors(len(V_base), E)
            arap_cache = {"E": E, "neighbors": neighbors}

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


# ══════════════════════════════════════════════════════════════════════════════
# Migrated from run_installation.py (gesture/interaction config + the two
# skeleton-mesh interaction-loop / selection-filter functions)
# ══════════════════════════════════════════════════════════════════════════════

# Recomputed independently from run_installation.py's copy: this file lives
# one directory deeper, so it needs its own extra ".." to reach installation_mirror/assets/.
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")

# Output canvas size + output rotation. run_installation.py injects the real
# OUTPUT_W/OUTPUT_H values into this module's namespace right after import.
OUTPUT_W = 1080
OUTPUT_H = 1920
ROTATE_DEG = 270
ROTATE_DIR = "ccw"

USE_ANIMATED_BACKGROUND = False
ANIM_BG_SPEED = 0.3
SHOW_RENDER_GROUP_DEBUG = False
RENDER_GROUP_DEBUG_ALPHA = 0.28
SHOW_MESH_OUTLINE = False
SHOW_LAYERED_RENDER = False
LAYER_MASK_DILATE_KSIZE = 0
LAYER_MASK_BLUR_KSIZE = 0
USE_DYNAMIC_YAW_RENDER_ORDER = True
BRUSH_RADIUS_MIN = 25
BRUSH_RADIUS_MAX = 300
_RESTART_MARGIN  = 30    # px from top and right edges
_RESTART_W       = 220   # button width  (px, output coords)
_RESTART_H       = 90    # button height (px, output coords)
_RESTART_CORNER  = 18    # rounded corner radius
_RESTART_OUTLINE = 5     # outline thickness
_RESTART_FRAMES  = 50    # frames to dwell before triggering finish
_MODE_BTN_W = 220
_MODE_BTN_H = 90
_MODE_BTN_MARGIN = 30
_MODE_BTN_GAP = 20
SESSION_DURATION_SECONDS = 180.0   # 3 minutes per session
TIMEOUT_MESSAGE_DURATION = 5.0     # seconds to show the message

def filter_selection_disallow_same_side_arm(selection_vertices, selection_triangles,
                                             T, binding, handedness_label):
    if handedness_label not in ("Left", "Right"):
        return selection_vertices, selection_triangles
    seg_ids = binding["vertex_segment"]
    if handedness_label == "Left":
        blocked_seg_ids = {TMh.SEGMENT_INDEX["left_upper_arm"],
                           TMh.SEGMENT_INDEX["left_lower_arm"], TMh.SEGMENT_INDEX["left_palm"]}
    else:
        blocked_seg_ids = {TMh.SEGMENT_INDEX["right_upper_arm"],
                           TMh.SEGMENT_INDEX["right_lower_arm"], TMh.SEGMENT_INDEX["right_palm"]}
    blocked_vertices = np.isin(seg_ids, list(blocked_seg_ids))
    tri_touches_blocked = np.any(blocked_vertices[T], axis=1)
    filtered_triangles = selection_triangles & (~tri_touches_blocked)
    filtered_vertices = np.zeros_like(selection_vertices)
    if np.any(filtered_triangles):
        filtered_vertices[np.unique(T[filtered_triangles].ravel())] = True
    return filtered_vertices, filtered_triangles

def run_hand_brush_drag_arap_loop_skeleton(
    cap, segmenter, hands, pose, h, w,
    V_base, V_def, T,
    step, thresh, feather, show_mask, brush_radius,
    interaction_state, arap_cache,
    inference_thread=None,
    window_name="Hand brush drag on skeleton mesh",
    _peace_detector=None,
    _pinch_tracker=None,
):
    global SHOW_MESH_OUTLINE
    print("Entered run_hand_brush_drag_arap_loop_skeleton")
    binding = interaction_state["binding"]

    _session_start     = time.time()
    _timeout_fired     = False
    _timeout_msg_start = 0.0
    _timeout_new_bs    = ""

    # ── Load beauty-standard reference image ─────────────────────────────
    _bs_overlay      = None
    _bs_overlay_x    = 0
    _bs_overlay_y    = 0
    _BS_ALPHA        = 0.35
    _BS_FIGURE_FRAC  = 0.82   # figure fills full image height

    if Sc.current_beauty_standard:
        _bs_path = os.path.join(Sc._BS_IMAGE_DIR, f"{Sc.current_beauty_standard}.png")
        _bs_raw  = cv2.imread(_bs_path, cv2.IMREAD_UNCHANGED)
        if _bs_raw is None:
            print(f"WARNING: could not load BS image: {_bs_path}")
        else:
            _cam_h_px  = interaction_state.get("person_height_px", h * 0.75)
            _scale_cam_to_out = OUTPUT_H / h
            _person_out_px    = _cam_h_px * _scale_cam_to_out
            _src_h            = _bs_raw.shape[0]
            _target_img_h     = int(_person_out_px / _BS_FIGURE_FRAC)
            _target_img_w     = int(_bs_raw.shape[1] * _target_img_h / _src_h)
            _bs_scaled        = cv2.resize(_bs_raw,
                                           (_target_img_w, _target_img_h),
                                           interpolation=cv2.INTER_AREA)
            _BS_BOTTOM_MARGIN = 60
            _bs_overlay_x = (OUTPUT_W - _target_img_w) // 2
            _bs_overlay_y = OUTPUT_H - _target_img_h - _BS_BOTTOM_MARGIN
            _bs_overlay   = _bs_scaled
            print(f"BS overlay: {_target_img_w}x{_target_img_h} "
                  f"at ({_bs_overlay_x},{_bs_overlay_y}), "
                  f"person_height={_person_out_px:.0f}px")

    # ── Finish button + auto-return state ───────────────────────────────
    _restart_frames     = 0
    interaction_mode = "drag"

    _DRAG_UI_DURATION = 150
    _drag_ui_timer = _DRAG_UI_DURATION
    _mode_click_cooldown = 0
    _no_detect_frames   = 0
    _NO_DETECT_LIMIT    = 10
    # Finish pinch-click tracking
    _finish_prev_pinch_dist = 1.0
    _finish_pinch_click_cooldown = 0

    _FINISH_PINCH_OPEN_DIST = 0.05     # fingers considered open
    _FINISH_PINCH_CLOSED_DIST = 0.02  # fingers considered clicked/pinched
    _FINISH_PINCH_DROP_MIN = 0.025     # required distance drop
    _FINISH_CLICK_COOLDOWN = 10        # prevents repeated clicks

    LEG_OVERLAP_TRIGGER_FRAC = 0.015
    LEG_FRONT_SCORE_DEADBAND = 8.0
    ARM_OVERLAP_TRIGGER_FRAC = 0.020
    ARM_FRONT_SCORE_DEADBAND = 6.0

    # ── Finish button images (load once) ─────────────────────────────
    finish_btn_normal = cv2.imread(os.path.join(_ASSETS_DIR, "buttons", "finish.png"), cv2.IMREAD_UNCHANGED)
    finish_btn_hover  = cv2.imread(os.path.join(_ASSETS_DIR, "buttons", "finish_hover.png"), cv2.IMREAD_UNCHANGED)
    finish_btn_click  = cv2.imread(os.path.join(_ASSETS_DIR, "buttons", "finish_clicked.png"), cv2.IMREAD_UNCHANGED)

    finish_btn_normal = cv2.resize(finish_btn_normal, (_RESTART_W, _RESTART_H))
    finish_btn_hover  = cv2.resize(finish_btn_hover,  (_RESTART_W, _RESTART_H))
    finish_btn_click  = cv2.resize(finish_btn_click,  (_RESTART_W, _RESTART_H))

    brush_btn_normal = cv2.imread(os.path.join(_ASSETS_DIR, "buttons", "brush.png"), cv2.IMREAD_UNCHANGED)
    brush_btn_hover  = cv2.imread(os.path.join(_ASSETS_DIR, "buttons", "brush_hover.png"), cv2.IMREAD_UNCHANGED)
    brush_btn_click  = cv2.imread(os.path.join(_ASSETS_DIR, "buttons", "brush_clicked.png"), cv2.IMREAD_UNCHANGED)

    drag_btn_normal = cv2.imread(os.path.join(_ASSETS_DIR, "buttons", "drag.png"), cv2.IMREAD_UNCHANGED)
    drag_btn_hover  = cv2.imread(os.path.join(_ASSETS_DIR, "buttons", "drag_hover.png"), cv2.IMREAD_UNCHANGED)
    drag_btn_click  = cv2.imread(os.path.join(_ASSETS_DIR, "buttons", "drag_clicked.png"), cv2.IMREAD_UNCHANGED)

    brush_btn_normal = cv2.resize(brush_btn_normal, (_MODE_BTN_W, _MODE_BTN_H))
    brush_btn_hover  = cv2.resize(brush_btn_hover,  (_MODE_BTN_W, _MODE_BTN_H))
    brush_btn_click  = cv2.resize(brush_btn_click,  (_MODE_BTN_W, _MODE_BTN_H))

    drag_btn_normal = cv2.resize(drag_btn_normal, (_MODE_BTN_W, _MODE_BTN_H))
    drag_btn_hover  = cv2.resize(drag_btn_hover,  (_MODE_BTN_W, _MODE_BTN_H))
    drag_btn_click  = cv2.resize(drag_btn_click,  (_MODE_BTN_W, _MODE_BTN_H))

    pinch_icon = cv2.imread(os.path.join(_ASSETS_DIR, "UI_gestures", "pinch.png"), cv2.IMREAD_UNCHANGED)
    open_palm_icon = cv2.imread(os.path.join(_ASSETS_DIR, "UI_gestures", "open_palm.png"), cv2.IMREAD_UNCHANGED)
    fist_icon = cv2.imread(os.path.join(_ASSETS_DIR, "UI_gestures", "fist.png"), cv2.IMREAD_UNCHANGED)
    fist_drag_icon = cv2.imread(os.path.join(_ASSETS_DIR, "UI_gestures", "fist_drag.png"), cv2.IMREAD_UNCHANGED)

    _PINCH_ICON_H = 42
    if pinch_icon is not None:
        scale = _PINCH_ICON_H / pinch_icon.shape[0]
        pinch_icon = cv2.resize(
            pinch_icon,
            (int(pinch_icon.shape[1] * scale), _PINCH_ICON_H),
            interpolation=cv2.INTER_AREA
        )

    from PIL import ImageFont as _FF2
    _FBOLD2 = [
        'C:/Windows/Fonts/segoeuisb.ttf',
        'C:/Windows/Fonts/segoeuib.ttf',
        'C:/Windows/Fonts/calibrib.ttf',
        'C:/Windows/Fonts/arialbd.ttf',
    ]
    _finish_font = None
    for _fp2 in _FBOLD2:
        if os.path.exists(_fp2):
            try: _finish_font = _FF2.truetype(_fp2, 36); break
            except: pass
    if _finish_font is None: _finish_font = _FF2.load_default()
    _rbx0 = OUTPUT_W - _RESTART_MARGIN - _RESTART_W
    _rbx1 = OUTPUT_W - _RESTART_MARGIN
    _rby0 = _RESTART_MARGIN + 80
    _rby1 = _rby0 + _RESTART_H

    # ── Brush / Drag button positions ────────────────────────────────
    brush_x0 = _MODE_BTN_MARGIN
    brush_y0 = _rby0
    brush_x1 = brush_x0 + _MODE_BTN_W
    brush_y1 = brush_y0 + _MODE_BTN_H

    drag_x0 = brush_x1 + _MODE_BTN_GAP
    drag_y0 = _rby0
    drag_x1 = drag_x0 + _MODE_BTN_W
    drag_y1 = drag_y0 + _MODE_BTN_H

    while True:
        brush_radius = interaction_state.get('brush_radius', brush_radius)

        ok, frame_raw = cap.read()
        if not ok:
            print("Main loop: cap.read() failed, breaking"); break
        frame_raw = Dh.rotate_frame_for_output(frame_raw, deg=ROTATE_DEG, direction=ROTATE_DIR)
        if frame_raw.shape[:2] != (h, w):
            frame_raw = cv2.resize(frame_raw, (w, h), interpolation=cv2.INTER_LINEAR)
        frame_raw = cv2.flip(frame_raw, 1)

        if inference_thread is not None:
            inference_thread.submit_frame(frame_raw)
            result = inference_thread.latest_result()
        else:
            result = None

        if result is not None:
            seg_mask, rgb, cur_pts, hand_state = result
        else:
            seg_mask = np.zeros((h, w), dtype=np.float32)
            rgb      = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
            cur_pts  = None
            hand_state = {"center":None,"is_open":False,"is_fist":False,
                          "over_body":False,"detected":False,"handedness":None}
            
        _now = time.time()
        _elapsed_session = _now - _session_start

        if not _timeout_fired and _elapsed_session >= SESSION_DURATION_SECONDS:
            _timeout_new_bs    = Sc._pick_new_beauty_standard(Sc.current_beauty_standard)
            _timeout_fired     = True
            _timeout_msg_start = _now
            print(f"Session timeout: rotating BS '{Sc.current_beauty_standard}' -> '{_timeout_new_bs}'")

        if _timeout_fired:
            if (_now - _timeout_msg_start) >= TIMEOUT_MESSAGE_DURATION:
                Sc.current_beauty_standard = _timeout_new_bs
                # reload the BS overlay with the new standard
                if Sc.current_beauty_standard:
                    _bs_path = os.path.join(Sc._BS_IMAGE_DIR, f"{Sc.current_beauty_standard}.png")
                    _bs_raw  = cv2.imread(_bs_path, cv2.IMREAD_UNCHANGED)
                    if _bs_raw is not None:
                        _cam_h_px      = interaction_state.get("person_height_px", h * 0.75)
                        _person_out_px = _cam_h_px * (OUTPUT_H / h)
                        _target_img_h  = int(_person_out_px / _BS_FIGURE_FRAC)
                        _target_img_w  = int(_bs_raw.shape[1] * _target_img_h / _bs_raw.shape[0])
                        _bs_overlay    = cv2.resize(_bs_raw, (_target_img_w, _target_img_h), interpolation=cv2.INTER_AREA)
                        _bs_overlay_x  = (OUTPUT_W - _target_img_w) // 2
                        _bs_overlay_y  = OUTPUT_H - _target_img_h - 60
                _session_start = _now
                _timeout_fired = False
                print(f"BS switched to '{Sc.current_beauty_standard}', timer reset.")

        _hand_also_gone = not hand_state.get('detected', False)
        _person_visible = (cur_pts is not None) or (not _hand_also_gone)
        if _person_visible:
            _no_detect_frames = 0
        else:
            _no_detect_frames += 1
            if _no_detect_frames >= _NO_DETECT_LIMIT:
                print(f"No person/hand detected for {_NO_DETECT_LIMIT} frames — returning to welcome screen")
                return "restart"

        V_track = None; frames = None
        stable_pts = PTh.smooth_pose_points(cur_pts, interaction_state, alpha=0.88, max_jump_px=120.0, hold_frames=1)

        if stable_pts is not None:
            frames = PTh.build_segment_frames(stable_pts)
            if frames is not None:
                if TMh.frames_moved_enough(interaction_state["cached_frames"], frames, threshold_px=1.5):
                    frame_arrays = TMh.build_frame_arrays(frames, binding["vertex_segment"], len(V_base))
                    interaction_state["cached_frame_arrays"] = frame_arrays
                    interaction_state["cached_frames"]       = frames
                else:
                    frame_arrays = interaction_state["cached_frame_arrays"]
                V_track  = TMh.reconstruct_tracked_mesh_vectorized(binding, frame_arrays)
                V_def[:] = TMh.reconstruct_deformed_mesh_vectorized(binding, frame_arrays)

        active = binding["tri_active"]

        hand_center         = hand_state["center"]
        hand_is_open        = hand_state["is_open"]
        hand_is_fist        = hand_state["is_fist"]
        hand_over_body      = hand_state["over_body"]
        hand_detected       = hand_state["detected"]
        hand_pinch_dist     = hand_state.get("pinch_dist", 1.0)
        hand_handedness_raw = hand_state.get("handedness")
        if hand_handedness_raw == "Left":     hand_handedness = "Right"
        elif hand_handedness_raw == "Right":  hand_handedness = "Left"
        else:                                  hand_handedness = None

        _restart_tip_out = None
        if hand_detected and hand_center is not None:
            _rx = int(hand_center[0] * OUTPUT_W / w)
            _ry = int(hand_center[1] * OUTPUT_H / h)
            _restart_tip_out = (_rx, _ry)
            if not hasattr(run_hand_brush_drag_arap_loop_skeleton, '_dbg'):
                run_hand_brush_drag_arap_loop_skeleton._dbg = 0
            run_hand_brush_drag_arap_loop_skeleton._dbg += 1
            if run_hand_brush_drag_arap_loop_skeleton._dbg % 90 == 0:
                print(f'[BTN] hand_out={_restart_tip_out} zone=[{_rbx0-200}-{_rbx1+200}, {_rby0-200}-{_rby1+200}]')
            if _rx < 0 or _rx > OUTPUT_W or _ry < 0 or _ry > OUTPUT_H:
                pass

        active_hand  = interaction_state.get("active_hand", None)
        hand_allowed = (
            not hand_detected
            or active_hand is None
            or hand_handedness == active_hand
            or hand_handedness is None
        )

        hand_is_peace = hand_state.get("is_peace", False)
        if hand_detected:
            peace_fired = _peace_detector.update_raw(hand_is_peace)
            if peace_fired:
                cur = interaction_state.get("active_hand", None)
                if cur is None:
                    nxt = hand_handedness
                else:
                    nxt = None
                interaction_state["active_hand"] = nxt
                label = nxt if nxt is not None else "Either hand"
                interaction_state["gesture_feedback_msg"]    = f"Active hand: {label}"
                interaction_state["gesture_feedback_frames"] = 90
                print(f"Peace sign: active hand -> {nxt}")
        else:
            _peace_detector.reset()

        if interaction_mode == "brush" and hand_detected:
            pinch_delta, is_pinching = _pinch_tracker.update_from_dist(hand_pinch_dist)

            if is_pinching:
                old_r = interaction_state.get("brush_radius", brush_radius)
                new_r = int(np.clip(old_r + pinch_delta, BRUSH_RADIUS_MIN, BRUSH_RADIUS_MAX))
                if new_r != old_r:
                    interaction_state["brush_radius"] = new_r
                    brush_radius = new_r

            if _pinch_tracker.is_locked:
                interaction_mode = "drag"
                _drag_ui_timer = _DRAG_UI_DURATION
                _pinch_tracker.reset()
        else:
            _pinch_tracker.reset()

        new_preview_vertices  = np.zeros(len(V_def), dtype=bool)
        new_preview_triangles = np.zeros(len(T),     dtype=bool)

        if hand_detected and hand_allowed:
            if (not interaction_state["dragging"]) and hand_is_open and hand_over_body:
                new_preview_vertices, new_preview_triangles = TMh.compute_brush_selection(
                    V=V_def, T=T, active=active, center=hand_center, radius=brush_radius)
                new_preview_vertices, new_preview_triangles = filter_selection_disallow_same_side_arm(
                    new_preview_vertices, new_preview_triangles, T, binding, hand_handedness)

            if ((not interaction_state["dragging"]) and interaction_state["hand_was_open"]
                    and hand_is_fist and np.any(interaction_state["preview_vertices"])):
                print(">>> DRAG STARTED")
                interaction_state["dragging"]    = True
                interaction_state["drag_vertices"] = interaction_state["preview_vertices"].copy()
                interaction_state["drag_triangles"] = np.any(interaction_state["drag_vertices"][T], axis=1)
                interaction_state["prev_hand_center"] = hand_center

            elif (interaction_state["dragging"] and hand_is_fist
                  and hand_center is not None and interaction_state["prev_hand_center"] is not None):
                dx = hand_center[0] - interaction_state["prev_hand_center"][0]
                dy = hand_center[1] - interaction_state["prev_hand_center"][1]
                interaction_state["prev_hand_center"] = hand_center
                if abs(dx) >= 1 or abs(dy) >= 1:
                    print("dx dy:", dx, dy)
                    V_new = TMh.apply_arap_drag_step_2(
                        V_track=V_track, V_def=V_def, T=T, active_triangles=active,
                        drag_vertices=interaction_state["drag_vertices"],
                        delta_xy=np.array([dx, dy], dtype=np.float32),
                        arap_cache=arap_cache, region_rings=6, n_iters=5, falloff_power=1.6)
                    if frames is not None and interaction_state.get("cached_frame_arrays") is not None:
                        TMh.update_local_offsets_vectorized(binding, interaction_state["cached_frame_arrays"], V_new)
                    V_rebuilt = TMh.reconstruct_deformed_mesh_vectorized(binding, interaction_state["cached_frame_arrays"])
                    if V_rebuilt is not None: V_def[:] = V_rebuilt

            elif interaction_state["dragging"] and not hand_is_fist:
                interaction_state["dragging"] = False
                interaction_state["drag_vertices"][:] = False
                interaction_state["drag_triangles"][:] = False
                interaction_state["prev_hand_center"] = None

            if not interaction_state["dragging"]:
                interaction_state["preview_vertices"]  = new_preview_vertices
                interaction_state["preview_triangles"] = new_preview_triangles
            interaction_state["hand_was_open"] = hand_is_open

        else:
            if interaction_state["dragging"]:
                interaction_state["dragging"] = False
                interaction_state["drag_vertices"][:] = False
                interaction_state["drag_triangles"][:] = False
                interaction_state["prev_hand_center"] = None
            interaction_state["preview_vertices"][:]  = False
            interaction_state["preview_triangles"][:] = False
            interaction_state["hand_was_open"] = False

        layered_layers=None; render_order_used=TMh.FIXED_RENDER_ORDER
        yaw_amount=0.0; yaw_sign=0.0; yaw_state_txt="frontal"
        yaw_debug={"sign_value":0.0,"sign_value_smooth":0.0,"width_ratio":1.0}
        leg_overlap_frac=0.0; leg_front_score=0.0; leg_override_used=False
        left_arm_overlap_frac=0.0; right_arm_overlap_frac=0.0
        left_arm_score=0.0; right_arm_score=0.0; arm_override_used=False
        bg_plate = interaction_state.get("bg_plate")

        if USE_ANIMATED_BACKGROUND:
            bg_plate = Sc.make_gradient_background(
                h, w,
                t=time.time() * ANIM_BG_SPEED,
            )

        if V_track is not None and bg_plate is not None:
            if SHOW_LAYERED_RENDER and ("tri_render_group" in binding):
                if USE_DYNAMIC_YAW_RENDER_ORDER and stable_pts is not None and "body_metrics" in interaction_state:
                    render_order_used, yaw_amount, yaw_sign, yaw_state_txt, yaw_debug = \
                        TMh.compute_render_order_from_yaw(stable_pts, interaction_state["body_metrics"], interaction_state)
                else:
                    render_order_used = TMh.FIXED_RENDER_ORDER
                _min_area = interaction_state.get('warp_min_area', 4.0)
                layered_layers = TMh.build_layered_body_layers_fast(frame_raw, V_track, V_def, T, binding,
                                                                  _min_area, 3.0, LAYER_MASK_DILATE_KSIZE, LAYER_MASK_BLUR_KSIZE)
                if stable_pts is not None and layered_layers is not None:
                    render_order_used, leg_overlap_frac, leg_front_score, leg_override_used = \
                        TMh.compute_render_order_with_leg_override(render_order_used, layered_layers, stable_pts, yaw_state_txt)
                    render_order_used, left_arm_overlap_frac, right_arm_overlap_frac, left_arm_score, right_arm_score, arm_override_used = \
                        TMh.compute_render_order_with_arm_override(render_order_used, layered_layers, stable_pts, yaw_state_txt)
                mesh_mask_u8 = TMh.build_active_mesh_mask_fast(h, w, V_def, T, active)
                base_hole_filled, _ = TMh.composite_mesh_with_background_holefill(
                    frame_raw, bg_plate, None, mesh_mask_u8, seg_mask, 0.50, 7, 5, False)
                warped_frame = TMh.composite_prebuilt_layers(base_hole_filled, layered_layers, render_order_used)
            else:
                mesh_mask_u8 = TMh.build_active_mesh_mask_fast(h, w, V_def, T, active)
                base_hole_filled, _ = TMh.composite_mesh_with_background_holefill(
                    frame_raw, bg_plate, None, mesh_mask_u8, seg_mask, 0.50, 7, 5, False)
                _min_area = interaction_state.get('warp_min_area', 4.0)
                warped_frame = TMh.warp_mesh_piecewise(frame_raw, V_track, V_def, T, active,
                                                    base_hole_filled.copy(), _min_area, 3.0)
        else:
            warped_frame = frame_raw.copy()

        affected_vertices  = interaction_state["drag_vertices"]  if interaction_state["dragging"] else interaction_state["preview_vertices"]
        affected_triangles = (interaction_state["drag_triangles"] & active) if interaction_state["dragging"] else interaction_state["preview_triangles"]
        vis = warped_frame.copy()

        if SHOW_RENDER_GROUP_DEBUG and "tri_render_group" in binding:
            vis = TMh.draw_triangle_render_group_overlay(vis, V_def, T, active, binding["tri_render_group"],
                                                      RENDER_GROUP_DEBUG_ALPHA, 1)
        if SHOW_MESH_OUTLINE:
            vis = TMh._draw_triangle_overlay(vis, V_def, T, active, affected_triangles,
                                              (0,0,0), (0,0,0), 0.12, 0.42, 1)
            if np.any(affected_vertices):
                for x, y in np.round(V_def[affected_vertices]).astype(np.int32):
                    cv2.circle(vis, (x, y), 3, (0,255,0), -1, cv2.LINE_AA)
            if hand_center is not None and not interaction_state["dragging"]:
                cx, cy = hand_center
                brush_color = (0,255,0) if (hand_is_open and hand_over_body) else (180,180,180)
                cv2.circle(vis, (cx,cy), brush_radius, brush_color, 2, cv2.LINE_AA)
                cv2.circle(vis, (cx,cy), 4, brush_color, -1, cv2.LINE_AA)
        else:
            if np.any(affected_triangles):
                vis = TMh.draw_filled_triangle_highlight(vis, V_def, T, affected_triangles,
                    fill_color=(255,150,246),
                    fill_alpha=0.65 if interaction_state["dragging"] else 0.14,
                    edge_color=(255,150,246), edge_thickness=1, edge_alpha=0.95)

        vis_display = cv2.resize(vis, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)

        if interaction_state.get("gesture_feedback_frames", 0) > 0:
            interaction_state["gesture_feedback_frames"] -= 1
            msg = interaction_state.get("gesture_feedback_msg", "")
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
            tx = (OUTPUT_W - tw) // 2
            ty = OUTPUT_H - 80
            cv2.putText(vis_display, msg, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)

        if hand_detected and hand_is_peace:
            prog = _peace_detector.progress()
            bar_w = int(OUTPUT_W * 0.5 * prog)
            bar_x = (OUTPUT_W - int(OUTPUT_W * 0.5)) // 2
            bar_y = OUTPUT_H - 130
            cv2.rectangle(vis_display,
                          (bar_x, bar_y), (bar_x + bar_w, bar_y + 18),
                          (0, 255, 255), -1)

        _hand_near_button = (
            _restart_tip_out is not None and
            (_rbx0 - 200) <= _restart_tip_out[0] <= (_rbx1 + 200) and
            (_rby0 - 200) <= _restart_tip_out[1] <= (_rby1 + 200)
        )
        _show_brush_circle = (
            hand_detected and hand_center is not None and
            (_pinch_tracker.is_active or _pinch_tracker.is_locked) and
            not _hand_near_button
        )
        if _show_brush_circle:
            scx = int(hand_center[0] * OUTPUT_W / w)
            scy = int(hand_center[1] * OUTPUT_H / h)
            scale_factor = OUTPUT_H / h
            display_r    = max(1, int(brush_radius * scale_factor))
            _br_overlay  = vis_display.copy()
            if _pinch_tracker.is_locked:
                cv2.circle(_br_overlay, (scx, scy), display_r,
                           (160, 160, 160), 1, cv2.LINE_AA)
                cv2.addWeighted(_br_overlay, 0.35, vis_display, 0.65, 0, vis_display)
            else:
                cv2.circle(_br_overlay, (scx, scy), display_r,
                           (255, 255, 255), 2, cv2.LINE_AA)
                cv2.addWeighted(_br_overlay, 0.70, vis_display, 0.30, 0, vis_display)
                _prog = _pinch_tracker.settling_progress
                if _prog > 0.0:
                    _angle = int(360 * _prog)
                    cv2.ellipse(vis_display, (scx, scy), (display_r, display_r),
                                -90, 0, _angle, (0, 255, 255), 3, cv2.LINE_AA)

        # ── Beauty-standard reference overlay ───────────────────────────
        if _bs_overlay is not None and not _timeout_fired:
            ox, oy   = _bs_overlay_x, _bs_overlay_y
            ow, oh_  = _bs_overlay.shape[1], _bs_overlay.shape[0]
            x0 = max(ox, 0);             y0 = max(oy, 0)
            x1 = min(ox + ow, OUTPUT_W); y1 = min(oy + oh_, OUTPUT_H)
            sx0 = x0 - ox;  sy0 = y0 - oy
            sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)
            if x1 > x0 and y1 > y0:
                roi    = vis_display[y0:y1, x0:x1]
                patch  = _bs_overlay[sy0:sy1, sx0:sx1]
                if patch.shape[2] == 4:
                    a      = patch[:,:,3:4].astype(np.float32) / 255.0
                    a     *= _BS_ALPHA
                    bgr    = patch[:,:,:3].astype(np.float32)
                else:
                    a      = np.full((y1-y0, x1-x0, 1), _BS_ALPHA, dtype=np.float32)
                    bgr    = patch.astype(np.float32)
                vis_display[y0:y1, x0:x1] = np.clip(
                    a * bgr + (1.0 - a) * roi.astype(np.float32), 0, 255
                ).astype(np.uint8)
        if _timeout_fired:
            scrim = vis_display.copy()
            scrim[:] = (0, 0, 0)
            vis_display = cv2.addWeighted(scrim, 0.45, vis_display, 0.55, 0)
            vis_display = Sc._draw_timeout_message(vis_display, _timeout_new_bs)

        # ── Finish button ─────────────────────────────────────────────────
        # ── Finish button: point + pinch to click ──────────────────────────
        # ── Finish button: PNG states ────────────────────────────────────

        _BTN_PAD = 200

        _hovering_restart = (
            _restart_tip_out is not None and
            (_rbx0 - _BTN_PAD) <= _restart_tip_out[0] <= (_rbx1 + _BTN_PAD) and
            (_rby0 - _BTN_PAD) <= _restart_tip_out[1] <= (_rby1 + _BTN_PAD)
        )

        pinch_dist = hand_state.get("pinch_dist", 1.0)

        # ── Brush / Drag hover detection ────────────────────────────────
        _MODE_BTN_PAD = 40

        _hovering_brush = (
            _restart_tip_out is not None and
            (brush_x0 - _MODE_BTN_PAD) <= _restart_tip_out[0] <= (brush_x1 + _MODE_BTN_PAD) and
            (brush_y0 - _MODE_BTN_PAD) <= _restart_tip_out[1] <= (brush_y1 + _MODE_BTN_PAD)
        )

        _hovering_drag = (
            _restart_tip_out is not None and
            (drag_x0 - _MODE_BTN_PAD) <= _restart_tip_out[0] <= (drag_x1 + _MODE_BTN_PAD) and
            (drag_y0 - _MODE_BTN_PAD) <= _restart_tip_out[1] <= (drag_y1 + _MODE_BTN_PAD)
        )

        if _mode_click_cooldown > 0:
            _mode_click_cooldown -= 1

        mode_pinch_click = (
            hand_state.get("detected", False)
            and (_hovering_brush or _hovering_drag)
            and _mode_click_cooldown <= 0
            and _finish_prev_pinch_dist > _FINISH_PINCH_OPEN_DIST
            and pinch_dist < _FINISH_PINCH_CLOSED_DIST
            and (_finish_prev_pinch_dist - pinch_dist) > _FINISH_PINCH_DROP_MIN
            and not _timeout_fired
        )

        if mode_pinch_click:
            if _hovering_brush:
                interaction_mode = "brush"
            elif _hovering_drag:
                interaction_mode = "drag"
                _drag_ui_timer = _DRAG_UI_DURATION

            _mode_click_cooldown = _FINISH_CLICK_COOLDOWN

        pinch_click = (
            hand_state.get("detected", False)
            and _hovering_restart
            and _finish_pinch_click_cooldown <= 0
            and _finish_prev_pinch_dist > _FINISH_PINCH_OPEN_DIST
            and pinch_dist < _FINISH_PINCH_CLOSED_DIST
            and (_finish_prev_pinch_dist - pinch_dist) > _FINISH_PINCH_DROP_MIN
            and not _timeout_fired
        )

        if _finish_pinch_click_cooldown > 0:
            _finish_pinch_click_cooldown -= 1

        if pinch_click:
            _finish_pinch_click_cooldown = _FINISH_CLICK_COOLDOWN

        _finish_prev_pinch_dist = pinch_dist


        # ── Choose button state ──────────────────────────────────────────
        if not _timeout_fired:
            if pinch_click:
                btn_img = finish_btn_click
            elif _hovering_restart:
                btn_img = finish_btn_hover
            else:
                btn_img = finish_btn_normal


            # ── Draw button ──────────────────────────────────────────────────
            vis_display = Sc.overlay_bgra(vis_display, btn_img, _rbx0, _rby0)

            # ── Draw brush / drag buttons ───────────────────────────────────
            if interaction_mode == "brush":
                brush_img = brush_btn_click
            elif _hovering_brush:
                brush_img = brush_btn_hover
            else:
                brush_img = brush_btn_normal

            if interaction_mode == "drag":
                drag_img = drag_btn_click
            elif _hovering_drag:
                drag_img = drag_btn_hover
            else:
                drag_img = drag_btn_normal

            vis_display = Sc.overlay_bgra(vis_display, brush_img, brush_x0, brush_y0)
            vis_display = Sc.overlay_bgra(vis_display, drag_img, drag_x0, drag_y0)

            if _hovering_brush:
                vis_display = Sc.draw_pinch_hint_under_button(
                    vis_display, pinch_icon, brush_x0, brush_y1, _MODE_BTN_W
                )

            if _hovering_drag:
                vis_display = Sc.draw_pinch_hint_under_button(
                    vis_display, pinch_icon, drag_x0, drag_y1, _MODE_BTN_W
                )

            if interaction_mode == "brush":
                # ── MUCH bigger pinch icon ───────────────────────────────────
                icon_scale = 2.4   # ↑ bigger than before (was ~1.6)

                if pinch_icon is not None:
                    big_icon = cv2.resize(
                        pinch_icon,
                        (int(pinch_icon.shape[1] * icon_scale),
                        int(pinch_icon.shape[0] * icon_scale)),
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    big_icon = None

                # ── Position (left edge, slightly lower for breathing room) ──
                base_x = 30
                base_y = brush_y1 + 90   # push down a bit

                if big_icon is not None:
                    vis_display = Sc.overlay_bgra(vis_display, big_icon, base_x, base_y)

                # ── Larger, more spaced text ─────────────────────────────────
                lines = ["Pinch", "Unpinch", "Hold"]

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.05     # ↑ bigger text
                thickness = 3         # ↑ thicker for readability
                line_gap = 75         # ↑ much more spacing

                text_x = base_x
                text_y = base_y + (big_icon.shape[0] if big_icon is not None else 0) + 75

                for i, line in enumerate(lines):
                    y = text_y + i * line_gap

                    # shadow
                    cv2.putText(
                        vis_display,
                        line,
                        (text_x + 3, y + 3),
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness + 2,
                        cv2.LINE_AA
                    )

                    # main text
                    cv2.putText(
                        vis_display,
                        line,
                        (text_x, y),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA
                    )

            if interaction_mode == "drag" and _drag_ui_timer > 0:
                # ── Fade factor ─────────────────────────────────────────────
                alpha = (_drag_ui_timer / _DRAG_UI_DURATION) ** 1.5

                drag_steps = [
                    (open_palm_icon, "Hide one hand behind your body"),
                    (open_palm_icon, "Hover over your body"),
                    (open_palm_icon, "Look for the highlight"),
                    (fist_icon, "Slowly make a fist"),
                    (fist_drag_icon, "Slowly drag away"),
                    (open_palm_icon, "Let go"),
                ]

                icon_h = 95
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.9
                thickness = 3

                base_x = 30
                base_y = brush_y1 + 90
                row_gap = 95
                icon_text_gap = 22

                for i, (icon, text) in enumerate(drag_steps):
                    y = base_y + i * row_gap

                    if icon is not None:
                        scale = icon_h / icon.shape[0]
                        icon_resized = cv2.resize(
                            icon,
                            (int(icon.shape[1] * scale), icon_h),
                            interpolation=cv2.INTER_AREA
                        )

                        # ── Apply fade to icon ───────────────────────────────
                        if icon_resized.shape[2] == 4:
                            icon_resized = icon_resized.copy()
                            icon_resized[:, :, 3] = (
                                icon_resized[:, :, 3].astype(np.float32) * alpha
                            ).astype(np.uint8)

                        vis_display = Sc.overlay_bgra(vis_display, icon_resized, base_x, y)
                        text_x = base_x + icon_resized.shape[1] + icon_text_gap
                    else:
                        text_x = base_x

                    text_y = y + icon_h // 2 + 12

                    # ── Text colors with fade ───────────────────────────────
                    text_color = (
                        int(255 * alpha),
                        int(255 * alpha),
                        int(255 * alpha)
                    )

                    shadow_color = (
                        int(0 * alpha),
                        int(0 * alpha),
                        int(0 * alpha)
                    )

                    # shadow
                    cv2.putText(
                        vis_display,
                        text,
                        (text_x + 3, text_y + 3),
                        font,
                        font_scale,
                        shadow_color,
                        thickness + 2,
                        cv2.LINE_AA
                    )

                    # main text
                    cv2.putText(
                        vis_display,
                        text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        text_color,
                        thickness,
                        cv2.LINE_AA
                    )

            if _hovering_restart:
                hint_text = "Pinch to click"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2

                (tw, th), baseline = cv2.getTextSize(hint_text, font, font_scale, thickness)

                gap = 10
                icon_gap = 8

                icon_w = pinch_icon.shape[1] if pinch_icon is not None else 0
                icon_h = pinch_icon.shape[0] if pinch_icon is not None else 0

                total_w = icon_w + icon_gap + tw

                group_x = _rbx0 + (_RESTART_W - total_w) // 2
                group_y = _rby1 + gap

                icon_x = group_x
                icon_y = group_y

                text_x = group_x + icon_w + icon_gap
                text_y = group_y + (icon_h + th) // 2

                if pinch_icon is not None:
                    vis_display = Sc.overlay_bgra(vis_display, pinch_icon, icon_x, icon_y)

                cv2.putText(
                    vis_display,
                    hint_text,
                    (text_x + 2, text_y + 2),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness + 1,
                    cv2.LINE_AA
                )

                cv2.putText(
                    vis_display,
                    hint_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )


            # ── Optional pointer visual ──────────────────────────────────────
            if _restart_tip_out is not None and _hand_near_button:
                cv2.circle(vis_display, _restart_tip_out, 14,
                        (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(vis_display, _restart_tip_out, 14,
                        (180, 180, 255), 2, cv2.LINE_AA)


        # ── Trigger action ───────────────────────────────────────────────
        if pinch_click:
            print("Finish triggered — returning to welcome screen")
            return "restart"
        
        if _drag_ui_timer > 0:
            _drag_ui_timer -= 1

        cv2.imshow(window_name, vis_display)

        if show_mask:
            cv2.imshow("Segmentation mask",
                       cv2.resize((seg_mask*255).astype(np.uint8), (OUTPUT_W,OUTPUT_H), interpolation=cv2.INTER_NEAREST))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('m'): SHOW_MESH_OUTLINE = not SHOW_MESH_OUTLINE
        elif key == ord('r'):
            binding["vertex_local_uv_offset"][:] = 0.0
            interaction_state["preview_vertices"][:]  = False
            interaction_state["preview_triangles"][:] = False
            interaction_state["drag_vertices"][:]  = False
            interaction_state["drag_triangles"][:] = False
            interaction_state["dragging"] = False
            interaction_state["prev_hand_center"] = None
            interaction_state["hand_was_open"] = False
        elif key in (ord('+'), ord('=')):
            interaction_state['warp_min_area'] = min(
                interaction_state.get('warp_min_area', 4.0) + 2.0, 30.0)
            print(f"warp_min_area -> {interaction_state['warp_min_area']:.1f}")
        elif key in (ord('-'), ord('_')):
            interaction_state['warp_min_area'] = max(
                interaction_state.get('warp_min_area', 4.0) - 2.0, 1.0)
            print(f"warp_min_area -> {interaction_state['warp_min_area']:.1f}")

    print("Exiting run_hand_brush_drag_arap_loop_skeleton")

