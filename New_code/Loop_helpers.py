import cv2
import numpy as np
import mediapipe as mp
import time

import Pose_Tracking_helpers as PTh
import Triangle_Mesh_helpers as TMh
import Display_helpres as Dh

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