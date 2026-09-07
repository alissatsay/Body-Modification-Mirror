"""
Screen/UI-drawing helpers for run_installation.py: the welcome, home,
beauty-standard-selection and countdown screens, plus the small drawing
utilities they share (overlay compositing, timeout/countdown text,
pinch-hint icons, the animated gradient background).

Migrated out of run_installation.py so the ~3000-line pipeline script only
keeps the parts that are genuinely specific to it (the main loop and the
background pose-inference thread).
"""

import os
import time
import random
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

from helpers import display as Dh

# Same values as run_installation.py's copy (needed here since this module
# also rotates raw camera frames for these UI screens).
ROTATE_DEG = 270
ROTATE_DIR = "ccw"

# Recomputed independently from run_installation.py's copy: this file lives
# one directory deeper (installation_mirror/helpers/ vs installation_mirror/),
# so it needs its own extra ".." to land on the same installation_mirror/assets/.
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")

# Output canvas size. run_installation.py injects the real values into this
# module's namespace right after importing it (Sc.OUTPUT_W = OUTPUT_W, etc.)
# so these are just safe fallback defaults if this module is ever used alone.
OUTPUT_W = 1080
OUTPUT_H = 1920

# Recomputed independently from run_installation.py's copy, via this
# module's own _ASSETS_DIR above (same reasoning as _ASSETS_DIR/_BS_IMAGE_DIR).
WELCOME_BG_PATH = os.path.join(_ASSETS_DIR, "welcome_background.png")

# Same value as run_installation.py's copy (used there as the alpha passed
# into _composite_init_pose_overlay's call; kept here too since it's also
# this function's own default parameter value).
_INIT_POSE_ALPHA = 0.35   # same opacity as BS overlay

WELCOME_SCREEN_DURATION = 6.0
ANIM_BG_COLOR_A = (80,  10,  30)   # BGR — left/top colour
ANIM_BG_COLOR_B = (20, 100, 180)   # BGR — right/bottom colour
current_beauty_standard = ""
_BS_IMAGE_DIR = os.path.join(_ASSETS_DIR, "beauty_standard_images")  # recomputed via this module's own _ASSETS_DIR
_BS_ALL_NAMES = {
    "MBS": ["MBS_1", "MBS_2", "MBS_3"],
    "FBS": ["FBS_1", "FBS_2", "FBS_3"],
}
_WELCOME_FONTS      = None
_WELCOME_FONTS_SIZE = None
_SEL_MARGIN_X    = 60
_SEL_MARGIN_Y_TOP = 40
_SEL_MARGIN_Y_BOT = 60
_SEL_GAP_X       = 28
_SEL_GAP_Y       = 28
_SEL_COLS        = 2
_SEL_ROWS        = 3
_SEL_CORNER_R    = 22
_SEL_OUTLINE     = 6
_SEL_NAMES = [
    ["MBS_1", "FBS_1"],
    ["MBS_2", "FBS_2"],
    ["MBS_3", "FBS_3"],
]
_SEL_HOVER_FRAMES = 50

def _pick_new_beauty_standard(current):
    if not current:
        return current
    prefix = current[:3].upper()   # "MBS" or "FBS"
    candidates = _BS_ALL_NAMES.get(prefix, [])
    alternatives = [c for c in candidates if c != current]
    if not alternatives:
        return current
    return random.choice(alternatives)

def _draw_timeout_message(canvas, new_bs_name):
    from PIL import Image as _PI, ImageDraw as _PD, ImageFont as _PF
    BOLD_PATHS = ["C:/Windows/Fonts/segoeuisb.ttf","C:/Windows/Fonts/segoeuib.ttf",
                  "C:/Windows/Fonts/calibrib.ttf","C:/Windows/Fonts/arialbd.ttf","C:/Windows/Fonts/arial.ttf"]
    REG_PATHS  = ["C:/Windows/Fonts/segoeui.ttf","C:/Windows/Fonts/calibri.ttf","C:/Windows/Fonts/arial.ttf"]
    W, H = OUTPUT_W, OUTPUT_H
    usable_w = W - 120

    def _first(paths):
        for p in paths:
            if os.path.exists(p): return p
        return None

    def _fit(text, path, start=60, mn=18):
        if path is None: return _PF.load_default()
        tmp = _PD.Draw(_PI.new("RGB", (1, 1)))
        for sz in range(start, mn - 1, -1):
            try:
                f = _PF.truetype(path, sz)
                bb = tmp.textbbox((0, 0), text, font=f)
                if (bb[2] - bb[0]) <= usable_w: return f
            except Exception: continue
        return _PF.load_default()

    line1 = "Ooops, we're sorry but you ran out of time."
    line2 = "The body shape you were trying to match is no longer deemed beautiful."
    line3 = f"The current beauty standard is:"

    font1 = _fit(line1, _first(BOLD_PATHS), 60)
    font2 = _fit(line2, _first(REG_PATHS),  52)
    font3 = _fit(line3, _first(BOLD_PATHS), 56)

    img  = _PI.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = _PD.Draw(img)

    def _measure(text, font):
        try:
            bb = draw.textbbox((0, 0), text, font=font)
            return bb[2]-bb[0], bb[3]-bb[1]
        except AttributeError:
            return draw.textsize(text, font=font)

    w1,h1 = _measure(line1, font1)
    w2,h2 = _measure(line2, font2)
    w3,h3 = _measure(line3, font3)
    gap = int(H * 0.022)
    total_h = h1 + gap + h2 + gap + h3
    y = (H - total_h) // 2

    def _draw_line(text, font, y, color=(255,255,255)):
        w_, h_ = _measure(text, font)
        x = (W - w_) // 2
        draw.text((x+2, y+2), text, font=font, fill=(0,0,0))
        draw.text((x,   y  ), text, font=font, fill=color)
        return h_

    h_ = _draw_line(line1, font1, y, (255,255,255));  y += h_ + gap
    h_ = _draw_line(line2, font2, y, (220,220,220));  y += h_ + gap
    _draw_line(line3, font3, y, (220,200,255))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def capture_background_plate(cap, h, w, n_frames=20, window_name="Background capture"):
    acc = None
    got = 0
    for k in range(n_frames):
        ok, fr = cap.read()
        if not ok:
            continue
        fr = Dh.rotate_frame_for_output(fr, deg=ROTATE_DEG, direction=ROTATE_DIR)
        if fr.shape[:2] != (h, w):
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_LINEAR)
        fr = cv2.flip(fr, 1)
        acc = fr.astype(np.float32) if acc is None else acc + fr.astype(np.float32)
        got += 1
        vis_display = cv2.resize(fr, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)
        # ── CHANGE 1: no frame counter in countdown text ──────────────────
        vis_display = _draw_countdown_text(
            vis_display,
            "Capturing background...",
            OUTPUT_W, OUTPUT_H)
        cv2.imshow(window_name, vis_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if acc is None or got == 0:
        return None
    return np.clip(acc / float(got), 0, 255).astype(np.uint8)

def _draw_countdown_text(frame_bgr, text, out_w, out_h):
    """
    Render countdown / status text onto frame_bgr using PIL.
    Returns a new BGR image.
    """
    from PIL import Image as _PILImg, ImageDraw as _PILDraw, ImageFont as _PILFont
    BOLD_PATHS = [
        "C:/Windows/Fonts/segoeuisb.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    usable_w = out_w - 120
    font = None
    for fp in BOLD_PATHS:
        if not os.path.exists(fp):
            continue
        for sz in range(72, 18, -1):
            try:
                f  = _PILFont.truetype(fp, sz)
                _d = _PILDraw.Draw(_PILImg.new("RGB", (1, 1)))
                bb = _d.textbbox((0, 0), text, font=f)
                if (bb[2] - bb[0]) <= usable_w:
                    font = f
                    break
            except Exception:
                continue
        if font:
            break
    if font is None:
        font = _PILFont.load_default()

    img  = _PILImg.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = _PILDraw.Draw(img)
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except AttributeError:
        tw, th = draw.textsize(text, font=font)
    x = (out_w - tw) // 2
    y = (out_h - th) // 2
    draw.text((x + 3, y + 3), text, font=font, fill=(0, 0, 0))
    draw.text((x,     y    ), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def draw_pinch_hint_under_button(vis_display, pinch_icon, button_x0, button_y1, button_w):
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

    group_x = button_x0 + (button_w - total_w) // 2
    group_y = button_y1 + gap

    icon_x = group_x
    icon_y = group_y

    text_x = group_x + icon_w + icon_gap
    text_y = group_y + (icon_h + th) // 2

    if pinch_icon is not None:
        vis_display = overlay_bgra(vis_display, pinch_icon, icon_x, icon_y)

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

    return vis_display

def _draw_two_line_countdown_text(frame_bgr, line1, line2, out_w, out_h):
    """
    Render two lines of text centred on frame_bgr using PIL.
    line1 is drawn above line2 with a small gap.
    Returns a new BGR image.
    """
    from PIL import Image as _PILImg, ImageDraw as _PILDraw, ImageFont as _PILFont
    BOLD_PATHS = [
        "C:/Windows/Fonts/segoeuisb.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    usable_w = out_w - 120

    def _pick_font(text):
        for fp in BOLD_PATHS:
            if not os.path.exists(fp):
                continue
            for sz in range(72, 18, -1):
                try:
                    f  = _PILFont.truetype(fp, sz)
                    _d = _PILDraw.Draw(_PILImg.new("RGB", (1, 1)))
                    bb = _d.textbbox((0, 0), text, font=f)
                    if (bb[2] - bb[0]) <= usable_w:
                        return f
                except Exception:
                    continue
        return _PILFont.load_default()

    font1 = _pick_font(line1)
    font2 = _pick_font(line2)

    img  = _PILImg.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = _PILDraw.Draw(img)

    def _measure(text, font):
        try:
            bb = draw.textbbox((0, 0), text, font=font)
            return bb[2] - bb[0], bb[3] - bb[1]
        except AttributeError:
            return draw.textsize(text, font=font)

    tw1, th1 = _measure(line1, font1)
    tw2, th2 = _measure(line2, font2)
    gap = int(out_h * 0.015)
    total_h = th1 + gap + th2

    y1 = (out_h - total_h) // 2
    y2 = y1 + th1 + gap

    x1 = (out_w - tw1) // 2
    x2 = (out_w - tw2) // 2

    # line1 — instruction text, slightly smaller / lighter colour
    draw.text((x1 + 3, y1 + 3), line1, font=font1, fill=(0, 0, 0))
    draw.text((x1,     y1    ), line1, font=font1, fill=(220, 200, 255))

    # line2 — countdown text, white
    draw.text((x2 + 3, y2 + 3), line2, font=font2, fill=(0, 0, 0))
    draw.text((x2,     y2    ), line2, font=font2, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def _composite_init_pose_overlay(canvas, init_pose_img, alpha=_INIT_POSE_ALPHA):
    """
    Composite init_pose_img onto canvas using the same logic as the BS overlay:
    horizontally centred, bottom-aligned with a 60px margin, at `alpha` opacity.
    init_pose_img may be 3-channel BGR or 4-channel BGRA.
    """
    if init_pose_img is None:
        return canvas
    oh_, ow = init_pose_img.shape[:2]
    _BS_BOTTOM_MARGIN = 60
    ox = (OUTPUT_W - ow) // 2
    oy = OUTPUT_H - oh_ - _BS_BOTTOM_MARGIN

    x0 = max(ox, 0);              y0 = max(oy, 0)
    x1 = min(ox + ow, OUTPUT_W);  y1 = min(oy + oh_, OUTPUT_H)
    sx0 = x0 - ox;  sy0 = y0 - oy
    sx1 = sx0 + (x1 - x0);        sy1 = sy0 + (y1 - y0)

    if x1 <= x0 or y1 <= y0:
        return canvas

    roi   = canvas[y0:y1, x0:x1]
    patch = init_pose_img[sy0:sy1, sx0:sx1]

    if patch.shape[2] == 4:
        a   = patch[:, :, 3:4].astype(np.float32) / 255.0
        a  *= alpha
        bgr = patch[:, :, :3].astype(np.float32)
    else:
        a   = np.full((y1 - y0, x1 - x0, 1), alpha, dtype=np.float32)
        bgr = patch.astype(np.float32)

    canvas[y0:y1, x0:x1] = np.clip(
        a * bgr + (1.0 - a) * roi.astype(np.float32), 0, 255
    ).astype(np.uint8)
    return canvas

def show_countdown(cap, h, w, seconds, message, window_name,
                   bs_img=None, bs_size=120, bs_margin=18,
                   overlay_timer_start=None, overlay_timer_duration=180.0):
    start_time = time.time()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = Dh.rotate_frame_for_output(fr, deg=ROTATE_DEG, direction=ROTATE_DIR)
        if fr.shape[:2] != (h, w):
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_LINEAR)
        fr = cv2.flip(fr, 1)
        elapsed   = time.time() - start_time
        remaining = int(np.ceil(seconds - elapsed))
        if remaining <= 0:
            break
        # ── CHANGE 1: no frame counter ────────────────────────────────────
        text        = f"{message} in {remaining}"
        vis_display = cv2.resize(fr, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)
        vis_display = _composite_init_pose_overlay(vis_display, bs_img, _INIT_POSE_ALPHA)
        vis_display = _draw_countdown_text(vis_display, text, OUTPUT_W, OUTPUT_H)
        cv2.imshow(window_name, vis_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def _resolve_beauty_standard_image_path(filename="beauty_standard.png"):
    candidates = [os.path.join(_ASSETS_DIR, "beauty_standard_images", filename)]
    for p in candidates:
        if os.path.exists(p): return p
    return candidates[0]

def load_beauty_standard_overlay(size=120, filename="beauty_standard.png"):
    path = _resolve_beauty_standard_image_path(filename)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None, path
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA), path

def _draw_bs_timer_overlay(canvas, bs_img, size, margin, elapsed, duration):
    H, W = canvas.shape[:2]; x0=W-margin-size; y0=margin
    if bs_img is not None:
        roi = canvas[y0:y0+size, x0:x0+size]
        if bs_img.shape[2] == 4:
            alpha=bs_img[:,:,3:4].astype(np.float32)/255.0; rgb=bs_img[:,:,:3].astype(np.float32)
            canvas[y0:y0+size, x0:x0+size] = np.clip(alpha*rgb + (1.0-alpha)*roi.astype(np.float32), 0,255).astype(np.uint8)
        else:
            canvas[y0:y0+size, x0:x0+size] = bs_img
    frac=float(np.clip(elapsed/max(duration,1.0),0.0,1.0)); color=(203,120,255); thick=8; pad=thick//2+2
    rx0=x0-pad; ry0=y0-pad; rx1=x0+size+pad; ry1=y0+size+pad
    perimeter=4*(size+2*pad); draw_len=frac*perimeter
    segs=[((rx0,ry0),(rx1,ry0),rx1-rx0),((rx1,ry0),(rx1,ry1),ry1-ry0),
          ((rx1,ry1),(rx0,ry1),rx1-rx0),((rx0,ry1),(rx0,ry0),ry1-ry0)]
    remaining=draw_len
    for (sx,sy),(ex,ey),seg_len in segs:
        if remaining<=0: break
        t=min(remaining,seg_len)/seg_len
        cv2.line(canvas,(int(sx),int(sy)),(int(round(sx+t*(ex-sx))),int(round(sy+t*(ey-sy)))),
                 color,thick,lineType=cv2.LINE_AA)
        remaining-=seg_len
    return canvas

def make_gradient_background(h, w, t,
                              color_a=ANIM_BG_COLOR_A,
                              color_b=ANIM_BG_COLOR_B):
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    d = 0.5 * (xv + yv)
    phase = np.pi * (d - t % 1.0)
    alpha = np.sin(phase) ** 2
    a = np.array(color_a, dtype=np.float32)
    b = np.array(color_b, dtype=np.float32)
    frame = (alpha[:, :, None] * b[None, None, :] +
             (1.0 - alpha[:, :, None]) * a[None, None, :])
    return np.clip(frame, 0, 255).astype(np.uint8)

def _load_welcome_fonts(out_w, out_h, margin=90):
    from PIL import ImageFont, Image, ImageDraw

    usable_w = out_w - 2 * margin

    BOLD_PATHS = [
        "C:/Windows/Fonts/segoeuisb.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    REG_PATHS = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    LIGHT_PATHS = [
        "C:/Windows/Fonts/segoeuil.ttf",
        "C:/Windows/Fonts/calibril.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]

    def first_path(candidates):
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    bold_path  = first_path(BOLD_PATHS)
    reg_path   = first_path(REG_PATHS)
    light_path = first_path(LIGHT_PATHS)

    def fit_font(text, path, start_size, min_size=20):
        if path is None:
            return ImageFont.load_default()
        tmp_img  = Image.new("RGB", (1, 1))
        tmp_draw = ImageDraw.Draw(tmp_img)
        for sz in range(start_size, min_size - 1, -1):
            try:
                f  = ImageFont.truetype(path, sz)
                bb = tmp_draw.textbbox((0, 0), text, font=f)
                if (bb[2] - bb[0]) <= usable_w:
                    return f
            except Exception:
                continue
        return ImageFont.load_default()

    font_bold    = fit_font("Welcome",                                       bold_path,  160)
    font_regular = fit_font("Today, you are invited",                            reg_path,    72)
    font_body    = fit_font("to become the sculptors of your own body...\nif your body lets you.",      reg_path,  60)
    font_warn    = fit_font("WARNING: the display may cause you distress.",  reg_path,
                            font_body.size if hasattr(font_body, "size") else 60)

    return font_bold, font_regular, font_body, font_warn

def _make_welcome_frame(bg_base, t, out_w, out_h):
    H, W = out_h, out_w

    xs = np.linspace(0.0, 1.0, W, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, H, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    d     = 0.5 * (xv + yv)
    speed = 0.18
    phase = np.pi * (d - (t * speed) % 1.0)
    alpha = (np.sin(phase) ** 2) * 0.28
    shimmer_bgr = np.array([180.0, 50.0, 170.0], dtype=np.float32)
    shimmer = (alpha[:, :, None] * shimmer_bgr[None, None, :]).astype(np.float32)
    frame_bgr = np.clip(bg_base.astype(np.float32) + shimmer, 0, 255).astype(np.uint8)

    welcome_text_path = os.path.join(_ASSETS_DIR, "UI_gestures", "welcome_text.png")
    welcome_text_img  = cv2.imread(welcome_text_path, cv2.IMREAD_UNCHANGED)
    if welcome_text_img is not None:
        if welcome_text_img.shape[:2] != (H, W):
            welcome_text_img = cv2.resize(welcome_text_img, (W, H), interpolation=cv2.INTER_AREA)
        if welcome_text_img.shape[2] == 4:
            a   = welcome_text_img[:, :, 3:4].astype(np.float32) / 255.0
            bgr = welcome_text_img[:, :, :3].astype(np.float32)
            frame_bgr = np.clip(a * bgr + (1.0 - a) * frame_bgr.astype(np.float32), 0, 255).astype(np.uint8)
        else:
            frame_bgr = cv2.addWeighted(welcome_text_img, 1.0, frame_bgr, 0.0, 0)

    return frame_bgr

def show_welcome_screen(window_name, cap, duration=WELCOME_SCREEN_DURATION):
    if duration <= 0:
        return

    bg_img = cv2.imread(WELCOME_BG_PATH)
    if bg_img is None:
        bg_img = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)
        bg_img[:, :] = (60, 20, 10)
        print(f"WARNING: welcome background not found at '{WELCOME_BG_PATH}', using plain colour.")
    else:
        bg_img = cv2.resize(bg_img, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)

    start = time.time()
    no_detect_frames = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=False,
        enable_segmentation=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as pose_welcome:

        while True:
            t       = time.time() - start
            elapsed = t

            if elapsed >= duration:
                break

            ok, fr = cap.read()
            if ok:
                fr  = Dh.rotate_frame_for_output(fr, deg=ROTATE_DEG, direction=ROTATE_DIR)
                fr  = cv2.flip(fr, 1)
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                res = pose_welcome.process(rgb)
                if res.pose_landmarks is None:
                    no_detect_frames += 1
                    if no_detect_frames >= 10:
                        print("No person during welcome screen — returning to home screen")
                        return
                else:
                    no_detect_frames = 0

            frame = _make_welcome_frame(bg_img, t, OUTPUT_W, OUTPUT_H)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(16) & 0xFF
            if key != 255:
                break

def _load_bs_thumbnails(cell_w, cell_h):
    from PIL import Image as PILImage
    thumbs = []
    for row in range(_SEL_ROWS):
        row_imgs = []
        for col in range(_SEL_COLS):
            name = _SEL_NAMES[row][col]
            fpath = os.path.join(_BS_IMAGE_DIR, f"{name}.png")
            if not os.path.exists(fpath):
                row_imgs.append(None)
                print(f"WARNING: beauty standard image not found: {fpath}")
                continue
            img = PILImage.open(fpath).convert("RGBA")
            img_w, img_h = img.size
            scale = min((cell_w - _SEL_OUTLINE*2) / img_w,
                        (cell_h - _SEL_OUTLINE*2) / img_h)
            new_w = max(1, int(img_w * scale))
            new_h = max(1, int(img_h * scale))
            img   = img.resize((new_w, new_h), PILImage.LANCZOS)
            row_imgs.append(img)
        thumbs.append(row_imgs)
    return thumbs

def _cell_rect(row, col, grid_top, grid_left, cell_w, cell_h):
    x0 = grid_left + col * (cell_w + _SEL_GAP_X)
    y0 = grid_top  + row * (cell_h + _SEL_GAP_Y)
    return x0, y0, x0 + cell_w, y0 + cell_h

def _get_index_tip_norm(hand_results, project=2.5):
    if not hand_results.multi_hand_landmarks:
        return None, None
    lm = hand_results.multi_hand_landmarks[0].landmark
    wx,  wy  = lm[0].x, lm[0].y
    mx,  my  = lm[5].x, lm[5].y
    dx = mx - wx
    dy = my - wy
    px = mx + project * dx
    py = my + project * dy
    return float(np.clip(px, 0.0, 1.0)), float(np.clip(py, 0.0, 1.0))

def overlay_bgra(base_bgr, overlay_bgra, x, y):
    h, w = overlay_bgra.shape[:2]

    x1 = min(x + w, base_bgr.shape[1])
    y1 = min(y + h, base_bgr.shape[0])

    if x >= x1 or y >= y1:
        return base_bgr

    overlay_crop = overlay_bgra[0:(y1 - y), 0:(x1 - x)]
    roi = base_bgr[y:y1, x:x1]

    bgr = overlay_crop[:, :, :3].astype(np.float32)
    alpha = overlay_crop[:, :, 3:4].astype(np.float32) / 255.0

    roi[:] = (alpha * bgr + (1 - alpha) * roi.astype(np.float32)).astype(np.uint8)
    return base_bgr

def _make_selection_frame(bg_base, t, out_w, out_h,
                           font_hdr, thumbnails,
                           grid_top, grid_left, cell_w, cell_h,
                           hover_fill,
                           tip_px=None):
    from PIL import Image as PILImage, ImageDraw

    H, W = out_h, out_w

    xs = np.linspace(0.0, 1.0, W, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, H, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    d     = 0.5 * (xv + yv)
    speed = 0.18
    phase = np.pi * (d - (t * speed) % 1.0)
    alpha_sh = (np.sin(phase) ** 2) * 0.28
    shimmer_bgr = np.array([180.0, 50.0, 170.0], dtype=np.float32)
    shimmer = (alpha_sh[:, :, None] * shimmer_bgr[None, None, :]).astype(np.float32)
    frame_bgr = np.clip(bg_base.astype(np.float32) + shimmer, 0, 255).astype(np.uint8)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    canvas = PILImage.fromarray(frame_rgb).convert("RGBA")
    draw   = ImageDraw.Draw(canvas)

    header = "Please choose the beauty standard you would like to emulate"
    try:
        bb = draw.textbbox((0, 0), header, font=font_hdr)
        tw = bb[2] - bb[0]
    except AttributeError:
        tw, _ = draw.textsize(header, font=font_hdr)
    hx = (W - tw) // 2
    hy = _SEL_MARGIN_X
    draw.text((hx + 2, hy + 2), header, font=font_hdr, fill=(0, 0, 0, 200))
    draw.text((hx,     hy    ), header, font=font_hdr, fill=(255, 255, 255, 255))

    shimmer_speed = 0.6
    shimmer_wave  = (t * shimmer_speed) % 1.0

    overlay = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
    odraw   = ImageDraw.Draw(overlay)

    for row in range(_SEL_ROWS):
        for col in range(_SEL_COLS):
            x0, y0, x1, y1 = _cell_rect(row, col, grid_top, grid_left,
                                          cell_w, cell_h)
            fill_frac = hover_fill.get((row, col), 0.0)

            cell_diag  = (col / max(_SEL_COLS - 1, 1) + row / max(_SEL_ROWS - 1, 1)) * 0.5
            phase      = np.pi * (cell_diag - shimmer_wave)
            shimmer_v  = float(np.sin(phase) ** 2)

            base_out_a  = 120
            shimmer_add = int(shimmer_v * 100)
            hover_add   = int(fill_frac * 35)
            out_a       = min(255, base_out_a + shimmer_add + hover_add)
            out_w_px    = _SEL_OUTLINE + int(fill_frac * 4)

            r_out = 255
            g_out = int(255 - (1.0 - shimmer_v) * 30 * (1.0 - fill_frac))
            b_out = int(255 - (1.0 - shimmer_v) * 60 * (1.0 - fill_frac))

            fill_a = int(8 + fill_frac * 247)

            odraw.rounded_rectangle(
                [x0, y0, x1, y1],
                radius=_SEL_CORNER_R,
                fill=(255, 255, 255, fill_a),
                outline=(r_out, g_out, b_out, out_a),
                width=out_w_px,
            )

            thumb = thumbnails[row][col]
            if thumb is not None:
                tw_i, th_i = thumb.size
                px = x0 + (cell_w - tw_i) // 2
                py = y0 + (cell_h - th_i) // 2
                canvas.paste(thumb,
                             (max(px, x0 + _SEL_OUTLINE),
                              max(py, y0 + _SEL_OUTLINE)),
                             mask=thumb)

    canvas = PILImage.alpha_composite(canvas, overlay)

    result_bgr = cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)
    if tip_px is not None:
        tx, ty = tip_px
        cv2.circle(result_bgr, (tx, ty), 18, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(result_bgr, (tx, ty), 18, (180, 180, 255),  2, cv2.LINE_AA)

    return result_bgr

def show_selection_screen(window_name, bg_base, cap):
    global current_beauty_standard

    from PIL import ImageFont as PILFont, Image as _PI3, ImageDraw as _ID3

    out_w, out_h = OUTPUT_W, OUTPUT_H
    usable_w = out_w - 2 * _SEL_MARGIN_X

    BOLD_PATHS = [
        "C:/Windows/Fonts/segoeuisb.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    header_text = "Please choose the beauty standard you would like to emulate"
    font_hdr = None
    for fpath in BOLD_PATHS:
        if not os.path.exists(fpath):
            continue
        for sz in range(52, 14, -1):
            try:
                f  = PILFont.truetype(fpath, sz)
                _d = _ID3.Draw(_PI3.new("RGB", (1, 1)))
                bb = _d.textbbox((0, 0), header_text, font=f)
                if (bb[2] - bb[0]) <= usable_w:
                    font_hdr = f
                    break
            except Exception:
                continue
        if font_hdr is not None:
            break
    if font_hdr is None:
        font_hdr = PILFont.load_default()

    _dd = _ID3.Draw(_PI3.new("RGB", (1, 1)))
    try:
        hdr_bb = _dd.textbbox((0, 0), header_text, font=font_hdr)
        hdr_h  = hdr_bb[3] - hdr_bb[1]
    except AttributeError:
        _, hdr_h = _dd.textsize(header_text, font=font_hdr)

    grid_top  = _SEL_MARGIN_X + hdr_h + _SEL_MARGIN_Y_TOP
    grid_left = _SEL_MARGIN_X
    grid_w    = out_w - 2 * _SEL_MARGIN_X
    grid_h    = out_h - _SEL_MARGIN_Y_BOT - grid_top
    cell_w    = (grid_w - _SEL_GAP_X * (_SEL_COLS - 1)) // _SEL_COLS
    cell_h    = (grid_h - _SEL_GAP_Y * (_SEL_ROWS - 1)) // _SEL_ROWS

    thumbnails = _load_bs_thumbnails(cell_w, cell_h)

    key_to_cell = {
        ord('1'): (0,0), ord('2'): (1,0), ord('3'): (2,0),
        ord('4'): (0,1), ord('5'): (1,1), ord('6'): (2,1),
    }

    hover_fill   = {(r, c): 0.0 for r in range(_SEL_ROWS) for c in range(_SEL_COLS)}
    hover_frames = {(r, c): 0   for r in range(_SEL_ROWS) for c in range(_SEL_COLS)}
    DRAIN_SPEED  = 8

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=False,
        enable_segmentation=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as pose_sel, mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands_sel:

        start  = time.time()
        no_detect_frames_sel = 0
        result = None

        smooth_nx, smooth_ny = None, None
        SMOOTH_ALPHA = 0.35

        RENDER_EVERY   = 4
        infer_count    = 0
        cached_frame   = None

        while result is None:
            for _ in range(3):
                cap.grab()
            ok, fr = cap.read()
            if not ok:
                break
            fr  = Dh.rotate_frame_for_output(fr, deg=ROTATE_DEG, direction=ROTATE_DIR)
            if fr.shape[:2] != (out_h, out_w):
                fr = cv2.resize(fr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            fr  = cv2.flip(fr, 1)
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

            hand_res       = hands_sel.process(rgb)
            raw_nx, raw_ny = _get_index_tip_norm(hand_res)
            infer_count   += 1

            pose_res_sel = pose_sel.process(rgb)
            if pose_res_sel.pose_landmarks is None:
                no_detect_frames_sel += 1
                if no_detect_frames_sel >= 10:
                    print("No person during selection — returning to home screen")
                    result = "__skip__"
            else:
                no_detect_frames_sel = 0

            if raw_nx is not None:
                if smooth_nx is None:
                    smooth_nx, smooth_ny = raw_nx, raw_ny
                else:
                    smooth_nx = SMOOTH_ALPHA * raw_nx + (1 - SMOOTH_ALPHA) * smooth_nx
                    smooth_ny = SMOOTH_ALPHA * raw_ny + (1 - SMOOTH_ALPHA) * smooth_ny
            else:
                smooth_nx = smooth_ny = None

            tip_px = None
            if smooth_nx is not None:
                tip_px = (int(smooth_nx * out_w), int(smooth_ny * out_h))

            hovered_cell = None
            if tip_px is not None:
                tx, ty = tip_px
                for row in range(_SEL_ROWS):
                    for col in range(_SEL_COLS):
                        x0, y0, x1, y1 = _cell_rect(row, col, grid_top,
                                                      grid_left, cell_w, cell_h)
                        if x0 <= tx <= x1 and y0 <= ty <= y1:
                            hovered_cell = (row, col)
                            break
                    if hovered_cell:
                        break

            for cell in list(hover_fill.keys()):
                if cell == hovered_cell:
                    hover_frames[cell] = min(_SEL_HOVER_FRAMES,
                                             hover_frames[cell] + 1)
                    hover_fill[cell]   = hover_frames[cell] / _SEL_HOVER_FRAMES
                    if hover_frames[cell] >= _SEL_HOVER_FRAMES:
                        result = _SEL_NAMES[cell[0]][cell[1]]
                else:
                    hover_frames[cell] = max(0, hover_frames[cell] - DRAIN_SPEED)
                    hover_fill[cell]   = hover_frames[cell] / _SEL_HOVER_FRAMES

            if infer_count % RENDER_EVERY == 0 or cached_frame is None:
                t = time.time() - start
                cached_frame = _make_selection_frame(
                    bg_base, t, out_w, out_h,
                    font_hdr, thumbnails,
                    grid_top, grid_left, cell_w, cell_h,
                    hover_fill=hover_fill,
                    tip_px=tip_px,
                )

            cv2.imshow(window_name, cached_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in key_to_cell:
                result = _SEL_NAMES[key_to_cell[key][0]][key_to_cell[key][1]]
            if key in (ord('q'), 27):
                result = "__skip__"
                break

    if result == "__skip__" or result is None:
        current_beauty_standard = ""
        return None

    current_beauty_standard = result
    print(f"Beauty standard selected: {result}")
    return result

def show_home_screen(window_name, cap, pose):
    ABSENT_FRAMES = 5
    PRESENT_FRAMES = 3

    bg_img = cv2.imread(WELCOME_BG_PATH)
    if bg_img is None:
        bg_img = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)
        bg_img[:, :] = (60, 20, 10)
    else:
        bg_img = cv2.resize(bg_img, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)

    armed          = False
    present_count  = 0
    absent_count   = 0
    start          = time.time()

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=False,
        enable_segmentation=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as pose_home:

        while True:
            ok, fr = cap.read()
            if not ok:
                break
            fr  = Dh.rotate_frame_for_output(fr, deg=ROTATE_DEG, direction=ROTATE_DIR)
            if fr.shape[:2] != bg_img.shape[:2]:
                fr = cv2.resize(fr, (bg_img.shape[1], bg_img.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
            fr  = cv2.flip(fr, 1)
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

            res          = pose_home.process(rgb)
            person_here  = (res.pose_landmarks is not None)

            if person_here:
                present_count += 1
                if present_count >= 2:
                    print("Home screen: person detected — showing welcome screen")
                    show_welcome_screen(window_name, cap, duration=WELCOME_SCREEN_DURATION)
                    return
            else:
                present_count = 0

            t     = time.time() - start
            xs    = np.linspace(0.0, 1.0, OUTPUT_W, dtype=np.float32)
            ys    = np.linspace(0.0, 1.0, OUTPUT_H, dtype=np.float32)
            xv, yv = np.meshgrid(xs, ys)
            d     = 0.5 * (xv + yv)
            speed = 0.18
            phase = np.pi * (d - (t * speed) % 1.0)
            alpha = (np.sin(phase) ** 2) * 0.28
            shimmer_bgr = np.array([180.0, 50.0, 170.0], dtype=np.float32)
            shimmer = (alpha[:, :, None] * shimmer_bgr[None, None, :]).astype(np.float32)
            frame_out = np.clip(bg_img.astype(np.float32) + shimmer, 0, 255).astype(np.uint8)

            cv2.imshow(window_name, frame_out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

