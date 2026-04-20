import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import warnings
import ctypes

warnings.filterwarnings('ignore', message='SymbolDatabase.GetPrototype')

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

COLORS = {
    "Red":    (0,   0,   255),
    "Green":  (0,   220, 80),
    "Blue":   (255, 80,  0),
    "Yellow": (0,   230, 255),
    "Purple": (200, 0,   180),
    "White":  (255, 255, 255),
    "Eraser": (0,   0,   0),
}
COLOR_NAMES = list(COLORS.keys())
BRUSH_SIZES = [4, 10, 22]

TOOLBAR_H    = 90
CLR_RADIUS   = 22   # color swatch radius
CLR_GAP      = 58   # center-to-center
TB_PAD       = 22   # horizontal toolbar padding
BRUSH_W      = 66
BRUSH_H      = 48
CLEAR_W      = 88

SMOOTH_N            = 3
MAX_JUMP            = 80
DRAW_CONFIRM_FRAMES = 1


def get_screen_size():
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return 1280, 720



def rounded_rect(img, pt1, pt2, color, r=10, t=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    if t == -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.line(img, (x1+r, y1), (x2-r, y1), color, t)
        cv2.line(img, (x1+r, y2), (x2-r, y2), color, t)
        cv2.line(img, (x1, y1+r), (x1, y2-r), color, t)
        cv2.line(img, (x2, y1+r), (x2, y2-r), color, t)
        cv2.ellipse(img, (x1+r, y1+r), (r, r), 180,  0, 90, color, t)
        cv2.ellipse(img, (x2-r, y1+r), (r, r), 270,  0, 90, color, t)
        cv2.ellipse(img, (x1+r, y2-r), (r, r),  90,  0, 90, color, t)
        cv2.ellipse(img, (x2-r, y2-r), (r, r),   0,  0, 90, color, t)


def alpha_rect(img, pt1, pt2, color, alpha=0.82, r=0):
    overlay = img.copy()
    if r > 0:
        rounded_rect(overlay, pt1, pt2, color, r=r)
    else:
        cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)



def toolbar_layout(w):
    mid_y = TOOLBAR_H // 2

    # color swatch centers
    color_cx = [TB_PAD + CLR_RADIUS + i * CLR_GAP for i in range(len(COLOR_NAMES))]
    color_centers = [(x, mid_y) for x in color_cx]

    # divider after last swatch
    div_x = color_cx[-1] + CLR_RADIUS + TB_PAD

    # brush buttons
    bsy = mid_y - BRUSH_H // 2
    brush_rects = []
    for i in range(len(BRUSH_SIZES)):
        bx = div_x + TB_PAD + i * (BRUSH_W + 8)
        brush_rects.append((bx, bsy, bx + BRUSH_W, bsy + BRUSH_H))

    # clear button 
    cx1 = w - TB_PAD - CLEAR_W
    cy1 = mid_y - BRUSH_H // 2
    clear_rect = (cx1, cy1, cx1 + CLEAR_W, cy1 + BRUSH_H)

    return color_centers, div_x, brush_rects, clear_rect



def draw_toolbar(frame, current_color, current_brush):
    h, w = frame.shape[:2]

    # glass background
    alpha_rect(frame, (0, 0), (w, TOOLBAR_H), (10, 12, 16), alpha=0.88)
    cv2.line(frame, (0, TOOLBAR_H), (w, TOOLBAR_H), (45, 50, 62), 1)

    color_centers, div_x, brush_rects, clear_rect = toolbar_layout(w)

    # color swatches
    for i, (name, color) in enumerate(COLORS.items()):
        cx, cy = color_centers[i]
        selected = name == current_color

        if name == "Eraser":
            cv2.circle(frame, (cx, cy), CLR_RADIUS, (55, 58, 66), -1)
            cv2.circle(frame, (cx, cy), CLR_RADIUS, (90, 95, 108), 1)
            off = 7
            cv2.line(frame, (cx-off, cy-off), (cx+off, cy+off), (160, 160, 170), 2)
            cv2.line(frame, (cx+off, cy-off), (cx-off, cy+off), (160, 160, 170), 2)
        else:
            cv2.circle(frame, (cx, cy), CLR_RADIUS, color, -1)

        if selected:
            cv2.circle(frame, (cx, cy), CLR_RADIUS + 5, (255, 255, 255), 2)
        else:
            cv2.circle(frame, (cx, cy), CLR_RADIUS + 2, (50, 55, 68), 1)

    # divider
    cv2.line(frame, (div_x, 16), (div_x, TOOLBAR_H - 16), (45, 50, 62), 1)

    # brush buttons
    for i, (bx1, by1, bx2, by2) in enumerate(brush_rects):
        size = BRUSH_SIZES[i]
        selected = current_brush == size
        bg    = (52, 62, 82)   if selected else (22, 26, 34)
        border= (90, 120, 180) if selected else (42, 48, 62)
        rounded_rect(frame, (bx1, by1), (bx2, by2), bg, r=9)
        rounded_rect(frame, (bx1, by1), (bx2, by2), border, r=9, t=1)
        dot_r = max(2, min(size // 2 + 1, 12))
        dot_cx = (bx1 + bx2) // 2
        dot_cy = (by1 + by2) // 2 - 5
        dot_col = (220, 228, 245) if selected else (110, 118, 135)
        cv2.circle(frame, (dot_cx, dot_cy), dot_r, dot_col, -1)
        label_col = (180, 192, 220) if selected else (80, 88, 105)
        cv2.putText(frame, f"{size}px", (bx1 + 10, by2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, label_col, 1, cv2.LINE_AA)

    # clear button
    cx1, cy1, cx2, cy2 = clear_rect
    rounded_rect(frame, (cx1, cy1), (cx2, cy2), (28, 30, 68), r=9)
    rounded_rect(frame, (cx1, cy1), (cx2, cy2), (70, 78, 200), r=9, t=1)
    (tw, th), _ = cv2.getTextSize("CLEAR", cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    tx = cx1 + ((cx2 - cx1) - tw) // 2
    ty = cy1 + ((cy2 - cy1) + th) // 2
    cv2.putText(frame, "CLEAR", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (160, 168, 255), 1, cv2.LINE_AA)

    return color_centers, div_x, brush_rects, clear_rect



def hit_toolbar(px, py, color_centers, brush_rects, clear_rect):
    if py > TOOLBAR_H:
        return None, None
    for i, (cx, cy) in enumerate(color_centers):
        if np.hypot(px - cx, py - cy) <= CLR_RADIUS + 5:
            return "color", COLOR_NAMES[i]
    for i, (bx1, by1, bx2, by2) in enumerate(brush_rects):
        if bx1 <= px <= bx2 and by1 <= py <= by2:
            return "brush", BRUSH_SIZES[i]
    cx1, cy1, cx2, cy2 = clear_rect
    if cx1 <= px <= cx2 and cy1 <= py <= cy2:
        return "clear", None
    return None, None



def draw_status(frame, drawing, draw_confirm, current_color, current_brush, sw, sh):
    if drawing:
        dot, mode_txt, dot_col = "●", "DRAWING", (80, 220, 110)
    elif draw_confirm > 0:
        dot, mode_txt, dot_col = "◌", "CONFIRM", (80, 180, 255)
    else:
        dot, mode_txt, dot_col = "○", "HOVER",   (110, 120, 140)

    info = f"  {current_color}  ·  {current_brush}px  ·  [c] clear  [q] quit"
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1

    (mw, mh), _ = cv2.getTextSize(f"{dot} {mode_txt}", font, fs, ft)
    (iw, _),  _ = cv2.getTextSize(info, font, fs, ft)
    total_w = mw + iw
    px, py = 20, 10
    pill_w, pill_h = total_w + px * 2, mh + py * 2
    pill_x = (sw - pill_w) // 2
    pill_y = sh - pill_h - 16
    r = pill_h // 2

    alpha_rect(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h),
               (10, 12, 16), alpha=0.80, r=r)
    rounded_rect(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h),
                 (42, 48, 62), r=r, t=1)

    tx, ty = pill_x + px, pill_y + py + mh
    cv2.putText(frame, f"{dot} {mode_txt}", (tx, ty), font, fs, dot_col, ft, cv2.LINE_AA)
    cv2.putText(frame, info, (tx + mw, ty), font, fs, (120, 128, 148), ft, cv2.LINE_AA)



def draw_cursor(frame, ix, iy, current_color, current_brush, drawing):
    dot_col = COLORS[current_color] if current_color != "Eraser" else (140, 145, 160)
    r = current_brush + 4
    if drawing:
        cv2.circle(frame, (ix, iy), r, dot_col, -1)
        cv2.circle(frame, (ix, iy), r + 3, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.circle(frame, (ix, iy), 3, dot_col, -1, cv2.LINE_AA)
        cv2.circle(frame, (ix, iy), r + 3, (220, 225, 235), 2, cv2.LINE_AA)
        cv2.circle(frame, (ix, iy), r + 7, (60, 68, 82), 1, cv2.LINE_AA)



def is_finger_up(lm, tip, pip, dip, threshold=0.025):
    return (lm[pip].y - lm[tip].y) > threshold


def count_fingers_up(lm):
    return sum(
        is_finger_up(lm, tip, pip, dip)
        for tip, pip, dip in [(8,6,5),(12,10,9),(16,14,13),(20,18,17)]
    )


def smooth_point(history):
    if not history:
        return None, None
    return int(np.mean([p[0] for p in history])), int(np.mean([p[1] for p in history]))



def main():
    SCREEN_W, SCREEN_H = get_screen_size()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ret, frame = cap.read()
    if not ret:
        print("Cannot open camera.")
        return

    canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

    cv2.namedWindow("Virtual Drawing Canvas", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Virtual Drawing Canvas", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    current_color = "Red"
    current_brush = 10
    prev_x = prev_y = None
    drawing = False
    toolbar_cooldown = 0
    draw_confirm = 0
    ix = iy = None

    pos_history = deque(maxlen=SMOOTH_N)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    # pre-compute layout once 
    color_centers, div_x, brush_rects, clear_rect = toolbar_layout(SCREEN_W)

    print("Index finger → draw   |   2+ fingers → hover   |   c = clear   |   q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (SCREEN_W, SCREEN_H))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        ix = iy = None

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            lm = hand_lm.landmark

            mp_draw.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 200, 80),  thickness=1, circle_radius=2),
                mp_draw.DrawingSpec(color=(120, 130, 180), thickness=1),
            )

            raw_x = int(lm[8].x * SCREEN_W)
            raw_y = int(lm[8].y * SCREEN_H)
            pos_history.append((raw_x, raw_y))
            ix, iy = smooth_point(pos_history)

            fingers_up = count_fingers_up(lm)

            if iy < TOOLBAR_H:
                if toolbar_cooldown == 0:
                    kind, value = hit_toolbar(ix, iy, color_centers, brush_rects, clear_rect)
                    if kind == "color":
                        current_color = value
                        toolbar_cooldown = 18
                    elif kind == "brush":
                        current_brush = value
                        toolbar_cooldown = 18
                    elif kind == "clear":
                        canvas[:] = 0
                        toolbar_cooldown = 25
                prev_x = prev_y = None
                drawing = False
                draw_confirm = 0

            elif fingers_up == 1:
                draw_confirm = min(draw_confirm + 1, DRAW_CONFIRM_FRAMES)
                if draw_confirm >= DRAW_CONFIRM_FRAMES:
                    drawing = True
                    color = COLORS[current_color]
                    thickness = current_brush * 3 if current_color == "Eraser" else current_brush * 2
                    if prev_x is not None and np.hypot(ix - prev_x, iy - prev_y) < MAX_JUMP:
                        cv2.line(canvas, (prev_x, prev_y), (ix, iy), color, thickness)
                    prev_x, prev_y = ix, iy
            else:
                draw_confirm = 0
                drawing = False
                prev_x = prev_y = None
        else:
            prev_x = prev_y = None
            drawing = False
            draw_confirm = 0
            pos_history.clear()

        if toolbar_cooldown > 0:
            toolbar_cooldown -= 1

        # composite canvas onto frame
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY)
        combined = cv2.add(
            cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask)),
            cv2.bitwise_and(canvas, canvas, mask=mask),
        )

        draw_toolbar(combined, current_color, current_brush)
        if ix is not None:
            draw_cursor(combined, ix, iy, current_color, current_brush, drawing)
        draw_status(combined, drawing, draw_confirm, current_color, current_brush, SCREEN_W, SCREEN_H)

        cv2.imshow("Virtual Drawing Canvas", combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            canvas[:] = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
