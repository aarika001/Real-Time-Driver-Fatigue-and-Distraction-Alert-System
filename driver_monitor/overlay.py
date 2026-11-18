"""HUD drawing: banners, reliability bars, probabilities, calibration, disclaimer, help."""
from dataclasses import dataclass, field
import cv2
import numpy as np
import config

FONT = cv2.FONT_HERSHEY_SIMPLEX
BANNER = {  # BGR
    0: ((60, 150, 60), "MONITORING: OK"),
    1: ((0, 200, 255), "SOFT ALERT"),
    2: ((0, 140, 255), "ALERT"),
    3: ((0, 0, 220), "URGENT"),
}


@dataclass
class Hud:
    fps: float = 0.0
    percept: object = None
    rel: object = None
    tout: object = None
    decision: object = None
    calib_progress: float = 1.0          # <1 while calibrating
    calib_waiting_face: bool = False
    effects: set = field(default_factory=set)
    severity: float = 0.6
    miss_ratio: float = config.MISS_RATIO
    muted: bool = False
    show_help: bool = True
    show_landmarks: bool = True
    stage_ms: dict = field(default_factory=dict)
    lm_scale: float = 1.0                # landmark coords -> drawn-frame coords


def put(img, text, org, scale=0.5, color=(255, 255, 255), thick=None):
    # Stroke widths follow the font size: a fixed 3 px outline around a ~7 px glyph (small
    # frame stretched by the window) swallows the white fill and smears the text.
    if thick is None:
        thick = max(1, int(round(scale * 2)))
    halo = max(1, thick // 2 + 1) if scale >= 0.6 else 0
    if halo:
        cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + halo * 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def bar(img, x, y, w, h, val, color, label):
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + int(w * float(np.clip(val, 0, 1))), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    put(img, f"{label} {val:.2f}", (x + w + 8, y + h - 2), h / 40 + 0.4)


def banner(img, text, color, s):
    H, W = img.shape[:2]
    bh = int(42 * s)
    cv2.rectangle(img, (0, 0), (W, bh), color, -1)
    scale = 0.8 * s
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    put(img, text, ((W - tw) // 2, (bh + th) // 2), scale, (255, 255, 255), max(2, int(round(scale * 2.5))))


def draw_hud(img, h: Hud):
    H, W = img.shape[:2]
    s = W / 640.0                      # HUD is laid out for 640 px wide, scaled to the frame
    d = h.decision

    # landmarks (eyes + mouth) so the cues are visible
    if h.show_landmarks and h.percept is not None and h.percept.landmarks_px is not None:
        for i in config.LM_R_EYE + config.LM_L_EYE + config.LM_MOUTH_ALL:
            x, y = h.percept.landmarks_px[i]
            cv2.circle(img, (int(x * h.lm_scale), int(y * h.lm_scale)), max(2, int(2 * s)), (0, 255, 255), -1)

    # top banner
    if h.calib_progress < 1.0 or h.calib_waiting_face:
        if h.calib_waiting_face:
            banner(img, "CALIBRATING: no face found, look at the camera", (150, 90, 20), s)
        else:
            banner(img, f"CALIBRATING {int(h.calib_progress * 100)}%: look at the road, relax", (150, 90, 20), s)
        bh = int(42 * s)
        cv2.rectangle(img, (0, bh), (int(W * h.calib_progress), bh + int(6 * s)), (255, 200, 80), -1)
    elif d is not None and d.degraded:
        banner(img, "MONITORING DEGRADED", (130, 60, 130), s)
    elif d is not None:
        color, text = BANNER[d.action_idx]
        if d.action_idx > 0:
            text += f": {d.dominant.upper()}"
        banner(img, text, color, s)

    # left panel (dark translucent backing so it stays readable on any background)
    x, y, lh = int(10 * s), int(70 * s), int(22 * s)
    px1, py1, py2 = int(300 * s), int(50 * s), int(318 * s)
    roi = img[py1:py2, 0:px1]
    img[py1:py2, 0:px1] = cv2.addWeighted(roi, 0.45, np.zeros_like(roi), 0.55, 0)
    bw, bhh = int(110 * s), int(12 * s)
    p = h.percept
    put(img, f"FPS {h.fps:4.1f}", (x, y), 0.55 * s, (0, 255, 0) if h.fps >= 15 else (0, 0, 255))
    y += lh
    if p is not None and p.face_found:
        put(img, f"EAR {p.ear:.2f}  MAR {p.mar:.2f}", (x, y), 0.5 * s)
        y += lh
        put(img, f"yaw {p.yaw:+.0f}  pitch {p.pitch:+.0f}", (x, y), 0.5 * s)
    else:
        put(img, "NO FACE", (x, y), 0.55 * s, (0, 0, 255))
    y += int(lh * 1.3)

    if h.rel is not None:
        put(img, "Cue reliability", (x, y), 0.45 * s, (200, 200, 200))
        y += int(lh * 0.7)
        for name, v in (("eye", h.rel.eye), ("mouth", h.rel.mouth), ("head", h.rel.head)):
            col = (0, 200, 0) if v >= config.R_MIN else (0, 0, 220)
            bar(img, x, y, bw, bhh, v, col, name)
            y += int(lh * 0.9)
        y += int(lh * 0.4)
    if h.tout is not None and h.calib_progress >= 1.0:
        put(img, "P(state)", (x, y), 0.45 * s, (200, 200, 200))
        y += int(lh * 0.7)
        for name, v, col in zip(config.STATES, h.tout.probs, ((0, 200, 0), (0, 165, 255), (255, 120, 0))):
            bar(img, x, y, bw, bhh, v, col, name[:5])
            y += int(lh * 0.9)
    put(img, f"miss:FA {h.miss_ratio:.0f}:1" + ("  MUTED" if h.muted else ""), (x, y + int(lh * 0.5)), 0.45 * s,
        (220, 220, 220))

    # active stress effects (right)
    if h.effects:
        put(img, f"STRESS: {', '.join(sorted(h.effects))} @{h.severity:.1f}", (W - int(330 * s), int(70 * s)),
            0.5 * s, (0, 200, 255))

    # help
    if h.show_help:
        lines = ["[1] glare [2] low light [3] occlusion [4] face lost [5] off-axis [0] clear  [+/-] severity",
                 "[c] recalibrate  [ / ] miss:FA ratio  [m] mute  [l] landmarks  [h] help  [q] quit"]
        for i, t in enumerate(lines):
            put(img, t, (int(10 * s), H - int((52 - i * 18) * s)), 0.42 * s, (220, 220, 220))

    put(img, "Research prototype - not a certified safety system", (int(10 * s), H - int(10 * s)), 0.45 * s,
        (0, 200, 255))
    return img
