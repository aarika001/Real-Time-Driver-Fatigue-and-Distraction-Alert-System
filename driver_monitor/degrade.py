"""Live stress-test effects on the video feed (toggle with keys). Lets you demo the
'monitoring degraded' banner without the real estimator. Real severity sweeps for the
report belong to R2/R5's offline generators; this is for the live demo."""
from dataclasses import dataclass, field
import cv2
import numpy as np
import config
from interfaces import Percept

KEYS = {ord("1"): "glare", ord("2"): "lowlight", ord("3"): "occlusion",
        ord("4"): "facelost", ord("5"): "offaxis"}


@dataclass
class DegradeState:
    active: set = field(default_factory=set)
    severity: float = 0.6

    def toggle(self, name):
        self.active ^= {name}

    def bump(self, d):
        self.severity = float(np.clip(self.severity + d, 0.1, 1.0))


def anchors_from(p: Percept):
    """Eye-midpoint and mouth positions from the last frame (for placing effects)."""
    if p.landmarks_px is None:
        return None
    pts = p.landmarks_px
    return {"eyes": tuple(pts[[33, 263]].mean(axis=0)), "mouth": tuple(pts[13])}


def apply(frame, st: DegradeState, anchors=None):
    if not st.active:
        return frame
    h, w = frame.shape[:2]
    s = st.severity
    out = frame.copy()
    eyes = anchors["eyes"] if anchors else (w * 0.5, h * 0.4)
    mouth = anchors["mouth"] if anchors else (w * 0.5, h * 0.65)

    if "offaxis" in st.active:                       # approx. camera placed off to the side
        dx, dy = w * 0.2 * s, h * 0.12 * s
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[0, 0], [w - dx, dy], [w - dx, h - dy], [0, h]])
        out = cv2.warpPerspective(out, cv2.getPerspectiveTransform(src, dst), (w, h))
    if "lowlight" in st.active:
        x = (out / 255.0) ** (1.0 + 3.0 * s) * (1.0 - 0.5 * s)
        x += np.random.normal(0, 0.03 * s, x.shape)
        out = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    if "glare" in st.active:
        mask = np.zeros((h, w), np.float32)
        r = int(w * (0.03 + 0.07 * s))
        for cx, cy in ((eyes[0] - r * 1.2, eyes[1]), (eyes[0] + r * 1.2, eyes[1])):
            cv2.circle(mask, (int(cx), int(cy)), r, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(r * 0.25, 1))[..., None]
        out = (out * (1 - mask) + 255 * mask).astype(np.uint8)
    if "occlusion" in st.active:                     # hand / mask over the mouth and nose
        bw, bh = int(w * 0.30 * (0.5 + s)), int(h * 0.22 * (0.5 + s))
        x0, y0 = int(mouth[0] - bw / 2), int(mouth[1] - bh * 0.6)
        cv2.rectangle(out, (x0, y0), (x0 + bw, y0 + bh), (90, 120, 160), -1)
    if "facelost" in st.active:
        out[:] = 0
    return out
