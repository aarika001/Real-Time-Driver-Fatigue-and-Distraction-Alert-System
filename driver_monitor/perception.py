"""MediaPipe Face Landmarker -> Percept (EAR, MAR, head pose, quality signals).

This is the real B1 front-end. R4 (eye CNN) and R5 (head pose) refine pieces of it.
"""
import math
import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config
import eye_model
from interfaces import Percept

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")


def ensure_model() -> str:
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("Downloading face_landmarker.task (~4 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def _d(a, b):
    return float(np.linalg.norm(a - b))


def _ear(pts, idx):
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    return (_d(p2, p6) + _d(p3, p5)) / (2.0 * _d(p1, p4) + 1e-6)


def _mar(pts):
    v = sum(_d(pts[a], pts[b]) for a, b in config.LM_MOUTH_V) / len(config.LM_MOUTH_V)
    return v / (_d(pts[config.LM_MOUTH_H[0]], pts[config.LM_MOUTH_H[1]]) + 1e-6)


def _wrap(a):
    # Canonical-face transforms can put pitch near +/-180; fold it back near 0.
    if abs(a) > 90:
        a -= math.copysign(180.0, a)
    return a


def head_angles(matrix):
    """(yaw, pitch) in degrees from MediaPipe's 4x4 facial transformation matrix.
    VERIFY the signs on your webcam and flip YAW_SIGN / PITCH_SIGN in config.py if needed."""
    R = np.array(matrix, dtype=np.float64)[:3, :3]
    R = R / (np.linalg.norm(R, axis=0, keepdims=True) + 1e-9)
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])))
    return config.YAW_SIGN * _wrap(yaw), config.PITCH_SIGN * _wrap(pitch)


def _box(pts_sub, shape, margin):
    h, w = shape
    x0, y0 = pts_sub.min(axis=0)
    x1, y1 = pts_sub.max(axis=0)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    xa, xb = int(max(x0 - mx, 0)), int(min(x1 + mx, w))
    ya, yb = int(max(y0 - my, 0)), int(min(y1 + my, h))
    return xa, ya, xb, yb


class Perception:
    def __init__(self):
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1
        self._prev_eye = None

    # ------------------------------------------------------------------
    def process(self, frame_bgr, t: float) -> Percept:
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ts = max(int(t * 1000), self._last_ts + 1)      # MediaPipe needs strictly increasing ms
        self._last_ts = ts
        res = self.lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        if not res.face_landmarks:
            self._prev_eye = None
            return Percept(t=t, face_found=False, quality=self._quality(gray, None))

        pts = np.array([[l.x * w, l.y * h] for l in res.face_landmarks[0]], dtype=np.float32)
        ear = 0.5 * (_ear(pts, config.LM_R_EYE) + _ear(pts, config.LM_L_EYE))
        mar = _mar(pts)
        yaw = pitch = 0.0
        if res.facial_transformation_matrixes:
            yaw, pitch = head_angles(res.facial_transformation_matrixes[0])
        crops = (self._crop(gray, pts, config.LM_L_EYE), self._crop(gray, pts, config.LM_R_EYE))
        return Percept(
            t=t, face_found=True,
            p_eye_closed=eye_model.predict_eye_closed(crops, ear),
            ear=ear, mar=mar, yaw=yaw, pitch=pitch,
            quality=self._quality(gray, pts), landmarks_px=pts, eye_crops=crops,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _crop(gray, pts, idx, size=64):
        xa, ya, xb, yb = _box(pts[idx], gray.shape, 0.6)
        if xb - xa < 4 or yb - ya < 4:
            return None
        return cv2.resize(gray[ya:yb, xa:xb], (size, size))

    def _quality(self, gray, pts):
        """All outputs in [0, 1]. These feed the reliability estimator (R7)."""
        if pts is None:
            region, face, jitter = gray, gray, 1.0
        else:
            eye_idx = config.LM_R_EYE + config.LM_L_EYE
            xa, ya, xb, yb = _box(pts[eye_idx], gray.shape, 0.5)
            region = gray[ya:yb, xa:xb]
            fx0, fy0, fx1, fy1 = _box(pts, gray.shape, 0.0)
            face = gray[fy0:fy1, fx0:fx1]
            if region.size == 0 or face.size == 0:
                region, face = gray, gray
            # landmark jitter: motion of the eye points after removing global translation
            e = pts[eye_idx]
            rel = e - e.mean(axis=0)
            iod = _d(pts[33], pts[263]) + 1e-6
            jitter = 0.0 if self._prev_eye is None else float(
                np.abs(rel - self._prev_eye).mean() / iod / 0.03)
            self._prev_eye = rel
        face_small = cv2.resize(face, (128, 128))
        return {
            "brightness": float(region.mean() / 255.0),
            "contrast": float(min(region.std() / 80.0, 1.0)),
            "glare_frac": float((region >= 245).mean()),
            "sharpness": float(min(cv2.Laplacian(face_small, cv2.CV_32F).var() / 500.0, 1.0)),
            "jitter": float(min(jitter, 1.0)),
            "det_conf": 1.0 if pts is not None else 0.0,
        }
