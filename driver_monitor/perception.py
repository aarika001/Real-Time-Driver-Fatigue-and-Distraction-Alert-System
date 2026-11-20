"""MediaPipe Face Landmarker -> Percept (R3/R4/R5-aligned cues and quality signals)."""
import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config
import eye_model
from cues.eye_crop import crop_eye
from interfaces import Percept

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_MODEL = os.path.abspath(os.path.join(HERE, "..", "face_landmarker.task"))
LOCAL_MODEL = os.path.join(HERE, "models", "face_landmarker.task")

# Exact R3 / R4 / R5 landmark indices and 3D head model
MOUTH_IDXS = [61, 291, 39, 181, 0, 17, 269, 405]
LEFT_EYE_IDXS = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE_IDXS = [362, 263, 387, 386, 385, 373, 374, 380]
POSE_2D_IDXS = [1, 152, 33, 263, 61, 291]
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (1)
    (0.0, -330.0, -65.0),        # Chin (152)
    (-225.0, 170.0, -135.0),     # Left eye corner (33)
    (225.0, 170.0, -135.0),      # Right eye corner (263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (61)
    (150.0, -150.0, -125.0),     # Right mouth corner (291)
], dtype=np.float64)


def ensure_model() -> str:
    if os.path.isfile(ROOT_MODEL):
        return ROOT_MODEL
    if not os.path.exists(LOCAL_MODEL):
        os.makedirs(os.path.dirname(LOCAL_MODEL), exist_ok=True)
        print("Downloading face_landmarker.task (~4 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, LOCAL_MODEL)
    return LOCAL_MODEL


def _d(a, b):
    return float(np.linalg.norm(a - b))


def _ear(pts, idx):
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    return (_d(p2, p6) + _d(p3, p5)) / (2.0 * _d(p1, p4) + 1e-6)


def _mar_r3(pts):
    """Exact outer-lip MAR from RLDD_preprocess/preprocess-rldd.py."""
    v1 = _d(pts[MOUTH_IDXS[2]], pts[MOUTH_IDXS[3]])
    v2 = _d(pts[MOUTH_IDXS[4]], pts[MOUTH_IDXS[5]])
    v3 = _d(pts[MOUTH_IDXS[6]], pts[MOUTH_IDXS[7]])
    horiz = _d(pts[MOUTH_IDXS[0]], pts[MOUTH_IDXS[1]])
    return float((v1 + v2 + v3) / (3.0 * horiz + 1e-6))


def _head_pose_pnp(pts, w: int, h: int):
    """Exact solvePnP (pitch, yaw, roll) matching R3 preprocess-rldd.py and R5 head_pose_core.py."""
    pts_2d = np.array([pts[i] for i in POSE_2D_IDXS], dtype=np.float64)
    focal = float(w)
    cam = np.array([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ok, rot_vec, _ = cv2.solvePnP(FACE_3D_MODEL, pts_2d, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return float(angles[0]), float(angles[1]), float(angles[2])


class Perception:
    def __init__(self):
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=False,
        )
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1
        self._prev_eye = None

    def process(self, frame_bgr, t: float) -> Percept:
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ts = max(int(t * 1000), self._last_ts + 1)
        self._last_ts = ts
        res = self.lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        if not res.face_landmarks:
            self._prev_eye = None
            return Percept(
                t=t, face_found=False,
                quality=self._quality_no_face(gray),
                raw_quality=(0.0, 0.0, 1.0, 0.0),
            )

        pts = np.array([[l.x * w, l.y * h] for l in res.face_landmarks[0]], dtype=np.float32)
        ear = 0.5 * (_ear(pts, config.LM_R_EYE) + _ear(pts, config.LM_L_EYE))
        mar = _mar_r3(pts)
        pitch, yaw, roll = _head_pose_pnp(pts, w, h)

        cl, ql, vl = crop_eye(gray, pts[LEFT_EYE_IDXS])
        cr, qr, vr = crop_eye(gray, pts[RIGHT_EYE_IDXS])
        raw_q = tuple((a + b) / 2.0 for a, b in zip(ql.as_list(), qr.as_list()))
        crops = (cl if vl else None, cr if vr else None)

        return Percept(
            t=t, face_found=True,
            p_eye_closed=eye_model.predict_eye_closed(crops, ear),
            ear=ear, mar=mar, yaw=yaw, pitch=pitch, roll=roll,
            quality=self._quality_from_eye(raw_q, pts),
            landmarks_px=pts, eye_crops=crops, raw_quality=raw_q,
        )

    def _quality_no_face(self, gray):
        small = cv2.resize(gray, (128, 128))
        return {
            "brightness": float(gray.mean() / 255.0),
            "contrast": float(min(gray.std() / 80.0, 1.0)),
            "glare_frac": float((gray >= 245).mean()),
            "sharpness": float(min(cv2.Laplacian(small, cv2.CV_32F).var() / 500.0, 1.0)),
            "jitter": 1.0,
            "det_conf": 0.0,
        }

    def _quality_from_eye(self, raw_q, pts):
        eye_idx = config.LM_R_EYE + config.LM_L_EYE
        e = pts[eye_idx]
        rel = e - e.mean(axis=0)
        iod = _d(pts[33], pts[263]) + 1e-6
        jitter = 0.0 if self._prev_eye is None else float(
            np.abs(rel - self._prev_eye).mean() / iod / 0.03)
        self._prev_eye = rel
        return {
            "brightness": float(raw_q[0]),
            "contrast": float(min((raw_q[1] * 255.0) / 80.0, 1.0)),
            "glare_frac": float(raw_q[2]),
            "sharpness": float(min(raw_q[3] / 500.0, 1.0)),
            "jitter": float(min(jitter, 1.0)),
            "det_conf": 1.0,
        }