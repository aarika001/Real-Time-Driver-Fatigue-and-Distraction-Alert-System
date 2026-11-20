"""Thin wrapper around MediaPipe Face Landmarker (Tasks API).

Used by R4's offline scripts and the live pipeline so landmark detection is
set up the same way everywhere. Returns landmarks as a (478, 3) pixel array
(see ``landmarks.landmarks_to_array``) or None when no face is found.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .landmarks import landmarks_to_array

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "face_landmarker.task"


class FaceLandmarker:
    """Single-face landmarker.

    mode="video": uses MediaPipe's tracking between frames (faster, smoother);
        timestamps passed to ``detect`` must strictly increase.
    mode="image": every frame is independent (what R3's RLDD script uses).
    """

    def __init__(self, model_path: Optional[str | Path] = None, mode: str = "video",
                 min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model not found: {path}. It is in the repo root as face_landmarker.task.")
        if mode not in ("video", "image"):
            raise ValueError("mode must be 'video' or 'image'")

        self._mp = mp
        self.mode = mode
        running_mode = vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=running_mode,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: Optional[int] = None) -> Optional[np.ndarray]:
        """Landmarks for one RGB uint8 frame, as (478, 3) pixels, or None."""
        h, w = rgb_frame.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(rgb_frame))
        if self.mode == "video":
            if timestamp_ms is None:
                raise ValueError("timestamp_ms is required in video mode")
            ts = max(int(timestamp_ms), self._last_ts + 1)   # MediaPipe needs strictly increasing
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        return landmarks_to_array(result.face_landmarks[0], w, h)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceLandmarker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
