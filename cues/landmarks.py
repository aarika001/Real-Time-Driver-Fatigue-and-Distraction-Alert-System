"""MediaPipe Face Landmarker index contract for R4 (eye and mouth cues).

Every module in the team that needs eye or mouth landmarks should import the
indices from here, so there is exactly one definition in the codebase.

Conventions
-----------
* MediaPipe Face Landmarker (Tasks API) returns 478 landmarks per face:
  468 face-mesh points + 10 iris points (468-477).
* "IMG_LEFT" / "IMG_RIGHT" mean the eye that appears on the left / right side
  of the (un-mirrored) image. IMG_LEFT is anatomically the subject's RIGHT eye.
  R3's RLDD preprocessing calls IMG_LEFT "left_eye" -- same points, same order.
* Landmarks arrive normalised to [0, 1]. ``landmarks_to_array`` converts them to
  pixel units so that distances are not distorted by a non-square frame.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

NUM_LANDMARKS: int = 478

# --- Eye contours used for cropping (identical to R3's LEFT/RIGHT_EYE_IDXS) --
# order: outer/inner corner, 3 upper-lid points, 3 lower-lid points
EYE_IMG_LEFT: tuple[int, ...] = (33, 133, 160, 159, 158, 144, 145, 153)
EYE_IMG_RIGHT: tuple[int, ...] = (362, 263, 387, 386, 385, 373, 374, 380)

# --- Six-point eye aspect ratio (Soukupova & Cech 2016), order p1..p6 -------
# p1, p4 = horizontal corners; (p2, p6) and (p3, p5) = vertical pairs
EAR_IMG_LEFT: tuple[int, ...] = (33, 160, 158, 133, 153, 144)
EAR_IMG_RIGHT: tuple[int, ...] = (362, 385, 387, 263, 373, 380)

# --- Mouth ----------------------------------------------------------------
# Outer lip, order: left corner, right corner, then 3 (upper, lower) pairs.
# Identical to R3's MOUTH_IDXS so values match the processed RLDD archives.
MOUTH_OUTER: tuple[int, ...] = (61, 291, 39, 181, 0, 17, 269, 405)
# Inner lip (the actual opening, excludes lip thickness), same ordering.
MOUTH_INNER: tuple[int, ...] = (78, 308, 82, 87, 13, 14, 312, 317)

# Outer eye corners: inter-ocular distance is a stable face-scale reference.
INTEROCULAR: tuple[int, int] = (33, 263)


def landmarks_to_array(landmarks: Sequence, width: int, height: int) -> np.ndarray:
    """Convert one face's MediaPipe landmarks to a (478, 3) float32 pixel array.

    Args:
        landmarks: ``result.face_landmarks[0]`` from the Tasks API -- a sequence
            of objects with normalised ``.x``, ``.y``, ``.z`` attributes.
        width: frame width in pixels.
        height: frame height in pixels.

    Returns:
        Array of shape (478, 3): columns are x_px, y_px, z. MediaPipe's z uses
        roughly the same scale as x, so it is multiplied by ``width`` to keep
        3-D distances consistent.
    """
    if len(landmarks) < NUM_LANDMARKS:
        raise ValueError(f"Expected {NUM_LANDMARKS} landmarks, got {len(landmarks)}")
    arr = np.array([(p.x, p.y, p.z) for p in landmarks[:NUM_LANDMARKS]], dtype=np.float32)
    arr[:, 0] *= width
    arr[:, 1] *= height
    arr[:, 2] *= width
    return arr
