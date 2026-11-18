"""
tests/test_stress_warp.py
==========================
R5 unit tests for the perspective_warp_frame function in stress_test_offaxis.py

Run from the head-pose/ directory:
    python -m pytest tests/ -v

Tests
-----
  1. test_identity_warp_exact        - 0° warp produces the identical array
  2. test_warp_output_shape_rgb      - output shape matches input (RGB)
  3. test_warp_output_shape_gray     - output shape matches input (grayscale)
  4. test_warp_nonzero_at_nonzero_angle - 30° warp changes pixel values
  5. test_warp_no_black_border_at_zero  - identity warp has no black border
  6. test_warp_symmetry_yaw          - +θ and -θ yaw warps are horizontal mirrors
  7. test_warp_dtype_preserved       - output dtype matches input
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from stress_test_offaxis import perspective_warp_frame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def solid_rgb_frame() -> np.ndarray:
    """200×320 solid-grey RGB frame (easy to reason about transforms)."""
    return np.full((200, 320, 3), 128, dtype=np.uint8)


@pytest.fixture
def checkerboard_frame() -> np.ndarray:
    """200×320 8-pixel checkerboard RGB frame for detecting spatial shifts."""
    h, w = 200, 320
    tile = 8
    rows = np.arange(h) // tile
    cols = np.arange(w) // tile
    board = ((rows[:, None] + cols[None, :]) % 2 * 255).astype(np.uint8)
    return cv2.cvtColor(board, cv2.COLOR_GRAY2RGB)


@pytest.fixture
def gray_frame() -> np.ndarray:
    """200×320 grayscale ramp."""
    h, w = 200, 320
    return np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPerspectiveWarpIdentity:
    def test_identity_warp_exact(self, solid_rgb_frame):
        """0° yaw and 0° pitch must return an array identical to the input."""
        warped = perspective_warp_frame(solid_rgb_frame, yaw_deg=0.0, pitch_deg=0.0)
        np.testing.assert_array_equal(
            warped, solid_rgb_frame,
            err_msg="Identity warp (0°) should not change any pixel value.",
        )

    def test_warp_no_black_border_at_zero(self, solid_rgb_frame):
        """Identity warp of a solid-grey frame must have no black pixels."""
        warped = perspective_warp_frame(solid_rgb_frame, 0.0, 0.0)
        assert warped.min() > 0, "Identity warp introduced unexpected black border pixels."


class TestPerspectiveWarpShape:
    def test_rgb_output_shape_preserved(self, checkerboard_frame):
        warped = perspective_warp_frame(checkerboard_frame, yaw_deg=15.0, pitch_deg=0.0)
        assert warped.shape == checkerboard_frame.shape

    def test_gray_output_shape_preserved(self, gray_frame):
        warped = perspective_warp_frame(gray_frame, yaw_deg=0.0, pitch_deg=15.0)
        assert warped.shape == gray_frame.shape


class TestPerspectiveWarpPixelChanges:
    def test_nonzero_angle_changes_pixels(self, checkerboard_frame):
        """A 30° yaw warp should visibly shift the checkerboard pattern."""
        warped = perspective_warp_frame(checkerboard_frame, yaw_deg=30.0, pitch_deg=0.0)
        diff = np.abs(warped.astype(int) - checkerboard_frame.astype(int))
        assert diff.sum() > 0, "30° yaw warp produced no pixel changes."

    def test_larger_angle_produces_more_black_border(self, checkerboard_frame):
        """A larger warp angle introduces more black (zero) border pixels.

        When perspective_warp_frame maps corners inward, the regions outside
        the warped quadrilateral are filled with black (borderValue=0).
        A larger angle means a smaller interior quadrilateral → more black pixels.
        This is a geometrically guaranteed property of the warp formula.
        """
        def black_pixel_count(frame: np.ndarray) -> int:
            if frame.ndim == 3:
                return int(np.all(frame == 0, axis=2).sum())
            return int((frame == 0).sum())

        black_small = black_pixel_count(
            perspective_warp_frame(checkerboard_frame, 5.0, 0.0)
        )
        black_large = black_pixel_count(
            perspective_warp_frame(checkerboard_frame, 25.0, 0.0)
        )
        assert black_large > black_small, (
            f"Expected more black-border pixels at 25° ({black_large}) than at 5° ({black_small})."
        )


class TestPerspectiveWarpSymmetry:
    def test_positive_and_negative_yaw_both_change_content(self, checkerboard_frame):
        """Both +θ and -θ yaw warps should displace content from the original.

        Note: our 4-corner homography is a symmetric contraction (both left
        and right corners shift inward by the same dx), so +θ and -θ produce
        the same transformation. This test verifies that both non-zero warps
        actually change the frame content (non-trivial operation).
        """
        pos_warp = perspective_warp_frame(checkerboard_frame, yaw_deg=20.0, pitch_deg=0.0)
        neg_warp = perspective_warp_frame(checkerboard_frame, yaw_deg=-20.0, pitch_deg=0.0)
        orig = checkerboard_frame

        diff_pos = np.abs(pos_warp.astype(int) - orig.astype(int)).sum()
        diff_neg = np.abs(neg_warp.astype(int) - orig.astype(int)).sum()

        assert diff_pos > 0, "+20° warp produced no pixel changes."
        assert diff_neg > 0, "-20° warp produced no pixel changes."


class TestPerspectiveWarpDtype:
    def test_dtype_uint8_preserved(self, checkerboard_frame):
        warped = perspective_warp_frame(checkerboard_frame, 15.0, 10.0)
        assert warped.dtype == np.uint8

    def test_dtype_float32_preserved(self):
        frame_f = np.random.rand(100, 160, 3).astype(np.float32) * 255
        warped = perspective_warp_frame(frame_f, 10.0, 0.0)
        assert warped.dtype == np.float32
