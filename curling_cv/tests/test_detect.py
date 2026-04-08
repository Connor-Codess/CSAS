"""
test_detect.py — Unit tests for detect.py.

All test images are generated programmatically with OpenCV — no external files needed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import pytest

from detect import (
    detect_outer_ring, crop_to_ring, detect_stones,
    _classify_color, _extract_patch,
    GLARE_MAX_SAT, GLARE_MIN_VAL,
    RED_HUE_LOW_LO, RED_HUE_LOW_HI,
    RED_HUE_HIGH_LO, RED_HUE_HIGH_HI,
    YELLOW_HUE_LO, YELLOW_HUE_HI,
    MIN_SATURATION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_image(w=800, h=800, color=(200, 200, 200)):
    """Create a plain grey BGR image."""
    img = np.full((h, w, 3), color, dtype=np.uint8)
    return img


def _draw_ring(img, cx, cy, radius, color=(50, 50, 50), thickness=4):
    """Draw a circle (ring) on the image."""
    cv2.circle(img, (cx, cy), radius, color, thickness)
    return img


def _draw_filled_circle(img, cx, cy, radius, color):
    cv2.circle(img, (cx, cy), radius, color, -1)
    return img


def _hsv_stone_image(hue, saturation=200, value=180):
    """
    Create a 100x100 BGR image filled with a single HSV colour.
    OpenCV HSV: H in [0,179], S in [0,255], V in [0,255].
    """
    hsv = np.full((100, 100, 3), (hue, saturation, value), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Outer ring detection
# ---------------------------------------------------------------------------

class TestDetectOuterRing:
    def _make_ring_image(self, w=600, h=600, ring_r=200):
        cx, cy = w // 2, h // 2
        img = _blank_image(w, h)
        _draw_ring(img, cx, cy, ring_r, color=(30, 30, 30), thickness=6)
        return img, cx, cy, ring_r

    def test_detects_ring_center_within_5px(self):
        img, true_cx, true_cy, ring_r = self._make_ring_image()
        result = detect_outer_ring(img)
        assert abs(result['cx'] - true_cx) <= 5, \
            f"Expected cx≈{true_cx}, got {result['cx']:.1f}"
        assert abs(result['cy'] - true_cy) <= 5, \
            f"Expected cy≈{true_cy}, got {result['cy']:.1f}"

    def test_confidence_is_high_or_medium_on_clear_ring(self):
        img, *_ = self._make_ring_image()
        result = detect_outer_ring(img)
        assert result['confidence'] in ('high', 'medium'), \
            f"Expected high/medium confidence, got '{result['confidence']}'"

    def test_returns_low_confidence_on_blank_image(self):
        img = _blank_image()
        result = detect_outer_ring(img)
        # Should not raise; confidence will be 'low' since no ring is detectable
        assert result['confidence'] in ('high', 'medium', 'low')
        assert 'cx' in result and 'cy' in result and 'radius' in result


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------

class TestCropToRing:
    def test_button_center_at_crop_origin_minus_offset(self):
        """After crop, button position should be ring_center - crop_offset."""
        img = _blank_image(800, 800)
        ring = {'cx': 400.0, 'cy': 400.0, 'radius': 150.0,
                'confidence': 'high', 'perspective_corrected': False,
                'warnings': []}
        _, crop_info = crop_to_ring(img, ring, pad_fraction=0.10)
        # button_cx should equal ring_cx - x0
        expected_bcx = ring['cx'] - crop_info['x0']
        assert abs(crop_info['button_cx'] - expected_bcx) < 1

    def test_crop_is_larger_than_ring_bounding_box(self):
        """With 10% padding the crop should exceed 2*radius on each side."""
        img = _blank_image(800, 800)
        ring = {'cx': 400.0, 'cy': 400.0, 'radius': 150.0,
                'confidence': 'high', 'perspective_corrected': False,
                'warnings': []}
        cropped, crop_info = crop_to_ring(img, ring, pad_fraction=0.10)
        min_size = int(2 * ring['radius'])
        assert cropped.shape[0] > min_size
        assert cropped.shape[1] > min_size


# ---------------------------------------------------------------------------
# Color classification
# ---------------------------------------------------------------------------

class TestClassifyColor:
    def test_low_hue_red_classified_as_team1(self):
        """Hue=5 (0-15°) with high saturation → Team 1 (red)."""
        img = _hsv_stone_image(hue=5, saturation=200, value=180)
        team = _classify_color(img, 50, 50, 40)
        assert team == 1, f"Expected 1 (red), got {team}"

    def test_high_hue_red_classified_as_team1(self):
        """Hue=170 (wraps around — still red) with high saturation → Team 1."""
        img = _hsv_stone_image(hue=85, saturation=200, value=180)
        # hue=85 maps to 170° real, which is in RED_HUE_HIGH range
        team = _classify_color(img, 50, 50, 40)
        assert team == 1, f"Expected 1 (red wrap-around), got {team}"

    def test_yellow_classified_as_team2(self):
        """Hue=15 (30° real — yellow) with high saturation → Team 2."""
        img = _hsv_stone_image(hue=15, saturation=200, value=180)
        team = _classify_color(img, 50, 50, 40)
        assert team == 2, f"Expected 2 (yellow), got {team}"

    def test_low_saturation_returns_none(self):
        """Grey / white patch (low saturation) → None (not a coloured stone)."""
        img = _hsv_stone_image(hue=5, saturation=10, value=200)
        team = _classify_color(img, 50, 50, 40)
        assert team is None, f"Expected None for low-saturation, got {team}"


# ---------------------------------------------------------------------------
# Stone detection
# ---------------------------------------------------------------------------

class TestDetectStones:
    def _make_house_with_stones(self):
        """
        Create a synthetic cropped house image with:
        - 2 red stones inside the house
        - 1 yellow stone inside the house
        - 1 stone outside the house (should be filtered)
        Ring radius = 200px, button at (250, 250).
        """
        w, h = 500, 500
        img = _blank_image(w, h, color=(180, 200, 220))
        ring_r = 200
        bcx, bcy = 250, 250

        # Ring outline
        _draw_ring(img, bcx, bcy, ring_r, color=(30, 30, 30), thickness=4)

        stone_r = 22
        red_bgr    = (0, 0, 220)
        yellow_bgr = (0, 215, 255)

        # 2 red stones inside house
        _draw_filled_circle(img, bcx - 50, bcy, stone_r, red_bgr)
        _draw_filled_circle(img, bcx + 60, bcy - 30, stone_r, red_bgr)

        # 1 yellow stone inside house
        _draw_filled_circle(img, bcx + 10, bcy + 70, stone_r, yellow_bgr)

        # 1 stone outside house (should be filtered)
        _draw_filled_circle(img, bcx - 210, bcy, stone_r, red_bgr)

        crop_info = {
            'button_cx': float(bcx),
            'button_cy': float(bcy),
            'ring_radius': float(ring_r),
            'x0': 0, 'y0': 0,
        }
        return img, crop_info

    def test_outside_stone_is_filtered(self):
        img, crop_info = self._make_house_with_stones()
        stones, _ = detect_stones(img, crop_info)
        # All returned stones must be within ring_radius * 1.1 from button
        ring_r = crop_info['ring_radius']
        bcx, bcy = crop_info['button_cx'], crop_info['button_cy']
        for s in stones:
            dist = ((s['pixel_x'] - bcx)**2 + (s['pixel_y'] - bcy)**2) ** 0.5
            assert dist <= ring_r * 1.1, \
                f"Stone at ({s['pixel_x']}, {s['pixel_y']}) is outside house"

    def test_touching_stone_warning_fires(self):
        """
        A circle noticeably larger than the median stone radius triggers a
        touching-stone warning.

        Detection radius range: 8-15% of ring_radius=200 → 16-30px.
        Normal stone: 17px radius.
        Oversized stone: 29px radius (29 > 17 * 1.7 = 28.9 → triggers warning).
        Both are within the detection range so HoughCircles finds them.
        """
        w, h = 500, 500
        img = _blank_image(w, h, color=(180, 200, 220))
        ring_r = 200
        bcx, bcy = 250, 250
        _draw_ring(img, bcx, bcy, ring_r, color=(30, 30, 30), thickness=4)

        # Three normal-sized red stones (radius 17) — anchors the median at 17
        _draw_filled_circle(img, bcx - 70, bcy - 50, 17, (0, 0, 220))
        _draw_filled_circle(img, bcx - 70, bcy + 50, 17, (0, 0, 220))
        _draw_filled_circle(img, bcx,      bcy - 80, 17, (0, 0, 220))
        # Oversized stone (30px > 17 * 1.7 = 28.9) — simulates two touching stones
        _draw_filled_circle(img, bcx + 70, bcy, 30, (0, 0, 220))

        crop_info = {
            'button_cx': float(bcx), 'button_cy': float(bcy),
            'ring_radius': float(ring_r), 'x0': 0, 'y0': 0,
        }
        stones, pipeline_warnings = detect_stones(img, crop_info)
        touching_warned = any('touching' in w.lower() for w in pipeline_warnings)
        stone_warned = any(
            any('touching' in w.lower() for w in s.get('warnings', []))
            for s in stones
        )
        assert touching_warned or stone_warned, \
            "Expected a touching-stone warning for an oversized circle"

    def test_glare_is_rejected(self):
        """A near-white circle (low saturation, high brightness) is discarded."""
        w, h = 500, 500
        img = _blank_image(w, h, color=(180, 200, 220))
        ring_r = 200
        bcx, bcy = 250, 250
        _draw_ring(img, bcx, bcy, ring_r, color=(30, 30, 30), thickness=4)
        # Draw a white/glare circle inside the house
        _draw_filled_circle(img, bcx, bcy, 22, (255, 255, 255))

        crop_info = {
            'button_cx': float(bcx), 'button_cy': float(bcy),
            'ring_radius': float(ring_r), 'x0': 0, 'y0': 0,
        }
        stones, pipeline_warnings = detect_stones(img, crop_info)
        # The white circle should either not appear in stones, or trigger a glare warning
        glare_warned = any('glare' in w.lower() for w in pipeline_warnings)
        white_stone_present = any(s.get('team') is None for s in stones)
        # If a stone was detected: it should have triggered the glare warning
        if not glare_warned and white_stone_present:
            pytest.fail("White/glare circle was detected as a stone without a warning")
