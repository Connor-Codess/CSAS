"""
test_transform.py — Unit tests for transform.py.

All tests are purely mathematical — no images needed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import pytest

from transform import (
    transform_stone, transform_all_stones, is_in_house,
    BUTTON_MODEL_X, BUTTON_TOP_Y, BUTTON_BOTTOM_Y, HOUSE_RADIUS_MODEL,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CROP_INFO = {
    'button_cx':   250.0,
    'button_cy':   250.0,
    'ring_radius': 200.0,
    'x0': 0,
    'y0': 0,
}

# scale = 250.0 / 200.0 = 1.25


# ---------------------------------------------------------------------------
# transform_stone
# ---------------------------------------------------------------------------

class TestTransformStone:
    def test_button_pixel_maps_to_model_button_top(self):
        """A stone at the button pixel position should map to (750, BUTTON_TOP_Y)."""
        result = transform_stone(250, 250, CROP_INFO, sheet_end='top')
        assert abs(result['model_x'] - BUTTON_MODEL_X) < 0.01
        assert abs(result['model_y'] - BUTTON_TOP_Y)   < 0.01
        assert abs(result['distance_to_button']) < 0.01

    def test_button_pixel_maps_to_model_button_bottom(self):
        """Same pixel position, bottom end → maps to (750, BUTTON_BOTTOM_Y)."""
        result = transform_stone(250, 250, CROP_INFO, sheet_end='bottom')
        assert abs(result['model_x'] - BUTTON_MODEL_X) < 0.01
        assert abs(result['model_y'] - BUTTON_BOTTOM_Y) < 0.01
        assert abs(result['distance_to_button']) < 0.01

    def test_ring_edge_maps_to_distance_250(self):
        """
        A stone at ring_radius pixels to the right of button should map to
        distance_to_button = HOUSE_RADIUS_MODEL (250 units).
        """
        # button_cx=250, ring_radius=200 → stone at pixel x=450
        result = transform_stone(450, 250, CROP_INFO, sheet_end='top')
        assert abs(result['distance_to_button'] - HOUSE_RADIUS_MODEL) < 0.5

    def test_bottom_end_y_flip(self):
        """
        For the bottom end, a stone above the button pixel (lower y value)
        should map to a model_y HIGHER than BUTTON_BOTTOM_Y (further from viewer).
        """
        # stone at pixel y=200 (above button at 250) → dy = -50
        # bottom end: model_y = BUTTON_BOTTOM_Y - (-50 * scale) = BUTTON_BOTTOM_Y + 62.5
        result = transform_stone(250, 200, CROP_INFO, sheet_end='bottom')
        assert result['model_y'] > BUTTON_BOTTOM_Y, \
            f"Expected model_y > {BUTTON_BOTTOM_Y}, got {result['model_y']:.1f}"

    def test_scale_factor_applied_correctly(self):
        """
        dx=100 pixels at scale=1.25 should give model_x = 750 + 125.
        """
        result = transform_stone(350, 250, CROP_INFO, sheet_end='top')
        expected_model_x = BUTTON_MODEL_X + 100 * (HOUSE_RADIUS_MODEL / CROP_INFO['ring_radius'])
        assert abs(result['model_x'] - expected_model_x) < 0.01


# ---------------------------------------------------------------------------
# transform_all_stones
# ---------------------------------------------------------------------------

class TestTransformAllStones:
    def test_updates_all_stones_in_list(self):
        stones = [
            {'team': 1, 'pixel_x': 250, 'pixel_y': 250, 'radius': 20, 'warnings': []},
            {'team': 2, 'pixel_x': 350, 'pixel_y': 250, 'radius': 20, 'warnings': []},
        ]
        result = transform_all_stones(stones, CROP_INFO, sheet_end='top')
        for stone in result:
            assert 'model_x' in stone
            assert 'model_y' in stone
            assert 'distance_to_button' in stone

    def test_returns_same_list_object(self):
        stones = [
            {'team': 1, 'pixel_x': 250, 'pixel_y': 250, 'radius': 20, 'warnings': []},
        ]
        result = transform_all_stones(stones, CROP_INFO)
        assert result is stones  # In-place update


# ---------------------------------------------------------------------------
# is_in_house
# ---------------------------------------------------------------------------

class TestIsInHouse:
    def test_stone_at_button_is_in_house(self):
        stone = {'distance_to_button': 0.0}
        assert is_in_house(stone)

    def test_stone_at_ring_edge_is_in_house(self):
        stone = {'distance_to_button': 250.0}
        assert is_in_house(stone)

    def test_stone_outside_ring_is_not_in_house(self):
        stone = {'distance_to_button': 251.0}
        assert not is_in_house(stone)

    def test_stone_without_distance_is_not_in_house(self):
        stone = {}
        assert not is_in_house(stone)
