"""
transform.py — Map cropped-image pixel positions into the Stones.csv coordinate space.

Stones.csv coordinate system:
  - Button (top end):    (750, 650)
  - Button (bottom end): (750, 1916)
  - House radius:        250.0 units
  - hammer_closest_dist training range: 4 – 1097 units

All distances output by this module are in those same units so they feed
directly into the trained XGBoost models without any scaling.
"""

import math

# Stones.csv coordinate anchors
BUTTON_MODEL_X      = 750.0
BUTTON_TOP_Y        = 650.0
BUTTON_BOTTOM_Y     = 1916.0
HOUSE_RADIUS_MODEL  = 250.0  # model-space units


def transform_stone(pixel_x, pixel_y, crop_info, sheet_end='top'):
    """
    Convert a stone's position in the cropped image to Stones.csv model coordinates.

    Args:
        pixel_x, pixel_y : stone center in cropped-image pixels
        crop_info        : dict from detect.crop_to_ring()
                           keys: button_cx, button_cy, ring_radius
        sheet_end        : 'top' or 'bottom' — which house is shown

    Returns:
        dict with model_x, model_y, distance_to_button
    """
    bcx = crop_info['button_cx']
    bcy = crop_info['button_cy']
    ring_r = crop_info['ring_radius']

    # Scale factor: outer ring pixel radius → 250 model units
    scale = HOUSE_RADIUS_MODEL / ring_r

    # Offset from button in cropped-image pixels
    dx = pixel_x - bcx
    dy = pixel_y - bcy

    button_model_y = BUTTON_TOP_Y if sheet_end == 'top' else BUTTON_BOTTOM_Y

    model_x = BUTTON_MODEL_X + dx * scale

    # Bottom end: camera Y increases downward but model Y should increase
    # toward the far end of the sheet, so flip the Y axis.
    if sheet_end == 'top':
        model_y = button_model_y + dy * scale
    else:
        model_y = button_model_y - dy * scale

    dist = math.sqrt((model_x - BUTTON_MODEL_X) ** 2 + (model_y - button_model_y) ** 2)

    return {
        'model_x': model_x,
        'model_y': model_y,
        'distance_to_button': dist,
    }


def transform_all_stones(stones, crop_info, sheet_end='top'):
    """
    Apply transform_stone to every stone in the list.
    Adds model_x, model_y, distance_to_button keys to each stone dict in-place.
    Returns the updated list.
    """
    for stone in stones:
        coords = transform_stone(
            stone['pixel_x'], stone['pixel_y'],
            crop_info, sheet_end
        )
        stone.update(coords)
    return stones


def is_in_house(stone):
    """Return True if the stone's distance_to_button is within the house (<=250 model units)."""
    return stone.get('distance_to_button', 999) <= HOUSE_RADIUS_MODEL
