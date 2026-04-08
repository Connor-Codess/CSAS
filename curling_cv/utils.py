"""
utils.py — Annotated detection image drawing for the confirmation step.

The user sees this image before trusting any model output, so they can
visually verify stone positions and detection quality.
"""

import cv2
import numpy as np


def draw_detections(cropped_image, crop_info, stones, warnings=None):
    """
    Draw detected outer ring, button, and stones onto the cropped image.

    Args:
        cropped_image : numpy array (BGR) — the padded crop from detect.crop_to_ring()
        crop_info     : dict from detect.crop_to_ring() (button_cx, button_cy, ring_radius)
        stones        : list of stone dicts with pixel_x, pixel_y, radius, team, warnings
        warnings      : list of pipeline-level warning strings (shown as text overlay)

    Returns:
        annotated numpy array (BGR)
    """
    out = cropped_image.copy()
    bcx = int(crop_info['button_cx'])
    bcy = int(crop_info['button_cy'])
    r   = int(crop_info['ring_radius'])

    # Outer ring
    cv2.circle(out, (bcx, bcy), r, (255, 255, 255), 2)

    # Button centre — white disc + crosshair so coaches can see what to drag
    cv2.circle(out, (bcx, bcy), 10, (0, 0, 0), 2)        # outer border
    cv2.circle(out, (bcx, bcy), 10, (255, 255, 255), -1)  # white fill
    cv2.line(out, (bcx - 16, bcy), (bcx + 16, bcy), (0, 0, 0), 2)
    cv2.line(out, (bcx, bcy - 16), (bcx, bcy + 16), (0, 0, 0), 2)
    _put_label(out, "CTR", bcx + 13, bcy - 13, (255, 255, 255), scale=0.42)

    # Stones
    for stone in stones:
        cx   = int(stone['pixel_x'])
        cy   = int(stone['pixel_y'])
        sr   = int(stone['radius'])
        team = stone.get('team')
        sw   = stone.get('warnings', [])

        if team == 1:
            color = (0, 0, 220)       # Red (BGR)
            label_color = (0, 0, 180)
        elif team == 2:
            color = (0, 200, 220)     # Yellow (BGR)
            label_color = (0, 150, 180)
        else:
            color = (160, 160, 160)   # Grey = unknown
            label_color = (100, 100, 100)

        # Touching stone warning → orange dashed effect (thicker ring)
        is_touching = any('touching' in w.lower() for w in sw)
        thickness = 3 if is_touching else 2
        cv2.circle(out, (cx, cy), sr, color, thickness)

        if is_touching:
            # Draw a second slightly larger circle in orange for the warning ring
            cv2.circle(out, (cx, cy), sr + 4, (0, 140, 255), 1)

        # Distance label
        dist = stone.get('distance_to_button')
        if dist is not None:
            label = f"{dist:.0f}u"
        elif team is None:
            label = "?"
        else:
            label = f"T{team}"

        _put_label(out, label, cx + sr + 4, cy - 4, label_color)

        # Warning symbol for touching stones
        if is_touching:
            _put_label(out, "! touching?", cx, cy + sr + 14, (0, 100, 220), scale=0.45)

        # Unknown team symbol
        if team is None:
            _put_label(out, "?", cx - 6, cy + 5, (80, 80, 80), scale=0.8, thickness=2)

    # Pipeline warning overlay (top-left)
    if warnings:
        y_offset = 18
        for w in warnings[:5]:  # Cap at 5 lines to avoid overflow
            text = w[:80]  # Truncate long lines
            cv2.putText(out, text, (8, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, text, (8, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (0, 220, 255), 1, cv2.LINE_AA)
            y_offset += 16

    return out


def _put_label(image, text, x, y, color, scale=0.5, thickness=1):
    """Draw text with a thin black shadow for readability on any background."""
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)
