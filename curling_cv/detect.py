"""
detect.py — Outer ring detection, image crop, stone detection, color classification.

Coordinate system: all outputs are in cropped-image pixel space.
transform.py maps these into the Stones.csv model coordinate space.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HSV hue ranges (OpenCV hue is 0-179, not 0-360)
RED_HUE_LOW_LO  = 0
RED_HUE_LOW_HI  = 8    # 0-15 degrees
RED_HUE_HIGH_LO = 83   # 165-180 degrees (170/2=85, use 83 for buffer)
RED_HUE_HIGH_HI = 179
YELLOW_HUE_LO   = 10   # 20-50 degrees (orange-yellow stones sit around 38°)
YELLOW_HUE_HI   = 25
MIN_SATURATION   = 80   # Below this → white/grey/glare, not a coloured stone

GLARE_MAX_SAT    = 30
GLARE_MIN_VAL    = 200

TOUCHING_RATIO   = 1.7  # If circle radius > median * this, flag as touching pair


# ---------------------------------------------------------------------------
# Outer ring detection
# ---------------------------------------------------------------------------

def detect_outer_ring(image):
    """
    Detect the outer boundary of the curling house.

    Strategy:
      1. HoughCircles with large radius range (primary)
      2. Canny + largest contour (fallback)

    Also checks for camera tilt: if the detected region is elliptical
    (aspect ratio deviates >10% from 1.0), applies a perspective warp to
    correct it and warns the caller.

    Returns:
        dict with keys:
            cx, cy          — ring center in ORIGINAL image pixels
            radius          — ring radius in ORIGINAL image pixels
            confidence      — 'high' | 'medium' | 'low'
            perspective_corrected — bool
            warnings        — list of warning strings
            corrected_image — perspective-corrected image (may be same as input)
    """
    warnings = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    h, w = image.shape[:2]
    img_center = (w / 2, h / 2)

    ring = None
    confidence = 'low'

    # --- Primary: HoughCircles ---
    min_r = int(w * 0.30)
    max_r = int(w * 0.60)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=w,          # only one ring
        param1=50,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        # Pick circle whose center is closest to image center
        best = min(circles, key=lambda c: (c[0] - img_center[0])**2 + (c[1] - img_center[1])**2)
        ring = {'cx': float(best[0]), 'cy': float(best[1]), 'radius': float(best[2])}
        confidence = 'high'

    # --- Fallback: Canny + largest contour ---
    if ring is None:
        edges = cv2.Canny(blurred, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            (cx, cy), radius = cv2.minEnclosingCircle(largest)
            ring = {'cx': float(cx), 'cy': float(cy), 'radius': float(radius)}
            confidence = 'medium'
            warnings.append("Outer ring detected via contour fallback — verify the annotated image.")

    # --- Last resort: image center ---
    if ring is None:
        ring = {'cx': img_center[0], 'cy': img_center[1], 'radius': min(w, h) * 0.4}
        confidence = 'low'
        warnings.append("Outer ring detection failed — using image center as fallback. Results unreliable.")

    # --- Ellipse / perspective check ---
    perspective_corrected = False
    corrected_image = image.copy()

    if confidence in ('high', 'medium'):
        _, ellipse = _try_fit_ellipse(blurred, ring)
        if ellipse is not None:
            _, (ma, Mi), _ = ellipse
            aspect = min(ma, Mi) / max(ma, Mi) if max(ma, Mi) > 0 else 1.0
            if aspect < 0.90:  # >10% deviation from circle
                warnings.append(
                    f"Camera tilt detected (ellipse aspect {aspect:.2f}). "
                    "Perspective correction applied — verify the annotated image."
                )
                corrected_image, ring = _apply_perspective_correction(image, ellipse, ring)
                perspective_corrected = True

    ring['confidence'] = confidence
    ring['perspective_corrected'] = perspective_corrected
    ring['warnings'] = warnings
    ring['corrected_image'] = corrected_image
    return ring


def _try_fit_ellipse(blurred, ring):
    """Attempt to fit an ellipse to the outer ring region."""
    cx, cy, r = int(ring['cx']), int(ring['cy']), int(ring['radius'])

    mask = np.zeros_like(blurred)
    cv2.circle(mask, (cx, cy), int(r * 1.1), 255, int(r * 0.15))
    edges = cv2.Canny(cv2.bitwise_and(blurred, blurred, mask=mask), 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return None, None

    try:
        ellipse = cv2.fitEllipse(largest)
        return blurred, ellipse
    except cv2.error:
        return None, None


def _apply_perspective_correction(image, ellipse, ring):
    """
    Warp an elliptical house to appear circular using perspective transform.
    Returns (corrected_image, updated_ring).
    """
    (ex, ey), (ma, Mi), _ = ellipse
    h, w = image.shape[:2]

    src_pts = np.float32([
        [ex - ma / 2, ey - Mi / 2],
        [ex + ma / 2, ey - Mi / 2],
        [ex + ma / 2, ey + Mi / 2],
        [ex - ma / 2, ey + Mi / 2],
    ])
    new_r = max(ma, Mi) / 2
    dst_pts = np.float32([
        [ex - new_r, ey - new_r],
        [ex + new_r, ey - new_r],
        [ex + new_r, ey + new_r],
        [ex - new_r, ey + new_r],
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected = cv2.warpPerspective(image, M, (w, h))

    updated_ring = {
        'cx': float(ex),
        'cy': float(ey),
        'radius': float(new_r),
    }
    return corrected, updated_ring


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------

def crop_to_ring(image, ring, pad_fraction=0.10):
    """
    Crop the image to the outer ring bounding box with padding.

    pad_fraction: fraction of ring radius added on each side (default 10%).

    Returns:
        cropped_image   — numpy array
        crop_info dict:
            x0, y0      — top-left corner of crop in original image
            button_cx   — ring center X in cropped image
            button_cy   — ring center Y in cropped image
            ring_radius — ring radius (unchanged)
    """
    cx, cy, r = ring['cx'], ring['cy'], ring['radius']
    pad = r * pad_fraction

    h, w = image.shape[:2]
    x0 = max(0, int(cx - r - pad))
    y0 = max(0, int(cy - r - pad))
    x1 = min(w, int(cx + r + pad))
    y1 = min(h, int(cy + r + pad))

    cropped = image[y0:y1, x0:x1]

    crop_info = {
        'x0': x0,
        'y0': y0,
        'button_cx': cx - x0,
        'button_cy': cy - y0,
        'ring_radius': r,
    }
    return cropped, crop_info


# ---------------------------------------------------------------------------
# Stone detection
# ---------------------------------------------------------------------------

def detect_stones(cropped_image, crop_info, max_stones=None):
    """
    Detect stones in the cropped house image.

    Uses HoughCircles with radius range expressed as a fraction of the ring
    radius (more stable across venues than fraction of image width).

    max_stones: if set, at most this many stones are returned. If more are
                detected, the most suspicious (button-like) are dropped first.

    Returns:
        stones  — list of dicts: {team, pixel_x, pixel_y, radius, warnings}
        warnings — list of pipeline-level warning strings
    """
    pipeline_warnings = []
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    ring_r = crop_info['ring_radius']
    bcx    = crop_info['button_cx']
    bcy    = crop_info['button_cy']

    min_stone_r = int(ring_r * 0.08)
    max_stone_r = int(ring_r * 0.15)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min_stone_r * 1.8),  # prevent double-detection of same stone
        param1=50,
        param2=25,
        minRadius=min_stone_r,
        maxRadius=max_stone_r,
    )

    if circles is None:
        pipeline_warnings.append("No stones detected in image.")
        return [], pipeline_warnings

    circles = np.round(circles[0]).astype(int)

    # Filter to in-house only; also drop phantom detections of the white button
    # disc itself. The button disc registers as HoughCircles' minimum detectable
    # radius, so: discard a button-center circle only when its radius is at or
    # near the minimum (≤ min_stone_r + 1). A real stone sitting on the button
    # will have a larger radius and is kept.
    in_house = []
    for cx, cy, r in circles:
        dist_to_button = np.sqrt((cx - bcx)**2 + (cy - bcy)**2)
        if dist_to_button < min_stone_r and r <= min_stone_r + 1:
            continue  # phantom detection of the button centre disc
        if dist_to_button <= ring_r * 1.1:
            in_house.append((cx, cy, r))

    if not in_house:
        pipeline_warnings.append("No stones detected inside the house.")
        return [], pipeline_warnings

    # Compute expected stone radius from median
    radii = [r for _, _, r in in_house]
    median_r = float(np.median(radii))

    stones = []
    for cx, cy, r in in_house:
        stone_warnings = []

        # Touching stone check
        if r > median_r * TOUCHING_RATIO:
            stone_warnings.append(
                f"Large circle at ({cx}, {cy}) — may be two touching stones. Verify manually."
            )
            pipeline_warnings.append(stone_warnings[-1])

        # Color classification — try primary + voting first.
        # If a valid colour is confirmed, keep the stone unconditionally;
        # artifact checks are only applied to unclassified circles.
        team = _classify_color(cropped_image, cx, cy, r, inner_fraction=0.4)

        if team is None:
            # Fix 1: Inner vs. outer saturation gradient check.
            # A real stone has a coloured handle in its centre (high inner sat)
            # and grey granite in its body (lower outer sat).  The white button
            # disc has near-zero inner saturation while the surrounding red
            # 4-foot ring gives HIGH outer saturation — the opposite gradient.
            # We only apply this to unclassified circles; confirmed stones skip it.
            if not _is_stone_not_ring_artifact(cropped_image, cx, cy, r):
                pipeline_warnings.append(
                    f"Discarded ring artifact at ({cx}, {cy}) — inner sat < outer sat "
                    "(button/ring false positive)."
                )
                continue

            # Fix 2: Grey body ring verification.
            # Real curling stones show grey granite in the annular region 35–85%
            # of their radius.  Solid house-ring markings and the button disc don't.
            if not _has_grey_body(cropped_image, cx, cy, r):
                pipeline_warnings.append(
                    f"Discarded non-stone at ({cx}, {cy}) — no grey granite body detected."
                )
                continue

            # Glare filter — discard if patch looks like ice/glare reflection.
            patch = _extract_patch(cropped_image, cx, cy, r, inner_fraction=0.6)
            if patch is not None and patch.size > 0:
                hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                mean_s = float(np.mean(hsv_patch[:, :, 1]))
                mean_v = float(np.mean(hsv_patch[:, :, 2]))
                if mean_s < GLARE_MAX_SAT and mean_v > GLARE_MIN_VAL:
                    pipeline_warnings.append(
                        f"Discarded likely glare at ({cx}, {cy}) — low saturation, high brightness."
                    )
                    continue

            stone_warnings.append(f"Stone at ({cx}, {cy}) — color unrecognised (not red or yellow).")
            pipeline_warnings.append(stone_warnings[-1])

        stones.append({
            'team':    team,
            'pixel_x': int(cx),
            'pixel_y': int(cy),
            'radius':  int(r),
            'warnings': stone_warnings,
        })

    # Fix 3: Shot count hard cap — drop most suspicious detections if over limit.
    if max_stones is not None and len(stones) > max_stones:
        pipeline_warnings.append(
            f"Detected {len(stones)} stones but shot_number cap is {max_stones}. "
            "Dropping most suspicious detections."
        )
        stones = _drop_most_suspicious(stones, cropped_image, crop_info, max_stones)

    return stones, pipeline_warnings


# ---------------------------------------------------------------------------
# Color classification helpers
# ---------------------------------------------------------------------------

def _extract_patch(image, cx, cy, radius, inner_fraction=0.6):
    """Extract a square patch of the inner portion of a detected circle."""
    inner_r = max(1, int(radius * inner_fraction))
    h, w = image.shape[:2]
    x0 = max(0, cx - inner_r)
    y0 = max(0, cy - inner_r)
    x1 = min(w, cx + inner_r)
    y1 = min(h, cy + inner_r)
    if x1 <= x0 or y1 <= y0:
        return None
    return image[y0:y1, x0:x1]


def _classify_color(image, cx, cy, radius, inner_fraction=0.4):
    """
    Classify a stone as Team 1 (red), Team 2 (yellow), or None (unknown).

    Primary method: samples the inner 40% of the stone radius, evaluates the
    top-25% most-saturated pixels to find handle colour (handles glare/button overlap).

    Fallback method (pixel voting): when the primary returns None — e.g., for
    a stone sitting exactly on the white button where the inner patch is all
    white — counts red and yellow pixels across a 1.3x-radius circular region.
    Returns the winner if it has >5% of total pixels AND >2x the runner-up count.

    Red hue in OpenCV HSV wraps around: 0-8 AND 83-179 (scaled from 0-179).
    Yellow/orange stones sit around hue 10-25 (20-50° real).
    """
    patch = _extract_patch(image, cx, cy, radius, inner_fraction=inner_fraction)
    if patch is None or patch.size == 0:
        return _classify_color_vote(image, cx, cy, radius)

    hsv  = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hues = hsv[:, :, 0].flatten().astype(float)
    sats = hsv[:, :, 1].flatten().astype(float)

    # Use only the top-25% most-saturated pixels to find handle colour.
    sat_cutoff = float(np.percentile(sats, 75))
    mask = sats >= max(sat_cutoff, 1.0)
    if not np.any(mask):
        return _classify_color_vote(image, cx, cy, radius)

    mean_top_sat = float(np.mean(sats[mask]))
    mean_top_hue = float(np.mean(hues[mask]))

    # Require the top-percentile pixels to have at least minimal saturation.
    if mean_top_sat < 20:
        return _classify_color_vote(image, cx, cy, radius)

    # Red: hue 0-8 OR 83-179
    is_red    = (RED_HUE_LOW_LO  <= mean_top_hue <= RED_HUE_LOW_HI) or \
                (RED_HUE_HIGH_LO <= mean_top_hue <= RED_HUE_HIGH_HI)
    # Yellow: hue 10-25 (20-50° real)
    is_yellow = YELLOW_HUE_LO <= mean_top_hue <= YELLOW_HUE_HI

    if is_red:
        return 1
    if is_yellow:
        return 2
    # Primary failed to classify — try pixel voting
    return _classify_color_vote(image, cx, cy, radius)


def _classify_color_vote(image, cx, cy, radius):
    """
    Pixel-voting colour fallback: count red vs yellow pixels in a circular
    region of 1.3x the detected radius.  Returns 1 (red) or 2 (yellow) if
    the winner has >5% of total circle pixels AND >2x the runner-up count,
    else None.
    """
    h, w = image.shape[:2]
    sample_r = radius * 1.3
    Y, X = np.ogrid[:h, :w]
    circle_mask = (X - cx) ** 2 + (Y - cy) ** 2 <= sample_r ** 2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(float)
    sat = hsv[:, :, 1].astype(float)

    red_mask = circle_mask & (sat > 80) & ((hue <= RED_HUE_LOW_HI) | (hue >= RED_HUE_HIGH_LO))
    yel_mask = circle_mask & (sat > 80) & (hue >= YELLOW_HUE_LO) & (hue <= YELLOW_HUE_HI)

    n_red   = int(red_mask.sum())
    n_yel   = int(yel_mask.sum())
    n_total = int(circle_mask.sum())
    min_px  = max(1, int(n_total * 0.05))

    if n_red >= min_px and n_red > n_yel * 2:
        return 1
    if n_yel >= min_px and n_yel > n_red * 2:
        return 2
    return None


def _is_stone_not_ring_artifact(image, cx, cy, radius):
    """
    Return True if the detected circle looks like a real stone (keep it),
    False if it looks like a house-ring or button artifact (discard it).

    Logic: a real stone has a coloured handle in its centre (high inner
    saturation) surrounded by grey granite (lower outer saturation).
    A false positive such as the white button disc has near-zero inner
    saturation while the surrounding painted ring gives high outer saturation
    — the exact opposite gradient.

    Threshold: if inner_mean_sat < outer_mean_sat * 0.6 → artifact.
    """
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - cx) ** 2 + (Y - cy) ** 2

    inner_mask = dist_sq <= (radius * 0.40) ** 2
    outer_mask = (dist_sq > (radius * 0.40) ** 2) & (dist_sq <= (radius * 0.85) ** 2)

    if inner_mask.sum() == 0 or outer_mask.sum() == 0:
        return True  # can't judge — don't discard

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(float)

    inner_sat = float(np.mean(sat[inner_mask]))
    outer_sat = float(np.mean(sat[outer_mask]))

    # Artifact: inner is much less saturated than outer
    return inner_sat >= outer_sat * 0.6


def _has_grey_body(image, cx, cy, radius):
    """
    Return True if the annular region (35–85% of stone radius) contains
    enough grey pixels to be consistent with a granite stone body.

    Grey pixel criterion: saturation < 80 AND 60 < value < 220.
    Threshold: at least 20% of annulus pixels must be grey.

    House rings are solid colours (no grey); the button disc is white
    surrounded by red (no grey annulus).  Real curling stones always show
    grey granite in this region.
    """
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - cx) ** 2 + (Y - cy) ** 2

    annulus_mask = (dist_sq > (radius * 0.35) ** 2) & (dist_sq <= (radius * 0.85) ** 2)
    n_annulus = int(annulus_mask.sum())
    if n_annulus == 0:
        return True  # too small to judge — don't discard

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(float)
    val = hsv[:, :, 2].astype(float)

    grey_mask = annulus_mask & (sat < 80) & (val > 60) & (val < 220)
    grey_fraction = float(grey_mask.sum()) / n_annulus

    return grey_fraction >= 0.20


def _drop_most_suspicious(stones, image, crop_info, max_stones):
    """
    If more stones than max_stones were detected, rank each by a suspicion
    score and drop the most suspicious until the count reaches max_stones.

    Suspicion score (0–3, higher = more likely a false positive):
      - Low inner saturation (sat < 40 in inner 40% of radius) → +1
      - Very close to button centre (dist < ring_r * 0.12) → +1
      - Radius is an outlier vs. median (> 1.4× or < 0.6× median) → +1
    """
    if len(stones) <= max_stones:
        return stones

    bcx    = crop_info['button_cx']
    bcy    = crop_info['button_cy']
    ring_r = crop_info['ring_radius']

    radii  = [s['radius'] for s in stones]
    median_r = float(np.median(radii))

    h, w = image.shape[:2]
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat  = hsv[:, :, 1].astype(float)
    Y, X = np.ogrid[:h, :w]

    def _suspicion(stone):
        cx, cy, r = stone['pixel_x'], stone['pixel_y'], stone['radius']
        score = 0

        # Low inner saturation
        inner_mask = (X - cx) ** 2 + (Y - cy) ** 2 <= (r * 0.40) ** 2
        if inner_mask.sum() > 0 and np.mean(sat[inner_mask]) < 40:
            score += 1

        # Near button centre
        dist = np.sqrt((cx - bcx) ** 2 + (cy - bcy) ** 2)
        if dist < ring_r * 0.12:
            score += 1

        # Radius outlier
        if r > median_r * 1.4 or r < median_r * 0.6:
            score += 1

        return score

    ranked = sorted(stones, key=_suspicion, reverse=True)
    return ranked[len(ranked) - max_stones:]  # keep the least suspicious
