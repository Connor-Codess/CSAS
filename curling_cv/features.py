"""
features.py — Engineer XGBoost model features from detected stone positions.

Feature names and value ranges match end_state_model_df.csv exactly:
  hammer_closest_dist       float  (4 – 1097 model units)
  nonhammer_closest_dist    float
  hammer_stones_in_house    int    (0 – 3 in training data; can exceed at later shots)
  nonhammer_stones_in_house int
  hammer_house_control_diff int    (-2 to +2 in training data)
  end_num                   int    (1 – 10)
  end_parity                int    (0 or 1)
  powerplay                 float  (1.0, 2.0, or NaN)
  hammer_is_team1           int    (0 or 1)
  hammer_is_team2           int    (0 or 1)
"""

import math
import warnings as _warnings

HOUSE_RADIUS_MODEL = 250.0
NO_STONE_DIST      = 999.0  # sentinel when a team has no stones in house


def compute_features(stones, hammer_team, end_num, powerplay=None):
    """
    Compute the 10 model input features from a list of transformed stones.

    Args:
        stones      : list of stone dicts (must have 'team' and 'distance_to_button')
        hammer_team : 1 or 2
        end_num     : integer end number (1-10)
        powerplay   : None | 1 | 2  (which team activated power play, or None)

    Returns:
        dict of features ready for XGBoost predict_proba()
    """
    hammer_dists = [
        s['distance_to_button'] for s in stones
        if s.get('team') == hammer_team and s['distance_to_button'] <= HOUSE_RADIUS_MODEL
    ]
    opp_team = 2 if hammer_team == 1 else 1
    opp_dists = [
        s['distance_to_button'] for s in stones
        if s.get('team') == opp_team and s['distance_to_button'] <= HOUSE_RADIUS_MODEL
    ]

    hammer_closest  = min(hammer_dists) if hammer_dists else NO_STONE_DIST
    opp_closest     = min(opp_dists)    if opp_dists    else NO_STONE_DIST
    hammer_in_house = len(hammer_dists)
    opp_in_house    = len(opp_dists)
    control_diff    = hammer_in_house - opp_in_house

    # powerplay column: NaN when not active (matches training distribution)
    if powerplay in (1, 2):
        pp_value = float(powerplay)
    else:
        pp_value = float('nan')

    return {
        'hammer_closest_dist':       hammer_closest,
        'nonhammer_closest_dist':    opp_closest,
        'hammer_stones_in_house':    hammer_in_house,
        'nonhammer_stones_in_house': opp_in_house,
        'hammer_house_control_diff': control_diff,
        'end_num':                   int(end_num),
        'end_parity':                int(end_num) % 2,
        'powerplay':                 pp_value,
        'hammer_is_team1':           int(hammer_team == 1),
        'hammer_is_team2':           int(hammer_team == 2),
    }


def validate_stone_counts(stones, expected_team1, expected_team2):
    """
    Optional validation: compare detected stone counts against user-provided
    expected counts. Returns a list of warning strings (empty if all match).

    Args:
        stones          : list of stone dicts with 'team' key
        expected_team1  : int or None
        expected_team2  : int or None
    """
    warnings = []
    detected_t1 = sum(1 for s in stones if s.get('team') == 1)
    detected_t2 = sum(1 for s in stones if s.get('team') == 2)
    detected_unknown = sum(1 for s in stones if s.get('team') is None)

    if expected_team1 is not None and detected_t1 != expected_team1:
        warnings.append(
            f"Team 1 (red): expected {expected_team1} stones, detected {detected_t1}. "
            "A stone may have been knocked out or misclassified."
        )
    if expected_team2 is not None and detected_t2 != expected_team2:
        warnings.append(
            f"Team 2 (yellow): expected {expected_team2} stones, detected {detected_t2}. "
            "A stone may have been knocked out or misclassified."
        )
    if detected_unknown > 0:
        warnings.append(
            f"{detected_unknown} stone(s) could not be colour-classified. "
            "Check the annotated image."
        )
    return warnings
