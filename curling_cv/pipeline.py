"""
pipeline.py — End-to-end orchestration.

Connects: image → detect → transform → features → XGBoost prediction
                                                 → GameState seed → simulation → GIF
"""

import os
import csv

import cv2
import numpy as np

from detect    import detect_outer_ring, crop_to_ring, detect_stones
from transform import transform_all_stones
from features  import compute_features, validate_stone_counts
from simulation import (load_models, seed_game_state_from_cv,
                        simulate_end_from_state, animate_simulation,
                        get_confidence_level)
from utils     import draw_detections

# Paths to analysis CSVs (relative to this file's directory)
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, '..')

OPENING_SEQ_PATH   = os.path.join(_DATA, 'opening_sequences_analysis.csv')
DEFENSIVE_PATH     = os.path.join(_DATA, 'defensive_strategies_analysis.csv')
SIM_STATES_PATH    = os.path.join(_DATA, 'simulation_states.csv')
MODELS_DIR         = os.path.join(_HERE, 'models')


def _load_csv(path):
    """Load a CSV as a pandas DataFrame, or None if pandas/file unavailable."""
    try:
        import pandas as pd
        if os.path.exists(path):
            return pd.read_csv(path)
    except ImportError:
        pass
    return None


def _build_advice(scoring_prob, steal_prob, blank_prob, features=None):
    f = features or {}
    h_in   = f.get('hammer_stones_in_house', 0)
    nh_in  = f.get('nonhammer_stones_in_house', 0)
    h_dist = f.get('hammer_closest_dist', 999)
    nh_dist= f.get('nonhammer_closest_dist', 999)
    ctrl   = f.get('hammer_house_control_diff', 0)

    # ── Situation header ─────────────────────────────────────────────────
    if scoring_prob > 0.65 and steal_prob < 0.35:
        situation = "STRONG POSITION"
        sit_detail = (
            f"The model gives the hammer team a {scoring_prob:.0%} chance of scoring this end "
            f"with only a {steal_prob:.0%} steal risk — this is a favourable end to be in."
        )
    elif steal_prob > 0.50:
        situation = "DANGER — HIGH STEAL RISK"
        sit_detail = (
            f"The model flags a {steal_prob:.0%} probability the non-hammer team steals this end. "
            f"The hammer team is currently not counting and the end needs to be cleaned up immediately."
        )
    elif blank_prob > 0.28 and scoring_prob < 0.52:
        situation = "NEUTRAL — CONSIDER BLANKING"
        sit_detail = (
            f"Neither team holds a clear edge. The model gives only a {scoring_prob:.0%} scoring chance "
            f"and a {blank_prob:.0%} blank probability — retaining hammer may be worth more than a contested one."
        )
    elif scoring_prob > 0.50:
        situation = "COMPETITIVE — HAMMER SLIGHT EDGE"
        sit_detail = (
            f"The hammer team holds a modest {scoring_prob:.0%} scoring probability. "
            f"The end is still very much in play for both sides, so execution on the next shots is critical."
        )
    else:
        situation = "DIFFICULT POSITION"
        sit_detail = (
            f"Scoring probability has dropped to {scoring_prob:.0%}. "
            f"The hammer team needs to reshape the end — trading stones or drawing back into the house should be the priority."
        )

    # ── Hammer team guidance ─────────────────────────────────────────────
    if h_in == 0 and nh_in == 0:
        hammer_advice = (
            "The house is completely clear. Your next shot should be a draw to the 4-foot or button — placing a stone "
            "in the centre of the house early forces the opposition to play reactively and gives you first scoring "
            "position. A centre draw with a slight handle preference to your strong side is ideal here."
        )
    elif h_in > 0 and nh_in == 0:
        hammer_advice = (
            f"You have {h_in} stone(s) in the house with no opposition stones — you are counting. "
            "Your next shot should be a guard out front to protect your scoring stones and make it harder for the "
            "opposition to hit and roll to safety. If you already have a guard, consider a freeze to the top stone "
            "or a second scoring draw to build a multi-point end."
        )
    elif nh_in > 0 and nh_dist < h_dist:
        opp_str = f"~{int(nh_dist)} units from the button" if nh_dist < 999 else "in the house"
        hammer_advice = (
            f"The opposition has the shot stone ({opp_str}). Your immediate priority is to reclaim shot. "
            "If the opposition stone is exposed, execute a precise takeout — aim to take it out cleanly "
            "and roll your stone toward the 8-foot to stay in scoring position. If it's buried behind a guard, "
            "consider a raise or a come-around draw to sit on top. Leaving the opposition counting heading into "
            "the last few shots will put serious pressure on your hammer."
        )
    elif h_in > nh_in and h_dist < nh_dist:
        pts = h_in - nh_in
        hammer_advice = (
            f"You are counting {h_in} stone(s) and lead in the house by {pts}. This is the time to build — "
            "draw a second stone into the top of the 4-foot to set up a potential multi-point end, "
            "or lay a guard if your scoring stones are exposed. Make the opposition play aggressive "
            "takeout shots; the more they have to come through, the more chances for a miss."
        )
    elif nh_in > h_in:
        hammer_advice = (
            f"The opposition has more stones in the house ({nh_in} vs your {h_in}). "
            "You need a takeout to reduce their count. Target the closest opposition stone — "
            "a clean hit-and-roll into the 4-foot keeps you in the house after the shot. "
            "If multiple opposition stones are clustered, look for a double takeout angle."
        )
    else:
        hammer_advice = (
            "The end is closely contested in the house. Focus on placing your next stone in the top of the 4-foot "
            "with a slight angle so it's difficult to remove without leaving an opening. "
            "Playing an in-off or a tight draw is safer than a high-risk peel in this situation."
        )

    # ── Non-hammer team guidance ─────────────────────────────────────────
    if steal_prob > 0.50:
        nonhammer_advice = (
            f"You have a {steal_prob:.0%} steal probability — a real chance. Stay aggressive. "
            "If you have a stone close to the button, guard it immediately to force the hammer team into a "
            "difficult come-around. If you don't have shot, play a straight-on takeout and roll to the "
            "button — even one counting stone close to centre significantly increases steal odds. "
            "The opposition will likely attempt a draw or raise to reclaim shot; be ready for them to play "
            "to your open side."
        )
    elif nh_in > 0 and nh_dist < h_dist:
        nonhammer_advice = (
            "You have the shot stone — use this momentum. Place a guard directly in front of your scoring stone "
            "to force the hammer team to play around or through it. If they miss, you score; if they hit, "
            "chances are good they leave you in a reasonable position. Expect the hammer team to attempt "
            "a peel or a come-around draw on their next shot."
        )
    elif blank_prob > 0.28 and scoring_prob < 0.52:
        nonhammer_advice = (
            "A blank end is a real possibility here — that means the hammer team keeps the hammer. "
            "Your priority is to prevent a blank by getting a stone in the house. Play a draw to the "
            "button or 4-foot; forcing the hammer team to play around your stone disrupts their blank "
            "attempt and increases your steal chances. Don't give them a clean runway to blank the end."
        )
    else:
        nonhammer_advice = (
            f"You are behind in the house ({nh_in} stone(s) vs the hammer team's {h_in}). "
            "Play to disrupt — a well-placed centre guard or an aggressive takeout to remove a hammer stone "
            "and roll behind cover can completely change the end. The hammer team will try to draw to the "
            "open side or peel your guards; make every stone count and force them to earn every point."
        )

    # ── Blank guidance appendix ──────────────────────────────────────────
    blank_note = ""
    if blank_prob > 0.30 and scoring_prob < 0.50:
        blank_note = (
            f"\nNOTE — BLANK END: With a {blank_prob:.0%} blank probability and a below-average scoring chance, "
            "blanking is a legitimate strategic play for the hammer team. Draw through the house cleanly "
            "to retain hammer going into the next end rather than forcing a low-percentage score attempt."
        )
    elif scoring_prob > 0.70 and h_in >= 2:
        blank_note = (
            f"\nMULTI-POINT OPPORTUNITY: With {h_in} hammer stones counting and a {scoring_prob:.0%} scoring probability, "
            "this end has genuine multi-point potential. Protect what you have and let the model's edge play out."
        )

    sections = [
        f"SITUATION: {situation}\n{sit_detail}",
        f"HAMMER TEAM — What to do next:\n{hammer_advice}",
        f"NON-HAMMER TEAM — What to expect and how to respond:\n{nonhammer_advice}",
    ]
    if blank_note:
        sections.append(blank_note.strip())

    return "\n\n".join(sections)


def run_pipeline(
    image_bgr,
    hammer_team,
    end_num,
    shot_number,
    team1_score,
    team2_score,
    sheet_end,
    powerplay=None,
    team1_stone_count=None,
    team2_stone_count=None,
):
    """
    Full pipeline: image → predictions + simulation GIF.

    Args:
        image_bgr        : numpy BGR array (from cv2.imread or Gradio)
        hammer_team      : 1 or 2
        end_num          : int (1-10)
        shot_number      : int — how many stones have already been thrown
        team1_score      : int
        team2_score      : int
        sheet_end        : 'top' or 'bottom'
        powerplay        : None | 1 | 2
        team1_stone_count: int or None — optional validation count
        team2_stone_count: int or None — optional validation count

    Returns:
        dict with keys:
            annotated_image  — BGR numpy array (detection confirmation)
            features         — dict of engineered features
            scoring_prob     — float
            steal_prob       — float
            blank_prob       — float
            magnitude_probs  — list [p0, p1, p2, p3plus] or None
            advice           — str
            confidence       — str
            warnings         — list of warning strings
            simulation_gif   — bytes or None
            final_score      — dict {team1, team2} or None
    """
    all_warnings = []

    # ------------------------------------------------------------------ #
    # Resize very large images — HoughCircles slows exponentially on them
    # ------------------------------------------------------------------ #
    MAX_DIM = 1200
    h0, w0 = image_bgr.shape[:2]
    if max(h0, w0) > MAX_DIM:
        scale = MAX_DIM / max(h0, w0)
        image_bgr = cv2.resize(image_bgr, (int(w0 * scale), int(h0 * scale)),
                               interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------ #
    # Step 1 — Detect outer ring
    # ------------------------------------------------------------------ #
    ring = detect_outer_ring(image_bgr)
    all_warnings.extend(ring.get('warnings', []))

    # Use perspective-corrected image for everything downstream
    working_image = ring.pop('corrected_image')

    # ------------------------------------------------------------------ #
    # Step 2 — Crop to house
    # ------------------------------------------------------------------ #
    cropped, crop_info = crop_to_ring(working_image, ring, pad_fraction=0.10)

    # ------------------------------------------------------------------ #
    # Step 3 — Detect stones
    # ------------------------------------------------------------------ #
    stones, stone_warnings = detect_stones(cropped, crop_info, max_stones=shot_number)
    all_warnings.extend(stone_warnings)

    # ------------------------------------------------------------------ #
    # Step 4 — Transform to model coordinate space
    # ------------------------------------------------------------------ #
    stones = transform_all_stones(stones, crop_info, sheet_end=sheet_end)

    # ------------------------------------------------------------------ #
    # Step 5 — Optional stone count validation
    # ------------------------------------------------------------------ #
    if team1_stone_count is not None or team2_stone_count is not None:
        count_warnings = validate_stone_counts(stones, team1_stone_count, team2_stone_count)
        all_warnings.extend(count_warnings)

    # ------------------------------------------------------------------ #
    # Step 6 — Feature engineering
    # ------------------------------------------------------------------ #
    features = compute_features(stones, hammer_team, end_num, powerplay)

    # ------------------------------------------------------------------ #
    # Step 7 — XGBoost predictions
    # ------------------------------------------------------------------ #
    models = load_models(MODELS_DIR)

    scoring_prob    = None
    steal_prob      = None
    blank_prob      = None
    magnitude_probs = None

    try:
        import pandas as pd
        import pickle as _pickle

        # Use saved feature column order if available (matches training exactly)
        _cols_path = os.path.join(MODELS_DIR, 'feature_cols.pkl')
        if os.path.exists(_cols_path):
            with open(_cols_path, 'rb') as _f:
                _feature_cols = _pickle.load(_f)
        else:
            from simulation import Q1_Q5_FEATURE_COLS as _feature_cols

        # Fill powerplay NaN → 0 to match notebook preprocessing
        _features_for_model = dict(features)
        import math as _math
        if _math.isnan(_features_for_model.get('powerplay', 0) or 0):
            _features_for_model['powerplay'] = 0
            _features_for_model['powerplay_missing'] = 1
        else:
            _features_for_model['powerplay_missing'] = 0

        fd  = {col: _features_for_model.get(col, 0) for col in _feature_cols}
        fdf = pd.DataFrame([fd])[_feature_cols]

        if models.get('q1') is not None:
            p = models['q1'].predict_proba(fdf)[0]
            scoring_prob = float(p[1] if len(p) > 1 else p[0])

        if models.get('q5') is not None:
            p = models['q5'].predict_proba(fdf)[0]
            steal_prob = float(p[1] if len(p) > 1 else p[0])

        if models.get('q4') is not None:
            p = models['q4'].predict_proba(fdf)[0]
            blank_prob = float(p[1] if len(p) > 1 else p[0])

        if models.get('q3') is not None:
            p = models['q3'].predict_proba(fdf)[0]
            magnitude_probs = [float(x) for x in p]

    except Exception as e:
        all_warnings.append(f"Model prediction unavailable: {e}. Using heuristic fallback.")

    # Heuristic fallback values
    if scoring_prob is None:
        h_in  = features['hammer_stones_in_house']
        nh_in = features['nonhammer_stones_in_house']
        scoring_prob = 0.5 + 0.1 * (h_in - nh_in)
        scoring_prob = max(0.1, min(0.9, scoring_prob))
    if steal_prob  is None: steal_prob  = 0.3
    if blank_prob  is None: blank_prob  = 0.1

    advice     = _build_advice(scoring_prob, steal_prob, blank_prob, features=features)
    confidence = get_confidence_level(shot_number)

    # ------------------------------------------------------------------ #
    # Step 8 — Annotated confirmation image
    # ------------------------------------------------------------------ #
    for stone in stones:
        stone['distance_to_button'] = stone.get('distance_to_button')

    annotated = draw_detections(cropped, crop_info, stones, warnings=all_warnings)

    # ------------------------------------------------------------------ #
    # Step 9 — Seed GameState and simulate remaining shots
    # ------------------------------------------------------------------ #
    simulation_gif = None
    final_score    = None

    try:
        opening_seqs    = _load_csv(OPENING_SEQ_PATH)
        defensive_strats = _load_csv(DEFENSIVE_PATH)
        sim_states      = _load_csv(SIM_STATES_PATH)

        game_state = seed_game_state_from_cv(
            cv_stones=stones,
            hammer_team=hammer_team,
            end_num=end_num,
            shot_number=shot_number,
            team1_score=team1_score,
            team2_score=team2_score,
            powerplay=powerplay,
            sheet_end=sheet_end,
        )

        final_state, game_log = simulate_end_from_state(
            game_state,
            models_q1_q5=models,
            opening_sequences=opening_seqs,
            defensive_strategies=defensive_strats,
            simulation_states=sim_states,
        )

        final_score    = game_log.get('final_score')
        simulation_gif = animate_simulation(game_log, final_state)

    except Exception as e:
        all_warnings.append(f"Simulation failed: {e}")

    return {
        'annotated_image': annotated,
        'features':        features,
        'scoring_prob':    scoring_prob,
        'steal_prob':      steal_prob,
        'blank_prob':      blank_prob,
        'magnitude_probs': magnitude_probs,
        'advice':          advice,
        'confidence':      confidence,
        'warnings':        all_warnings,
        'simulation_gif':  simulation_gif,
        'final_score':     final_score,
        # Exposed for manual correction flow
        'cropped_bgr':     cropped,
        'crop_info':       crop_info,
        'stones':          stones,
    }


def run_from_stones(
    stones_crop,
    cropped_bgr,
    crop_info,
    hammer_team,
    end_num,
    shot_number,
    team1_score,
    team2_score,
    sheet_end,
    powerplay=None,
):
    """
    Re-run the pipeline from step 4 onward using a manually corrected stone list.

    stones_crop: list of stone dicts with pixel_x, pixel_y, radius, team
                 in cropped-image coordinate space (as returned by detect_stones
                 or as placed by the manual correction UI).

    Skips image loading, ring detection, and stone detection entirely.
    """
    all_warnings = []

    # Deep-copy so we don't mutate the caller's list
    import copy
    stones = copy.deepcopy(stones_crop)

    # Step 4 — Transform to model coordinates
    stones = transform_all_stones(stones, crop_info, sheet_end=sheet_end)

    # Step 5 — Feature engineering
    features = compute_features(stones, hammer_team, end_num, powerplay)

    # Step 6 — XGBoost predictions
    models = load_models(MODELS_DIR)

    scoring_prob    = None
    steal_prob      = None
    blank_prob      = None
    magnitude_probs = None

    try:
        import pandas as pd
        import pickle as _pickle

        _cols_path = os.path.join(MODELS_DIR, 'feature_cols.pkl')
        if os.path.exists(_cols_path):
            with open(_cols_path, 'rb') as _f:
                _feature_cols = _pickle.load(_f)
        else:
            from simulation import Q1_Q5_FEATURE_COLS as _feature_cols

        _features_for_model = dict(features)
        import math as _math
        if _math.isnan(_features_for_model.get('powerplay', 0) or 0):
            _features_for_model['powerplay'] = 0
            _features_for_model['powerplay_missing'] = 1
        else:
            _features_for_model['powerplay_missing'] = 0

        fd  = {col: _features_for_model.get(col, 0) for col in _feature_cols}
        fdf = pd.DataFrame([fd])[_feature_cols]

        if models.get('q1') is not None:
            p = models['q1'].predict_proba(fdf)[0]
            scoring_prob = float(p[1] if len(p) > 1 else p[0])
        if models.get('q5') is not None:
            p = models['q5'].predict_proba(fdf)[0]
            steal_prob = float(p[1] if len(p) > 1 else p[0])
        if models.get('q4') is not None:
            p = models['q4'].predict_proba(fdf)[0]
            blank_prob = float(p[1] if len(p) > 1 else p[0])
        if models.get('q3') is not None:
            p = models['q3'].predict_proba(fdf)[0]
            magnitude_probs = [float(x) for x in p]

    except Exception as e:
        all_warnings.append(f"Model prediction unavailable: {e}. Using heuristic fallback.")

    if scoring_prob is None:
        h_in  = features['hammer_stones_in_house']
        nh_in = features['nonhammer_stones_in_house']
        scoring_prob = max(0.1, min(0.9, 0.5 + 0.1 * (h_in - nh_in)))
    if steal_prob  is None: steal_prob  = 0.3
    if blank_prob  is None: blank_prob  = 0.1

    advice     = _build_advice(scoring_prob, steal_prob, blank_prob, features=features)
    confidence = get_confidence_level(shot_number)

    annotated = draw_detections(cropped_bgr, crop_info, stones, warnings=all_warnings)

    # Simulation
    simulation_gif = None
    final_score    = None
    try:
        opening_seqs     = _load_csv(OPENING_SEQ_PATH)
        defensive_strats = _load_csv(DEFENSIVE_PATH)
        sim_states       = _load_csv(SIM_STATES_PATH)

        game_state = seed_game_state_from_cv(
            cv_stones=stones,
            hammer_team=hammer_team,
            end_num=end_num,
            shot_number=shot_number,
            team1_score=team1_score,
            team2_score=team2_score,
            powerplay=powerplay,
            sheet_end=sheet_end,
        )
        final_state, game_log = simulate_end_from_state(
            game_state, models_q1_q5=models,
            opening_sequences=opening_seqs,
            defensive_strategies=defensive_strats,
            simulation_states=sim_states,
        )
        final_score    = game_log.get('final_score')
        simulation_gif = animate_simulation(game_log, final_state)

    except Exception as e:
        all_warnings.append(f"Simulation failed: {e}")

    return {
        'annotated_image': annotated,
        'features':        features,
        'scoring_prob':    scoring_prob,
        'steal_prob':      steal_prob,
        'blank_prob':      blank_prob,
        'magnitude_probs': magnitude_probs,
        'advice':          advice,
        'confidence':      confidence,
        'warnings':        all_warnings,
        'simulation_gif':  simulation_gif,
        'final_score':     final_score,
        'cropped_bgr':     cropped_bgr,
        'crop_info':       crop_info,
        'stones':          stones,
    }
