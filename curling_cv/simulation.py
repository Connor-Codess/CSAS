"""
simulation.py — Importable simulation module refactored from game_simulation.ipynb.

Extracts GameState, move generation, model evaluation, and animation into a
standalone module. Adds seed_game_state_from_cv() and simulate_end_from_state()
so the CV pipeline can seed a real board position and simulate forward.
"""

import copy
import io
import math
import os
import pickle

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Gradio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------------------------------------------------------
# Constants (match XGBoost.ipynb cell 16 exactly)
# ---------------------------------------------------------------------------

BUTTON_X     = 750.0
BUTTON_Y     = 800.0
HOUSE_RADIUS = 250.0
HOG_LINE_Y   = 1200.0

# Feature columns the Q1-Q5 models were trained on
Q1_Q5_FEATURE_COLS = [
    'hammer_closest_dist',
    'nonhammer_closest_dist',
    'hammer_stones_in_house',
    'nonhammer_stones_in_house',
    'hammer_house_control_diff',
    'end_num',
    'end_parity',
    'powerplay',
    'hammer_is_team1',
    'hammer_is_team2',
    'missing_shot3_snapshot',
    'any_missing_coordinates',
]

CONFIDENCE_MAP = {
    range(1, 4):  'High confidence — matches model training (Shot 1–3)',
    range(4, 7):  'Moderate confidence — slight distribution shift (Shot 4–6)',
    range(7, 11): 'Low confidence — outside training distribution (Shot 7–10)',
}


def get_confidence_level(shot_number):
    """Return a confidence string based on how far the shot number is from training data."""
    for r, label in CONFIDENCE_MAP.items():
        if shot_number in r:
            return label
    return 'Unknown shot number'


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------

class GameState:
    """Tracks the complete state of a curling end. Refactored from game_simulation.ipynb cell 3."""

    def __init__(self, team1_id, team2_id, hammer_team, powerplay=None,
                 end_num=1, team1_score=0, team2_score=0,
                 skip_prepositioned=False):
        self.team1_id    = team1_id
        self.team2_id    = team2_id
        self.hammer_team = hammer_team
        self.powerplay   = powerplay
        self.end_num     = end_num
        self.team1_score = team1_score
        self.team2_score = team2_score

        self.stones          = [None] * 12
        self.shot_number     = 0
        self.current_team    = hammer_team
        self.delivered_shots = 0
        self.shot_history    = []

        if not skip_prepositioned:
            self._place_prepositioned_stones()

    def _place_prepositioned_stones(self):
        four_foot = HOUSE_RADIUS * 0.4
        if self.hammer_team == self.team1_id:
            self.stones[0] = _make_stone(BUTTON_X, BUTTON_Y - four_foot, self.team1_id, prepos=True)
            self.stones[6] = _make_stone(BUTTON_X, BUTTON_Y - HOUSE_RADIUS - 50, self.team2_id, prepos=True)
        else:
            self.stones[6] = _make_stone(BUTTON_X, BUTTON_Y - four_foot, self.team2_id, prepos=True)
            self.stones[0] = _make_stone(BUTTON_X, BUTTON_Y - HOUSE_RADIUS - 50, self.team1_id, prepos=True)

    def get_stones_in_play(self, team=None):
        result = [s for s in self.stones if s is not None and s.get('in_play', True)]
        if team is not None:
            result = [s for s in result if s['team'] == team]
        return result

    def get_stones_in_house(self, team=None):
        return [s for s in self.get_stones_in_play(team)
                if _dist(s['x'], s['y'], BUTTON_X, BUTTON_Y) <= HOUSE_RADIUS]

    def get_closest_stone(self, team=None):
        stones = self.get_stones_in_play(team)
        if not stones:
            return None, float('inf')
        closest = min(stones, key=lambda s: _dist(s['x'], s['y'], BUTTON_X, BUTTON_Y))
        return closest, _dist(closest['x'], closest['y'], BUTTON_X, BUTTON_Y)

    def get_house_control(self):
        t1 = len(self.get_stones_in_house(self.team1_id))
        t2 = len(self.get_stones_in_house(self.team2_id))
        _, d1 = self.get_closest_stone(self.team1_id)
        _, d2 = self.get_closest_stone(self.team2_id)
        if d1 == float('inf'): d1 = HOUSE_RADIUS * 2
        if d2 == float('inf'): d2 = HOUSE_RADIUS * 2

        diff = t1 - t2
        if d1 < d2:   diff += 0.5
        elif d2 < d1: diff -= 0.5

        controlling = None
        if diff > 0:   controlling = self.team1_id
        elif diff < 0: controlling = self.team2_id

        return {
            'team1_stones': t1, 'team2_stones': t2,
            'team1_closest_dist': d1, 'team2_closest_dist': d2,
            'control_diff': diff, 'controlling_team': controlling,
        }

    def get_game_situation(self):
        if self.delivered_shots <= 3:   phase = 'opening'
        elif self.delivered_shots <= 7: phase = 'mid'
        else:                           phase = 'late'

        score_diff = self.team1_score - self.team2_score
        ahead = self.team1_id if score_diff > 0 else (self.team2_id if score_diff < 0 else None)
        pp_team = (self.team1_id if self.powerplay == 1 else self.team2_id) if self.powerplay else None

        return {
            'phase': phase,
            'shot_number': self.delivered_shots,
            'score_diff': score_diff,
            'ahead_team': ahead,
            'is_hammer_team': self.current_team == self.hammer_team,
            'is_powerplay': self.powerplay is not None,
            'is_defensive_team': (self.powerplay is not None and self.current_team != pp_team),
            'free_guard_zone': self.delivered_shots < 3,
        }

    def apply_shot(self, move):
        # Enforce Free Guard Zone
        if self.delivered_shots < 3 and move['shot_type'] in [
            'Take-out', 'Double Take-out', 'Clearing', 'Hit and Roll', 'Promotion Take-out'
        ]:
            move = move.copy()
            move['shot_type'] = 'Draw'
            move['reasoning'] = move.get('reasoning', '') + ' (FGZ: converted to draw)'

        # Determine target with placement noise
        if move['shot_type'] in ['Take-out', 'Double Take-out', 'Clearing', 'Hit and Roll']:
            if 'takeout_target' in move and move['takeout_target'] is not None:
                idx = move['takeout_target']
                if idx < len(self.stones) and self.stones[idx] is not None:
                    self.stones[idx]['in_play'] = False
            new_x = move['target_x'] + np.random.normal(0, 20)
            new_y = move['target_y'] + np.random.normal(0, 20)
        else:
            new_x = move['target_x'] + np.random.normal(0, 30)
            new_y = move['target_y'] + np.random.normal(0, 30)

        # Find slot
        if self.current_team == self.team1_id:
            start, end = 1, 6
        else:
            start, end = 7, 12

        thrown = [i for i in range(start, end) if self.stones[i] is not None]
        if len(thrown) >= 5:
            return

        stone_idx = next((i for i in range(start, end) if self.stones[i] is None), None)
        if stone_idx is None:
            return

        self.delivered_shots += 1
        self.shot_number = self.delivered_shots
        self.stones[stone_idx] = _make_stone(new_x, new_y, self.current_team,
                                             shot_num=self.shot_number)
        self.shot_history.append({
            'shot_num': self.shot_number,
            'team': self.current_team,
            'move': move,
            'result': {'x': new_x, 'y': new_y},
        })
        self.current_team = self.team2_id if self.current_team == self.team1_id else self.team1_id

    def is_terminal(self):
        if self.delivered_shots >= 10:
            return True
        t1 = sum(1 for i in range(1, 6) if self.stones[i] is not None)
        t2 = sum(1 for i in range(7, 12) if self.stones[i] is not None)
        return t1 >= 5 and t2 >= 5

    def calculate_score(self):
        closest, _ = self.get_closest_stone()
        if closest is None:
            return (0, 0)
        scoring_team = closest['team']
        opp = self.team2_id if scoring_team == self.team1_id else self.team1_id
        _, opp_dist = self.get_closest_stone(opp)
        if opp_dist == float('inf'):
            opp_dist = HOUSE_RADIUS * 2

        pts = sum(
            1 for s in self.get_stones_in_play(scoring_team)
            if _dist(s['x'], s['y'], BUTTON_X, BUTTON_Y) < opp_dist
        )
        pts = min(pts, 6)
        return (pts, 0) if scoring_team == self.team1_id else (0, pts)

    def to_model_features(self):
        ctrl = self.get_house_control()
        if self.hammer_team == self.team1_id:
            h_stones = ctrl['team1_stones'];  h_dist = ctrl['team1_closest_dist']
            nh_stones = ctrl['team2_stones']; nh_dist = ctrl['team2_closest_dist']
            ctrl_diff = ctrl['control_diff']
        else:
            h_stones = ctrl['team2_stones'];  h_dist = ctrl['team2_closest_dist']
            nh_stones = ctrl['team1_stones']; nh_dist = ctrl['team1_closest_dist']
            ctrl_diff = -ctrl['control_diff']

        return {
            'hammer_closest_dist':       h_dist,
            'nonhammer_closest_dist':    nh_dist,
            'hammer_stones_in_house':    h_stones,
            'nonhammer_stones_in_house': nh_stones,
            'hammer_house_control_diff': ctrl_diff,
            'end_num':                   self.end_num,
            'end_parity':                self.end_num % 2,
            'powerplay':                 self.powerplay if self.powerplay is not None else np.nan,
            'hammer_is_team1':           int(self.hammer_team == self.team1_id),
            'hammer_is_team2':           int(self.hammer_team == self.team2_id),
            'missing_shot3_snapshot':    0,
            'any_missing_coordinates':   0,
        }


# ---------------------------------------------------------------------------
# CV seeding (new — not in original notebook)
# ---------------------------------------------------------------------------

def seed_game_state_from_cv(cv_stones, hammer_team, end_num, shot_number,
                             team1_score, team2_score, powerplay=None,
                             sheet_end='top'):
    """
    Create a GameState seeded from CV-detected stone positions.

    cv_stones: list of stone dicts with keys team (1|2|None), model_x, model_y
    shot_number: how many stones have already been thrown (sets delivered_shots)
    sheet_end: 'top' or 'bottom' — needed to remap Stones.csv coords → sim space

    Stones.csv uses button_y=650 (top) or 1916 (bottom).
    Simulation always uses BUTTON_Y=800.  We remap here so the animation and
    distance calculations all use a consistent coordinate system.
    """
    # Stones.csv button Y anchors
    CSV_BUTTON_Y = 650.0 if sheet_end == 'top' else 1916.0

    def _to_sim(model_x, model_y):
        """Remap from Stones.csv space → simulation space (button at BUTTON_Y=800)."""
        sim_x = BUTTON_X + (model_x - BUTTON_X)   # X anchor is the same (750)
        dy_csv = model_y - CSV_BUTTON_Y
        # Bottom end Y is already flipped in transform.py, so direction is consistent
        sim_y = BUTTON_Y + dy_csv
        return sim_x, sim_y

    gs = GameState(
        team1_id=1, team2_id=2,
        hammer_team=hammer_team,
        powerplay=powerplay,
        end_num=end_num,
        team1_score=team1_score,
        team2_score=team2_score,
        skip_prepositioned=True,
    )

    # Fill stone slots from CV detections
    t1_idx = 0   # slots 0-5 for team 1
    t2_idx = 6   # slots 6-11 for team 2

    for stone in cv_stones:
        team = stone.get('team')
        x, y = _to_sim(stone.get('model_x', BUTTON_X), stone.get('model_y', BUTTON_Y))

        if team == 1 and t1_idx < 6:
            gs.stones[t1_idx] = _make_stone(x, y, 1, shot_num=0, prepos=True)
            t1_idx += 1
        elif team == 2 and t2_idx < 12:
            gs.stones[t2_idx] = _make_stone(x, y, 2, shot_num=0, prepos=True)
            t2_idx += 1
        # Unknown-team stones are skipped (can't assign ownership)

    gs.delivered_shots = int(shot_number)
    gs.shot_number = int(shot_number)

    # Whose turn is it? Hammer throws first (shot 1), teams alternate.
    # Even delivered_shots → hammer team's turn; odd → non-hammer.
    non_hammer = 2 if hammer_team == 1 else 1
    gs.current_team = hammer_team if gs.delivered_shots % 2 == 0 else non_hammer

    return gs


def simulate_end_from_state(game_state, models_q1_q5=None,
                             opening_sequences=None, defensive_strategies=None,
                             simulation_states=None, verbose=False):
    """
    Run the simulation loop forward from an already-seeded GameState.
    Returns (final_game_state, game_log).
    """
    if models_q1_q5 is None:
        models_q1_q5 = {}

    game_log = {'shots': [], 'final_score': None}

    max_iterations = 15  # hard cap — prevents infinite loop if slots are full
    iterations = 0

    while not game_state.is_terminal() and iterations < max_iterations:
        iterations += 1
        situation = game_state.get_game_situation()
        team_before = game_state.current_team

        best_move, evaluation, _ = select_best_move(
            game_state,
            opening_sequences=opening_sequences,
            defensive_strategies=defensive_strategies,
            simulation_states=simulation_states,
            models_q1_q5=models_q1_q5,
        )

        delivered_before = game_state.delivered_shots
        game_state.apply_shot(best_move)
        # If apply_shot couldn't place a stone (all slots full) it returns
        # without incrementing delivered_shots — break to avoid infinite loop.
        if game_state.delivered_shots == delivered_before:
            break

        game_log['shots'].append({
            'shot_num': game_state.delivered_shots,
            'team': team_before,
            'move': best_move,
            'evaluation': evaluation,
            'game_state': {
                'house_control': game_state.get_house_control(),
                'situation': situation,
            },
        })

        if verbose:
            print(f"Shot {game_state.delivered_shots}/10: Team {team_before} — "
                  f"{best_move['shot_type']} | score={evaluation['score']:.3f}")

    t1, t2 = game_state.calculate_score()
    game_log['final_score'] = {'team1': t1, 'team2': t2}
    return game_state, game_log


# ---------------------------------------------------------------------------
# Move generation (from cell 5)
# ---------------------------------------------------------------------------

def generate_candidate_moves(game_state):
    situation = game_state.get_game_situation()
    candidates = []

    opp = game_state.team2_id if game_state.current_team == game_state.team1_id else game_state.team1_id
    opp_stones = game_state.get_stones_in_play(opp)

    if situation['phase'] == 'opening':
        candidates += [
            {'shot_type': 'Draw', 'target_x': BUTTON_X,      'target_y': BUTTON_Y, 'handle': 0, 'reasoning': 'Opening draw to button'},
            {'shot_type': 'Draw', 'target_x': BUTTON_X + 50, 'target_y': BUTTON_Y, 'handle': 1, 'reasoning': 'Opening draw open side'},
            {'shot_type': 'Draw', 'target_x': BUTTON_X - 50, 'target_y': BUTTON_Y, 'handle': 0, 'reasoning': 'Opening draw closed side'},
            {'shot_type': 'Guard','target_x': BUTTON_X,      'target_y': BUTTON_Y - 150, 'handle': 0, 'reasoning': 'Opening guard'},
        ]
    else:
        for x, y, desc in [
            (BUTTON_X,      BUTTON_Y,       'Button'),
            (BUTTON_X + 50, BUTTON_Y,       'Open side'),
            (BUTTON_X - 50, BUTTON_Y,       'Closed side'),
            (BUTTON_X,      BUTTON_Y - 100, '4-foot'),
        ]:
            candidates.append({'shot_type': 'Draw', 'target_x': x, 'target_y': y,
                                'handle': 0 if x <= BUTTON_X else 1,
                                'reasoning': f'Draw to {desc}'})

        for x, y, desc in [
            (BUTTON_X,       BUTTON_Y - 150, 'Centre guard'),
            (BUTTON_X + 100, BUTTON_Y - 150, 'Open side guard'),
            (BUTTON_X - 100, BUTTON_Y - 150, 'Closed side guard'),
        ]:
            candidates.append({'shot_type': 'Guard', 'target_x': x, 'target_y': y,
                                'handle': 0 if x <= BUTTON_X else 1, 'reasoning': desc})

        if not situation.get('free_guard_zone'):
            for stone in opp_stones[:5]:
                if stone and stone.get('in_play') and not stone.get('prepositioned'):
                    candidates.append({
                        'shot_type': 'Take-out',
                        'target_x': stone['x'], 'target_y': stone['y'],
                        'handle': 0, 'takeout_target': None,
                        'reasoning': f"Takeout at ({stone['x']:.0f},{stone['y']:.0f})",
                    })

        for stone in game_state.get_stones_in_play(game_state.current_team)[:3]:
            if stone:
                candidates.append({'shot_type': 'Freeze',
                                   'target_x': stone['x'] + 20, 'target_y': stone['y'] + 20,
                                   'handle': 0, 'reasoning': 'Freeze to own stone'})

    # Filter by game situation
    filtered = []
    for move in candidates:
        sdiff = situation['score_diff']
        ahead = situation['ahead_team']
        if sdiff > 2 and game_state.current_team == ahead:
            if move['shot_type'] in ['Take-out', 'Clearing', 'Double Take-out']:
                continue
        if sdiff < -1 and game_state.current_team != ahead:
            if move['shot_type'] == 'Guard' and situation['phase'] != 'opening':
                continue
        filtered.append(move)

    return filtered[:10]


# ---------------------------------------------------------------------------
# Model evaluation (from cell 7)
# ---------------------------------------------------------------------------

def evaluate_move_with_models(move, game_state, opening_sequences=None,
                               defensive_strategies=None, simulation_states=None,
                               models_q1_q5=None):
    if models_q1_q5 is None:
        models_q1_q5 = {}

    test_state = copy.deepcopy(game_state)
    test_state.apply_shot(move)

    situation = game_state.get_game_situation()
    features  = test_state.to_model_features()

    score        = 0.0
    scoring_prob = 0.5
    expected_pts = 0.0
    steal_prob   = 0.0
    reasoning    = []

    # Q8 opening sequences
    if situation['phase'] == 'opening' and opening_sequences is not None:
        try:
            import pandas as pd
            shot_type = move['shot_type']
            matches = opening_sequences[opening_sequences['sequence'].str.contains(shot_type, na=False)]
            if len(matches) > 0:
                avg_s = matches['scoring_rate'].mean()
                avg_p = matches['avg_points'].mean()
                score += avg_s * 0.4
                scoring_prob = avg_s
                expected_pts = avg_p
                reasoning.append(f"Q8: {shot_type} → {avg_s:.1%} scoring rate")
        except Exception:
            pass

    # Q9 defensive strategies
    if situation['is_defensive_team'] and defensive_strategies is not None:
        try:
            import pandas as pd
            matches = defensive_strategies[
                defensive_strategies['strategy_type'].str.contains(move['shot_type'], na=False)
            ]
            if len(matches) > 0:
                avg_pp = matches['avg_pp_points'].mean()
                score += (2.0 - avg_pp) * 0.3
                reasoning.append(f"Q9: allows {avg_pp:.2f} avg PP points")
        except Exception:
            pass

    # Q7 state-based mid-game
    if situation['phase'] == 'mid' and simulation_states is not None:
        try:
            import pandas as pd
            sh = min(int(features['hammer_stones_in_house']), 3)
            so = min(int(features['nonhammer_stones_in_house']), 3)
            cd = features['hammer_house_control_diff']
            ctrl = 'Hammer' if cd > 0 else ('Opponent' if cd < 0 else 'Neutral')
            sid = f"{sh}_{so}_{ctrl}_{int(features['end_parity'])}"
            match = simulation_states[simulation_states['state_id'] == sid]
            if len(match) > 0:
                sp = match['p_score'].iloc[0]
                ep = match['expected_points'].iloc[0]
                score += sp * 0.3
                scoring_prob = sp
                expected_pts = ep
                reasoning.append(f"Q7: state {sid} → {sp:.1%} scoring")
        except Exception:
            pass

    # Q1-Q5 XGBoost models
    if models_q1_q5 and any(v is not None for v in models_q1_q5.values()):
        try:
            import pandas as pd
            fd = {col: features.get(col, 0) for col in Q1_Q5_FEATURE_COLS}
            fdf = pd.DataFrame([fd])[Q1_Q5_FEATURE_COLS]

            if models_q1_q5.get('q1') is not None:
                p = models_q1_q5['q1'].predict_proba(fdf)[0]
                q1 = p[1] if len(p) > 1 else p[0]
                score += q1 * 0.3; scoring_prob = q1
                reasoning.append(f"Q1: {q1:.1%} scoring prob")

            if models_q1_q5.get('q3') is not None:
                p = models_q1_q5['q3'].predict_proba(fdf)[0]
                ep = sum(prob * pts for prob, pts in zip(p, [0, 1, 2, 3]))
                score += ep * 0.2; expected_pts = ep
                reasoning.append(f"Q3: {ep:.2f} expected pts")

            if models_q1_q5.get('q4') is not None:
                p = models_q1_q5['q4'].predict_proba(fdf)[0]
                bp = p[1] if len(p) > 1 else p[0]
                score += (1 - bp) * 0.1
                reasoning.append(f"Q4: {bp:.1%} blank prob")

            if models_q1_q5.get('q5') is not None:
                p = models_q1_q5['q5'].predict_proba(fdf)[0]
                sp = p[1] if len(p) > 1 else p[0]
                steal_prob = sp
                if situation['is_hammer_team']:
                    score += (1 - sp) * 0.15
                    reasoning.append(f"Q5: {sp:.1%} steal prob (lower=better for hammer)")
                else:
                    score += sp * 0.15
                    reasoning.append(f"Q5: {sp:.1%} steal prob (higher=better for non-hammer)")
        except Exception:
            pass

    # Heuristic fallback
    if features['hammer_stones_in_house'] > features['nonhammer_stones_in_house']:
        if situation['is_hammer_team']:
            scoring_prob = max(scoring_prob, 0.6); score += 0.2
        else:
            steal_prob = 0.3; score += 0.15
    if features['hammer_closest_dist'] < features['nonhammer_closest_dist']:
        if situation['is_hammer_team']:
            scoring_prob = max(scoring_prob, 0.55); score += 0.1

    if situation['is_hammer_team']:
        expected_pts = max(expected_pts, features['hammer_stones_in_house'] * 0.3)

    sdiff = situation['score_diff']
    ahead = situation['ahead_team']
    if sdiff > 2 and game_state.current_team == ahead:
        if move['shot_type'] in ['Guard', 'Draw']:
            score += 0.1
        else:
            score -= 0.1
        reasoning.append("Ahead: prefer conservative")
    elif sdiff < -1 and game_state.current_team != ahead:
        if move['shot_type'] in ['Take-out', 'Hit and Roll']:
            score += 0.15
        else:
            score -= 0.05
        reasoning.append("Behind: prefer aggressive")

    final_score = (score + scoring_prob * 0.3 + expected_pts * 0.2
                   - (features['nonhammer_closest_dist'] / 1000) * 0.1)

    return {
        'score': final_score,
        'scoring_prob': scoring_prob,
        'expected_points': expected_pts,
        'steal_prob': steal_prob,
        'reasoning': '; '.join(reasoning) if reasoning else 'Heuristic evaluation',
    }


# ---------------------------------------------------------------------------
# Decision engine (from cell 9)
# ---------------------------------------------------------------------------

def select_best_move(game_state, opening_sequences=None, defensive_strategies=None,
                     simulation_states=None, models_q1_q5=None):
    if models_q1_q5 is None:
        models_q1_q5 = {}

    candidates = generate_candidate_moves(game_state)
    if not candidates:
        return (
            {'shot_type': 'Draw', 'target_x': BUTTON_X, 'target_y': BUTTON_Y,
             'handle': 0, 'reasoning': 'Default draw (no candidates)'},
            {'score': 0.5}, []
        )

    evals = []
    for move in candidates:
        ev = evaluate_move_with_models(move, game_state,
                                       opening_sequences=opening_sequences,
                                       defensive_strategies=defensive_strategies,
                                       simulation_states=simulation_states,
                                       models_q1_q5=models_q1_q5)
        evals.append({'move': move, 'evaluation': ev})

    evals.sort(key=lambda e: e['evaluation']['score'], reverse=True)
    best = evals[0]
    best['move']['evaluation'] = best['evaluation']
    return best['move'], best['evaluation'], evals


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_models(models_dir):
    """
    Load Q1-Q5 XGBoost models from .pkl files in models_dir.
    Returns a dict {q1: model|None, q2: model|None, ...}
    """
    models = {}
    for q in ['q1', 'q2', 'q3', 'q4', 'q5']:
        path = os.path.join(models_dir, f'{q}_model.pkl')
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    models[q] = pickle.load(f)
            except Exception as e:
                print(f"Warning: could not load {path}: {e}")
                models[q] = None
        else:
            models[q] = None
    return models


# ---------------------------------------------------------------------------
# Animation (from cell 16) — returns GIF bytes for Gradio
# ---------------------------------------------------------------------------

def animate_simulation(game_log, game_state, throw_frames=10, pause_frames=4, fps=8):
    """
    Animate the simulation shots and return GIF as bytes (for Gradio display).

    Matches the style of game_simulation.ipynb cell 16:
      - Curved trajectory with curl effect and fading trail
      - Glow on the moving stone
      - Shot number labels on each stone
      - Highlight on newly landed stone
      - throw_frames frames of motion + pause_frames frames of result per shot
    """
    if not game_log.get('shots'):
        return None

    from PIL import Image as _PILImage

    # ------------------------------------------------------------------ #
    # State cache — reconstruct board at any shot index
    # ------------------------------------------------------------------ #
    state_cache = {}

    def _get_state(idx):
        if idx in state_cache:
            return state_cache[idx]
        s = GameState(game_state.team1_id, game_state.team2_id,
                      game_state.hammer_team, game_state.powerplay,
                      game_state.end_num, skip_prepositioned=True)
        s.stones = [st.copy() if st else None for st in game_state.stones]
        for i in range(len(s.stones)):
            if s.stones[i] and not s.stones[i].get('prepositioned', False):
                s.stones[i] = None
        s.delivered_shots = 0
        for i in range(idx):
            if i < len(game_log['shots']):
                s.apply_shot(game_log['shots'][i]['move'])
        state_cache[idx] = s
        return s

    # Pre-build all states up front
    n_shots = len(game_log['shots'])
    for i in range(n_shots + 1):
        _get_state(i)

    # ------------------------------------------------------------------ #
    # Drawing helpers (match notebook style)
    # ------------------------------------------------------------------ #
    circle_colors = ['#4169E1', '#1E90FF', '#00BFFF', '#0000CD']

    def _draw_sheet(ax):
        ax.set_facecolor('#e8f4f8')
        for i, radius in enumerate([HOUSE_RADIUS * 0.2, HOUSE_RADIUS * 0.4,
                                     HOUSE_RADIUS * 0.6, HOUSE_RADIUS]):
            ax.add_patch(patches.Circle(
                (BUTTON_X, BUTTON_Y), radius, fill=False,
                color=circle_colors[i],
                linewidth=2 if i == 3 else 1,
                linestyle='--' if radius < HOUSE_RADIUS else '-',
                alpha=0.7, zorder=1,
            ))
        # Button centre — small white disc with thin border (not a stone)
        ax.add_patch(patches.Circle(
            (BUTTON_X, BUTTON_Y), 6, facecolor='white',
            edgecolor='#555555', linewidth=1.5, zorder=10,
        ))
        ax.plot(BUTTON_X, BUTTON_Y, '+', color='#555555',
                markersize=8, markeredgewidth=1.5, zorder=11)
        # Hog line
        ax.axhline(y=HOG_LINE_Y, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(BUTTON_X - HOUSE_RADIUS * 1.85, HOG_LINE_Y + 8,
                'HOG LINE', fontsize=8, color='gray', alpha=0.8)

    def _draw_stones(ax, stone_list, highlight_snum=None):
        for st in stone_list:
            if not st or not st.get('in_play', True):
                continue
            is_new   = (highlight_snum is not None and st.get('shot_num') == highlight_snum)
            color    = 'red' if st['team'] == game_state.team1_id else 'yellow'
            msize    = 16 if is_new else 12
            ewidth   = 3  if is_new else 1
            bg_col   = 'lightgreen' if is_new else 'white'
            ax.plot(st['x'], st['y'], 'o', color=color,
                    markersize=msize, markeredgecolor='black',
                    markeredgewidth=ewidth, zorder=5)
            lbl = str(st.get('shot_num', '')) if st.get('shot_num') else ''
            if lbl:
                ax.text(st['x'] + 15, st['y'] + 15, lbl,
                        fontsize=9 if not is_new else 10,
                        color='darkred' if st['team'] == game_state.team1_id else 'darkorange',
                        fontweight='bold' if is_new else 'normal',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_col, alpha=0.8))

    def _set_axes(ax):
        ax.set_xlim(BUTTON_X - HOUSE_RADIUS * 2.2, BUTTON_X + HOUSE_RADIUS * 2.2)
        ax.set_ylim(BUTTON_Y - HOUSE_RADIUS * 2.4, BUTTON_Y + HOUSE_RADIUS * 1.6)
        ax.set_aspect('equal')
        ax.set_facecolor('#e8f4f8')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    def _capture(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=90)
        buf.seek(0)
        img = _PILImage.open(buf).copy()
        buf.close()
        return img

    # ------------------------------------------------------------------ #
    # Build trajectory path for a shot (30-point curve with curl)
    # ------------------------------------------------------------------ #
    def _trajectory(move):
        traj_x = np.linspace(BUTTON_X, move['target_x'], 30)
        traj_y = np.linspace(HOG_LINE_Y, move['target_y'], 30)
        for i in range(30):
            p = i / 30.0
            traj_x[i] += 40 * p * (1 - p) * 2 * np.sin(p * np.pi)
        return traj_x, traj_y

    # ------------------------------------------------------------------ #
    # Render all frames
    # ------------------------------------------------------------------ #
    frames    = []
    durations = []

    fig, ax = plt.subplots(1, 1, figsize=(10, 11))
    fig.patch.set_facecolor('#e8f4f8')

    for shot_idx, shot_log in enumerate(game_log['shots']):
        move    = shot_log['move']
        team    = shot_log['team']
        snum    = shot_log['shot_num']
        stype   = move['shot_type']
        s_color = '#FF0000' if team == game_state.team1_id else '#FFD700'

        state_before = _get_state(shot_idx)
        state_after  = _get_state(shot_idx + 1)
        traj_x, traj_y = _trajectory(move)

        # --- Throw frames: stone moving along trajectory ---
        for f in range(throw_frames):
            progress = f / float(throw_frames)
            idx      = int(progress * 29)

            ax.clear()
            _draw_sheet(ax)
            _draw_stones(ax, state_before.get_stones_in_play())

            # Fading trail
            trail_len = min(idx + 1, 8)
            trail_start = max(0, idx - trail_len + 1)
            for j in range(trail_start, idx + 1):
                alpha = (j - trail_start + 1) / trail_len * 0.5
                ax.plot(traj_x[j], traj_y[j], 'o', color=s_color,
                        markersize=4, alpha=alpha, zorder=3)

            # Glow + stone
            ax.plot(traj_x[idx], traj_y[idx], 'o', color=s_color,
                    markersize=28, alpha=0.25, zorder=9)
            ax.plot(traj_x[idx], traj_y[idx], 'o', color=s_color,
                    markersize=20, markeredgecolor='black',
                    markeredgewidth=2.5, zorder=10)

            ax.set_title(f"Shot {snum}/10: Team {team} — {stype}  (throwing...)",
                         fontsize=12, fontweight='bold', pad=10)
            _set_axes(ax)
            frames.append(_capture(fig))
            durations.append(int(1000 / fps))

        # --- Pause frames: stone landed, highlighted ---
        for f in range(pause_frames):
            ax.clear()
            _draw_sheet(ax)
            _draw_stones(ax, state_after.get_stones_in_play(), highlight_snum=snum)
            ax.set_title(f"Shot {snum}/10: Team {team} — {stype}  (landed)",
                         fontsize=12, fontweight='bold', pad=10)
            _set_axes(ax)
            frames.append(_capture(fig))
            durations.append(int(1000 / fps))

    # Final summary frame (hold 3 s)
    fs = game_log['final_score']
    ax.clear()
    _draw_sheet(ax)
    _draw_stones(ax, _get_state(n_shots).get_stones_in_play())
    ax.set_title(f"End complete — Team 1: {fs['team1']} pts   Team 2: {fs['team2']} pts",
                 fontsize=13, fontweight='bold', color='darkblue', pad=10)
    _set_axes(ax)
    frames.append(_capture(fig))
    durations.append(3000)

    plt.close(fig)

    if not frames:
        return None

    out = io.BytesIO()
    frames[0].save(
        out, format='GIF', save_all=True, append_images=frames[1:],
        duration=durations, loop=0, optimize=True,
    )
    return out.getvalue()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _make_stone(x, y, team, shot_num=0, prepos=False):
    return {'x': x, 'y': y, 'team': team, 'in_play': True,
            'shot_num': shot_num, 'prepositioned': prepos}


def _draw_stone(ax, x, y, team, team1_id, size, edge):
    color = '#FF4444' if team == team1_id else '#FFD700'
    ax.plot(x, y, 'o', color=color, markersize=size,
            markeredgecolor='black', markeredgewidth=edge, zorder=5)
