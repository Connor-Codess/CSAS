"""
train_models.py — Train Q1/Q3/Q4/Q5 XGBoost models and save to models/

Run from curling_cv/:
    python train_models.py

Reads:  ../end_state_model_df.csv
Writes: models/q1_model.pkl  (hammer scoring probability)
        models/q3_model.pkl  (points magnitude 0/1/2/3+)
        models/q4_model.pkl  (blank end probability)
        models/q5_model.pkl  (steal probability)
"""

import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

try:
    import xgboost as xgb
    XGBOOST = True
    print("XGBoost available")
except ImportError:
    XGBOOST = False
    print("XGBoost not found — using sklearn GradientBoostingClassifier")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, '..', 'end_state_model_df.csv')
OUT_DIR   = os.path.join(HERE, 'models')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load and prepare data  (mirrors XGBoost.ipynb cell 31)
# ---------------------------------------------------------------------------
print(f"\nLoading {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape}")

# Match notebook preprocessing exactly
df['powerplay_missing'] = df['powerplay'].isna().astype(int)
df['powerplay'] = df['powerplay'].fillna(0)

# Drop constant / identifier / target columns to get feature_cols
exclude = {
    'competitionid', 'sessionid', 'gameid', 'endid',
    'hammer_team_scored', 'hammer_team_points', 'hammer_points_bucket',
    'end_blank', 'steal', 'net_points_for_hammer',
    'missing_shot3_snapshot', 'any_missing_coordinates',
}
feature_cols = [c for c in df.columns if c not in exclude]
# Keep only numeric columns
feature_cols = [c for c in feature_cols if pd.to_numeric(df[c], errors='coerce').notna().all()]
print(f"Feature columns ({len(feature_cols)}): {feature_cols}\n")

group_col = 'gameid' if 'gameid' in df.columns else 'game_id'
splitter  = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)


def _make_xgb_clf(**kwargs):
    if XGBOOST:
        return xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='logloss',
            use_label_encoder=False, **kwargs
        )
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42, loss='log_loss'
    )


def _split(X, y, groups):
    train_idx, _ = next(splitter.split(X, y, groups))
    return X.iloc[train_idx], y.iloc[train_idx]


def _save(model, name):
    path = os.path.join(OUT_DIR, f'{name}_model.pkl')
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved {path}  ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# Q1 — Hammer scoring probability (binary)
# ---------------------------------------------------------------------------
print("Training Q1: hammer scoring probability ...")
y1     = df['hammer_team_scored'].dropna()
X1     = df.loc[y1.index, feature_cols]
g1     = df.loc[y1.index, group_col].values
X1t, y1t = _split(X1, y1, g1)

m1 = _make_xgb_clf()
m1.fit(X1t, y1t)
_save(m1, 'q1')
print(f"  Train samples: {len(X1t)}")


# ---------------------------------------------------------------------------
# Q3 — Scoring magnitude (multi-class 0/1/2/3+)
# ---------------------------------------------------------------------------
print("\nTraining Q3: scoring magnitude ...")
y3     = df['hammer_points_bucket'].dropna()
X3     = df.loc[y3.index, feature_cols]
g3     = df.loc[y3.index, group_col].values
X3t, y3t = _split(X3, y3, g3)

if XGBOOST:
    m3 = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='mlogloss',
        objective='multi:softprob', use_label_encoder=False,
    )
else:
    m3 = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42, loss='log_loss'
    )
m3.fit(X3t, y3t)
_save(m3, 'q3')
print(f"  Train samples: {len(X3t)}")


# ---------------------------------------------------------------------------
# Q4 — Blank end probability (binary, heavily imbalanced)
# ---------------------------------------------------------------------------
print("\nTraining Q4: blank end probability ...")
y4     = df['end_blank'].dropna()
X4     = df.loc[y4.index, feature_cols]
g4     = df.loc[y4.index, group_col].values
X4t, y4t = _split(X4, y4, g4)

pos_w  = len(y4t) / (2 * max(sum(y4t == 1), 1))
neg_w  = len(y4t) / (2 * max(sum(y4t == 0), 1))
sw     = compute_sample_weight('balanced', y4t)
spw    = pos_w / neg_w if neg_w > 0 else 1.0

if XGBOOST:
    m4 = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss',
        use_label_encoder=False, scale_pos_weight=spw,
    )
else:
    m4 = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42, loss='log_loss'
    )
m4.fit(X4t, y4t, sample_weight=sw)
_save(m4, 'q4')
print(f"  Train samples: {len(X4t)}  (pos={sum(y4t==1)} blank ends)")


# ---------------------------------------------------------------------------
# Q5 — Steal probability (binary)
# ---------------------------------------------------------------------------
print("\nTraining Q5: steal probability ...")
y5     = df['steal'].dropna()
X5     = df.loc[y5.index, feature_cols]
g5     = df.loc[y5.index, group_col].values
X5t, y5t = _split(X5, y5, g5)

m5 = _make_xgb_clf()
m5.fit(X5t, y5t)
_save(m5, 'q5')
print(f"  Train samples: {len(X5t)}")

# ---------------------------------------------------------------------------
# Save the feature column list so pipeline.py can use the exact same order
# ---------------------------------------------------------------------------
cols_path = os.path.join(OUT_DIR, 'feature_cols.pkl')
with open(cols_path, 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"\nSaved feature column list to {cols_path}")

print("\nAll models saved to models/")
print("Done.")
