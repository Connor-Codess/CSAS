# Curling Strategy Analysis System (CSAS)

A comprehensive machine learning system for analyzing curling game strategies and simulating optimal play using XGBoost models.

## 📋 File Summary

### Essential Files (10 total)

**Notebooks (2):**
- `XGBoost.ipynb` - Model training and analysis
- `game_simulation.ipynb` - Simulation system

**Data Files (5):**
- `Stones.csv` - Raw data
- `end_state_model_df.csv` - Processed model data
- `opening_sequences_analysis.csv` - Opening analysis (for simulation)
- `defensive_strategies_analysis.csv` - Defense analysis (for simulation)
- `simulation_states.csv` - State data (for simulation)

**Documentation (3):**
- `README.md` - This file
- `MODEL_INTERPRETABILITY_GUIDE.md` - Model guide
- `FINAL_DATA_VALIDATION_REPORT.md` - Validation report

**Legacy (1, optional):**
- `Stones.ipynb` - Legacy processing notebook (not imported)

---

## 📁 Project Structure

### Core Notebooks

#### `XGBoost.ipynb`
**Purpose**: Main data pipeline, feature engineering, and model development notebook.

**Key Functions**:
- Loads and cleans raw curling data (Stones, Ends, Games, Teams, etc.)
- Performs feature engineering (spatial features, contextual features, interaction features)
- Trains Q1-Q9 XGBoost models to answer competition questions:
  - **Q1**: Hammer value in predicting scoring
  - **Q2**: Hammer value changes with geometry
  - **Q3**: Predictors of scoring magnitude
  - **Q4**: Conditions for blank ends
  - **Q5**: House control → steal probability
  - **Q6**: Power play impact
  - **Q7**: State representation for simulation
  - **Q8**: Most effective opening sequences
  - **Q9**: Defensive strategies against power play
  - **Q10**: Power play strategic timing

**Outputs**:
- `end_state_model_df.csv` - Processed end-level data with features and targets
- `opening_sequences_analysis.csv` - Analysis of effective opening shot sequences
- `defensive_strategies_analysis.csv` - Analysis of defensive strategies
- `simulation_states.csv` - State representations for simulation
- Model files (`.pkl`) - Saved XGBoost models for Q1-Q5 (optional)

**Usage**: Run cells sequentially to build models. Models can be saved as `.pkl` files for use in simulation.

---

#### `game_simulation.ipynb`
**Purpose**: Optimal curling end simulation system with animated visualization.

**Key Components**:
1. **GameState Class**: Tracks complete game state (stones, teams, scores, powerplay)
2. **Move Generation**: Generates candidate moves based on game situation
3. **Decision Engine**: Uses Q1-Q5 models and analysis data to select optimal moves
4. **Simulation Controller**: Runs complete end simulations
5. **Animation System**: Creates animated GIFs showing stones being thrown

**Features**:
- Mixed Doubles curling rules (5 stones per team + pre-positioned stones)
- Free Guard Zone enforcement
- Power play support
- Q1-Q5 model integration for decision making
- Opening sequence analysis integration
- Defensive strategy analysis integration
- Animated visualization with stone throwing motion

**Inputs** (optional, uses heuristics if not available):
- `opening_sequences_analysis.csv`
- `defensive_strategies_analysis.csv`
- `simulation_states.csv`
- Q1-Q5 model `.pkl` files

**Outputs**:
- `curling_simulation_end{N}.gif` - Animated visualization of simulated end
- Console output with shot-by-shot decisions

**Usage**:
```python
# Basic simulation
final_state, game_log = simulate_end(
    team1_id=1, 
    team2_id=2, 
    hammer_team=1, 
    end_num=1
)

# View animation
view_animation()  # Shows most recent animation
```

---

### Data Files

#### `Stones.csv`
**Purpose**: Raw stone-level data from curling games.

**Contains**: Shot-by-shot data including positions, shot types, teams, etc.

**Used by**: `XGBoost.ipynb` for feature engineering and model training.

**Note**: The notebook loads data from `/Users/connorbrady/Desktop/Curling_files/Stones.csv`. If your data is elsewhere, update the file path in `XGBoost.ipynb`.

---

#### `end_state_model_df.csv`
**Purpose**: Processed end-level dataset with engineered features and target variables.

**Contains**:
- Spatial features (closest distances, stones in house, house control)
- Contextual features (end number, powerplay, hammer indicators)
- Target variables (hammer_team_scored, hammer_team_points, steal, end_blank)

**Created by**: `XGBoost.ipynb`

**Used by**: Model training in `XGBoost.ipynb`

---

#### `opening_sequences_analysis.csv`
**Purpose**: Analysis of effective opening shot sequences (first 3 shots).

**Contains**: Sequence patterns, expected scoring rates, occurrence counts.

**Created by**: `XGBoost.ipynb` (Q8 analysis)

**Used by**: `game_simulation.ipynb` for opening phase decision making.

---

#### `defensive_strategies_analysis.csv`
**Purpose**: Analysis of defensive strategies against power play attacks.

**Contains**: Strategy types, effectiveness metrics, risk/reward analysis.

**Created by**: `XGBoost.ipynb` (Q9 analysis)

**Used by**: `game_simulation.ipynb` for defensive decision making.

---

#### `simulation_states.csv`
**Purpose**: State representations for mid-game simulation.

**Contains**: State IDs, feature vectors, outcome probabilities.

**Created by**: `XGBoost.ipynb` (Q7 analysis)

**Used by**: `game_simulation.ipynb` for mid-game decision making.

---

### Documentation

#### `MODEL_INTERPRETABILITY_GUIDE.md`
**Purpose**: Guide for interpreting XGBoost model results and feature importance.

**Contains**: Explanations of model outputs, feature meanings, interpretation guidelines.

---

## 🚀 Quick Start

### 1. Train Models
1. Open `XGBoost.ipynb`
2. Run all cells to train Q1-Q9 models
3. (Optional) Save Q1-Q5 models as `.pkl` files for simulation:
   ```python
   import pickle
   with open('q1_model.pkl', 'wb') as f:
       pickle.dump(xgb_q1, f)
   # Repeat for q2-q5
   ```

### 2. Run Simulation
1. Open `game_simulation.ipynb`
2. Run all cells to set up the simulation system
3. Run the test simulation cell to see an animated end
4. Use `view_animation()` to view saved animations

### 3. Analyze Results
- Check model outputs in `XGBoost.ipynb`
- View simulation animations in `game_simulation.ipynb`
- Review analysis CSVs for strategic insights

---

## 📊 Model Overview

### Q1-Q5: Core Prediction Models
- **Q1**: Binary classification - Will hammer team score?
- **Q2**: Hammer value by geometry (uses Q1 with feature interactions)
- **Q3**: Multi-class classification - Scoring magnitude (0, 1, 2, 3+)
- **Q4**: Binary classification - Will end be blank?
- **Q5**: Binary classification - Will non-hammer team steal?

### Q6-Q10: Strategic Analysis
- **Q6**: Power play impact analysis
- **Q7**: State representation for simulation
- **Q8**: Opening sequence effectiveness
- **Q9**: Defensive strategy analysis
- **Q10**: Power play timing optimization

---

## 🔧 Dependencies

### Required Packages
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `xgboost` - Gradient boosting models
- `matplotlib` - Visualization
- `scikit-learn` - Model evaluation
- `seaborn` - Statistical visualization

### Optional Packages
- `pillow` - For saving animation GIFs
- `ipympl` - For interactive matplotlib in Jupyter

### Installation
```bash
pip install pandas numpy xgboost matplotlib scikit-learn seaborn pillow
```

---

## 📝 Notes

- Raw data files (Stones.csv, Ends.csv, etc.) are expected to be in `/Users/connorbrady/Desktop/Curling_files/`
- Update file paths in `XGBoost.ipynb` if your data is located elsewhere
- Model `.pkl` files are optional - simulation works with heuristics if models aren't available
- Animation GIFs are automatically saved when running simulations (can be deleted after viewing)
- `Stones.ipynb` is a legacy data processing notebook - not required for current system

---

## 🎯 Key Features

- **Comprehensive Analysis**: 10 different models answering strategic curling questions
- **Optimal Play Simulation**: AI-driven decision making for each shot
- **Mixed Doubles Rules**: Full implementation of mixed doubles curling rules
- **Animated Visualization**: Watch stones being thrown with realistic motion
- **Model Integration**: Uses trained models for intelligent decision making
- **Strategic Insights**: Analysis of opening sequences, defensive strategies, and power play timing

---

## 📧 Support

For questions or issues, refer to the documentation in each notebook or the `MODEL_INTERPRETABILITY_GUIDE.md`.

