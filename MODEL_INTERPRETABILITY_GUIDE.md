# Model Interpretability Guide
## Comprehensive Guide to Understanding All 7 Models and Visualizations

---

## Table of Contents

1. [Overview](#overview)
2. [Q1: Hammer Value in Predicting Scoring](#q1-hammer-value-in-predicting-scoring)
3. [Q2: Hammer Value Changes with Geometry](#q2-hammer-value-changes-with-geometry)
4. [Q3: Predictors of Scoring Magnitude](#q3-predictors-of-scoring-magnitude)
5. [Q4: Conditions for Blank Ends](#q4-conditions-for-blank-ends)
6. [Q5: House Control → Steal Probability](#q5-house-control--steal-probability)
7. [Q6: Power Play Impact](#q6-power-play-impact)
8. [Q7: State Representation for Simulation](#q7-state-representation-for-simulation)
9. [Common Metrics and Interpretations](#common-metrics-and-interpretations)
10. [Feature Importance Guide](#feature-importance-guide)

---

## Overview

This guide provides detailed explanations for interpreting all 7 models built to answer curling competition questions. Each model uses different techniques and produces different types of outputs. Understanding how to read these models and their visualizations is crucial for extracting actionable insights.

**Key Concepts:**
- **Hammer**: The last stone advantage in an end
- **Shot 3 Snapshot**: The state of the game after the 3rd shot of the end
- **House Control**: Which team has more stones closer to the button (center)
- **Steal**: When the non-hammer team scores
- **Blank End**: An end where no team scores

---

## Q1: Hammer Value in Predicting Scoring

### Question
**"How valuable is hammer (last stone advantage) in predicting whether the hammer team scores in the end?"**

### Model Type
**Binary Classification using XGBoost**
- **Target Variable**: `hammer_team_scored` (1 = hammer team scored, 0 = did not score)
- **Input Features**: 17 spatial and contextual features from Shot 3 snapshot
- **Algorithm**: XGBoost Classifier

### Model Performance Metrics

**ROC-AUC: 0.623 (62.3%)**
- **Interpretation**: The model can distinguish between scoring and non-scoring ends 62.3% better than random guessing
- **Scale**: 0.5 = random, 1.0 = perfect prediction
- **Assessment**: Moderate predictive power - early game state (Shot 3) provides some signal but many factors influence final outcome

**Accuracy: 60.9%**
- **Interpretation**: The model correctly predicts whether the hammer team will score in 60.9% of ends
- **Baseline**: 56.4% (proportion of ends where hammer team actually scored)
- **Improvement**: 4.5 percentage points above baseline

**F1-Score: 0.672**
- **Interpretation**: Balanced measure of precision and recall
- **Scale**: 0 to 1, higher is better
- **Assessment**: Good balance between correctly identifying scoring and non-scoring ends

### How to Read Q1_Hammer_Value_Analysis.png

The visualization contains 4 panels:

#### Panel 1: Feature Importance Bar Chart
- **What it shows**: The top 10 most important features for predicting scoring
- **How to read**: 
  - Longer bars = more important features
  - Values range from 0 to ~0.07 (importance scores)
  - **Key features**:
    - `log_hammer_dist` (0.070): Logarithm of hammer team's closest stone distance
    - `hammer_closest_dist` (0.070): Distance from hammer team's closest stone to button
    - `stones_ratio` (0.069): Ratio of hammer to non-hammer stones in house
- **Interpretation**: Distance metrics are most predictive - closer stones = higher scoring probability

#### Panel 2: ROC Curve
- **What it shows**: Model's ability to distinguish between scoring and non-scoring ends at different probability thresholds
- **How to read**:
  - X-axis: False Positive Rate (incorrectly predicting score when no score)
  - Y-axis: True Positive Rate (correctly predicting score)
  - Diagonal line: Random guessing baseline
  - Blue curve: Model performance
- **Interpretation**: 
  - Curve above diagonal = better than random
  - Area under curve (AUC) = 0.623 means model has moderate discriminative power
  - Steeper curve in upper left = better at identifying high-probability scoring situations

#### Panel 3: Precision-Recall Curve
- **What it shows**: Trade-off between precision (accuracy of positive predictions) and recall (coverage of actual positives)
- **How to read**:
  - X-axis: Recall (what % of actual scores we catch)
  - Y-axis: Precision (what % of our predictions are correct)
  - Higher curve = better performance
- **Interpretation**: 
  - Shows model can achieve good precision (few false positives) or good recall (catch most scores), but not both simultaneously
  - Useful for setting decision thresholds based on business needs

#### Panel 4: Confusion Matrix
- **What it shows**: Actual vs predicted outcomes in a 2x2 grid
- **How to read**:
  - Top-left: True Negatives (correctly predicted no score)
  - Top-right: False Positives (predicted score but didn't score)
  - Bottom-left: False Negatives (predicted no score but scored)
  - Bottom-right: True Positives (correctly predicted score)
- **Interpretation**: 
  - Diagonal cells (top-left, bottom-right) = correct predictions
  - Off-diagonal = errors
  - Larger numbers in diagonal = better model

### Key Insights from Q1

1. **Hammer provides moderate advantage**: 62.3% predictive power suggests hammer is valuable but not deterministic
2. **Distance is critical**: Closest stone distance is the strongest predictor
3. **Early game state matters**: Shot 3 geometry provides meaningful signal about final outcome
4. **Baseline scoring rate**: Hammer team scores in 56.4% of ends (slight advantage)

### Practical Applications

- **Strategy**: Teams with hammer should prioritize getting stones close to button early
- **Risk assessment**: When hammer team's closest stone is far (>300 units), scoring probability drops significantly
- **Game planning**: Use model predictions to inform shot selection and risk tolerance

---

## Q2: Hammer Value Changes with Geometry

### Question
**"How does hammer's value change depending on early-house geometry after Shot 3?"**

### Model Type
**Interaction Analysis + Conditional Effects**
- **Approach**: Uses Q1 model to analyze how scoring probability varies across different house geometry conditions
- **Not a separate model**: Analyzes conditional probabilities from Q1 model predictions

### How to Read Q2_Hammer_Value_by_Geometry.png

The visualization contains 4 panels showing how hammer value (scoring probability) changes with geometry:

#### Panel 1: Actual Scoring Rate Heatmap
- **What it shows**: Real scoring rates by house geometry combinations
- **Axes**:
  - Y-axis: Hammer stones in house (0, 1, 2, 3+)
  - X-axis: House control (Opponent Control, Neutral, Hammer Control)
- **Color scale**: 
  - Red = low scoring probability (<0.4)
  - Yellow = medium (0.4-0.6)
  - Green = high (>0.6)
- **How to read**:
  - Each cell shows actual scoring rate for that geometry combination
  - Numbers in cells = probability (e.g., 0.622 = 62.2% scoring rate)
- **Key patterns**:
  - **Highest scoring**: 2+ hammer stones with Hammer Control or Neutral
  - **Lowest scoring**: 0 hammer stones with Opponent Control
  - **Range**: Best geometry (0.8+) vs worst (0.4-) = ~40 percentage point difference

#### Panel 2: Predicted Scoring Probability Heatmap
- **What it shows**: Model predictions for scoring probability by geometry
- **Structure**: Same as Panel 1 but shows what the model predicts
- **Interpretation**: 
  - Compare to Panel 1 to see if model captures real patterns
  - Close match = model understands geometry effects
  - Differences = areas where model may need improvement

#### Panel 3: Scoring Probability by Distance to Button
- **What it shows**: How scoring probability changes with distance from button
- **X-axis**: Distance bins (Very Close, Close, Far, Very Far)
- **Y-axis**: Scoring probability (0 to 1)
- **Bars**: 
  - Blue = Actual scoring rate
  - Coral = Model prediction
- **Interpretation**:
  - Closer to button = higher scoring probability
  - Steep decline as distance increases
  - Model predictions should track actual rates closely

#### Panel 4: Scoring Probability vs House Control Difference
- **What it shows**: Relationship between house control difference and scoring probability
- **X-axis**: House control difference (negative = opponent control, positive = hammer control)
- **Y-axis**: Scoring probability
- **Lines**:
  - Blue circles = Actual rates
  - Coral squares = Model predictions
- **Interpretation**:
  - Positive control difference = higher scoring probability
  - Relationship may not be perfectly linear (interactions with other factors)
  - Horizontal line at 0.5 = break-even point

### Key Insights from Q2

1. **Geometry dramatically affects hammer value**:
   - Best case: 2+ stones + Hammer Control = ~80% scoring probability
   - Worst case: 0 stones + Opponent Control = ~40% scoring probability
   - **40 percentage point swing** based on geometry alone

2. **Distance is critical**:
   - Very Close (<100 units): ~70% scoring rate
   - Very Far (>300 units): ~40% scoring rate
   - **30 percentage point difference**

3. **House control matters**:
   - Hammer Control: Higher scoring rates across all stone counts
   - Opponent Control: Lower scoring rates, especially with fewer stones

4. **Non-linear interactions**:
   - Having 2 stones doesn't always mean higher probability than 1 stone
   - Control status interacts with stone count in complex ways

### Practical Applications

- **Shot selection**: Prioritize shots that improve geometry (more stones, better control)
- **Risk assessment**: When geometry is poor, consider defensive strategies
- **Game state evaluation**: Use geometry to assess current advantage/disadvantage
- **Strategy adjustment**: Modify aggression based on geometry (aggressive when favorable, conservative when unfavorable)

---

## Q3: Predictors of Scoring Magnitude

### Question
**"Which early spatial features after Shot 3 are the strongest predictors of scoring magnitude (0, 1, 2, 3+ points)?"**

### Model Type
**Multi-class Classification using XGBoost**
- **Target Variable**: `hammer_points_bucket` (0, 1, 2, 3+ points)
- **Classes**: 4 classes representing different scoring magnitudes
- **Algorithm**: XGBoost Classifier with multi-class objective

### Model Performance Metrics

**Accuracy: 42.9%**
- **Interpretation**: Model correctly predicts exact point bucket in 42.9% of ends
- **Baseline**: 25% (random guessing among 4 classes)
- **Improvement**: 17.9 percentage points above random
- **Assessment**: Moderate performance - predicting exact magnitude is harder than binary scoring

**F1-Score (Macro): 0.290**
- **Interpretation**: Average F1 across all 4 classes
- **Assessment**: Lower than binary models because multi-class is inherently harder
- **Note**: Some classes (like 3+ points) are rare, making them harder to predict

**F1-Score (Weighted): 0.404**
- **Interpretation**: F1 weighted by class frequency
- **Assessment**: Better than macro because it accounts for class imbalance
- **More relevant**: Reflects performance on common outcomes

### How to Read Q3_Scoring_Magnitude_Predictors.png

The visualization contains 4 panels:

#### Panel 1: Top 15 Feature Importance
- **What it shows**: Most important features for predicting scoring magnitude
- **How to read**:
  - Horizontal bars, longer = more important
  - Top features:
    - `log_hammer_dist` (0.073): Logarithm of hammer distance
    - `stones_ratio` (0.066): Ratio of stones in house
    - `hammer_stones_x_dist` (0.065): Interaction of stones and distance
- **Interpretation**: 
  - Distance and stone ratios are key
  - Interactions between features matter (e.g., stones × distance)
  - End number is also important (game context)

#### Panel 2: Confusion Matrix
- **What it shows**: Actual vs predicted point buckets in a 4×4 grid
- **How to read**:
  - Rows = Actual point buckets
  - Columns = Predicted point buckets
  - Diagonal = correct predictions
  - Off-diagonal = errors
- **Color intensity**: Darker = more cases
- **Key patterns**:
  - Model is best at predicting 0 and 1 point buckets (most common)
  - Tends to confuse adjacent buckets (e.g., predicts 1 when actual is 2)
  - Rare events (3+ points) are harder to predict accurately

#### Panel 3: Average Points by Top Feature
- **What it shows**: How average points scored varies with the most important feature
- **X-axis**: Feature value bins or categories
- **Y-axis**: Average points scored
- **Bars**: Show average points for each feature value
- **Interpretation**:
  - Steeper differences = feature is more predictive
  - Non-linear relationships may appear
  - Helps understand feature effects on magnitude

#### Panel 4: Prediction Confidence by Points Bucket
- **What it shows**: Box plots of prediction probabilities for each actual point bucket
- **X-axis**: Actual point buckets (0, 1, 2, 3+)
- **Y-axis**: Predicted probability for that bucket
- **Box plots**: Show distribution of probabilities
- **Interpretation**:
  - Higher boxes = model is more confident
  - Narrow boxes = consistent predictions
  - Wide boxes = uncertain predictions
  - Ideal: High probability boxes on diagonal, low elsewhere

### Key Insights from Q3

1. **Top predictors of magnitude**:
   - Distance metrics (log_hammer_dist, hammer_closest_dist)
   - Stone ratios and interactions
   - House control differences
   - End number (game context matters)

2. **Average points by key features**:
   - **Hammer stones in house**:
     - 0 stones: 1.09 points
     - 1 stone: 1.01 points
     - 2 stones: 1.19 points
     - 3 stones: 1.00 points
   - **House control**:
     - Negative: 0.96 points
     - Neutral: 1.15 points
     - Positive: 1.15 points

3. **Model limitations**:
   - Predicting exact magnitude is challenging (42.9% accuracy)
   - Model is better at distinguishing 0 vs non-zero than exact amounts
   - Rare events (3+ points) are hardest to predict

4. **Feature interactions matter**:
   - Simple features (distance, stones) combined create stronger signals
   - Non-linear relationships exist (e.g., 2 stones doesn't always mean more points than 1)

### Practical Applications

- **Shot selection**: Features that predict higher magnitudes should be prioritized
- **Risk assessment**: When features suggest low magnitude, consider defensive play
- **Game planning**: Use magnitude predictions to inform strategy (aggressive for high magnitude, conservative for low)
- **Expectation setting**: Understand that exact point prediction is difficult, but direction (high/low) is more reliable

---

## Q4: Conditions for Blank Ends

### Question
**"Under what early-house conditions is a blank end most likely?"**

### Model Type
**Binary Classification with Class Weights (XGBoost)**
- **Target Variable**: `end_blank` (1 = blank end, 0 = scored end)
- **Challenge**: Highly imbalanced data (99.8% scored, 0.2% blank)
- **Solution**: Uses class weights to handle imbalance
- **Algorithm**: XGBoost Classifier with `scale_pos_weight` parameter

### Model Performance Metrics

**ROC-AUC: N/A (too few blank ends)**
- **Issue**: Only 5 blank ends in dataset (0.19%)
- **Interpretation**: Insufficient data for reliable model evaluation
- **Note**: Model may not generalize well due to extreme rarity

**F1-Score: 0.000**
- **Interpretation**: Model struggles to identify blank ends
- **Reason**: Extreme class imbalance makes prediction very difficult
- **Assessment**: Model has limited utility for blank end prediction

**Precision: 1.000, Recall: 0.000**
- **Interpretation**: 
  - When model predicts blank, it's always right (high precision)
  - But it almost never predicts blank (low recall)
  - Model is very conservative due to rarity

### How to Read Q4_Blank_End_Conditions.png

The visualization contains 4 panels:

#### Panel 1: Top 10 Feature Importance
- **What it shows**: Features most important for predicting blank ends
- **Top features**:
  - `stones_ratio` (0.134): Ratio of stones in house
  - `hammer_stones_x_dist` (0.091): Interaction of stones and distance
  - `log_hammer_dist` (0.089): Logarithm of distance
- **Interpretation**: 
  - Stone ratios and distances are key
  - Interactions between features matter
  - Model tries to find patterns despite limited data

#### Panel 2: Stones in House Comparison
- **What it shows**: Distribution of hammer stones in house for blank vs scored ends
- **X-axis**: Number of hammer stones in house (0, 1, 2, 3+)
- **Y-axis**: Proportion of ends
- **Bars**:
  - Dark red = Blank ends
  - Blue = Scored ends
- **Interpretation**:
  - Blank ends tend to have fewer stones in house
  - Average: 0.80 stones for blank vs 1.28 for scored
  - Pattern suggests defensive/neutral play leads to blanks

#### Panel 3: House Control Difference Comparison
- **What it shows**: Box plot comparing house control for blank vs scored ends
- **Y-axis**: House control difference (negative = opponent control, positive = hammer control)
- **Boxes**: Show distribution of control values
- **Interpretation**:
  - Blank ends: Average -0.20 (slight opponent control)
  - Scored ends: Average +0.03 (slight hammer control)
  - Neutral/opponent control favors blanks

#### Panel 4: Distance to Button Comparison
- **What it shows**: Box plot comparing distances for blank vs scored ends
- **Y-axis**: Closest distance to button
- **Boxes**: Show distribution of distances
- **Interpretation**:
  - Blank ends: Average 214.4 units
  - Scored ends: Average 167.5 units
  - **Further from button = more likely blank**

### Key Insights from Q4

1. **Blank end characteristics**:
   - **Fewer stones in house**: 0.80 vs 1.28 average
   - **Slight opponent control**: -0.20 vs +0.03 average
   - **Further from button**: 214.4 vs 167.5 units average
   - **Fewer non-hammer stones**: 1.00 vs 1.25 average

2. **Model limitations**:
   - Only 5 blank ends in dataset (extreme rarity)
   - Model cannot reliably predict blanks
   - Patterns identified may be spurious due to small sample

3. **Top predictive features**:
   - Stone ratios (balanced house = more blanks)
   - Distance metrics (far from button = more blanks)
   - Feature interactions (complex relationships)

4. **Practical implications**:
   - Blank ends are very rare (0.2% of ends)
   - Conditions that favor blanks: neutral house, far from button, few stones
   - These conditions suggest defensive/neutral play

### Practical Applications

- **Strategy**: When aiming for blank, maintain neutral house, avoid close stones
- **Risk assessment**: Recognize that blank prediction is extremely difficult
- **Game planning**: Understand that blanks are rare and hard to force
- **Caution**: Model predictions for blanks should be treated with skepticism due to data limitations

---

## Q5: House Control → Steal Probability

### Question
**"How does early house control relate to the probability that the non-hammer team steals?"**

### Model Type
**Binary Classification using XGBoost**
- **Target Variable**: `steal` (1 = non-hammer team scored, 0 = hammer team scored or blank)
- **Balance**: Relatively balanced (55.6% no steal, 44.4% steal)
- **Algorithm**: XGBoost Classifier

### Model Performance Metrics

**ROC-AUC: 0.632 (63.2%)**
- **Interpretation**: Model can distinguish steal vs no-steal 63.2% better than random
- **Assessment**: Moderate predictive power - similar to Q1
- **Comparison**: Slightly better than hammer scoring prediction (0.623)

**Accuracy: 61.3%**
- **Interpretation**: Model correctly predicts steal outcome in 61.3% of ends
- **Baseline**: 55.6% (proportion with no steal)
- **Improvement**: 5.7 percentage points above baseline

**F1-Score: 0.536**
- **Interpretation**: Balanced precision and recall
- **Assessment**: Moderate performance, room for improvement

### How to Read Q5_Steal_Probability.png

The visualization contains 4 panels:

#### Panel 1: Top 10 Feature Importance
- **What it shows**: Most important features for predicting steals
- **Top features**:
  - `hammer_closest_dist` (0.069): Distance of hammer's closest stone
  - `log_nonhammer_dist` (0.069): Logarithm of non-hammer distance
  - `hammer_house_control_diff` (0.067): House control difference
- **Interpretation**: 
  - Distance metrics are critical (both teams' distances matter)
  - House control is important but not dominant
  - Opponent stones in house also matter

#### Panel 2: Steal Probability vs House Control
- **What it shows**: How steal probability changes with house control difference
- **X-axis**: House control difference bins (negative = opponent control, positive = hammer control)
- **Y-axis**: Steal probability (0 to 1)
- **Line**: Shows relationship between control and steal rate
- **Horizontal line at 0.5**: Break-even point
- **Interpretation**:
  - Negative control (opponent advantage) = higher steal probability
  - Positive control (hammer advantage) = lower steal probability
  - Relationship may not be perfectly linear
  - **Key finding**: Even with hammer control, steals still occur ~43-49%

#### Panel 3: Steal Probability by Opponent Stones
- **What it shows**: How steal probability varies with number of opponent stones in house
- **X-axis**: Opponent stones in house (0, 1, 2, 3+)
- **Y-axis**: Steal probability
- **Bars**: Show steal rate for each stone count
- **Interpretation**:
  - More opponent stones = generally higher steal probability
  - But relationship may not be monotonic
  - Other factors (distance, control) interact with stone count

#### Panel 4: ROC Curve
- **What it shows**: Model's ability to distinguish steals at different thresholds
- **X-axis**: False Positive Rate
- **Y-axis**: True Positive Rate
- **Blue curve**: Model performance (AUC = 0.632)
- **Diagonal line**: Random baseline
- **Interpretation**:
  - Curve above diagonal = better than random
  - Moderate discriminative power
  - Can identify high-risk steal situations

### Key Insights from Q5

1. **Steal rates by house control**:
   - **Negative control (< -0.5)**: 48.7% steal rate
   - **Neutral (-0.5 to 0.5)**: 43.3% steal rate
   - **Positive control (> 0.5)**: 43.5% steal rate
   - **Finding**: Even with hammer control, steals occur ~43-49% of the time

2. **Conditions that favor steals**:
   - **Opponent control**: Higher steal probability
   - **More opponent stones**: Generally higher steal rate
   - **Hammer stones far from button**: Higher steal probability
   - **Average conditions**:
     - Steal: -0.00 control diff, 1.24 opponent stones
     - No steal: +0.06 control diff, 1.25 opponent stones

3. **Top predictive features**:
   - Distance metrics (both teams matter)
   - House control difference
   - Opponent stones in house
   - Stone ratios and interactions

4. **Model performance**:
   - Moderate predictive power (63.2% AUC)
   - Better than random but room for improvement
   - Can identify high-risk situations

### Practical Applications

- **Defensive strategy**: When opponent has control, be aware of steal risk
- **Risk assessment**: Use model to evaluate steal probability in current game state
- **Shot selection**: Prioritize shots that reduce opponent's steal opportunities
- **Game planning**: Understand that steals are common (44% of ends) even with hammer

---

## Q6: Power Play Impact

### Question
**"How do power play situations (if present) shift optimal aggression/defense and scoring outcomes?"**

### Model Type
**Comparative Analysis (Not a predictive model)**
- **Approach**: Compare outcomes between Power Play and Normal Play ends
- **Method**: Statistical comparison of outcome distributions
- **Note**: Only 11% of ends have Power Play data (290 ends)

### Data Summary

- **Power Play ends**: 290 (11.0%)
- **Normal ends**: 2,347 (89.0%)
- **Limitation**: Small sample size for Power Play analysis

### How to Read Q6_Power_Play_Impact.png

The visualization contains 4 panels:

#### Panel 1: Outcome Comparison Bar Chart
- **What it shows**: Side-by-side comparison of outcomes for Power Play vs Normal Play
- **Outcomes compared**:
  - Scoring Rate: % of ends where hammer team scored
  - Avg Points: Average points scored by hammer team
  - Steal Rate: % of ends where non-hammer team stole
  - Blank Rate: % of blank ends
- **Bars**:
  - Orange = Power Play
  - Blue = Normal Play
- **Interpretation**:
  - **Scoring Rate**: Power Play 54.5% vs Normal 56.6% (-3.8% difference)
  - **Avg Points**: Power Play 1.166 vs Normal 1.106 (+5.4% difference)
  - **Steal Rate**: Power Play 46.9% vs Normal 44.1% (+6.2% difference)
  - **Blank Rate**: Power Play 0.3% vs Normal 0.2% (+102% difference, but very small absolute)

#### Panel 2: Points Distribution Comparison
- **What it shows**: Distribution of points scored (0, 1, 2, 3+) for Power Play vs Normal
- **X-axis**: Points scored
- **Y-axis**: Proportion of ends
- **Bars**:
  - Orange = Power Play
  - Blue = Normal Play
- **Interpretation**:
  - Compare distributions to see if Power Play changes point patterns
  - Look for shifts in distribution (e.g., more high-scoring ends)

#### Panel 3: House Geometry Comparison
- **What it shows**: Average house geometry features for Power Play vs Normal
- **Features compared**:
  - Hammer stones in house
  - Non-hammer stones in house
  - House control difference
- **Bars**:
  - Orange = Power Play
  - Blue = Normal Play
- **Interpretation**:
  - Compare geometry to see if Power Play changes game setup
  - Differences may explain outcome differences

#### Panel 4: Power Play Type Analysis
- **What it shows**: Outcomes by Power Play type (if multiple types exist)
- **X-axis**: Power Play type (1, 2, etc.)
- **Y-axis**: Scoring rate or average points
- **Interpretation**:
  - Compare different Power Play types
  - May show "Insufficient Power Play type variation" if only one type exists

### Key Insights from Q6

1. **Scoring outcomes**:
   - **Scoring rate**: Slightly lower in Power Play (54.5% vs 56.6%)
   - **Average points**: Higher in Power Play (1.166 vs 1.106, +5.4%)
   - **Interpretation**: When hammer team scores in Power Play, they score more points on average

2. **Steal rates**:
   - **Power Play**: 46.9% steal rate
   - **Normal**: 44.1% steal rate
   - **Difference**: +6.2% higher steal rate in Power Play
   - **Interpretation**: Power Play may favor non-hammer team slightly

3. **Blank ends**:
   - **Power Play**: 0.3% blank rate
   - **Normal**: 0.2% blank rate
   - **Difference**: Very small absolute difference
   - **Interpretation**: Power Play doesn't significantly change blank rate

4. **Overall pattern**:
   - Power Play may create more volatile outcomes (higher average points when scoring, higher steal rate)
   - Slightly lower scoring rate but higher magnitude when scoring
   - Suggests Power Play changes game dynamics

### Practical Applications

- **Strategy adjustment**: In Power Play, be aware of higher steal risk
- **Risk assessment**: Power Play may increase volatility (higher highs, lower lows)
- **Game planning**: Adjust expectations - scoring rate slightly lower but magnitude higher
- **Caution**: Small sample size (11% of ends) limits generalizability

---

## Q7: State Representation for Simulation

### Question
**"Can we build a state representation after Shot 3 that supports simulated play (Markov-style) to compare strategies under different initial states?"**

### Model Type
**State Space Analysis (Not a predictive model)**
- **Approach**: Create discrete state representation from continuous features
- **Method**: Group similar game states and calculate transition probabilities
- **Output**: State table with outcome probabilities for each state

### State Definition

States are defined by 4 dimensions:
1. **Hammer stones in house**: 0, 1, 2, 3+ (clipped to 3)
2. **Opponent stones in house**: 0, 1, 2, 3+ (clipped to 3)
3. **House control**: Hammer, Neutral, or Opponent (based on control difference sign)
4. **End parity**: 0 (even end) or 1 (odd end)

**State ID format**: `{hammer_stones}_{opp_stones}_{control}_{parity}`
Example: `2_1_Hammer_0` = 2 hammer stones, 1 opponent stone, Hammer control, even end

### State Space Summary

- **Total unique states**: 19 states identified
- **Most common state**: `2_2_Neutral_1` (413 occurrences, 15.7% of ends)
- **State coverage**: Top 10 states cover ~60% of all ends

### How to Read Q7_State_Representation.png

The visualization contains 4 panels:

#### Panel 1: Top 20 Most Common States
- **What it shows**: Frequency distribution of game states
- **Y-axis**: State IDs (formatted for readability)
- **X-axis**: Frequency (number of occurrences)
- **Bars**: Horizontal bars showing how often each state occurs
- **Interpretation**:
  - Most common states represent typical game situations
  - Rare states may be high-risk or high-reward situations
  - Distribution shows which states to focus on for strategy

#### Panel 2: Scoring Probability by State
- **What it shows**: Scoring probability for top 10 states (sorted by scoring rate)
- **Y-axis**: State labels (simplified format: e.g., "2H-1O-H" = 2 hammer, 1 opponent, Hammer control)
- **X-axis**: Scoring probability (0 to 1)
- **Bars**: Horizontal bars showing scoring probability
- **Vertical line at 0.5**: Break-even point
- **Interpretation**:
  - States above 0.5 = favorable for hammer team
  - States below 0.5 = unfavorable (steal risk)
  - Range shows how much state matters

#### Panel 3: Average Points by State
- **What it shows**: Average points scored for top 10 states (by point magnitude)
- **X-axis**: State labels (simplified)
- **Y-axis**: Average points
- **Bars**: Show average points for each state
- **Interpretation**:
  - Higher bars = states that lead to more points
  - Identifies high-value states to target
  - Shows expected value of different states

#### Panel 4: Transition Heatmap (Stones → Points)
- **What it shows**: Probability of scoring different point amounts given hammer stones in house
- **Y-axis**: Hammer stones in house (0, 1, 2, 3+)
- **X-axis**: Points scored (0, 1, 2, 3+)
- **Color intensity**: Probability (darker = higher probability)
- **Numbers in cells**: Exact probabilities
- **Interpretation**:
  - Each row sums to 1.0 (probabilities)
  - Diagonal or upper-right = expected pattern (more stones → more points)
  - Off-diagonal patterns = interesting transitions

### State Table Structure

The `simulation_states.csv` file contains:
- **state_id**: Unique state identifier
- **p_score**: Probability hammer team scores
- **expected_points**: Expected points scored by hammer team
- **p_steal**: Probability of steal (non-hammer scores)
- **p_blank**: Probability of blank end
- **count**: Number of times this state occurred

### Key Insights from Q7

1. **Most common states**:
   - `2_2_Neutral_1`: 413 occurrences (15.7%) - balanced, neutral control
   - `2_2_Neutral_0`: 370 occurrences (14.0%) - similar, even end
   - `2_1_Hammer_1`: 242 occurrences (9.2%) - hammer advantage

2. **Best scoring states**:
   - States with 2+ hammer stones and Hammer control
   - Example: `2_0_Hammer_1` has 78.1% scoring rate, 1.84 average points

3. **Worst scoring states**:
   - States with 0-1 hammer stones and Opponent control
   - Example: `1_2_Opponent_0` has 44.8% scoring rate, 0.91 average points

4. **State probabilities**:
   - Each state has associated outcome probabilities
   - Can be used for Markov chain simulation
   - Enables strategy comparison across different initial states

### Practical Applications

- **Strategy simulation**: Use state table to simulate game outcomes
- **State evaluation**: Quickly assess value of current game state
- **Decision making**: Compare expected outcomes of different shot choices
- **Game planning**: Understand which states to target or avoid
- **Markov simulation**: Build game simulations to test strategies

### Using the State Table for Simulation

1. **Identify current state**: Determine state_id from current game position
2. **Look up probabilities**: Use p_score, expected_points, p_steal, p_blank
3. **Simulate outcomes**: Use probabilities to run Monte Carlo simulations
4. **Compare strategies**: Test different shot choices by comparing resulting states
5. **Optimize decisions**: Choose shots that lead to highest-value states

---

## Common Metrics and Interpretations

### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

**What it measures**: Model's ability to distinguish between classes

**Scale**: 0.0 to 1.0
- **0.5**: Random guessing (no predictive power)
- **0.6-0.7**: Moderate predictive power
- **0.7-0.8**: Good predictive power
- **0.8-0.9**: Very good predictive power
- **0.9-1.0**: Excellent predictive power

**Interpretation**:
- Higher AUC = better at ranking predictions (ordering cases by probability)
- Useful for binary classification problems
- Doesn't depend on classification threshold

**Example**: Q1 has AUC of 0.623, meaning it can distinguish scoring vs non-scoring ends 62.3% better than random.

### Accuracy

**What it measures**: Percentage of correct predictions

**Formula**: (True Positives + True Negatives) / Total

**Interpretation**:
- Simple and intuitive
- Can be misleading with imbalanced classes
- Compare to baseline (majority class) to assess improvement

**Example**: Q1 has 60.9% accuracy vs 56.4% baseline = 4.5% improvement.

### F1-Score

**What it measures**: Harmonic mean of precision and recall

**Formula**: 2 × (Precision × Recall) / (Precision + Recall)

**Scale**: 0.0 to 1.0
- **0.0**: Worst (no true positives or all false)
- **1.0**: Perfect (all predictions correct)

**Interpretation**:
- Balances precision (accuracy of positive predictions) and recall (coverage)
- Useful when both false positives and false negatives matter
- Better than accuracy for imbalanced classes

**Variants**:
- **Macro F1**: Average F1 across all classes (treats all classes equally)
- **Weighted F1**: Average F1 weighted by class frequency (accounts for imbalance)

### Precision

**What it measures**: Of all positive predictions, what % are correct?

**Formula**: True Positives / (True Positives + False Positives)

**Interpretation**:
- High precision = few false positives
- Important when false positives are costly
- Trade-off with recall

### Recall (Sensitivity)

**What it measures**: Of all actual positives, what % did we catch?

**Formula**: True Positives / (True Positives + False Negatives)

**Interpretation**:
- High recall = few false negatives
- Important when missing positives is costly
- Trade-off with precision

### Confusion Matrix

**What it shows**: Breakdown of predictions vs actuals

**Structure**: N×N grid where N = number of classes
- **Rows**: Actual classes
- **Columns**: Predicted classes
- **Diagonal**: Correct predictions
- **Off-diagonal**: Errors

**How to read**:
- Larger numbers on diagonal = better model
- Off-diagonal patterns show common mistakes
- Can identify which classes are confused

### Feature Importance

**What it measures**: How much each feature contributes to predictions

**XGBoost importance types**:
- **Gain**: Average improvement in accuracy when feature is used
- **Weight**: Number of times feature is used in splits
- **Cover**: Average coverage of observations when feature is used

**Interpretation**:
- Higher importance = more predictive power
- Relative importance matters (compare features)
- Absolute values depend on model and data

**Limitations**:
- Importance doesn't show direction (positive/negative effect)
- Correlated features may have split importance
- Importance is model-specific

---

## Feature Importance Guide

### Understanding Feature Names

**Distance Features**:
- `hammer_closest_dist`: Distance from hammer team's closest stone to button (in coordinate units)
- `nonhammer_closest_dist`: Distance from non-hammer team's closest stone to button
- `log_hammer_dist`: Logarithm of hammer closest distance (handles skew, emphasizes close stones)
- `log_nonhammer_dist`: Logarithm of non-hammer closest distance
- **Interpretation**: Lower values = closer to button = better position

**Stone Count Features**:
- `hammer_stones_in_house`: Number of hammer team stones in house (within 250 units of button)
- `nonhammer_stones_in_house`: Number of non-hammer team stones in house
- **Interpretation**: More stones = better control, but interactions matter

**Control Features**:
- `hammer_house_control_diff`: Difference in house control (hammer - opponent)
  - Positive = hammer control
  - Negative = opponent control
  - Zero = neutral
- **Interpretation**: Higher values = hammer advantage

**Ratio Features**:
- `stones_ratio`: Ratio of hammer to non-hammer stones in house
- `dist_ratio`: Ratio of hammer to non-hammer closest distances
- **Interpretation**: 
  - stones_ratio > 1 = hammer has more stones
  - dist_ratio < 1 = hammer is closer (better)

**Interaction Features**:
- `hammer_stones_x_dist`: Interaction of hammer stones and distance
- `house_control_x_parity`: Interaction of control and end parity
- **Interpretation**: Captures non-linear relationships (e.g., stones matter more when close)

**Context Features**:
- `end_num`: End number (1-9)
- `end_parity`: 0 (even) or 1 (odd) - affects hammer assignment
- `powerplay`: Power play indicator (1, 2, or NaN)
- `powerplay_missing`: Indicator for missing power play data
- `hammer_is_team1`, `hammer_is_team2`: Team indicators

### How to Interpret Feature Importance

1. **Relative importance**: Compare features to each other, not absolute values
2. **Top features**: Focus on top 5-10 features for each model
3. **Consistency**: Features important across multiple models are most reliable
4. **Interactions**: Interaction features show complex relationships
5. **Context**: Some features (end_num, parity) provide game context

### Feature Importance Patterns Across Models

**Consistently Important**:
- Distance metrics (hammer_closest_dist, log_hammer_dist)
- Stone ratios (stones_ratio)
- House control (hammer_house_control_diff)

**Model-Specific**:
- Q1 (Scoring): Distance and stone ratios dominate
- Q3 (Magnitude): Interactions become more important
- Q4 (Blank): Ratios and interactions matter most
- Q5 (Steal): Both teams' distances matter

---

## Conclusion

This interpretability guide provides comprehensive explanations for understanding all 7 models and their visualizations. Key takeaways:

1. **Model performance varies**: Binary classification (Q1, Q5) performs better than multi-class (Q3) or rare event prediction (Q4)

2. **Distance is critical**: Closest stone distance is consistently the most important feature across models

3. **Geometry matters**: House geometry (stones, control) dramatically affects outcomes (40+ percentage point swings)

4. **Early game state provides signal**: Shot 3 snapshot has meaningful predictive power, though many factors influence final outcomes

5. **State representation enables simulation**: Q7 creates a framework for comparing strategies and simulating game outcomes

6. **Power Play changes dynamics**: Power Play situations show different outcome patterns, though sample size is limited

Use this guide to interpret model outputs, understand visualizations, and extract actionable insights for curling strategy and decision-making.

---

## Quick Reference: Model Summary

| Question | Model Type | Key Metric | Performance | Best Use Case |
|----------|-----------|------------|-------------|---------------|
| Q1: Hammer Value | Binary Classification | ROC-AUC: 0.623 | Moderate | Predicting if hammer team scores |
| Q2: Geometry Effects | Interaction Analysis | Geometry range: ~40% | N/A | Understanding conditional effects |
| Q3: Scoring Magnitude | Multi-class | Accuracy: 42.9% | Moderate | Predicting point buckets (0,1,2,3+) |
| Q4: Blank Ends | Binary (Weighted) | Limited (5 cases) | Poor | Identifying blank conditions (limited utility) |
| Q5: Steal Probability | Binary Classification | ROC-AUC: 0.632 | Moderate | Predicting steal outcomes |
| Q6: Power Play | Comparative Analysis | Outcome differences | N/A | Comparing Power Play vs Normal |
| Q7: State Representation | State Space | 19 states | N/A | Simulation and strategy comparison |

---

*Last Updated: Based on XGBoost.ipynb analysis*
*For questions or clarifications, refer to the model code and data validation reports*

