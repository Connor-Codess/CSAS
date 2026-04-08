I am building a computer vision pipeline that takes an image of a curling house
(the circular target area) and converts stone positions into coordinates that feed
into a trained XGBoost model for game state evaluation.

The specific approach:

- Use OpenCV Hough circle detection to find all circular objects in the image
- Automatically detect the button (center of the house) as the smallest detected
  circle near the image center — use this as the coordinate origin (0, 0)
- Detect the outer ring of the house to establish the scale reference
- Classify detected stones by team using HSV color segmentation
- Convert all stone pixel positions to real-world coordinates relative to the button
- Output a structured dictionary of features matching the XGBoost model's expected inputs

---

STEP 1 — Environment setup

Create a requirements.txt with:
opencv-python
numpy
scipy
scikit-learn
xgboost
Pillow

Create the following file structure:
curling_cv/
**init**.py
detect.py # stone and button detection
transform.py # coordinate conversion
features.py # feature engineering
pipeline.py # end-to-end orchestration
utils.py # visualization helpers
app.py # Gradio interface
models/ # directory for saved XGBoost .pkl files
tests/
test_detect.py
test_transform.py
sample_images/ # placeholder directory

---

STEP 2 — Button detection (detect.py)

Write a function detect_button(image) that:

1. Converts the image to grayscale
2. Applies Gaussian blur (kernel size 9x9) to reduce noise
3. Runs cv2.HoughCircles with the following strategy:
   - Use HOUGH_GRADIENT method
   - Set minRadius and maxRadius to a tight range for small circles
     (the button is roughly 0.5–2% of the image width)
   - Set minDist high enough that only one button candidate is returned
   - Use param1=50 (Canny edge threshold) and param2=30 (accumulator threshold)
4. Among all detected small circles, select the one closest to the image center
   (compute Euclidean distance from each circle center to image center, take minimum)
5. Return (cx, cy, radius) of the button, or raise a clear ValueError if none found

Also write detect_outer_ring(image) that:

1. Runs HoughCircles with a large radius range (30–60% of image width)
2. Selects the circle closest to the image center
3. Returns (cx, cy, radius) representing the house boundary
4. This radius becomes the scale reference — in real curling the house radius is 6 feet

---

STEP 3 — Stone detection and team classification (detect.py continued)

Write detect_stones(image, button_center, outer_ring_radius) that:

1. Runs HoughCircles with a radius range appropriate for stones
   (stones are roughly 3–6% of image width)
2. Filters out any detected circle whose center is more than
   outer_ring_radius \* 1.1 pixels from button_center
   (ignore stones outside the house entirely — they do not score)
3. For each remaining circle, classifies it as team 1, team 2, or button:
   - Extract a small patch (the inner 60% of the circle radius) around the center
   - Convert patch to HSV color space
   - Compute the mean hue of the patch
   - On first call, prompt the user (or accept parameters) for the hue ranges
     of team 1 and team 2 colors
   - Assign team label based on which hue range the mean falls into
   - If neither matches, label as unknown and log a warning
4. Return a list of dicts:
   [
   {'team': 1, 'pixel_x': 412, 'pixel_y': 387, 'radius': 23},
   {'team': 2, 'pixel_x': 501, 'pixel_y': 392, 'radius': 22},
   ...
   ]

---

STEP 4 — Coordinate transformation (transform.py)

Write pixels_to_coordinates(pixel_x, pixel_y, button_px, button_py,
outer_ring_px_radius) that:

1. Computes offset from button in pixels:
   dx = pixel_x - button_px
   dy = pixel_y - button_py
2. Converts to real-world feet using the scale factor:
   scale = 6.0 / outer_ring_px_radius (house radius = 6 feet in real curling)
   real_x = dx _ scale
   real_y = dy _ scale
3. Returns (real_x, real_y) in feet relative to button as origin

Write transform_all_stones(stones, button, outer_ring) that:

1. Calls pixels_to_coordinates for every stone in the list
2. Adds 'real_x' and 'real_y' keys to each stone dict
3. Computes distance_to_button for each stone:
   dist = sqrt(real_x^2 + real_y^2)
4. Adds 'distance_to_button' key to each stone dict
5. Returns the updated list

---

STEP 5 — Feature engineering (features.py)

Write compute_features(stones, hammer_team, end_num, score_diff, powerplay) that:

The house radius is 6 feet — only stones within 6 feet of the button score.
Stones are already filtered to within the house from Step 3 but double-check here.

Compute:
hammer_stones = [s for s in stones if s['team'] == hammer_team]
opp_stones = [s for s in stones if s['team'] != hammer_team]

hammer_dists = [s['distance_to_button'] for s in hammer_stones]
opp_dists = [s['distance_to_button'] for s in opp_stones]

hammer_closest = min(hammer_dists) if hammer_dists else 999
opp_closest = min(opp_dists) if opp_dists else 999
hammer_in_house = len(hammer_stones)
opp_in_house = len(opp_stones)
house_control = hammer_in_house - opp_in_house
stones_ratio = hammer_in_house / max(opp_in_house, 1)

hammer_x_dist = mean of (s['real_x'] \* s['distance_to_button'])
for s in hammer_stones, else 0
(this approximates hammer_stones_x_dist from the model)

Return a dict with EXACT key names matching the trained XGBoost feature columns:
{
'log_hammer_dist': np.log1p(hammer_closest),
'hammer_closest_dist': hammer_closest,
'log_nonhammer_dist': np.log1p(opp_closest),
'nonhammer_closest_dist': opp_closest,
'hammer_stones_in_house': hammer_in_house,
'nopp_stones_in_house': opp_in_house,
'hammer_house_control_diff': house_control,
'stones_ratio': stones_ratio,
'hammer_stones_x_dist': hammer_x_dist,
'end_num': end_num,
'hammer_is_team1': int(hammer_team == 1),
'hammer_is_team2': int(hammer_team == 2),
'powerplay': int(powerplay),
'powerplay_missing': int(not powerplay),
}

---

STEP 6 — Pipeline orchestration (pipeline.py)

Write run_pipeline(image_path, hammer_team, end_num, score_t1, score_t2,
powerplay, team1_hue_range, team2_hue_range) that:

1. Loads image with cv2.imread
2. Calls detect_button — if it fails, raise a clear error with instructions
   (image may need to be more top-down, or better lit)
3. Calls detect_outer_ring
4. Calls detect_stones with team hue ranges passed in
5. Calls transform_all_stones
6. Calls compute_features with score_diff = score_t1 - score_t2
7. Loads XGBoost models from models/ directory using pickle
8. Runs predict_proba on each sub-model
9. Returns a result dict:
   {
   'features': feature_dict,
   'stones': transformed_stones,
   'button': (cx, cy),
   'scoring_prob': float,
   'steal_prob': float,
   'blank_prob': float,
   'magnitude_probs': [p0, p1, p2, p3plus],
   'pp_timing_prob': float,
   'advice': str, # generated by threshold logic below
   }

Advice generation logic:
if scoring_prob > 0.65 and steal_prob < 0.40:
advice = "Strong position — push for multiple points."
elif steal_prob > 0.50:
advice = "High steal risk — prioritize defensive placement."
elif blank_prob > 0.25:
advice = "Neutral end — consider blanking to retain hammer."
elif scoring_prob < 0.45:
advice = "Difficult position — play conservatively."
else:
advice = "Balanced end — standard shot selection applies."

---

STEP 7 — Visualization helper (utils.py)

Write draw_detections(image, button, outer_ring, stones) that:

1. Draws the button as a small filled circle in white
2. Draws the outer ring as a circle outline in white
3. For each stone, draws a circle in team color (green for team 1, red for team 2)
4. Annotates each stone with its distance to button in feet (1 decimal place)
5. Returns the annotated image as a numpy array

This is the confirmation view shown to the user before they trust any output.

---

STEP 8 — Gradio interface (app.py)

Build a Gradio interface with gr.Blocks() layout:

Row 1 — inputs:

- gr.Image(label="Upload curling end image", type="numpy")
- gr.Radio(["Team 1", "Team 2"], label="Which team has the hammer?")
- gr.Slider(1, 8, step=1, value=1, label="End number")
- gr.Number(value=0, label="Team 1 score")
- gr.Number(value=0, label="Team 2 score")
- gr.Checkbox(label="Power play active?")
- gr.ColorPicker(label="Team 1 stone color")
- gr.ColorPicker(label="Team 2 stone color")
- gr.Button("Analyze")

Row 2 — outputs:

- gr.Image(label="Detection confirmation — verify stones before trusting output")
- gr.Label(label="Scoring probability")
- gr.Label(label="Steal probability")
- gr.Label(label="Blank end probability")
- gr.JSON(label="Full model output")
- gr.Textbox(label="Recommendation")

Wire the button to run_pipeline and draw_detections.
Convert the ColorPicker hex output to HSV hue ranges
(accept +/- 15 degrees around the picked hue as the valid range).

---

STEP 9 — Tests (tests/test_detect.py and test_transform.py)

Write unit tests for:

- detect_button returns correct center on a synthetic image
  (draw a known circle with cv2.circle, run detect_button, assert center within 5px)
- pixels_to_coordinates returns (0, 0) when pixel position equals button position
- pixels_to_coordinates returns (6, 0) when pixel position is exactly
  outer_ring_radius pixels to the right of button
- compute_features returns all expected keys
- compute_features handles empty stone lists without crashing

Use pytest. Do not use any external image files in tests — generate synthetic
test images programmatically with OpenCV.

---

KNOWN LIMITATIONS TO DOCUMENT IN CODE COMMENTS

Add a LIMITATIONS.md file noting:

- Hough circle detection assumes a roughly top-down camera angle (< 30 degrees tilt)
- Color segmentation fails under strong yellow/warm broadcast lighting
- Overlapping or touching stones may be detected as one large circle
- Coordinate scaling assumes the outer ring is fully visible in the frame
- The button may be obscured by a stone sitting on it — handle this gracefully
  by falling back to image center as button position if detection fails,
  and log a warning to the user
- This is a v1 prototype — YOLO-based detection is the recommended upgrade path
  for production accuracy
