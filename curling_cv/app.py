"""
app.py — Gradio dashboard for the curling CV pipeline.

Run with: python app.py
"""

import copy
import os

import cv2
import gradio as gr
import numpy as np

from pipeline import run_pipeline, run_from_stones
from utils import draw_detections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bgr_to_rgb(bgr_array):
    return cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)


def _confidence_badge(confidence_str):
    if confidence_str.startswith('High'):
        return f"✅ {confidence_str}"
    elif confidence_str.startswith('Moderate'):
        return f"⚠️  {confidence_str}"
    else:
        return f"🔴 {confidence_str}"


def _result_to_outputs(result):
    """Convert a pipeline result dict into the 7 Gradio output values."""
    annotated_rgb = _bgr_to_rgb(result['annotated_image'])
    confidence_text = _confidence_badge(result['confidence'])

    sp = result['scoring_prob']
    stp = result['steal_prob']
    bp = result['blank_prob']
    mp = result['magnitude_probs']

    probs_lines = [
        f"Hammer team scores:  {sp:.0%}",
        f"Non-hammer steals:   {stp:.0%}",
        f"End blanked:         {bp:.0%}",
    ]
    if mp and len(mp) >= 4:
        probs_lines += [
            "",
            "Points breakdown (if scoring):",
            f"  0 pts  {mp[0]:.0%}",
            f"  1 pt   {mp[1]:.0%}",
            f"  2 pts  {mp[2]:.0%}",
            f"  3+ pts {mp[3]:.0%}",
        ]
    probs = "\n".join(probs_lines)

    ft = result['features']
    h_in  = ft.get('hammer_stones_in_house', 0)
    nh_in = ft.get('nonhammer_stones_in_house', 0)
    h_d   = ft.get('hammer_closest_dist', 999)
    nh_d  = ft.get('nonhammer_closest_dist', 999)
    ctrl  = ft.get('hammer_house_control_diff', 0)
    pp    = ft.get('powerplay')

    features_display = "\n".join([
        f"Hammer stones in house:      {h_in}",
        f"Non-hammer stones in house:  {nh_in}",
        f"House control (hammer – opp): {ctrl:+d}",
        f"Hammer closest stone:        {'None' if h_d >= 999 else f'{h_d:.0f} units from button'}",
        f"Non-hammer closest stone:    {'None' if nh_d >= 999 else f'{nh_d:.0f} units from button'}",
        f"End number:                  {ft.get('end_num', '?')}",
        f"Power play:                  {'Active' if pp in (1.0, 2.0) else 'No'}",
    ])

    fs = result.get('final_score')
    final_score_text = (f"Simulated end result — Team 1: {fs['team1']}  Team 2: {fs['team2']}"
                        if fs else "Simulation not available.")

    gif_html = ""
    if result['simulation_gif']:
        import base64
        b64 = base64.b64encode(result['simulation_gif']).decode()
        gif_html = (
            f'<img src="data:image/gif;base64,{b64}" '
            f'style="width:100%;max-width:700px;border-radius:8px;" />'
        )

    return (
        annotated_rgb,
        confidence_text,
        probs,
        features_display,
        result['advice'],
        final_score_text,
        gif_html,
    )


def _stone_count_label(stones):
    r = sum(1 for s in stones if s.get('team') == 1)
    y = sum(1 for s in stones if s.get('team') == 2)
    u = sum(1 for s in stones if s.get('team') is None)
    parts = [f"🔴 {r} red", f"🟡 {y} yellow"]
    if u:
        parts.append(f"⬜ {u} unknown")
    return "  |  ".join(parts)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(
    image,
    hammer_team_label,
    end_num,
    shot_number,
    sheet_end,
    team1_score,
    team2_score,
    powerplay_active,
    team1_stone_count_str,
    team2_stone_count_str,
):
    empty = (None, "", "", "", "", "", None,
             [], {}, None, "Upload an image and click Analyze.", "")

    if image is None:
        return empty

    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hammer_team    = 1 if hammer_team_label == "Team 1 (Red)" else 2
    sheet_end_key  = 'top' if sheet_end == "Top end" else 'bottom'
    powerplay      = hammer_team if powerplay_active else None
    t1_count = int(team1_stone_count_str) if team1_stone_count_str.strip().isdigit() else None
    t2_count = int(team2_stone_count_str) if team2_stone_count_str.strip().isdigit() else None

    try:
        result = run_pipeline(
            image_bgr=image_bgr,
            hammer_team=hammer_team,
            end_num=int(end_num),
            shot_number=int(shot_number),
            team1_score=int(team1_score),
            team2_score=int(team2_score),
            sheet_end=sheet_end_key,
            powerplay=powerplay,
            team1_stone_count=t1_count,
            team2_stone_count=t2_count,
        )
    except Exception as e:
        return (None, "", "", "", "", "", None,
                [], {}, None, f"Pipeline error: {e}", "")

    # Store correction state: stones in crop-pixel space (before transform)
    # result['stones'] have already been transformed; we need pre-transform
    # copies. Re-extract from the original detected set via pixel_x/pixel_y.
    stones_for_edit = [
        {
            'team':    s['team'],
            'pixel_x': s['pixel_x'],
            'pixel_y': s['pixel_y'],
            'radius':  s['radius'],
            'warnings': [],
        }
        for s in result['stones']
    ]

    outputs = _result_to_outputs(result)
    count_label = _stone_count_label(stones_for_edit)
    ci = result['crop_info']
    init_center = f"House center: ({int(ci['button_cx'])}, {int(ci['button_cy'])})"

    return (
        *outputs,                        # 7 main outputs
        stones_for_edit,                 # corrected_stones state
        ci,                              # crop_info state
        result['cropped_bgr'],           # cropped_bgr state
        count_label,                     # stone count label
        init_center,                     # center label
    )


# ---------------------------------------------------------------------------
# Manual stone correction — click handler
# ---------------------------------------------------------------------------

def on_annotated_click(
    evt: gr.SelectData,
    action,
    corrected_stones,
    crop_info,
    cropped_bgr,
):
    """Called when the user clicks on the annotated image."""
    if not crop_info or cropped_bgr is None:
        return gr.update(), corrected_stones, gr.update(), "No detection data yet."

    x, y = int(evt.index[0]), int(evt.index[1])
    stones = copy.deepcopy(corrected_stones)
    info   = copy.deepcopy(crop_info)
    est_r  = max(8, int(info.get('ring_radius', 100) * 0.11))

    if action == "Add Red Stone":
        stones.append({'team': 1, 'pixel_x': x, 'pixel_y': y,
                       'radius': est_r, 'warnings': []})
    elif action == "Add Yellow Stone":
        stones.append({'team': 2, 'pixel_x': x, 'pixel_y': y,
                       'radius': est_r, 'warnings': []})
    elif action == "Remove Nearest":
        if stones:
            dists = [((s['pixel_x'] - x) ** 2 + (s['pixel_y'] - y) ** 2) ** 0.5
                     for s in stones]
            stones.pop(int(np.argmin(dists)))
    elif action == "Set House Center":
        info['button_cx'] = float(x)
        info['button_cy'] = float(y)

    # Redraw
    annotated = draw_detections(cropped_bgr, info, stones, warnings=[])
    annotated_rgb = _bgr_to_rgb(annotated)
    count_label = _stone_count_label(stones)
    center_label = f"House center: ({int(info['button_cx'])}, {int(info['button_cy'])})"

    return annotated_rgb, stones, info, count_label, center_label


# ---------------------------------------------------------------------------
# Rerun with corrected stones
# ---------------------------------------------------------------------------

def rerun(
    corrected_stones,
    crop_info,
    cropped_bgr,
    hammer_team_label,
    end_num,
    shot_number,
    sheet_end,
    team1_score,
    team2_score,
    powerplay_active,
):
    if not corrected_stones or not crop_info or cropped_bgr is None:
        return (None, "", "", "", "", None,
                "No stones to rerun with.")

    hammer_team   = 1 if hammer_team_label == "Team 1 (Red)" else 2
    sheet_end_key = 'top' if sheet_end == "Top end" else 'bottom'
    powerplay     = hammer_team if powerplay_active else None

    try:
        result = run_from_stones(
            stones_crop=corrected_stones,
            cropped_bgr=cropped_bgr,
            crop_info=crop_info,
            hammer_team=hammer_team,
            end_num=int(end_num),
            shot_number=int(shot_number),
            team1_score=int(team1_score),
            team2_score=int(team2_score),
            sheet_end=sheet_end_key,
            powerplay=powerplay,
        )
    except Exception as e:
        return (None, "", "", "", "", None,
                f"Rerun error: {e}")

    outputs = _result_to_outputs(result)
    count_label = _stone_count_label(corrected_stones)
    return (*outputs, count_label)


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Curling CV Dashboard") as demo:
    gr.Markdown("# Curling Strategy Dashboard\nUpload an overhead photo of the house to get AI-driven shot recommendations and a simulated continuation.")

    # Hidden state for manual correction
    corrected_stones_state = gr.State([])
    crop_info_state        = gr.State({})
    cropped_bgr_state      = gr.State(None)

    # ── Inputs ──────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Inputs")
            image_input = gr.Image(label="Upload curling house image", type="numpy")

            with gr.Row():
                hammer_input = gr.Radio(
                    ["Team 1 (Red)", "Team 2 (Yellow)"],
                    label="Which team has the hammer?",
                    value="Team 1 (Red)",
                )
                end_input = gr.Slider(1, 10, step=1, value=1, label="End number")

            with gr.Row():
                shot_input = gr.Slider(1, 10, step=1, value=3, label="Stones thrown so far")
                sheet_end_input = gr.Radio(
                    ["Top end", "Bottom end"],
                    label="Which end of the sheet?",
                    value="Top end",
                )

            with gr.Row():
                score_t1 = gr.Number(value=0, label="Team 1 score (Red)", precision=0)
                score_t2 = gr.Number(value=0, label="Team 2 score (Yellow)", precision=0)

            powerplay_input = gr.Checkbox(label="Power play active (hammer team)?", value=False)

            with gr.Row():
                t1_count_input = gr.Textbox(label="Team 1 stones thrown (optional — for validation)", value="")
                t2_count_input = gr.Textbox(label="Team 2 stones thrown (optional — for validation)", value="")

            analyze_btn = gr.Button("Analyze", variant="primary")

            gr.Examples(
                examples=[["Test_Image_huggingFace.png", "Team 2 (Yellow)", 4, 5, "Top end", 0, 0, False, "", ""]],
                inputs=[image_input, hammer_input, end_input, shot_input,
                        sheet_end_input, score_t1, score_t2,
                        powerplay_input, t1_count_input, t2_count_input],
                label="▶  Try a sample image — click the row below then hit Analyze",
            )

        # ── Detection + Correction ──────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Detection Confirmation")
            gr.Markdown("*After analyzing, click on the image to manually add or remove stones if needed.*")

            annotated_output = gr.Image(
                label="Detected stones — click to correct",
                interactive=False,  # display only; .select() handles clicks
            )

            stone_count_label = gr.Textbox(
                label="Detected stones", interactive=False, value=""
            )

            with gr.Row():
                correction_action = gr.Radio(
                    ["Add Red Stone", "Add Yellow Stone", "Remove Nearest", "Set House Center"],
                    label="Click action",
                    value="Add Red Stone",
                )

            center_label = gr.Textbox(
                label="House center position",
                interactive=False,
                value="",
                placeholder="Click 'Set House Center' then click the true button centre on the image",
            )

            rerun_btn = gr.Button("🔄  Rerun with corrected stones", variant="secondary")

            confidence_output = gr.Textbox(label="Model confidence", interactive=False)

    # ── Results ─────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Predictions")
            probs_output    = gr.Textbox(label="Scoring probabilities", lines=8, interactive=False)
            features_output = gr.Textbox(label="Position summary", lines=8, interactive=False)

        with gr.Column():
            gr.Markdown("### Recommendation")
            advice_output      = gr.Textbox(label="Strategic advice", lines=16, interactive=False)
            final_score_output = gr.Textbox(label="Simulated end result", interactive=False)

    gr.Markdown("### Simulated Remaining Shots")
    gr.Markdown("*Optimal play for shots not yet thrown, according to the model.*")
    gif_output = gr.HTML(label="Simulation animation")

    # ── Event wiring ─────────────────────────────────────────────────────────

    _main_outputs = [
        annotated_output, confidence_output,
        probs_output, features_output, advice_output,
        final_score_output, gif_output,
    ]

    analyze_btn.click(
        fn=analyze,
        inputs=[
            image_input, hammer_input, end_input, shot_input,
            sheet_end_input, score_t1, score_t2,
            powerplay_input, t1_count_input, t2_count_input,
        ],
        outputs=[
            *_main_outputs,
            corrected_stones_state,
            crop_info_state,
            cropped_bgr_state,
            stone_count_label,
            center_label,
        ],
    )

    # Click on annotated image → add/remove stone or reposition house center
    annotated_output.select(
        fn=on_annotated_click,
        inputs=[
            correction_action,
            corrected_stones_state,
            crop_info_state,
            cropped_bgr_state,
        ],
        outputs=[
            annotated_output,
            corrected_stones_state,
            crop_info_state,
            stone_count_label,
            center_label,
        ],
    )

    # Rerun button → full model + simulation with corrected stones
    rerun_btn.click(
        fn=rerun,
        inputs=[
            corrected_stones_state,
            crop_info_state,
            cropped_bgr_state,
            hammer_input, end_input, shot_input,
            sheet_end_input, score_t1, score_t2,
            powerplay_input,
        ],
        outputs=[
            *_main_outputs,
            stone_count_label,
        ],
    )


if __name__ == "__main__":
    demo.launch()
