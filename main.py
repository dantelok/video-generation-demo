from typing import Dict

import gradio as gr
import json

from cinematic_planning import build_scene_sequence
from generation import generate_video
from prompt_template_control import generate_video_prompt_with_template
from storyboard import generate_multiple_storyboards


def save_storyboard_choice(choice: Dict[str, str]):
    # Save the full dictionary as JSON (append mode)
    with open("selected_storyboards.json", "a") as f:
        f.write(json.dumps(choice) + "\n")
    return f"✅ Saved your selection to selected_storyboards.json:\n\n{json.dumps(choice, indent=2)}"


# Connect button
def run_pseudo_video_workflow(scene, shot_type, emotion, model_choice, num_keyframes):
    # Build storyboard dict
    storyboard = {
        "scene": scene,
        "shot_type": shot_type,
        "emotion": emotion
    }

    # Call your iterative builder
    scene_sequence = build_scene_sequence(
        storyboard, model_choice, num_keyframes=num_keyframes
    )

    # Format result as text
    result_text = ""
    for i, step in enumerate(scene_sequence):
        result_text += f"\nKeyframe {i + 1}:\n"
        result_text += f"Transition: {step['transition_text']}\n"
        result_text += f"Video Path: {step['video_path']}\n"

    return result_text


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("# 🎥 Video Generator")

        # Video Generator Interface
        with gr.Row():
            with gr.Column():
                video_prompt = gr.Textbox(label="Enter your video prompt")
                negative_prompt = gr.Textbox(label="Enter your negative prompt (optional: Wan2.1 Only)")
                model_choice = gr.Radio(
                    choices=["SkyReels-V2", "Wan2.1", "Veo-2", "T2V-01-Director"],
                    label="Choose the video generation model"
                )
                generate_btn = gr.Button("Generate Video")
            with gr.Column():
                video_output = gr.Video(label="Generated Video")

        generate_btn.click(
            generate_video,
            inputs=[video_prompt, model_choice, negative_prompt],
            outputs=video_output
        )

        # Divider
        gr.Markdown("---")

        # Narrative to Storyboard interface
        gr.Markdown("# 🎬 Narrative to Storyboard Grounding")
        narrative_input = gr.Textbox(label="Enter your narrative")
        generate_storyboards_btn = gr.Button("Generate 5 Storyboards")
        storyboards_output = gr.Radio(
            choices=[],
            label="Select your preferred storyboard"
        )
        save_choice_btn = gr.Button("Save Selection")
        save_output = gr.Textbox(label="Save Output", interactive=False)

        # Generate the storyboards
        def update_storyboards(narrative):
            cards = generate_multiple_storyboards(narrative)
            return gr.update(choices=cards)


        generate_storyboards_btn.click(
            update_storyboards,
            inputs=narrative_input,
            outputs=storyboards_output
        )

        # Save the choice
        save_choice_btn.click(
            save_storyboard_choice,
            inputs=storyboards_output,
            outputs=save_output
        )

        gr.Markdown("---")

        # Prompt Injection + Template Control
        gr.Markdown("# 🎥 Prompt Injection + Template Control (LLM + T2V)")

        # Modular controls
        role_input = gr.Textbox(label="Role", placeholder="e.g., Product demo")
        setting_input = gr.Textbox(label="Setting", placeholder="e.g., Urban bar")
        emotion_input = gr.Textbox(label="Emotion", placeholder="e.g., Energetic")
        shot_input = gr.Textbox(label="Shot Type", placeholder="e.g., Front-facing")
        duration_input = gr.Textbox(label="Duration", placeholder="e.g., 5s loop")

        # Model selection
        model_choice = gr.Radio(
            choices=["SkyReels-V2", "Veo-2", "Runway", "T2V-01-Director"],
            label="Choose video generation model"
        )

        # Generate final natural language prompt
        generate_prompt_btn = gr.Button("Generate Final Prompt")
        final_prompt_output = gr.Textbox(label="Final Video Prompt", interactive=False)

        # Generate video
        generate_video_btn = gr.Button("Generate Video")
        video_output = gr.Video(label="Generated Video")

        # Connect callbacks
        generate_prompt_btn.click(
            generate_video_prompt_with_template,
            inputs=[role_input, setting_input, emotion_input, shot_input, duration_input],
            outputs=final_prompt_output
        )

        generate_video_btn.click(
            generate_video,
            inputs=[final_prompt_output, model_choice, negative_prompt],
            outputs=video_output
        )

        gr.Markdown("# 🎞️ Pseudo Video Workflow (Storyboard → Scene Builder)")

        # Storyboard inputs
        pseudo_scene_input = gr.Textbox(label="Scene", placeholder="e.g., Misty forest")
        pseudo_shot_input = gr.Textbox(label="Shot Type", placeholder="e.g., Wide shot")
        pseudo_emotion_input = gr.Textbox(label="Emotion", placeholder="e.g., Mysterious")

        pseudo_model_choice = gr.Radio(
            choices=["SkyReels-V2", "Wan2.1", "Veo-2", "T2V-01-Director"],
            label="Choose video generation model"
        )

        num_keyframes_input = gr.Slider(minimum=1, maximum=20, value=12, label="Number of Keyframes")

        run_pseudo_video_btn = gr.Button("Build Pseudo Video Workflow")

        pseudo_output = gr.Textbox(label="Workflow Result", lines=10)

        # Hook to Gradio button
        run_pseudo_video_btn.click(
            run_pseudo_video_workflow,
            inputs=[
                pseudo_scene_input,
                pseudo_shot_input,
                pseudo_emotion_input,
                pseudo_model_choice,
                num_keyframes_input
            ],
            outputs=pseudo_output
        )

    demo.launch()
