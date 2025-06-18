from typing import Dict

import gradio as gr
import json

from prompt_template_control import generate_video_prompt_with_template
from storyboard import generate_multiple_storyboards
from generation import wan_text_to_video, gcp_veo, hailuo_text_to_video


def generate_video(prompt, model_id, negative_prompt=None):
    if model_id == "Wan2.1":
        video_path = wan_text_to_video(prompt, negative_prompt)
    elif model_id == "SkyReels-V2":
        raise ValueError("SkyReels-V2 model not yet implemented.")
    elif model_id == "Veo-2":
        video_path = gcp_veo(prompt, local_output_path="output/cat_reading_book.mp4")
    elif model_id == "T2V-01-Director":
        video_path = hailuo_text_to_video(prompt)
    return video_path


def save_storyboard_choice(choice: Dict[str, str]):
    # Save the full dictionary as JSON (append mode)
    with open("selected_storyboards.json", "a") as f:
        f.write(json.dumps(choice) + "\n")
    return f"✅ Saved your selection to selected_storyboards.json:\n\n{json.dumps(choice, indent=2)}"


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

    demo.launch()
