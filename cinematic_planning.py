import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

from main import generate_video

# Load env for OpenAI
load_dotenv()
client = OpenAI()


def storyboard_to_pseudo_video(storyboard):
    return {
        "scene": storyboard["scene"],
        "characters": [
            {
                "id": "main",
                "emoji": "👧",
                "action": "walk",
                "path": "left_to_right",
                "emotion": storyboard["emotion"]
            }
        ],
        "duration_sec": 5,
        "camera": storyboard["shot_type"]
    }


# Generate natural language transition description
def generate_transition_description(previous_state, next_state, i):
    # You can replace this with GPT for smarter descriptions
    return f"Transition {i+1}: The character continues to walk through the {next_state['scene']} with a {next_state['characters'][0]['emotion']} expression."


# Convert pseudo-video spec to text prompt
def pseudo_video_to_prompt(pseudo_video):
    scene = pseudo_video["scene"]
    emotion = pseudo_video["characters"][0]["emotion"]
    camera = pseudo_video["camera"]
    action = pseudo_video["characters"][0]["action"]
    path = pseudo_video["characters"][0]["path"]
    duration = pseudo_video["duration_sec"]

    prompt = (
        f"Create a {duration}-second video showing a {emotion} scene in a {scene}. "
        f"A character (represented by emoji) performs the action '{action}' across the screen from {path.replace('_', ' ')}. "
        f"Use a {camera} to capture the atmosphere."
    )
    return prompt


# Iterative Process
def build_scene_sequence(storyboard, model_id, num_keyframes=12):
    pseudo_video = storyboard_to_pseudo_video(storyboard)
    print("Pseudo-Video Spec:\n", json.dumps(pseudo_video, indent=2))

    previous_state = pseudo_video
    scene_sequence = []

    for i in range(num_keyframes):
        # 1️⃣ Generate transition text
        transition_text = generate_transition_description(previous_state, pseudo_video, i)

        # 2️⃣ Generate video prompt
        video_prompt = pseudo_video_to_prompt(pseudo_video)

        # 3️⃣ Generate video clip
        video_path = generate_video(video_prompt, model_id)

        # 4️⃣ Save this step
        scene_sequence.append({
            "transition_text": transition_text,
            "prompt": video_prompt,
            "video_path": video_path
        })

        # Optional: Update pseudo_video for next iteration if needed
        # Example: character moves deeper, emotion changes, etc.

    return scene_sequence


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # Example storyboard
    storyboard = {
        "scene": "misty forest",
        "shot_type": "wide shot",
        "emotion": "mysterious"
    }

    scene_sequence = build_scene_sequence(storyboard, model_id="Veo-2", num_keyframes=3)

    print("\n--- Final Scene Sequence ---")
    for i, step in enumerate(scene_sequence):
        print(f"\nKeyframe {i+1}:")
        print("Transition:", step["transition_text"])
        print("Video:", step["video_path"])
