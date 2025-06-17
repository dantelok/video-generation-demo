import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dotenv import load_dotenv
from openai import OpenAI
import os

# Load env for OpenAI
load_dotenv()
client = OpenAI()


# --- Step 1: Convert Storyboard to Pseudo-Video ---
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


# --- Step 2: Render Animation using Matplotlib ---
def render_pseudo_video(pseudo_video, save_path="preview.gif"):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.set_title(f"{pseudo_video['scene']} ({pseudo_video['camera']}, {pseudo_video['characters'][0]['emotion']})")

    # Setup character
    character = pseudo_video['characters'][0]
    emoji = character['emoji']
    duration = pseudo_video['duration_sec']
    frames = duration * 10  # 10 fps

    text_obj = ax.text(0, 0.5, emoji, fontsize=30)

    def update(frame):
        x = (frame / frames) * 10
        text_obj.set_position((x, 0.5))
        return text_obj,

    ani = animation.FuncAnimation(fig, update, frames=int(frames), blit=True)
    ani.save(save_path, writer="pillow", fps=10)
    plt.close()
    print(f"Saved animation to {save_path}")


# --- Step 3: Convert Pseudo-Video to Gen-2 Prompt ---
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


if __name__ == "__main__":
    # Simulate storyboard input
    storyboard = {
        "scene": "misty forest",
        "shot_type": "wide shot",
        "emotion": "mysterious"
    }

    pseudo_video = storyboard_to_pseudo_video(storyboard)
    print("Pseudo-Video Spec:\n", json.dumps(pseudo_video, indent=2))

    render_pseudo_video(pseudo_video, save_path="forest_demo.gif")

    gen2_prompt = pseudo_video_to_prompt(pseudo_video)
    print("\nPrompt for Gen-2:\n", gen2_prompt)