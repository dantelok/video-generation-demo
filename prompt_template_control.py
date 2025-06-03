from openai import OpenAI
from dotenv import load_dotenv

# Env variable
load_dotenv()

# Initialize OpenAI client
client = OpenAI()


def generate_video_prompt_with_template(role: str, setting: str, emotion: str, shot: str, duration: str) -> str:
    system_prompt = (
        "You are a video director who converts structured metadata into detailed, natural language video prompts.\n"
        "Here are examples:\n\n"
        "Example 1:\n"
        "- Role: Product demo\n"
        "- Setting: Urban bar\n"
        "- Emotion: Energetic\n"
        "- Shot: Front-facing, 5s loop\n"
        "Output: \"Create a short 5-second video of a product demo in an energetic tone. "
        "The scene takes place in an urban bar setting, using a front-facing camera to capture the vibrant atmosphere.\"\n\n"
        "Example 2:\n"
        "- Role: Storytelling\n"
        "- Setting: Forest, misty\n"
        "- Emotion: Mysterious\n"
        "- Shot: Wide shot\n"
        "Output: \"Create a video showing a mysterious scene in a misty forest. Use a wide shot to capture the atmosphere and suspense.\"\n\n"
        "Now, create a natural language video prompt for the following:\n"
    )

    user_prompt = (
        f"- Role: {role}\n"
        f"- Setting: {setting}\n"
        f"- Emotion: {emotion}\n"
        f"- Shot: {shot}, {duration}\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    final_prompt = response.choices[0].message.content.strip()
    return final_prompt
