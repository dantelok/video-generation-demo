import json
import re
from typing import List, Dict

from dotenv import load_dotenv
from typing import Dict
from openai import OpenAI


# Env variable
load_dotenv()

# Initialize OpenAI client
client = OpenAI()


def narrative_to_storyboard(narrative: str) -> Dict[str, str]:
    """
    Converts a narrative prompt into a structured storyboard dict
    with scene, shot type, and emotion using an LLM.
    """
    system_prompt = (
        "You are a professional storyboard artist and cinematographer. "
        "Given a narrative, extract and describe:\n"
        "- scene: The environment and visual setting\n"
        "- shot_type: The camera angle or framing (e.g., wide shot, close-up)\n"
        "- emotion: The overall mood or emotional tone\n\n"
        "Return the result as a JSON dictionary with keys: scene, shot_type, emotion."
    )

    user_prompt = f"Narrative: {narrative}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    content = response.choices[0].message.content.strip()

    # Use regex to extract JSON block
    json_match = re.search(r"\{[\s\S]*\}", content)
    if json_match:
        json_str = json_match.group(0)
        try:
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return {"scene": "", "shot_type": "", "emotion": ""}
    else:
        print("No JSON block found in response.")
        return {"scene": "", "shot_type": "", "emotion": ""}


def generate_multiple_storyboards(narrative: str, num_versions: int = 5) -> List[Dict[str, str]]:
    """
    Generate multiple storyboards for the same narrative by calling the LLM-based
    narrative_to_storyboard() function multiple times with slight variations.
    """
    storyboards = []

    for i in range(num_versions):
        # Add variation to the narrative to encourage different outputs
        variant_narrative = f"{narrative}\nPlease provide a different creative version #{i+1}."
        storyboard = narrative_to_storyboard(variant_narrative)
        storyboard['version'] = i + 1  # Track version number
        storyboards.append(storyboard)

    return storyboards


if __name__ == "__main__":
    print("Testing Narrative to Storyboard...")
    narrative_text = "A girl walks into a dark forest on a misty night."
    # storyboard_output = narrative_to_storyboard(narrative_text)
    # print(storyboard_output)

    print("Generate 5 Storyboard based on the narrative...")
    storyboard_list = generate_multiple_storyboards(narrative_text)
    for i, sb in enumerate(storyboard_list):
        print(f"Version {i + 1}: {sb}")
