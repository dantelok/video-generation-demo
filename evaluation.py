import json
import numpy as np
import cv2

from dotenv import load_dotenv
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from skimage.metrics import structural_similarity as ssim


# Load env for OpenAI
load_dotenv()
client = OpenAI()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def evaluate_with_gpt4(storyboard, video_description):
    system_prompt = (
        "You are a film critic evaluating how well a video matches a storyboard.\n"
        "Rate each of the following from 1 to 10:\n"
        "- Story Consistency: Does the video follow the scene and emotion described?\n"
        "- Shot Variety: Does it use interesting or varied camera angles?\n"
        "- Relevance: Does it suit the intended purpose (role, setting, emotion)?\n\n"
        "Provide scores and brief justifications for each.\n\n"
        "Format output as:\n"
        "{\n"
        "  \"story_consistency\": <score>,\n"
        "  \"shot_variety\": <score>,\n"
        "  \"relevance\": <score>,\n"
        "  \"justification\": \"...\"\n"
        "}"
    )

    user_prompt = (
        f"Storyboard:\n"
        f"Scene: {storyboard['scene']}\n"
        f"Shot: {storyboard['shot_type']}\n"
        f"Emotion: {storyboard['emotion']}\n\n"
        f"Video Description:\n{video_description}"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    content = response.choices[0].message.content.strip()
    return json.loads(content)


def compute_clip_similarity(image_path, text_prompt):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[text_prompt], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    similarity = logits_per_image.softmax(dim=1).item()
    return similarity


def compute_motion_score(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    motion_values = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_values.append(np.mean(magnitude))

        prev_gray = gray

    cap.release()
    return np.mean(motion_values) if motion_values else 0


def compute_temporal_coherence(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    ssim_scores = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            score = ssim(prev_frame, gray)
            ssim_scores.append(score)

        prev_frame = gray

    cap.release()
    return np.mean(ssim_scores) if ssim_scores else 0


def evaluate_video(storyboard, video_description, video_path, thumbnail_path, text_prompt):
    gpt_eval = evaluate_with_gpt4(storyboard, video_description)
    clip_score = compute_clip_similarity(thumbnail_path, text_prompt)
    motion_score = compute_motion_score(video_path)
    coherence_score = compute_temporal_coherence(video_path)

    return {
        "gpt_eval": gpt_eval,
        "metrics": {
            "clip_similarity": clip_score,
            "motion_score": motion_score,
            "temporal_coherence": coherence_score
        }
    }
