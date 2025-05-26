import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

import os
import time
import urllib

from PIL import Image as PIL_Image
from google import genai
from google.genai import types
from google.cloud import aiplatform
from google.cloud import storage
import matplotlib.pyplot as plt
import mediapy as media


def wan_text_to_video(prompt, negative_prompt):
    # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    # model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = scheduler
    pipe.to("cpu")

    prompt = ("A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the "
              "dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through "
              "the window.")
    negative_prompt = ("Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
                       "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
                       "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
                       "misshapen limbs, fused fingers, still picture, messy background, three legs, many people in "
                       "the background, walking backwards")

    output = pipe(
         prompt=prompt,
         negative_prompt=negative_prompt,
         height=720,
         width=1280,
         num_frames=81,
         guidance_scale=5.0,
        ).frames[0]
    export_to_video(output, "output.mp4", fps=16)

    return "output.mp4"


def gcp_veo_3(prompt: str = "a cat reading a book", local_output_path: str = "./generated_video.mp4"):
    PROJECT_ID = "dante-test-461016"
    LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    BUCKET_NAME = "dante-test-461016-output"  # Replace with your GCS bucket name
    OUTPUT_GCS_PATH = f"gs://{BUCKET_NAME}/videos/output_{int(time.time())}.mp4"

    # Initialize Vertex AI
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # Initialize Generative AI client
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    video_model = "veo-2.0-generate-001"
    # video_model = "veo-3.0-generate-preview"
    aspect_ratio = "16:9"

    operation = client.models.generate_videos(
        model=video_model,
        prompt=prompt,
        config=types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            output_gcs_uri=OUTPUT_GCS_PATH,
            number_of_videos=1,
            duration_seconds=5,
            person_generation="allow_adult",
            enhance_prompt=True,
        ),
    )

    # Poll until operation is complete
    print("Generating video...")
    while not operation.done:
        time.sleep(15)
        operation = client.operations.get(operation)
        print(f"Operation status: {operation}")

    # Check for errors
    if operation.error:
        raise Exception(f"Video generation failed: {operation.error}")

    # Get the generated video URI
    if operation.response and operation.result.generated_videos:
        video_uri = operation.result.generated_videos[0].video.uri
        print(f"Video generated at: {video_uri}")

        # Download the video from GCS to local
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        blob_name = video_uri.replace(f"gs://{BUCKET_NAME}/", "")
        blob = bucket.blob(blob_name)

        # Ensure local directory exists
        os.makedirs(os.path.dirname(local_output_path), exist_ok=True)

        # Download the video
        blob.download_to_filename(local_output_path)
        print(f"Video downloaded to: {local_output_path}")

        # Optionally, delete the file from GCS to clean up
        blob.delete()
        print(f"Video deleted from GCS: {video_uri}")

        return local_output_path
    else:
        raise Exception("No video generated or response is empty")


def show_video(gcs_uri):
    file_name = gcs_uri.split("/")[-1]
    # !gsutil cp {gcs_uri} {file_name}
    media.show_video(media.read_video(file_name), height=500)


def display_images(image) -> None:
    fig, axis = plt.subplots(1, 1, figsize=(12, 6))
    axis.imshow(image)
    axis.set_title("Starting Image")
    axis.axis("off")
    plt.show()


# Only available for cuda / cpu
# wan_text_to_video()


# if __name__ == "__main__":
#     try:
#         local_path = gcp_veo_3(
#             prompt="a cat reading a book",
#             local_output_path="output/cat_reading_book.mp4"
#         )
#         print(f"Success! Video saved at: {local_path}")
#     except Exception as e:
#         print(f"Error: {e}")
