import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

import os
import time
import requests
import json

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


def gcp_veo(prompt: str = "a cat reading a book", local_output_path: str = "./generated_video.mp4"):
    PROJECT_ID = "gcp-credit-applying-to-g-suite"
    LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    BUCKET_NAME = "dante-test-123456-output"
    OUTPUT_GCS_PATH = f"gs://{BUCKET_NAME}/videos/output_{int(time.time())}.mp4"

    # Initialize Vertex AI
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # Initialize Generative AI client
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    # Video Generation Pipeline
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

    # Error Handling
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

        # Delete the file from GCS
        blob.delete()
        print(f"Video deleted from GCS: {video_uri}")

        return local_output_path
    else:
        raise Exception("No video generated or response is empty")


def hailuo_text_to_video(
        prompt: str,
        model: str = "T2V-01-Director",
        output_file_name: str = "output.mp4",
        api_key: str = ""
) -> str:
    def invoke_video_generation()->str:
        print("-----------------Submit video generation task-----------------")
        url = "https://api.minimaxi.chat/v1/video_generation"
        payload = json.dumps({
          "prompt": prompt,
          "model": model
        })
        headers = {
          'authorization': 'Bearer ' + api_key,
          'content-type': 'application/json',
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.text)
        task_id = response.json()['task_id']
        print("Video generation task submitted successfully, task ID.："+task_id)
        return task_id

    def query_video_generation(task_id: str):
        url = "https://api.minimaxi.chat/v1/query/video_generation?task_id="+task_id
        headers = {
          'authorization': 'Bearer ' + api_key
        }
        response = requests.request("GET", url, headers=headers)
        status = response.json()['status']
        if status == 'Preparing':
            print("...Preparing...")
            return "", 'Preparing'
        elif status == 'Queueing':
            print("...In the queue...")
            return "", 'Queueing'
        elif status == 'Processing':
            print("...Generating...")
            return "", 'Processing'
        elif status == 'Success':
            return response.json()['file_id'], "Finished"
        elif status == 'Fail':
            return "", "Fail"
        else:
            return "", "Unknown"


    def fetch_video_result(file_id: str):
        print("---------------Video generated successfully, downloading now---------------")
        url = "https://api.minimaxi.chat/v1/files/retrieve?file_id="+file_id
        headers = {
            'authorization': 'Bearer '+api_key,
        }

        response = requests.request("GET", url, headers=headers)
        print(response.text)

        download_url = response.json()['file']['download_url']
        print("Video download link：" + download_url)
        with open(output_file_name, 'wb') as f:
            f.write(requests.get(download_url).content)
        print("THe video has been downloaded in："+os.getcwd()+'/'+output_file_name)


    task_id = invoke_video_generation()
    print("-----------------Video generation task submitted -----------------")
    while True:
        time.sleep(10)

        file_id, status = query_video_generation(task_id)
        if file_id != "":
            fetch_video_result(file_id)
            print("---------------Successful---------------")
            break
        elif status == "Fail" or status == "Unknown":
            print("---------------Failed---------------")
            break

    return os.getcwd()+'/'+output_file_name


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
