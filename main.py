from generation import wan_text_to_video, gcp_veo_3
import gradio as gr


def generate(prompt, negative_prompt, model_id):
    if model_id == "Wan2.1":
        video_path = wan_text_to_video(prompt, negative_prompt)
    elif model_id == "SkyReels-V2":
        raise ValueError("SkyReels-V2 model not yet implemented.")
    elif model_id == "Veo-3":
        video_path = gcp_veo_3(prompt, local_output_path="output/cat_reading_book.mp4")
    return video_path


iface = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Enter your video prompt"),
        gr.Textbox(label="Enter your negative prompt (optional)"),
        gr.Radio(
            choices=["SkyReels-V2", "Wan2.1", "Veo-3"],
            label="Choose the video generation model"
        )
    ],
    outputs=gr.Video(label="Generated Video"),
    title="Video Generator",
    description="Generate a short video from your prompt using SkyReels-V2!"
)

iface.launch()
