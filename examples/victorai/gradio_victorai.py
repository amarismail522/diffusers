import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

def inference(prompt, negative_prompt, num_samples, height, width, num_inference_steps, guidance_scale):
    with torch.autocast("cuda"), torch.inference_mode():
        return pipe(
            prompt, height=int(height), width=int(width),
            negative_prompt=negative_prompt,
            num_images_per_prompt=int(num_samples),
            num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
            generator=g_cuda
        ).images

WEIGHTS_DIR = "path to your weights directory"  # Replace with the path to your model directory
pipe = StableDiffusionPipeline.from_pretrained(WEIGHTS_DIR, safety_checker=None, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.enable_xformers_memory_efficient_attention()
g_cuda = None
g_cuda = torch.Generator(device='cuda')
seed = -1  # You can specify a seed here if needed
g_cuda.manual_seed(seed)

def launch_gradio_app():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="photo of a realistic, unique, shoe design")
                negative_prompt = gr.Textbox(label="Negative Prompt", value="")
                run = gr.Button(value="Generate")
                with gr.Row():
                    num_samples = gr.Number(label="Number of Samples", value=4)
                    guidance_scale = gr.Number(label="Guidance Scale", value=7)
                with gr.Row():
                    height = gr.Number(label="Height", value=512)
                    width = gr.Number(label="Width", value=512)
                num_inference_steps = gr.Slider(label="Steps", value=50)
            with gr.Column():
                gallery = gr.Gallery()

        run.click(inference, inputs=[prompt, negative_prompt, num_samples, height, width, num_inference_steps, guidance_scale], outputs=gallery)

    demo.launch(debug=True, share=True)

if __name__ == "__main__":
    launch_gradio_app()
