import gradio as gr
import torch
from PIL import Image
from VictorAI import AutoPipelineForImage2Image

model_path = "/content/modelfiles/victorai_shoe_model/1000"
def inference_image2image(image_path, prompt, negative_prompt, guidance_scale, strength):
    # Load the pre-trained model
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    pipeline.enable_model_cpu_offload()
    
    # Load the input image
    init_image = Image.open(image_path).convert("RGB")
    
    # Generate the image based on user prompts and additional parameters
    generated_images = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        guidance_scale=guidance_scale,
        strength=strength
    ).images[0]
    
    return generated_images

def launch_gradio_app_image2image():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                # model_path = gr.Textbox(label="Model Path", value="path to your model directory")
                image_path = gr.File(label="Input Image", type="filepath")
                prompt = gr.Textbox(label="Prompt", value="photo of a realistic, unique, shoe design")
                negative_prompt = gr.Textbox(label="Negative Prompt", value="")
                guidance_scale = gr.Number(label="Guidance Scale", value=1.0)
                strength = gr.Number(label="Strength", value=1.0)
                run = gr.Button(value="Generate")
            with gr.Column():
                output_image = gr.Image()
                
        run.click(
            inference_image2image,
            inputs=[image_path, prompt, negative_prompt, guidance_scale, strength],
            outputs=output_image
        )

    demo.launch(debug=True, share=True)

if __name__ == "__main__":
    launch_gradio_app_image2image()
