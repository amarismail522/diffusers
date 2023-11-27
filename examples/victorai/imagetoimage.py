import argparse
import torch
from PIL import Image
from VictorAI import AutoPipelineForImage2Image

def main(args):
    # Load the pre-trained model
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    pipeline.enable_model_cpu_offload()
    
    # Load the input image
    init_image = Image.open(args.image_path).convert("RGB")
    
    # Generate the image based on user prompts and additional parameters
    generated_images = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=init_image,
        guidance_scale=args.guidance_scale,
        strength=args.strength
    ).images[0]
    
    # Save the generated image
    generated_images.save(args.output_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an image using a trained model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text for image generation.")
    parser.add_argument("--negative_prompt", type=str, help="Negative prompt for image generation.")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale for image generation.")
    parser.add_argument("--strength", type=float, default=1.0, help="Strength parameter for image generation.")
    parser.add_argument("--output_image", type=str, required=True, help="Path to save the generated image.")

    args = parser.parse_args()
    main(args)
