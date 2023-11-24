import argparse
import os
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Generate images")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder to save generated images")
    parser.add_argument("--prompt", type=str, default="photo of shoe design, masterpiece, best quality, chaoxie, SNEAKERS, shimmer, gradient color, luminous color, 8k, Photography, super detailed, hyper-realistic, masterpiece,")
    parser.add_argument("--negative_prompt", type=str, default="fingers, hands,")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--guidance_scale", type=int, default=5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)

    args = parser.parse_args()

    model_path = args.model_path
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    g_cuda = None

    g_cuda = torch.Generator(device='cuda')
    seed = -1  # You can specify a seed here if needed
    g_cuda.manual_seed(seed)

    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            negative_prompt=args.negative_prompt,
            num_images_per_prompt=args.num_samples,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=g_cuda
        ).images

    for i, img in enumerate(images):
        # Save each generated image to the specified folder
        img.save(os.path.join(output_folder, f"generated_image_{i}.png"))

if __name__ == "__main__":
    main()
