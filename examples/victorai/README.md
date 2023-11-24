# victor_ai

Trained weights are currently saved on this drive link and can be downloaded:

https://drive.google.com/file/d/1aydBxD7RlleFKB95-2qdEmMoTwVvrInT/view?usp=drive_link

A trained weights directory should be specified which contains previous model weights and is used for inference or re-training:
```python
WEIGHTS_DIR = "/path/to/weights/directory"
```
## Dependencies:
install the necessary dependencies and also check the `requirements.txt` file for training and additional libraries installation.
```python
pip install -r requirements.txt
```

## Gradio app:

Run the following command for gradio app inference:

Text to Image Pipeline Gradio Interface
```python
python gradio_victorai.py
```
Image to Image Pipeline Gradio Interface
```python
python img2img_gradio.py
```
## inference:
use the following commands for inference:

### Text to Image:
```python
python inference.py --model_path victorai_shoe_model/1000/ --prompt "photo of blood-red color shoe design, air jordan JOGGERS, masterpiece, best quality, shimmer, gradient color, luminous color, 8k,Photography, super detailed,hyper realistic,masterpiece" --negative_prompt "blur, half image" --num_samples 4 --guidance_scale 7 --num_inference_steps 30 --height 512 --width 512 --output_folder ./txt2img_results
```
Parameters/Arguments:
- model_path: Path to the pre-trained model directory.
- prompt: Textual description of the desired image.
- negative_prompt: Textual description of undesirable features in the image.
- num_samples: Number of images to generate.
- guidance_scale: Strength of the guidance during image generation.
- num_inference_steps: Number of steps to run the image generation process.
- height: Output image height.
- width: Output image width.
- output_folder: Directory to save the generated images.


### Image to image Pipeline:
```python
 python imagetoimage.py --model_path "victorai_shoe_model/1000" --image_path "/image.png" --prompt "shoe with black and white colors, detailed, 8k" --output_image "/img2img_results/output_image.png"
```
Parameters/Arguments:
- model_path: Path to the pre-trained model directory.
- image_path: Path to the input image.
- prompt: Textual description of the desired image modifications.
- output_image: Path to save the generated output image.

Additional Args:
- strength
- guidance_scale

To use the script with command-line arguments, you can follow these guidelines. Start by specifying the `--model_path` argument, which is required and should point to the model directory you want to use. Optionally, you can customize the image generation process using the following arguments: `--prompt` for the text prompt used to generate images, `--negative_prompt` for a negative text prompt, `--num_samples` to set the number of images to generate per prompt, `--guidance_scale` to adjust the guidance scale for the generation process, `--num_inference_steps` to control the number of inference steps, and `--height` and `--width` to set the dimensions of the generated images. Replace the placeholders with the specific values and paths that suit your needs. Ensure that the required dependencies are installed, as mentioned in the README, and execute the script with the chosen command-line arguments to generate customized images.

`Inference.py` script utilizes a Stable Diffusion Pipeline to generate images and offers customization through command-line arguments. It requires the specification of essential parameters, including the model path, text prompts, image count, guidance scale, and image dimensions. By running the script with appropriate values for these arguments, users can generate images tailored to their preferences. Ensure that the necessary dependencies, such as `argparse`, `torch`, `diffusers`, and `IPython`, are installed to use the script effectively.

## train:
create a `concepts_list.json` file with the following code:
```python
concepts_list = [
    {
        "instance_prompt":      "photo of a shoe",
        "class_prompt":         "design",
        "instance_data_dir":    "/content/shoes1k",
        "class_data_dir":       "/content/data/shoes"
    },
]

# `instance_data_dir` contains the training images
# `class_data_dir` contains regularization images
import json
import os
for c in concepts_list:
    os.makedirs(c["instance_data_dir"], exist_ok=True)

with open("concepts_list.json", "w") as f:
    json.dump(concepts_list, f, indent=4)
```
train using the following command:
```python
 python3 train_victorai.py --pretrained_model_name_or_path="/victorai_shoe_model/1000" --pretrained_vae_name_or_path="victorai_shoe_model/1000/vae" --output_dir="weights_output" --with_prior_preservation --prior_loss_weight=1.0 --seed=1337 --resolution=512 --train_batch_size=1 --train_text_encoder --mixed_precision="fp16" --use_8bit_adam --gradient_accumulation_steps=1 --learning_rate=1e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --num_class_images=50 --sample_batch_size=4 --max_train_steps=1000 --save_interval=500 --save_sample_prompt="photo of a red shoe design with blue laces " --concepts_list="concepts_list.json"
```
List of argument names and their functions that can be used via CLI:

1. `--pretrained_model_name_or_path`: Path to the pretrained model.
2. `--pretrained_vae_name_or_path`: Path to pretrained VAE (Variational Autoencoder).
3. `--revision`: Revision of pretrained model identifier.
4. `--tokenizer_name`: Pretrained tokenizer name or path if different from `model_name`.
5. `--instance_data_dir`: Folder containing training data of instance images.
6. `--class_data_dir`: Folder containing training data of class images.
7. `--instance_prompt`: Prompt identifying the instance.
8. `--class_prompt`: Prompt specifying images in the same class as provided instance images.
9. `--save_sample_prompt`: Prompt used to generate sample outputs to save.
10. `--save_sample_negative_prompt`: Negative prompt used to generate sample outputs to save.
11. `--n_save_sample`: Number of samples to save.
12. `--save_guidance_scale`: CFG for save sample.
13. `--save_infer_steps`: Number of inference steps for save sample.
14. `--pad_tokens`: Flag to pad tokens to length 77.
15. `--with_prior_preservation`: Flag to add prior preservation loss.
16. `--prior_loss_weight`: Weight of prior preservation loss.
17. `--num_class_images`: Minimal class images for prior preservation loss.
18. `--output_dir`: Output directory for model predictions and checkpoints.
19. `--seed`: Seed for reproducible training.
20. `--resolution`: Resolution for input images.
21. `--center_crop`: Whether to center crop images before resizing.
22. `--train_text_encoder`: Whether to train the text encoder.
23. `--train_batch_size`: Batch size for the training dataloader.
24. `--sample_batch_size`: Batch size for sampling images.
25. `--num_train_epochs`: Number of training epochs.
26. `--max_train_steps`: Total number of training steps to perform.
27. `--gradient_accumulation_steps`: Number of updates steps to accumulate before backward pass.
28. `--gradient_checkpointing`: Whether to use gradient checkpointing.
29. `--learning_rate`: Initial learning rate.
30. `--scale_lr`: Scale the learning rate by certain factors.
31. `--lr_scheduler`: Type of scheduler to use.
32. `--lr_warmup_steps`: Number of steps for the warmup in the lr scheduler.
33. `--use_8bit_adam`: Whether to use 8-bit Adam from bitsandbytes.
34. `--adam_beta1`: Beta1 parameter for the Adam optimizer.
35. `--adam_beta2`: Beta2 parameter for the Adam optimizer.
36. `--adam_weight_decay`: Weight decay to use.
37. `--adam_epsilon`: Epsilon value for the Adam optimizer.
38. `--max_grad_norm`: Max gradient norm.
39. `--push_to_hub`: Whether to push the model to the Hub.
40. `--hub_token`: Token to use to push to the Model Hub.
41. `--hub_model_id`: Name of the repository to keep in sync with the local `output_dir`.
42. `--logging_dir`: TensorBoard log directory.
43. `--log_interval`: Log every N steps.
44. `--save_interval`: Save weights every N steps.
45. `--save_min_steps`: Start saving weights after N steps.
46. `--mixed_precision`: Whether to use mixed precision.
47. `--not_cache_latents`: Do not precompute and cache latents from VAE.
48. `--hflip`: Apply horizontal flip data augmentation.
49. `--local_rank`: For distributed training: local_rank.
50. `--concepts_list`: Path to JSON containing multiple concepts, overwrites parameters like instance_prompt, class_prompt, etc.
51. `--read_prompts_from_txts`: Use prompt per image, prompts in the same directory as images.
