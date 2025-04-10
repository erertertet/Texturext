# https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3

import torch
from PIL import Image

from diffusers import StableDiffusion3Pipeline
from transformers import SiglipVisionModel, SiglipImageProcessor

image_encoder_id = "google/siglip-so400m-patch14-384"
ip_adapter_id = "InstantX/SD3.5-Large-IP-Adapter"

feature_extractor = SiglipImageProcessor.from_pretrained(
    image_encoder_id,
    torch_dtype=torch.float16
)
image_encoder = SiglipVisionModel.from_pretrained(
    image_encoder_id,
    torch_dtype=torch.float16
).to( "cuda")

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.float16,
    feature_extractor=feature_extractor,
    image_encoder=image_encoder,
).to("cuda")

pipe.load_ip_adapter(ip_adapter_id)
pipe.set_ip_adapter_scale(0.6)

ref_img = Image.open("image.jpg").convert('RGB')

image = pipe(
    width=1024,
    height=1024,
    prompt="a cat",
    negative_prompt="lowres, low quality, worst quality",
    num_inference_steps=24,
    guidance_scale=5.0,
    ip_adapter_image=ref_img
).images[0]

image.save("result.jpg")