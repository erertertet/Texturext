import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    "A texture pack of cobblestone, with a focus on realism and detail, smooth",
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("img/cobble_stone.png")