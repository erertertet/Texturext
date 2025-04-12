import torch
import torchvision.transforms as transforms
from diffusers import StableDiffusion3ControlNetPipeline,SD3ControlNetModel
from diffusers.utils import load_image

# add global options to models
def patch_conv(**patch):
    cls = torch.nn.Conv2d
    init = cls.__init__
    def __init__(self, *args, **kwargs):
        return init(self, *args, **kwargs, **patch)
    cls.__init__ = __init__

patch_conv(padding_mode='circular')
print("patched for tiling")

controlnet = SD3ControlNetModel.from_pretrained("stabilityai/stable-diffusion-3.5-large-controlnet-blur", torch_dtype=torch.float16)
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

control_image = load_image("img/result1.png")
gaussian_blur = transforms.GaussianBlur(kernel_size=501)
blurred_image = gaussian_blur(control_image)
blurred_image.save('img/blurred.png')
prompt = "cobblestone, 8k, Hyper realistic"

generator = torch.Generator(device="cpu").manual_seed(0)
image = pipe(
    prompt, 
    control_image=blurred_image, 
    negative_prompt="ugly, deformed, blurry, small repetition",
    guidance_scale=3.5,
    num_inference_steps=60,
    generator=generator,
    max_sequence_length=77,
    width=512,
    height=512
).images[0]
image.save('img/cobble.png')