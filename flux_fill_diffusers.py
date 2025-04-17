import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image

# image = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup.png")
# mask = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup_mask.png")

image_path="./assets/cup.png"
image = load_image(image_path)
mask_path = "./assets/cup_mask.png"
mask = load_image(mask_path)

model_id  = "/home/work/fuping-workspace/model_zoo/FLUX.1-Fill-dev"
pipe = FluxFillPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
image = pipe(
    prompt="a white paper cup",
    image=image,
    mask_image=mask,
    height=1632,
    width=1232,
    guidance_scale=30,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save(f"flux-fill-dev.png")
