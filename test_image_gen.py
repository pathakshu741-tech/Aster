from diffusers import AutoPipelineForText2Image
import torch

print("Loading model... this will download ~2GB the first time, then it's cached.")

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32   # change to torch.float16 below if you have an NVIDIA GPU
)
pipe = pipe.to("cpu")   # change to "cuda" if you have an NVIDIA GPU

print("Model loaded. Generating image...")

image = pipe(
    prompt="a futuristic city skyline",
    num_inference_steps=2,
    guidance_scale=0.0
).images[0]

image.save("test_output.png")
print("Done — check test_output.png")