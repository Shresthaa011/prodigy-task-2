import torch
from diffusers import StableDiffusionPipeline
import os

def generate_image(prompt, output_filename="output.png"):
    # Detect if a GPU is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the pre-trained model
    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"Loading model '{model_id}'...")
    
    # We use torch_dtype=torch.float16 if using GPU to save memory, otherwise float32 for CPU
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    # Disable safety checker if memory is an issue (optional)
    # pipe.safety_checker = None
    
    print(f"Generating image for prompt: '{prompt}'")
    # Generate the image
    # Note: On a CPU, this can take several minutes.
    image = pipe(prompt).images[0]
    
    # Save the image
    image.save(output_filename)
    print(f"Image saved successfully to {output_filename}")

if __name__ == "__main__":
    prompt = "A futuristic city skyline at sunset, cyberpunk style, high quality"
    generate_image(prompt)
