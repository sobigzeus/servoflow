# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("robotics-diffusion-transformer/rdt-1b", dtype="auto")
