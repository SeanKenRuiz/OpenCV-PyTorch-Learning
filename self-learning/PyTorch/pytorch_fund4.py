import torch
from torch import nn

print(f"Torch version: {torch.__version__}")

# Set up device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")