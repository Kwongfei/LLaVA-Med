import torch

if torch.cuda.is_available():
    print(f"CUDA is available. Version: {torch.version.cuda}")
else:
    print("CUDA is not available.")