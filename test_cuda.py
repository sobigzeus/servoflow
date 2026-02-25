import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
try:
    d = torch.device("cuda:0")
    print(f"Device: {d}")
    x = torch.tensor([1.0]).to(d)
    print("Tensor on device:", x)
    torch.cuda.reset_peak_memory_stats(d)
    print("Reset stats success")
except Exception as e:
    print(f"Error: {e}")
