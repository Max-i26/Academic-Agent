import torch
import transformers
import importlib

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    print("Device capability:", torch.cuda.get_device_capability(0))
    print("Total memory (MB):", torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
print("Accelerate available:", importlib.util.find_spec("accelerate") is not None)
print("BitsAndBytes available:", importlib.util.find_spec("bitsandbytes") is not None)
