"""Check CUDA availability for PyTorch."""
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("CUDA NOT available - PyTorch is CPU-only build!")
    print(f"torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled if hasattr(torch.backends, 'cudnn') else 'N/A'}")
