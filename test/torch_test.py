import sys
print("python:", sys.version)

try:
    import torch
    print("torch:", torch.__version__)
    print("torch cuda available:", torch.cuda.is_available())
    print("torch built with cuda:", torch.version.cuda)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch import failed:", repr(e))

try:
    import torchvision
    print("torchvision:", torchvision.__version__)
except Exception as e:
    print("torchvision import failed:", repr(e))