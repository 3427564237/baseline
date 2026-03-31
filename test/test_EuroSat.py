import sys
import torchvision
from torchvision.datasets import EuroSAT

print("python:", sys.version)

DATA_ROOT = "data"

dataset = EuroSAT(
    root=DATA_ROOT,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

image, label = dataset[0]
print("image shape:", image.shape)
print("label:", label)

# image shape: torch.Size([3, 64, 64])
# label: 0

print(dataset.classes)
print(len(dataset.classes))
