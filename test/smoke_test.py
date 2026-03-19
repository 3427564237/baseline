import torch
import open_clip

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

print("open_clip imported successfully")

pairs = [x for x in open_clip.list_pretrained() if x[0] == "ViT-B-16"]
print("ViT-B-16 pretrained options:")
for p in pairs[:10]:
    print(" ", p)