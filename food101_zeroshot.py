import os

import torch
import open_clip
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from tqdm import tqdm

# 1. Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-16"
PRETRAINED = "openai"

# 研究作业版本：直接跑 Food-101 test 全量数据，不做随机抽样
BATCH_SIZE = 64
NUM_WORKERS = min(8, os.cpu_count() or 1)
PIN_MEMORY = DEVICE == "cuda"

# 2. Load CLIP
print(f"Using device: {DEVICE}")

model, _, preprocess = open_clip.create_model_and_transforms(
	MODEL_NAME,
	pretrained=PRETRAINED
)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

model = model.to(DEVICE)
model.eval()

# 3. Load Food-101（使用全部 101 类，顺序与官方 label 一致）
dataset = Food101(
	root="data",
	split="test",
	download=True,
	transform=preprocess
)

selected_classes = dataset.classes

print("Selected classes:")
for idx, name in enumerate(selected_classes):
	print(f" - {name} -> {idx}")

print(f"Total images in test split: {len(dataset)}")

loader = DataLoader(
	dataset,
	batch_size=BATCH_SIZE,
	shuffle=False,
	num_workers=NUM_WORKERS,
	pin_memory=PIN_MEMORY,
	persistent_workers=NUM_WORKERS > 0
)

# 4. Build prompts（自动覆盖全部 101 类）
def label_to_text(label_name: str) -> str:
	return label_name.replace("_", " ")


prompts = [f"a photo of a dish of {label_to_text(name)}" for name in selected_classes]

print("\nPrompts:")
for p in prompts:
	print(" -", p)

# 5. Encode text
with torch.no_grad():
	text_tokens = tokenizer(prompts).to(DEVICE)
	text_features = model.encode_text(text_tokens)
	text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 6. Inference（标签直接使用官方编号，不再做额外映射）
correct = 0
total = 0

per_class_correct = [0] * len(selected_classes)
per_class_total = [0] * len(selected_classes)

for images, labels in tqdm(loader, desc="Running zero-shot inference"):
	images = images.to(DEVICE, non_blocking=PIN_MEMORY)
	labels = labels.to(DEVICE)

	with torch.no_grad():
		image_features = model.encode_image(images)
		image_features = image_features / image_features.norm(dim=-1, keepdim=True)

		logits = 100.0 * image_features @ text_features.T
		preds = logits.argmax(dim=1)

	correct += (preds == labels).sum().item()
	total += labels.size(0)

	for y_true, y_pred in zip(labels, preds):
		y_true = int(y_true.item())
		y_pred = int(y_pred.item())
		per_class_total[y_true] += 1
		if y_true == y_pred:
			per_class_correct[y_true] += 1

# 7. Print results
accuracy = correct / total
print("\n========================")
print(f"Overall Accuracy: {accuracy:.4f}")
print("========================")

for i, class_name in enumerate(selected_classes):
	class_acc = per_class_correct[i] / per_class_total[i]
	print(f"{class_name:12s} accuracy = {class_acc:.4f} ({per_class_correct[i]}/{per_class_total[i]})")
