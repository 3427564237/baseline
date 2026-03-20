import random
from collections import defaultdict

import torch
import open_clip
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import Food101
from tqdm import tqdm

# 1. Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-16"
PRETRAINED = "openai"

SELECTED_CLASSES = [
    "pizza",
    "sushi",
    "ice_cream",
    "hamburger",
    "omelette",
]

SAMPLES_PER_CLASS = 250
BATCH_SIZE = 16
SEED = 1

random.seed(SEED)
torch.manual_seed(SEED)

# 2. Load CLIP
print(f"Using device: {DEVICE}")

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED
)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

model = model.to(DEVICE)
model.eval()

# 3. Load Food-101
base_dataset = Food101(
    root="data",
    split="test",
    download=True,
    transform=None
)

class_to_idx = {name: i for i, name in enumerate(base_dataset.classes)}
selected_class_indices = [class_to_idx[name] for name in SELECTED_CLASSES]

print("Selected classes:")
for name in SELECTED_CLASSES:
    print(f" - {name} -> {class_to_idx[name]}")

# 4. 收集每个目标类别的样本下标
indices_by_class = defaultdict(list)

for idx in range(len(base_dataset)):
    _, label = base_dataset[idx]
    if label in selected_class_indices:
        indices_by_class[label].append(idx)

for class_idx in selected_class_indices:
    count = len(indices_by_class[class_idx])
    class_name = base_dataset.classes[class_idx]
    print(f"{class_name}: found {count} images")

# 5. 每类随机抽样
final_indices = []
for class_name in SELECTED_CLASSES:
    class_idx = class_to_idx[class_name]
    chosen = random.sample(indices_by_class[class_idx], SAMPLES_PER_CLASS)
    final_indices.extend(chosen)

print(f"Total sampled images: {len(final_indices)}")

# 6. 自定义子集 Dataset
class FoodSubsetDataset(Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_idx = self.indices[i]
        image, label = self.base_dataset[real_idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


subset_dataset = FoodSubsetDataset(
    base_dataset=base_dataset,
    indices=final_indices,
    transform=preprocess
)

loader = DataLoader(
    subset_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# 7. Build prompts
def label_to_text(label_name: str) -> str:
    return label_name.replace("_", " ")

prompts = [f"a photo of a dish of {label_to_text(name)}" for name in SELECTED_CLASSES]

print("\nPrompts:")
for p in prompts:
    print(" -", p)

# 8. Encode text
with torch.no_grad():
    text_tokens = tokenizer(prompts).to(DEVICE)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 把 Food-101 的全局 label id 映射成 0~4
global_to_local = {
    class_to_idx[name]: i for i, name in enumerate(SELECTED_CLASSES)
}

# 9. Inference
correct = 0
total = 0

per_class_correct = [0] * len(SELECTED_CLASSES)
per_class_total = [0] * len(SELECTED_CLASSES)

for images, labels in tqdm(loader, desc="Running zero-shot inference"):
    images = images.to(DEVICE)

    local_labels = torch.tensor(
        [global_to_local[int(x)] for x in labels],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.T
        preds = logits.argmax(dim=1)

    correct += (preds == local_labels).sum().item()
    total += local_labels.size(0)

    for y_true, y_pred in zip(local_labels, preds):
        y_true = int(y_true.item())
        y_pred = int(y_pred.item())
        per_class_total[y_true] += 1
        if y_true == y_pred:
            per_class_correct[y_true] += 1
            
# 10. Print results

accuracy = correct / total
print("\n========================")
print(f"Overall Accuracy: {accuracy:.4f}")
print("========================")

for i, class_name in enumerate(SELECTED_CLASSES):
    class_acc = per_class_correct[i] / per_class_total[i]
    print(f"{class_name:12s} accuracy = {class_acc:.4f} ({per_class_correct[i]}/{per_class_total[i]})")