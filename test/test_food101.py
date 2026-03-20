from torchvision.datasets import Food101

dataset = Food101(root="data", split="test", download=True)

print("dataset type:", type(dataset))
print("dataset size:", len(dataset))
print("num classes:", len(dataset.classes))
print("first 20 classes:", dataset.classes[:20])

target_names = ["pizza", "sushi", "ice_cream", "hamburger", "omelette"]
for name in target_names:
    print(name, "->", name in dataset.classes)
    
image, label = dataset[555]
print("image shape:", image.size)
print("image label:", label)
print("class name:", dataset.classes[label])

print("----------------------------------------------------")

# 1: 先拿到name label 键值对
class_to_idx = {name: idx for idx, name in enumerate(dataset.classes)}
# 拿到 name label 键值对，然后根据 target_names 列表中的类名，找到对应的label
target_names_idx = [class_to_idx[name] for name in target_names]

print("selected classes:", target_names)
print("selected class indices:", target_names_idx)


# 2: 拿到某个类在数据集中的位置
from collections import defaultdict

indices_by_class = defaultdict(list)

for sample_idx in range(len(dataset)):
    image, label = dataset[sample_idx]
    if label in target_names_idx:
        indices_by_class[label].append(sample_idx)

for class_idx in target_names_idx:
    class_name = dataset.classes[class_idx]
    print(class_name, "->", len(indices_by_class[class_idx]))
    print("first 5 sample indices:", indices_by_class[class_idx][:5])

# 3: 从每个类中随机选取50个样本, 总共250个样本    
import random

random.seed(114514)

samples_per_class = 50
selected_indices = []

for class_idx in target_names_idx:
    chosen = random.sample(indices_by_class[class_idx], samples_per_class)
    selected_indices.extend(chosen)
print("total selected samples:", len(selected_indices))
print("first 10 selected indices:", selected_indices[:10])

# 4: 从原始数据集中根据 selected_indices 创建一个新的子数据集
from torch.utils.data import Dataset
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
    
# 创建子数据集
subset_dataset = FoodSubsetDataset(
    base_dataset=dataset,
    indices=selected_indices,
    transform=None
)

print("subset size:", len(subset_dataset))

image, label = subset_dataset[0]
print("first subset image type:", type(image))
print("first subset label:", label)
print("first subset class name:", dataset.classes[label])

print("----------------------------------------------------")

# 5: 加载 CLIP 模型和预处理函数
import torch
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

model_name = "ViT-B-16"
pretrained = "openai"

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name,
    pretrained=pretrained
)

tokenizer = open_clip.get_tokenizer(model_name)

model = model.to(device)
model.eval()

print("model loaded")
print("tokenizer loaded")
print("preprocess:", preprocess)

print("----------------------------------------------------")
# 6: 对子数据集中的第一张图片进行预处理，并查看结果
image, label = subset_dataset[0]

print("before preprocess:")
print("image type:", type(image))
print("class name:", dataset.classes[label])

image_tensor = preprocess(image)

print("\nafter preprocess:")
print("tensor type:", type(image_tensor))
print("tensor shape:", image_tensor.shape)
print("tensor dtype:", image_tensor.dtype)