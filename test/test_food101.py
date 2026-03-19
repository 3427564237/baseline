from torchvision.datasets import Food101

dataset = Food101(root="data", split="test", download=True)

print("dataset size:", len(dataset))
print("num classes:", len(dataset.classes))
print("first 20 classes:", dataset.classes[:20])

target_names = ["pizza", "sushi", "ice_cream", "hamburger", "omelette"]
for name in target_names:
    print(name, "->", name in dataset.classes)