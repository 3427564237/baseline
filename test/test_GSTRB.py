from torchvision.datasets import OxfordIIITPet
dataset = OxfordIIITPet (root="data", split="test", download=True)
print("dataset type:", type(dataset))
print("dataset size:", len(dataset))
print("num classes:", len(dataset.classes))
print("first 20 classes:", dataset.classes[:20])

