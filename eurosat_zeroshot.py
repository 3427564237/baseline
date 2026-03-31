import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

import open_clip
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import EuroSAT
from tqdm import tqdm

try:
    from sklearn.metrics import confusion_matrix, f1_score
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for Macro F1 and confusion matrix. "
        "Install it with: micromamba -r /home/miku/.micromamba install -n pytorch-env scikit-learn"
    ) from exc

# 1. Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-16"
PRETRAINED = "openai"
DATASET_NAME = "EuroSAT"
DATA_ROOT = "data"
DATA_SPLIT = "all"
OUTPUT_DIR = Path("outputs")

# Prompt template is configurable for later experiments.
PROMPT_TEMPLATE = "a satellite image of {}"

# 研究作业版本：直接跑 EuroSAT 全量数据，不做随机抽样
BATCH_SIZE = 64
NUM_WORKERS = min(8, os.cpu_count() or 1)
PIN_MEMORY = DEVICE == "cuda"
TOP_K_VALUES = (3, 5)
TOP_CONFUSED_PAIRS_TO_SHOW = 20


def label_to_text(label_name: str) -> str:
    # EuroSAT class names are CamelCase, e.g. AnnualCrop -> annual crop.
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", label_name)
    return spaced.lower()


def load_model():
    print(f"Using device: {DEVICE}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    model = model.to(DEVICE)
    model.eval()
    return model, preprocess, tokenizer


def load_dataset(preprocess):
    dataset = EuroSAT(
        root=DATA_ROOT,
        download=True,
        transform=preprocess,
    )

    print("Selected classes:")
    for idx, name in enumerate(dataset.classes):
        print(f" - {name} -> {idx}")

    print(f"Total images in dataset: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
    )
    return dataset, loader


def build_text_prompts(class_names):
    prompts = [PROMPT_TEMPLATE.format(label_to_text(name)) for name in class_names]

    print("\nPrompts:")
    for prompt in prompts:
        print(" -", prompt)

    return prompts


def encode_text_features(model, tokenizer, prompts):
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(DEVICE)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def evaluate(model, loader, text_features, class_names):
    num_classes = len(class_names)
    max_k = min(max(TOP_K_VALUES), num_classes)

    correct = 0
    total = 0
    top_k_correct = {k: 0 for k in TOP_K_VALUES if k <= num_classes}

    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes
    all_true = []
    all_pred = []

    for images, labels in tqdm(loader, desc="Running zero-shot inference"):
        images = images.to(DEVICE, non_blocking=PIN_MEMORY)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = 100.0 * image_features @ text_features.T
            topk_indices = logits.topk(max_k, dim=1).indices
            preds = topk_indices[:, 0]

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for k in top_k_correct:
            matches = topk_indices[:, :k].eq(labels.unsqueeze(1))
            top_k_correct[k] += matches.any(dim=1).sum().item()

        labels_cpu = labels.cpu().tolist()
        preds_cpu = preds.cpu().tolist()
        all_true.extend(labels_cpu)
        all_pred.extend(preds_cpu)

        for y_true, y_pred in zip(labels_cpu, preds_cpu):
            per_class_total[y_true] += 1
            if y_true == y_pred:
                per_class_correct[y_true] += 1

    overall_accuracy = correct / total
    top_k_accuracy = {f"top{k}_accuracy": top_k_correct[k] / total for k in sorted(top_k_correct)}
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)

    per_class_rows = []
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_acc = per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0.0
        per_class_accuracy[class_name] = class_acc
        per_class_rows.append(
            {
                "class_name": class_name,
                "per_class_accuracy": class_acc,
                "sample_count": per_class_total[i],
                "correct_count": per_class_correct[i],
            }
        )

    conf_matrix = confusion_matrix(all_true, all_pred, labels=list(range(num_classes)))
    confused_pairs = extract_top_confused_pairs(conf_matrix, class_names)

    return {
        "overall_accuracy": overall_accuracy,
        "macro_f1": macro_f1,
        "top_k_accuracy": top_k_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "per_class_rows": per_class_rows,
        "all_true": all_true,
        "all_pred": all_pred,
        "confusion_matrix": conf_matrix,
        "confused_pairs": confused_pairs,
    }


def extract_top_confused_pairs(conf_matrix, class_names, top_n=TOP_CONFUSED_PAIRS_TO_SHOW):
    pairs = []
    for true_idx, true_class in enumerate(class_names):
        for pred_idx, pred_class in enumerate(class_names):
            if true_idx == pred_idx:
                continue

            count = int(conf_matrix[true_idx, pred_idx])
            if count > 0:
                pairs.append(
                    {
                        "true_class": true_class,
                        "predicted_class": pred_class,
                        "count": count,
                    }
                )

    pairs.sort(key=lambda item: item["count"], reverse=True)
    return pairs[:top_n]


def save_json(path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_results(results, class_names):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_stem = f"eurosat_zeroshot_{run_id}"

    top3_accuracy = results["top_k_accuracy"].get("top3_accuracy")
    top5_accuracy = results["top_k_accuracy"].get("top5_accuracy")

    summary = {
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "dataset_name": DATASET_NAME,
        "dataset_split": DATA_SPLIT,
        "prompt_template": PROMPT_TEMPLATE,
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "class_count": len(class_names),
        "overall_accuracy": results["overall_accuracy"],
        "top3_accuracy": top3_accuracy,
        "top5_accuracy": top5_accuracy,
        "macro_f1": results["macro_f1"],
        "per_class_accuracy": results["per_class_accuracy"],
    }

    json_path = OUTPUT_DIR / f"{file_stem}_summary.json"
    per_class_csv_path = OUTPUT_DIR / f"{file_stem}_per_class_accuracy.csv"
    confusion_csv_path = OUTPUT_DIR / f"{file_stem}_top_confused_pairs.csv"

    save_json(json_path, summary)
    save_csv(
        per_class_csv_path,
        fieldnames=["class_name", "per_class_accuracy", "sample_count", "correct_count"],
        rows=results["per_class_rows"],
    )
    save_csv(
        confusion_csv_path,
        fieldnames=["true_class", "predicted_class", "count"],
        rows=results["confused_pairs"],
    )

    return {
        "json_path": json_path,
        "per_class_csv_path": per_class_csv_path,
        "confusion_csv_path": confusion_csv_path,
    }


def print_results(results, class_names, saved_paths):
    per_class_row_map = {row["class_name"]: row for row in results["per_class_rows"]}

    print("\n========================")
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    if "top3_accuracy" in results["top_k_accuracy"]:
        print(f"Top-3 Accuracy:   {results['top_k_accuracy']['top3_accuracy']:.4f}")
    if "top5_accuracy" in results["top_k_accuracy"]:
        print(f"Top-5 Accuracy:   {results['top_k_accuracy']['top5_accuracy']:.4f}")
    print(f"Macro F1:         {results['macro_f1']:.4f}")
    print("========================")

    for class_name in class_names:
        class_acc = results["per_class_accuracy"][class_name]
        row = per_class_row_map[class_name]
        print(
            f"{class_name:24s} accuracy = {class_acc:.4f} "
            f"({row['correct_count']}/{row['sample_count']})"
        )

    print("\nTop confused class pairs:")
    if results["confused_pairs"]:
        for pair in results["confused_pairs"]:
            print(
                f" - true: {pair['true_class']:24s} | "
                f"pred: {pair['predicted_class']:24s} | count = {pair['count']}"
            )
    else:
        print(" - No confused class pairs found.")

    print("\nSaved files:")
    print(f" - JSON summary: {saved_paths['json_path']}")
    print(f" - Per-class CSV: {saved_paths['per_class_csv_path']}")
    print(f" - Confusion CSV: {saved_paths['confusion_csv_path']}")


def main():
    model, preprocess, tokenizer = load_model()
    dataset, loader = load_dataset(preprocess)
    prompts = build_text_prompts(dataset.classes)
    text_features = encode_text_features(model, tokenizer, prompts)
    results = evaluate(model, loader, text_features, dataset.classes)
    saved_paths = save_results(results, dataset.classes)
    print_results(results, dataset.classes, saved_paths)


if __name__ == "__main__":
    main()
