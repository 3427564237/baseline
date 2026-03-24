import argparse
import csv
import json
import os
from datetime import datetime
from itertools import combinations
from pathlib import Path

import open_clip
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from tqdm import tqdm

try:
    from sklearn.metrics import f1_score
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for Macro F1. "
        "Install it with: micromamba -r /home/miku/.micromamba install -n pytorch-env scikit-learn"
    ) from exc


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-16"
PRETRAINED = "openai"
DATA_ROOT = "data"
DATA_SPLIT = "test"
OUTPUT_DIR = Path("outputs")

BATCH_SIZE = 64
NUM_WORKERS = min(8, os.cpu_count() or 1)
PIN_MEMORY = DEVICE == "cuda"
TOP_K_VALUES = (3, 5)

DEFAULT_PROMPT_TEMPLATES = [
    "a photo of a dish of {}",
    "a photo of {}",
    "a close-up photo of {}",
    "a high-resolution photo of {}",
    "a food photo of {}",
    "a delicious dish of {}",
    "a restaurant style photo of {}",
    "a plated serving of {}",
    "a cooked dish of {}",
    "a professional food photography shot of {}",
    "a top-down photo of {}",
    "a menu photo of {}",
]


def label_to_text(name: str) -> str:
    return name.replace("_", " ")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search for a relatively best prompt set for Food-101 CLIP zero-shot"
    )
    parser.add_argument(
        "--max-set-size",
        type=int,
        default=4,
        help="Maximum prompt set size to search (default: 4)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many top prompt sets to save in the ranking CSV (default: 20)",
    )
    parser.add_argument(
        "--templates-file",
        type=str,
        default=None,
        help="Optional path to a text file with one prompt template per line",
    )
    return parser.parse_args()


def load_prompt_templates(templates_file=None):
    if templates_file is None:
        raw_templates = DEFAULT_PROMPT_TEMPLATES
    else:
        path = Path(templates_file)
        if not path.exists():
            raise FileNotFoundError(f"Templates file not found: {path}")

        raw_templates = []
        for line in path.read_text(encoding="utf-8").splitlines():
            template = line.strip()
            if not template or template.startswith("#"):
                continue
            if "{}" not in template:
                template = template + " {}"
            raw_templates.append(template)

    templates = []
    for template in raw_templates:
        if template not in templates:
            templates.append(template)

    if not templates:
        raise ValueError("No valid prompt templates loaded")

    return templates


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
    dataset = Food101(
        root=DATA_ROOT,
        split=DATA_SPLIT,
        download=True,
        transform=preprocess,
    )

    print(f"Total images in test split: {len(dataset)}")
    print(f"Class count: {len(dataset.classes)}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
    )
    return dataset, loader


def extract_image_features(model, loader):
    all_image_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting image features"):
            images = images.to(DEVICE, non_blocking=PIN_MEMORY)
            labels = labels.to(DEVICE)

            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_image_features.append(image_features)
            all_labels.append(labels)

    image_features = torch.cat(all_image_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return image_features, labels


def build_text_feature_bank(model, tokenizer, class_names, templates):
    text_feature_bank = {}

    for template in tqdm(templates, desc="Encoding text templates"):
        prompts = [template.format(label_to_text(class_name)) for class_name in class_names]

        with torch.no_grad():
            text_tokens = tokenizer(prompts).to(DEVICE)
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        text_feature_bank[template] = text_features

    return text_feature_bank


def average_text_features(text_feature_bank, selected_templates):
    features_to_average = []
    for template in selected_templates:
        features_to_average.append(text_feature_bank[template])

    combined_text_features = torch.stack(features_to_average, dim=0).mean(dim=0)
    combined_text_features = combined_text_features / combined_text_features.norm(dim=-1, keepdim=True)
    return combined_text_features


def evaluate_prompt_set(image_features, labels, text_features):
    logits = 100.0 * image_features @ text_features.T
    predictions = logits.argmax(dim=1)

    total = labels.size(0)
    correct = (predictions == labels).sum().item()

    result = {
        "overall_accuracy": correct / total,
        "macro_f1": f1_score(
            labels.cpu().tolist(),
            predictions.cpu().tolist(),
            average="macro",
            zero_division=0,
        ),
    }

    max_k = min(max(TOP_K_VALUES), text_features.size(0))
    topk_indices = logits.topk(max_k, dim=1).indices

    for k in TOP_K_VALUES:
        if k > text_features.size(0):
            continue
        is_correct_in_topk = topk_indices[:, :k].eq(labels.unsqueeze(1))
        topk_correct = is_correct_in_topk.any(dim=1).sum().item()
        result[f"top{k}_accuracy"] = topk_correct / total

    return result


def search_prompt_sets(image_features, labels, text_feature_bank, max_set_size):
    templates = list(text_feature_bank.keys())
    max_set_size = max(1, min(max_set_size, len(templates)))

    ranking_rows = []
    best_row = None

    for set_size in range(1, max_set_size + 1):
        prompt_sets = combinations(templates, set_size)

        for prompt_set in tqdm(list(prompt_sets), desc=f"Testing set size {set_size}"):
            combined_text_features = average_text_features(text_feature_bank, prompt_set)
            metrics = evaluate_prompt_set(image_features, labels, combined_text_features)

            row = {
                "set_size": len(prompt_set),
                "templates": " | ".join(prompt_set),
                "overall_accuracy": metrics["overall_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "top3_accuracy": metrics.get("top3_accuracy"),
                "top5_accuracy": metrics.get("top5_accuracy"),
            }
            ranking_rows.append(row)

            if best_row is None:
                best_row = row
            elif row["overall_accuracy"] > best_row["overall_accuracy"]:
                best_row = row
            elif (
                row["overall_accuracy"] == best_row["overall_accuracy"]
                and row["macro_f1"] > best_row["macro_f1"]
            ):
                best_row = row

    ranking_rows.sort(
        key=lambda row: (row["overall_accuracy"], row["macro_f1"], row["set_size"]),
        reverse=True,
    )
    return best_row, ranking_rows


def save_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_results(best_row, ranking_rows, templates, args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_stem = f"food101_prompt_search_{run_id}"

    summary = {
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "dataset_split": DATA_SPLIT,
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "template_count": len(templates),
        "max_set_size": args.max_set_size,
        "best_prompt_set": best_row,
    }

    summary_path = OUTPUT_DIR / f"{file_stem}_summary.json"
    ranking_path = OUTPUT_DIR / f"{file_stem}_ranking.csv"

    save_json(summary_path, summary)
    save_csv(
        ranking_path,
        fieldnames=[
            "set_size",
            "templates",
            "overall_accuracy",
            "macro_f1",
            "top3_accuracy",
            "top5_accuracy",
        ],
        rows=ranking_rows[: args.top_n],
    )
    return summary_path, ranking_path


def print_top_rows(rows, top_n=5):
    print("\nTop prompt sets:")
    for index, row in enumerate(rows[:top_n], start=1):
        print(
            f"{index:2d}. acc={row['overall_accuracy']:.4f} "
            f"macro_f1={row['macro_f1']:.4f} size={row['set_size']}\n"
            f"    {row['templates']}"
        )


def main():
    args = parse_args()

    templates = load_prompt_templates(args.templates_file)
    print("Prompt templates to evaluate:")
    for index, template in enumerate(templates):
        print(f" - [{index}] {template}")

    model, preprocess, tokenizer = load_model()
    dataset, loader = load_dataset(preprocess)
    image_features, labels = extract_image_features(model, loader)

    print("\nEncoding text features...")
    text_feature_bank = build_text_feature_bank(
        model=model,
        tokenizer=tokenizer,
        class_names=dataset.classes,
        templates=templates,
    )

    best_row, ranking_rows = search_prompt_sets(
        image_features=image_features,
        labels=labels,
        text_feature_bank=text_feature_bank,
        max_set_size=args.max_set_size,
    )

    summary_path, ranking_path = save_results(
        best_row=best_row,
        ranking_rows=ranking_rows,
        templates=templates,
        args=args,
    )

    print("\n========================")
    print("Best prompt set found:")
    print(f" - overall_accuracy: {best_row['overall_accuracy']:.4f}")
    print(f" - macro_f1:         {best_row['macro_f1']:.4f}")
    print(f" - top3_accuracy:    {best_row['top3_accuracy']:.4f}")
    print(f" - top5_accuracy:    {best_row['top5_accuracy']:.4f}")
    print(f" - set_size:         {best_row['set_size']}")
    print(" - templates:")
    for template in best_row["templates"].split(" | "):
        print(f"   * {template}")
    print("========================")

    print_top_rows(ranking_rows, top_n=min(5, len(ranking_rows)))

    print("\nSaved files:")
    print(f" - Summary JSON: {summary_path}")
    print(f" - Ranking CSV:  {ranking_path}")


if __name__ == "__main__":
    main()
