import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

import open_clip
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import EuroSAT
from tqdm import tqdm

try:
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for Macro F1, confusion matrix, and dataset splitting. "
        "Install it with: micromamba -r /home/miku/.micromamba install -n pytorch-env scikit-learn"
    ) from exc

# 1. Config
BASE_DIR = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-16"
PRETRAINED = "openai"
DATASET_NAME = "EuroSAT"
DATA_ROOT = BASE_DIR / "data"
DATA_SPLIT = "test"
OUTPUT_ROOT = BASE_DIR / "outputs"
SPLIT_DIR = BASE_DIR / "splits"

# Best prompt setting selected on the validation split via eurosat_prompt_search.py.
PROMPT_TEMPLATES = (
    "a remote sensing image of {}",
    "an overhead view of {}",
)
BASELINE_PROMPT_TEMPLATES = ("a satellite image of {}",)
CLASS_NAME_SET_NAME = "student_v1"

CLASS_NAME_SETS = {
    "default": {
        "AnnualCrop": "annual crop",
        "Forest": "forest",
        "HerbaceousVegetation": "herbaceous vegetation",
        "Highway": "highway",
        "Industrial": "industrial",
        "Pasture": "pasture",
        "PermanentCrop": "permanent crop",
        "Residential": "residential",
        "River": "river",
        "SeaLake": "sea lake",
    },
    "student_v1": {
        "AnnualCrop": "annual crop field",
        "Forest": "forest",
        "HerbaceousVegetation": "grassland",
        "Highway": "highway road",
        "Industrial": "industrial area",
        "Pasture": "pasture land",
        "PermanentCrop": "orchard or permanent crop",
        "Residential": "residential area",
        "River": "river",
        "SeaLake": "lake or sea water",
    },
    "student_v2": {
        "AnnualCrop": "farmland with annual crops",
        "Forest": "dense forest area",
        "HerbaceousVegetation": "low vegetation grassland",
        "Highway": "major highway road",
        "Industrial": "industrial buildings area",
        "Pasture": "grazing pasture field",
        "PermanentCrop": "orchard or vineyard area",
        "Residential": "residential buildings area",
        "River": "river water channel",
        "SeaLake": "lake or sea water body",
    },
}

# 固定 3:1:1 切分，便于后续做 baseline / validation / finetuning 对比
SPLIT_RATIOS = {"train": 3, "val": 1, "test": 1}
SPLIT_SEED = 42

BATCH_SIZE = 64
NUM_WORKERS = min(8, os.cpu_count() or 1)
PIN_MEMORY = DEVICE == "cuda"
TOP_K_VALUES = (3, 5)
TOP_CONFUSED_PAIRS_TO_SHOW = 20


def label_to_text(label_name: str) -> str:
    # EuroSAT class names are CamelCase, e.g. AnnualCrop -> annual crop.
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", label_name)
    return spaced.lower()


def get_class_name_map(class_name_set_name=CLASS_NAME_SET_NAME):
    if class_name_set_name not in CLASS_NAME_SETS:
        raise ValueError(f"Unknown CLASS_NAME_SET_NAME: {class_name_set_name}")
    return CLASS_NAME_SETS[class_name_set_name]


def get_ratio_tag():
    return "".join(str(SPLIT_RATIOS[name]) for name in ("train", "val", "test"))


def get_default_output_dir(data_split, class_name_set_name, prompt_templates):
    uses_baseline_setup = (
        class_name_set_name == "default"
        and tuple(prompt_templates) == BASELINE_PROMPT_TEMPLATES
    )
    prefix = "eurosat" if uses_baseline_setup else "eurosat_tuned"

    if data_split == "all":
        suffix = "all_test"
    else:
        suffix = f"{get_ratio_tag()}_{data_split}"

    return OUTPUT_ROOT / f"{prefix}_{suffix}"


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


def get_split_file_path():
    ratio_tag = "_".join(str(SPLIT_RATIOS[name]) for name in ("train", "val", "test"))
    return SPLIT_DIR / f"eurosat_split_{ratio_tag}_seed{SPLIT_SEED}.json"


def validate_split_indices(split_indices, dataset_size):
    all_indices = []
    for split_name in ("train", "val", "test"):
        all_indices.extend(split_indices[split_name])

    if len(all_indices) != dataset_size or len(set(all_indices)) != dataset_size:
        raise ValueError("Split indices are invalid: expected a full non-overlapping partition.")


def create_split_indices(labels):
    all_indices = list(range(len(labels)))

    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=SPLIT_RATIOS["test"] / sum(SPLIT_RATIOS.values()),
        random_state=SPLIT_SEED,
        shuffle=True,
        stratify=labels,
    )

    train_val_labels = [labels[idx] for idx in train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=SPLIT_RATIOS["val"] / (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]),
        random_state=SPLIT_SEED,
        shuffle=True,
        stratify=train_val_labels,
    )

    split_indices = {
        "train": sorted(train_indices),
        "val": sorted(val_indices),
        "test": sorted(test_indices),
    }
    validate_split_indices(split_indices, len(labels))
    return split_indices


def load_or_create_split_indices(dataset):
    split_path = get_split_file_path()
    split_path.parent.mkdir(parents=True, exist_ok=True)

    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        split_indices = {
            split_name: [int(idx) for idx in payload["indices"][split_name]]
            for split_name in ("train", "val", "test")
        }
        validate_split_indices(split_indices, len(dataset))
        return split_indices, split_path

    split_indices = create_split_indices(dataset.targets)
    payload = {
        "dataset_name": DATASET_NAME,
        "seed": SPLIT_SEED,
        "ratios": SPLIT_RATIOS,
        "sizes": {split_name: len(indices) for split_name, indices in split_indices.items()},
        "indices": split_indices,
    }
    save_json(split_path, payload)
    return split_indices, split_path


def load_dataset(preprocess, data_split):
    dataset = EuroSAT(
        root=DATA_ROOT,
        download=True,
        transform=preprocess,
    )

    print("Selected classes:")
    for idx, name in enumerate(dataset.classes):
        print(f" - {name} -> {idx}")

    print(f"Total images in dataset: {len(dataset)}")

    split_indices, split_path = load_or_create_split_indices(dataset)

    print("\nFixed split summary (3:1:1, stratified):")
    for split_name in ("train", "val", "test"):
        split_size = len(split_indices[split_name])
        print(f" - {split_name}: {split_size} images ({split_size / len(dataset):.2%})")
    print(f"Split file: {split_path}")

    if data_split == "all":
        selected_dataset = dataset
    else:
        if data_split not in split_indices:
            raise ValueError(f"Unsupported DATA_SPLIT: {data_split}")
        selected_dataset = Subset(dataset, split_indices[data_split])

    print(f"Running split: {data_split}")
    print(f"Images in selected split: {len(selected_dataset)}")

    loader = DataLoader(
        selected_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
    )
    split_metadata = {
        "split_seed": SPLIT_SEED,
        "split_ratios": SPLIT_RATIOS,
        "split_sizes": {split_name: len(indices) for split_name, indices in split_indices.items()},
        "split_file": str(split_path),
        "selected_split_size": len(selected_dataset),
    }
    return dataset.classes, loader, split_metadata


def build_prompt_groups(class_names, class_name_map=None, prompt_templates=None):
    if class_name_map is None:
        class_name_map = get_class_name_map()
    if prompt_templates is None:
        prompt_templates = PROMPT_TEMPLATES

    prompt_groups = {}
    for class_name in class_names:
        alias = class_name_map.get(class_name, label_to_text(class_name))
        prompt_groups[class_name] = [template.format(alias) for template in prompt_templates]
    return prompt_groups


def print_prompt_groups(prompt_groups, class_name_set_name, prompt_templates):
    print(f"\nClass name set: {class_name_set_name}")
    print("Prompt templates:")
    for template in prompt_templates:
        print(" -", template)

    print("\nPrompt groups:")
    for class_name, prompts in prompt_groups.items():
        print(f" - {class_name}:")
        for prompt in prompts:
            print(f"    {prompt}")


def encode_text_features(model, tokenizer, class_names, prompt_groups):
    text_feature_rows = []

    with torch.no_grad():
        for class_name in class_names:
            prompts = prompt_groups[class_name]
            text_tokens = tokenizer(prompts).to(DEVICE)
            prompt_features = model.encode_text(text_tokens)
            prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)

            class_feature = prompt_features.mean(dim=0)
            class_feature = class_feature / class_feature.norm()
            text_feature_rows.append(class_feature)

    return torch.stack(text_feature_rows, dim=0)


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


def save_results(
    results,
    class_names,
    split_metadata,
    prompt_groups,
    data_split,
    class_name_set_name,
    class_name_map,
    prompt_templates,
    output_dir,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_stem = f"eurosat_zeroshot_{run_id}"

    top3_accuracy = results["top_k_accuracy"].get("top3_accuracy")
    top5_accuracy = results["top_k_accuracy"].get("top5_accuracy")

    summary = {
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "dataset_name": DATASET_NAME,
        "dataset_split": data_split,
        "class_name_set_name": class_name_set_name,
        "class_name_map": class_name_map,
        "prompt_templates": list(prompt_templates),
        "prompt_groups": prompt_groups,
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "class_count": len(class_names),
        "split_seed": split_metadata["split_seed"],
        "split_ratios": split_metadata["split_ratios"],
        "split_sizes": split_metadata["split_sizes"],
        "split_file": split_metadata["split_file"],
        "selected_split_size": split_metadata["selected_split_size"],
        "overall_accuracy": results["overall_accuracy"],
        "top3_accuracy": top3_accuracy,
        "top5_accuracy": top5_accuracy,
        "macro_f1": results["macro_f1"],
        "per_class_accuracy": results["per_class_accuracy"],
    }

    json_path = output_dir / f"{file_stem}_summary.json"
    per_class_csv_path = output_dir / f"{file_stem}_per_class_accuracy.csv"
    confusion_csv_path = output_dir / f"{file_stem}_top_confused_pairs.csv"

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run EuroSAT zero-shot evaluation with configurable prompts."
    )
    parser.add_argument(
        "--data-split",
        choices=("train", "val", "test", "all"),
        default=DATA_SPLIT,
        help="Dataset split to evaluate. Default: test",
    )
    parser.add_argument(
        "--class-name-set",
        choices=tuple(CLASS_NAME_SETS),
        default=CLASS_NAME_SET_NAME,
        help="Class-name alias set to use for prompt construction.",
    )
    parser.add_argument(
        "--prompt-template",
        action="append",
        default=None,
        help=(
            "Prompt template to use. Pass multiple times for multi-prompt zero-shot. "
            "Defaults to the script's tuned prompt templates."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to a split-aware folder under outputs/.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prompt_templates = tuple(args.prompt_template) if args.prompt_template else PROMPT_TEMPLATES
    class_name_map = get_class_name_map(args.class_name_set)
    output_dir = args.output_dir or get_default_output_dir(
        args.data_split,
        args.class_name_set,
        prompt_templates,
    )

    model, preprocess, tokenizer = load_model()
    class_names, loader, split_metadata = load_dataset(preprocess, args.data_split)
    prompt_groups = build_prompt_groups(class_names, class_name_map, prompt_templates)
    print_prompt_groups(prompt_groups, args.class_name_set, prompt_templates)
    text_features = encode_text_features(model, tokenizer, class_names, prompt_groups)
    results = evaluate(model, loader, text_features, class_names)
    saved_paths = save_results(
        results,
        class_names,
        split_metadata,
        prompt_groups,
        args.data_split,
        args.class_name_set,
        class_name_map,
        prompt_templates,
        output_dir,
    )
    print_results(results, class_names, saved_paths)


if __name__ == "__main__":
    main()
