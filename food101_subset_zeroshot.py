import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path

import open_clip
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Food101
from tqdm import tqdm

try:
    from sklearn.metrics import confusion_matrix, f1_score
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for Macro F1 and confusion matrix. "
        "Install it with: micromamba -r /home/miku/.micromamba install -n pytorch-env scikit-learn"
    ) from exc


# 1. Config
BASE_DIR = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-16"
PRETRAINED = "openai"
DATASET_NAME = "Food-101"
DATA_ROOT = BASE_DIR / "data"
DATA_SPLIT = "test"
OUTPUT_ROOT = BASE_DIR / "outputs"
SPLIT_DIR = BASE_DIR / "splits"

SUBSET_CLASS_COUNT = 50
SUBSET_SEED = 42
VAL_SAMPLES_PER_CLASS = 250

PROMPT_TEMPLATES = ("a photo of a dish of {}",)
BASELINE_PROMPT_TEMPLATES = ("a photo of a dish of {}",)
CLASS_NAME_SET_NAME = "default"

CLASS_NAME_SETS = {
    "default": {},
    "student_v1": {
        "apple_pie": "apple pie",
        "beef_carpaccio": "beef carpaccio appetizer",
        "beef_tartare": "beef tartare",
        "caesar_salad": "caesar salad",
        "caprese_salad": "caprese salad",
        "cheese_plate": "cheese platter",
        "chicken_quesadilla": "chicken quesadilla",
        "chocolate_mousse": "chocolate mousse dessert",
        "clam_chowder": "clam chowder soup",
        "club_sandwich": "club sandwich",
        "creme_brulee": "creme brulee dessert",
        "croque_madame": "croque madame sandwich",
        "cup_cakes": "cupcakes",
        "fish_and_chips": "fish and chips",
        "fried_calamari": "fried calamari appetizer",
        "garlic_bread": "garlic bread",
        "hot_and_sour_soup": "hot and sour soup",
        "miso_soup": "miso soup",
        "onion_rings": "onion rings",
        "pork_chop": "pork chop",
        "red_velvet_cake": "red velvet cake",
        "shrimp_and_grits": "shrimp and grits",
        "spaghetti_bolognese": "spaghetti bolognese",
        "strawberry_shortcake": "strawberry shortcake",
    },
    "student_v2": {
        "apple_pie": "slice of apple pie",
        "beef_carpaccio": "thinly sliced beef carpaccio",
        "beef_tartare": "seasoned beef tartare",
        "caesar_salad": "classic caesar salad",
        "caprese_salad": "tomato mozzarella caprese salad",
        "cheese_plate": "assorted cheese platter",
        "chicken_quesadilla": "grilled chicken quesadilla",
        "chocolate_mousse": "rich chocolate mousse dessert",
        "clam_chowder": "creamy clam chowder soup",
        "club_sandwich": "triple-layer club sandwich",
        "creme_brulee": "caramelized creme brulee dessert",
        "croque_madame": "croque madame toasted sandwich",
        "cup_cakes": "decorated cupcakes",
        "fish_and_chips": "fried fish and chips meal",
        "fried_calamari": "crispy fried calamari rings",
        "garlic_bread": "toasted garlic bread",
        "hot_and_sour_soup": "chinese hot and sour soup",
        "miso_soup": "japanese miso soup",
        "onion_rings": "crispy onion rings",
        "pork_chop": "grilled pork chop",
        "red_velvet_cake": "slice of red velvet cake",
        "shrimp_and_grits": "shrimp served with grits",
        "spaghetti_bolognese": "spaghetti with bolognese sauce",
        "strawberry_shortcake": "strawberry shortcake dessert",
    },
}

BATCH_SIZE = 64
NUM_WORKERS = min(8, os.cpu_count() or 1)
PIN_MEMORY = DEVICE == "cuda"
TOP_K_VALUES = (3, 5)
TOP_CONFUSED_PAIRS_TO_SHOW = 20


class RemappedFoodSubsetDataset(Dataset):
    def __init__(self, base_dataset, indices, label_map):
        self.base_dataset = base_dataset
        self.indices = indices
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real_index = self.indices[index]
        image, original_label = self.base_dataset[real_index]
        return image, self.label_map[int(original_label)]


def label_to_text(label_name: str) -> str:
    return label_name.replace("_", " ")


def get_class_name_map(class_name_set_name=CLASS_NAME_SET_NAME):
    if class_name_set_name not in CLASS_NAME_SETS:
        raise ValueError(f"Unknown CLASS_NAME_SET_NAME: {class_name_set_name}")
    return CLASS_NAME_SETS[class_name_set_name]


def get_split_file_path():
    return SPLIT_DIR / f"food101_subset{SUBSET_CLASS_COUNT}_seed{SUBSET_SEED}_val{VAL_SAMPLES_PER_CLASS}.json"


def get_default_output_dir(data_split, class_name_set_name, prompt_templates):
    uses_baseline_setup = (
        class_name_set_name == "default"
        and tuple(prompt_templates) == BASELINE_PROMPT_TEMPLATES
    )
    prefix = "food101_subset50" if uses_baseline_setup else "food101_subset50_tuned"
    return OUTPUT_ROOT / f"{prefix}_{data_split}"


def save_json(path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix_csv(path, conf_matrix, class_names):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class", *class_names])
        for true_class, row_values in zip(class_names, conf_matrix.tolist()):
            writer.writerow([true_class, *row_values])


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


def get_dataset_labels(dataset):
    if hasattr(dataset, "_labels"):
        return [int(label) for label in dataset._labels]
    return [int(dataset[index][1]) for index in range(len(dataset))]


def collect_indices_by_class(labels, selected_class_ids):
    indices_by_class = {class_id: [] for class_id in selected_class_ids}
    selected_class_id_set = set(selected_class_ids)

    for index, label in enumerate(labels):
        if label in selected_class_id_set:
            indices_by_class[label].append(index)

    return indices_by_class


def create_subset_split_payload(train_dataset, test_dataset):
    rng = random.Random(SUBSET_SEED)
    total_class_count = len(train_dataset.classes)
    selected_original_class_ids = sorted(
        rng.sample(range(total_class_count), SUBSET_CLASS_COUNT)
    )
    selected_classes = [
        train_dataset.classes[class_id] for class_id in selected_original_class_ids
    ]
    local_label_map = {
        original_id: local_id
        for local_id, original_id in enumerate(selected_original_class_ids)
    }

    train_labels = get_dataset_labels(train_dataset)
    test_labels = get_dataset_labels(test_dataset)
    train_indices_by_class = collect_indices_by_class(train_labels, selected_original_class_ids)
    test_indices_by_class = collect_indices_by_class(test_labels, selected_original_class_ids)

    train_indices = []
    val_indices = []
    test_indices = []

    for class_id in selected_original_class_ids:
        class_train_indices = list(train_indices_by_class[class_id])
        class_test_indices = list(test_indices_by_class[class_id])
        rng.shuffle(class_train_indices)

        val_indices.extend(sorted(class_train_indices[:VAL_SAMPLES_PER_CLASS]))
        train_indices.extend(sorted(class_train_indices[VAL_SAMPLES_PER_CLASS:]))
        test_indices.extend(sorted(class_test_indices))

    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    payload = {
        "dataset_name": DATASET_NAME,
        "subset_seed": SUBSET_SEED,
        "class_count": SUBSET_CLASS_COUNT,
        "val_samples_per_class": VAL_SAMPLES_PER_CLASS,
        "selected_classes": selected_classes,
        "selected_original_class_ids": selected_original_class_ids,
        "local_label_map": {
            str(original_id): local_id for original_id, local_id in local_label_map.items()
        },
        "split_sizes": {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices),
        },
        "indices": {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        },
    }
    return payload


def validate_subset_split_payload(payload, train_dataset, test_dataset):
    selected_classes = payload["selected_classes"]
    selected_original_class_ids = [int(idx) for idx in payload["selected_original_class_ids"]]
    local_label_map = {
        int(original_id): int(local_id)
        for original_id, local_id in payload["local_label_map"].items()
    }
    split_indices = {
        split_name: [int(index) for index in payload["indices"][split_name]]
        for split_name in ("train", "val", "test")
    }

    if len(selected_classes) != SUBSET_CLASS_COUNT:
        raise ValueError("Subset split file has an unexpected number of selected classes.")
    if len(selected_original_class_ids) != SUBSET_CLASS_COUNT:
        raise ValueError("Subset split file has an unexpected number of class ids.")
    if len(set(selected_original_class_ids)) != SUBSET_CLASS_COUNT:
        raise ValueError("Subset split file contains duplicate selected class ids.")

    expected_classes = [
        train_dataset.classes[class_id] for class_id in selected_original_class_ids
    ]
    if expected_classes != selected_classes:
        raise ValueError("Selected class names do not match the stored original class ids.")

    expected_label_map = {
        class_id: local_id for local_id, class_id in enumerate(selected_original_class_ids)
    }
    if local_label_map != expected_label_map:
        raise ValueError("Local label mapping is inconsistent with the selected class ids.")

    train_labels = get_dataset_labels(train_dataset)
    test_labels = get_dataset_labels(test_dataset)
    selected_class_id_set = set(selected_original_class_ids)

    train_index_set = set(split_indices["train"])
    val_index_set = set(split_indices["val"])
    if train_index_set & val_index_set:
        raise ValueError("Train and validation subset indices overlap.")

    for split_name in ("train", "val"):
        for index in split_indices[split_name]:
            if not 0 <= index < len(train_dataset):
                raise ValueError(f"{split_name} split index out of range: {index}")
            if train_labels[index] not in selected_class_id_set:
                raise ValueError(f"{split_name} split contains a sample outside the selected subset.")

    for index in split_indices["test"]:
        if not 0 <= index < len(test_dataset):
            raise ValueError(f"test split index out of range: {index}")
        if test_labels[index] not in selected_class_id_set:
            raise ValueError("Test split contains a sample outside the selected subset.")

    expected_sizes = {
        "train": SUBSET_CLASS_COUNT * (750 - VAL_SAMPLES_PER_CLASS),
        "val": SUBSET_CLASS_COUNT * VAL_SAMPLES_PER_CLASS,
        "test": SUBSET_CLASS_COUNT * 250,
    }
    actual_sizes = {
        split_name: len(split_indices[split_name]) for split_name in ("train", "val", "test")
    }
    if actual_sizes != expected_sizes:
        raise ValueError(
            f"Unexpected split sizes. Expected {expected_sizes}, got {actual_sizes}."
        )

    if payload["split_sizes"] != expected_sizes:
        raise ValueError("Stored split sizes do not match the expected balanced subset sizes.")

    return {
        "selected_classes": selected_classes,
        "selected_original_class_ids": selected_original_class_ids,
        "local_label_map": local_label_map,
        "split_sizes": actual_sizes,
        "indices": split_indices,
    }


def load_or_create_subset_split(train_dataset, test_dataset):
    split_path = get_split_file_path()
    split_path.parent.mkdir(parents=True, exist_ok=True)

    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = create_subset_split_payload(train_dataset, test_dataset)
        save_json(split_path, payload)

    validated_payload = validate_subset_split_payload(payload, train_dataset, test_dataset)
    validated_payload["split_file"] = str(split_path)
    return validated_payload, split_path


def load_dataset(preprocess, data_split):
    train_dataset = Food101(
        root=DATA_ROOT,
        split="train",
        download=True,
        transform=preprocess,
    )
    test_dataset = Food101(
        root=DATA_ROOT,
        split="test",
        download=True,
        transform=preprocess,
    )
    split_payload, split_path = load_or_create_subset_split(train_dataset, test_dataset)
    selected_classes = split_payload["selected_classes"]

    print("Selected classes:")
    for local_id, class_name in enumerate(selected_classes):
        original_id = split_payload["selected_original_class_ids"][local_id]
        print(f" - {class_name} -> local {local_id} (original {original_id})")

    print("\nFixed subset split summary:")
    for split_name in ("train", "val", "test"):
        split_size = split_payload["split_sizes"][split_name]
        print(f" - {split_name}: {split_size} images")
    print(f"Split file: {split_path}")

    if data_split not in ("train", "val", "test"):
        raise ValueError(f"Unsupported DATA_SPLIT: {data_split}")

    base_dataset = train_dataset if data_split in ("train", "val") else test_dataset
    selected_dataset = RemappedFoodSubsetDataset(
        base_dataset,
        split_payload["indices"][data_split],
        split_payload["local_label_map"],
    )

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
        "subset_seed": SUBSET_SEED,
        "subset_class_count": SUBSET_CLASS_COUNT,
        "selected_classes": selected_classes,
        "selected_original_class_ids": split_payload["selected_original_class_ids"],
        "val_samples_per_class": VAL_SAMPLES_PER_CLASS,
        "split_sizes": split_payload["split_sizes"],
        "split_file": str(split_path),
        "selected_split_size": len(selected_dataset),
    }
    return selected_classes, loader, split_metadata


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
    for class_index, class_name in enumerate(class_names):
        class_acc = (
            per_class_correct[class_index] / per_class_total[class_index]
            if per_class_total[class_index] > 0
            else 0.0
        )
        per_class_accuracy[class_name] = class_acc
        per_class_rows.append(
            {
                "class_name": class_name,
                "per_class_accuracy": class_acc,
                "sample_count": per_class_total[class_index],
                "correct_count": per_class_correct[class_index],
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
        "confusion_matrix": conf_matrix,
        "confused_pairs": confused_pairs,
    }


def extract_top_confused_pairs(conf_matrix, class_names, top_n=TOP_CONFUSED_PAIRS_TO_SHOW):
    pairs = []
    for true_index, true_class in enumerate(class_names):
        for pred_index, pred_class in enumerate(class_names):
            if true_index == pred_index:
                continue

            count = int(conf_matrix[true_index, pred_index])
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
    file_stem = f"food101_subset50_zeroshot_{run_id}"

    summary = {
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "dataset_name": DATASET_NAME,
        "dataset_variant": "Food-101 50-class subset",
        "dataset_split": data_split,
        "class_name_set_name": class_name_set_name,
        "class_name_map": class_name_map,
        "prompt_templates": list(prompt_templates),
        "prompt_groups": prompt_groups,
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "class_count": len(class_names),
        "subset_class_count": split_metadata["subset_class_count"],
        "subset_seed": split_metadata["subset_seed"],
        "selected_classes": split_metadata["selected_classes"],
        "selected_original_class_ids": split_metadata["selected_original_class_ids"],
        "val_samples_per_class": split_metadata["val_samples_per_class"],
        "split_sizes": split_metadata["split_sizes"],
        "split_file": split_metadata["split_file"],
        "selected_split_size": split_metadata["selected_split_size"],
        "overall_accuracy": results["overall_accuracy"],
        "top3_accuracy": results["top_k_accuracy"].get("top3_accuracy"),
        "top5_accuracy": results["top_k_accuracy"].get("top5_accuracy"),
        "macro_f1": results["macro_f1"],
        "per_class_accuracy": results["per_class_accuracy"],
    }

    summary_path = output_dir / f"{file_stem}_summary.json"
    per_class_csv_path = output_dir / f"{file_stem}_per_class_accuracy.csv"
    confusion_csv_path = output_dir / f"{file_stem}_top_confused_pairs.csv"
    confusion_matrix_csv_path = output_dir / f"{file_stem}_confusion_matrix.csv"

    save_json(summary_path, summary)
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
    save_confusion_matrix_csv(
        confusion_matrix_csv_path,
        results["confusion_matrix"],
        class_names,
    )

    return {
        "summary_path": summary_path,
        "per_class_csv_path": per_class_csv_path,
        "confusion_csv_path": confusion_csv_path,
        "confusion_matrix_csv_path": confusion_matrix_csv_path,
    }


def print_results(results, class_names, saved_paths):
    per_class_row_map = {row["class_name"]: row for row in results["per_class_rows"]}

    print("\n========================")
    print("Food-101 50-Class Subset")
    print("========================")
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    if "top3_accuracy" in results["top_k_accuracy"]:
        print(f"Top-3 Accuracy:   {results['top_k_accuracy']['top3_accuracy']:.4f}")
    if "top5_accuracy" in results["top_k_accuracy"]:
        print(f"Top-5 Accuracy:   {results['top_k_accuracy']['top5_accuracy']:.4f}")
    print(f"Macro F1:         {results['macro_f1']:.4f}")
    print("========================")

    print("\nPer-class accuracy:")
    for class_name in class_names:
        row = per_class_row_map[class_name]
        print(
            f"{class_name:24s} accuracy = {row['per_class_accuracy']:.4f} "
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
    print(f" - JSON summary: {saved_paths['summary_path']}")
    print(f" - Per-class CSV: {saved_paths['per_class_csv_path']}")
    print(f" - Confusion CSV: {saved_paths['confusion_csv_path']}")
    print(f" - Full confusion matrix CSV: {saved_paths['confusion_matrix_csv_path']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Food-101 zero-shot evaluation on a fixed 50-class subset."
    )
    parser.add_argument(
        "--data-split",
        choices=("train", "val", "test"),
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
            "Defaults to the script's baseline prompt template."
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
