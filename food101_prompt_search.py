import argparse
import csv
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from sklearn.metrics import f1_score
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for Macro F1. "
        "Install it with: micromamba -r /home/miku/.micromamba install -n pytorch-env scikit-learn"
    ) from exc

from food101_subset_zeroshot import (
    CLASS_NAME_SETS,
    DATASET_NAME,
    DEVICE,
    MODEL_NAME,
    PIN_MEMORY,
    PRETRAINED,
    build_prompt_groups,
    encode_text_features,
    load_dataset,
    load_model,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "food101_subset50_prompt_optimize_val"

# Prompt search should use validation or train only.
# Keep the test split untouched until you choose the final prompt setting.
SEARCH_SPLIT = "val"
TOP_RESULTS_TO_SHOW = 10
COMBINATION_SIZES = (1, 2, 3)

PROMPT_TEMPLATES = [
    "a photo of {}",
    "a photo of a dish of {}",
    "a close-up photo of {}",
    "a food photo of {}",
]

ACTIVE_CLASS_NAME_SET_NAMES = ("default", "student_v1", "student_v2")


def save_json(path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_search_dataset(preprocess, search_split):
    if search_split == "test":
        raise ValueError(
            "Prompt search must not run on the test split. "
            "Use SEARCH_SPLIT='val' to avoid test leakage."
        )

    class_names, loader, split_metadata = load_dataset(preprocess, search_split)
    split_metadata = dict(split_metadata)
    split_metadata["search_split"] = search_split
    return class_names, loader, split_metadata


def get_template_combinations():
    template_combos = []
    for combo_size in COMBINATION_SIZES:
        for combo in combinations(PROMPT_TEMPLATES, combo_size):
            template_combos.append(list(combo))
    return template_combos


def get_active_class_name_sets():
    missing_names = [name for name in ACTIVE_CLASS_NAME_SET_NAMES if name not in CLASS_NAME_SETS]
    if missing_names:
        raise ValueError(f"Unknown class name set(s): {missing_names}")
    return {name: CLASS_NAME_SETS[name] for name in ACTIVE_CLASS_NAME_SET_NAMES}


def extract_image_features(model, loader, split_name):
    image_feature_batches = []
    label_batches = []

    for images, labels in tqdm(loader, desc=f"Extracting {split_name} image features"):
        images = images.to(DEVICE, non_blocking=PIN_MEMORY)

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        image_feature_batches.append(image_features.cpu())
        label_batches.append(labels.cpu())

    all_image_features = torch.cat(image_feature_batches, dim=0).to(DEVICE)
    all_labels = torch.cat(label_batches, dim=0).to(DEVICE)
    return all_image_features, all_labels


def evaluate_prompt_set(image_features, labels, text_features):
    with torch.no_grad():
        logits = 100.0 * image_features @ text_features.T
        preds = logits.argmax(dim=1)

    correct = (preds == labels).sum().item()
    total = labels.size(0)
    all_true = labels.cpu().tolist()
    all_pred = preds.cpu().tolist()

    return {
        "overall_accuracy": correct / total,
        "macro_f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
    }


def rank_results(results):
    ranked_results = [dict(item) for item in results]
    ranked_results.sort(key=lambda item: (item["overall_accuracy"], item["macro_f1"]), reverse=True)

    for rank, item in enumerate(ranked_results, start=1):
        item["rank"] = rank

    return ranked_results


def save_results(class_names, split_metadata, results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_stem = f"food101_subset50_prompt_search_{run_id}"
    ranked_results = rank_results(results)

    best_result = ranked_results[0] if ranked_results else None
    best_prompt_groups = {}
    if best_result is not None:
        best_templates = best_result["templates"].split(" || ")
        best_class_name_map = CLASS_NAME_SETS[best_result["class_name_set"]]
        best_prompt_groups = build_prompt_groups(class_names, best_class_name_map, best_templates)

    summary = {
        "dataset_name": DATASET_NAME,
        "dataset_variant": "Food-101 50-class subset",
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "device": DEVICE,
        "search_split": split_metadata["search_split"],
        "split_file": split_metadata["split_file"],
        "split_sizes": split_metadata["split_sizes"],
        "selected_classes": split_metadata["selected_classes"],
        "subset_seed": split_metadata["subset_seed"],
        "subset_class_count": split_metadata["subset_class_count"],
        "val_samples_per_class": split_metadata["val_samples_per_class"],
        "prompt_templates": PROMPT_TEMPLATES,
        "combination_sizes": list(COMBINATION_SIZES),
        "active_class_name_set_names": list(ACTIVE_CLASS_NAME_SET_NAMES),
        "class_name_sets": CLASS_NAME_SETS,
        "experiment_count": len(ranked_results),
        "best_result": best_result,
        "best_prompt_groups": best_prompt_groups,
        "top_results": ranked_results[:TOP_RESULTS_TO_SHOW],
    }

    summary_path = output_dir / f"{file_stem}_summary.json"
    ranking_path = output_dir / f"{file_stem}_ranking.csv"

    save_json(summary_path, summary)
    save_csv(
        ranking_path,
        fieldnames=[
            "rank",
            "experiment_id",
            "class_name_set",
            "template_count",
            "templates",
            "overall_accuracy",
            "macro_f1",
        ],
        rows=ranked_results,
    )
    return summary_path, ranking_path, ranked_results


def run_prompt_search(search_split, output_dir):
    model, preprocess, tokenizer = load_model()
    class_names, loader, split_metadata = load_search_dataset(preprocess, search_split)
    template_combos = get_template_combinations()
    active_class_name_sets = get_active_class_name_sets()
    image_features, labels = extract_image_features(
        model,
        loader,
        split_metadata["search_split"],
    )

    results = []
    total_experiments = len(active_class_name_sets) * len(template_combos)
    experiment_index = 0

    print(f"\nPrompt search is running on split: {split_metadata['search_split']}")
    print(f"Split file: {split_metadata['split_file']}")
    print(f"Active class name sets: {list(active_class_name_sets)}")
    print(f"Total experiments: {total_experiments}")
    for class_set_name, class_name_map in active_class_name_sets.items():
        for template_combo in template_combos:
            experiment_index += 1
            prompt_groups = build_prompt_groups(class_names, class_name_map, template_combo)
            text_features = encode_text_features(model, tokenizer, class_names, prompt_groups)
            metrics = evaluate_prompt_set(image_features, labels, text_features)

            result = {
                "rank": 0,
                "experiment_id": experiment_index,
                "class_name_set": class_set_name,
                "template_count": len(template_combo),
                "templates": " || ".join(template_combo),
                "overall_accuracy": metrics["overall_accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
            results.append(result)

            print(
                f"[{experiment_index:02d}/{total_experiments}] "
                f"{class_set_name:10s} | {len(template_combo)} prompts | "
                f"acc={metrics['overall_accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f}"
            )

    summary_path, ranking_path, ranked_results = save_results(
        class_names,
        split_metadata,
        results,
        output_dir,
    )
    return ranked_results, summary_path, ranking_path


def print_best_results(results):
    if not results:
        print("\nNo prompt results were saved.")
        return

    print("\nTop prompt settings:")
    for item in results[:TOP_RESULTS_TO_SHOW]:
        print(
            f"#{item['rank']:02d} "
            f"{item['class_name_set']:10s} | "
            f"{item['template_count']} prompts | "
            f"acc={item['overall_accuracy']:.4f} | "
            f"macro_f1={item['macro_f1']:.4f}"
        )
        print(f"    {item['templates']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search Food-101 subset zero-shot prompt settings on a validation split."
    )
    parser.add_argument(
        "--search-split",
        choices=("train", "val"),
        default=SEARCH_SPLIT,
        help="Split to use for prompt search. Default: val",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for prompt-search summaries and rankings.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results, summary_path, ranking_path = run_prompt_search(
        args.search_split,
        args.output_dir,
    )
    print_best_results(results)
    print("\nSaved files:")
    print(f" - Summary JSON: {summary_path}")
    print(f" - Ranking CSV:  {ranking_path}")


if __name__ == "__main__":
    main()
