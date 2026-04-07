import csv
import inspect
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import EuroSAT
from tqdm import tqdm

from eurosat_zeroshot import (
    BATCH_SIZE,
    DATA_ROOT,
    DATASET_NAME,
    DEVICE,
    MODEL_NAME,
    NUM_WORKERS,
    PIN_MEMORY,
    PRETRAINED,
    SPLIT_RATIOS,
    SPLIT_SEED,
    TOP_CONFUSED_PAIRS_TO_SHOW,
    TOP_K_VALUES,
    extract_top_confused_pairs,
    load_model,
    load_or_create_split_indices,
)


# 1. Config
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "eurosat_linear_probe"
LINEAR_PROBE_SEED = 42

# We keep the search space small on purpose so the experiment stays easy to reproduce.
CANDIDATE_C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)

# Keep test untouched while we are still optimizing.
# When you are ready for the final report, change this to "test".
EVAL_SPLIT = "val"

# Only refit on train+val for the final test run.
REFIT_ON_TRAIN_AND_VAL = EVAL_SPLIT == "test"


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_split_loaders(preprocess):
    dataset = EuroSAT(
        root=DATA_ROOT,
        download=True,
        transform=preprocess,
    )
    split_indices, split_path = load_or_create_split_indices(dataset)

    loaders = {}
    split_sizes = {}

    print("\nFixed split summary (3:1:1, stratified):")
    for split_name in ("train", "val", "test"):
        subset = Subset(dataset, split_indices[split_name])
        loaders[split_name] = DataLoader(
            subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=NUM_WORKERS > 0,
        )
        split_sizes[split_name] = len(subset)
        print(f" - {split_name}: {split_sizes[split_name]} images")

    split_metadata = {
        "split_seed": SPLIT_SEED,
        "split_ratios": SPLIT_RATIOS,
        "split_sizes": split_sizes,
        "split_file": str(split_path),
    }
    return dataset.classes, loaders, split_metadata


def extract_features(model, loader, split_name):
    feature_batches = []
    label_batches = []

    for images, labels in tqdm(loader, desc=f"Extracting {split_name} features"):
        images = images.to(DEVICE, non_blocking=PIN_MEMORY)

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        feature_batches.append(image_features.cpu())
        label_batches.append(labels.cpu())

    features = torch.cat(feature_batches, dim=0).numpy().astype(np.float32)
    labels = torch.cat(label_batches, dim=0).numpy().astype(np.int64)
    return features, labels


def build_classifier(c_value):
    # LogisticRegression is a simple linear classifier on top of frozen CLIP features.
    classifier_kwargs = {
        "C": c_value,
        "max_iter": 2000,
        "random_state": LINEAR_PROBE_SEED,
        "solver": "lbfgs",
        # Some environments use an older LogisticRegression implementation
        # that does not expose multi_class in __init__. If it is supported,
        # we still prefer multinomial for this multi-class task.
        "multi_class": "multinomial",
    }

    try:
        supported_params = inspect.signature(LogisticRegression).parameters
        filtered_kwargs = {
            name: value
            for name, value in classifier_kwargs.items()
            if name in supported_params
        }
    except (TypeError, ValueError):
        filtered_kwargs = {
            name: value
            for name, value in classifier_kwargs.items()
            if name != "multi_class"
        }

    return LogisticRegression(**filtered_kwargs)


def evaluate_classifier(classifier, features, labels, class_names):
    probabilities = classifier.predict_proba(features)
    predictions = probabilities.argmax(axis=1)

    correct = int((predictions == labels).sum())
    total = int(labels.shape[0])
    overall_accuracy = correct / total
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)

    top_k_accuracy = {}
    ranked_indices = np.argsort(-probabilities, axis=1)
    for k in TOP_K_VALUES:
        k = min(k, len(class_names))
        top_k_hits = (ranked_indices[:, :k] == labels[:, None]).any(axis=1)
        top_k_accuracy[f"top{k}_accuracy"] = float(top_k_hits.mean())

    per_class_rows = []
    per_class_accuracy = {}
    for class_index, class_name in enumerate(class_names):
        class_mask = labels == class_index
        class_total = int(class_mask.sum())
        class_correct = int((predictions[class_mask] == labels[class_mask]).sum())
        class_acc = class_correct / class_total if class_total > 0 else 0.0
        per_class_accuracy[class_name] = class_acc
        per_class_rows.append(
            {
                "class_name": class_name,
                "per_class_accuracy": class_acc,
                "sample_count": class_total,
                "correct_count": class_correct,
            }
        )

    conf_matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    confused_pairs = extract_top_confused_pairs(
        conf_matrix,
        class_names,
        top_n=TOP_CONFUSED_PAIRS_TO_SHOW,
    )

    return {
        "overall_accuracy": overall_accuracy,
        "macro_f1": macro_f1,
        "top_k_accuracy": top_k_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "per_class_rows": per_class_rows,
        "confusion_matrix": conf_matrix,
        "confused_pairs": confused_pairs,
    }


def search_best_c(train_features, train_labels, val_features, val_labels, class_names):
    search_rows = []
    best_result = None

    for c_value in CANDIDATE_C_VALUES:
        classifier = build_classifier(c_value)
        classifier.fit(train_features, train_labels)
        metrics = evaluate_classifier(classifier, val_features, val_labels, class_names)

        row = {
            "C": c_value,
            "overall_accuracy": metrics["overall_accuracy"],
            "macro_f1": metrics["macro_f1"],
        }
        search_rows.append(row)

        print(
            f"C={c_value:>6} | "
            f"val_acc={metrics['overall_accuracy']:.4f} | "
            f"val_macro_f1={metrics['macro_f1']:.4f}"
        )

        if best_result is None:
            best_result = row
            continue

        current_key = (row["overall_accuracy"], row["macro_f1"])
        best_key = (best_result["overall_accuracy"], best_result["macro_f1"])
        if current_key > best_key:
            best_result = row

    return best_result, search_rows


def fit_final_classifier(best_c, train_features, train_labels, val_features, val_labels):
    if REFIT_ON_TRAIN_AND_VAL:
        final_features = np.concatenate([train_features, val_features], axis=0)
        final_labels = np.concatenate([train_labels, val_labels], axis=0)
        training_split_name = "train+val"
    else:
        final_features = train_features
        final_labels = train_labels
        training_split_name = "train"

    classifier = build_classifier(best_c)
    classifier.fit(final_features, final_labels)
    return classifier, training_split_name, int(final_features.shape[0])


def save_results(results, class_names, split_metadata, best_result, search_rows, training_info):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_stem = f"eurosat_linear_probe_{EVAL_SPLIT}_{run_id}"

    summary = {
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "dataset_name": DATASET_NAME,
        "method": "linear_probe",
        "evaluation_split": EVAL_SPLIT,
        "is_final_test_result": EVAL_SPLIT == "test",
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "linear_probe_seed": LINEAR_PROBE_SEED,
        "candidate_c_values": list(CANDIDATE_C_VALUES),
        "best_validation_result": best_result,
        "refit_on_train_and_val": REFIT_ON_TRAIN_AND_VAL,
        "final_training_split": training_info["training_split_name"],
        "final_training_sample_count": training_info["training_sample_count"],
        "class_count": len(class_names),
        "split_seed": split_metadata["split_seed"],
        "split_ratios": split_metadata["split_ratios"],
        "split_sizes": split_metadata["split_sizes"],
        "split_file": split_metadata["split_file"],
        "overall_accuracy": results["overall_accuracy"],
        "top3_accuracy": results["top_k_accuracy"].get("top3_accuracy"),
        "top5_accuracy": results["top_k_accuracy"].get("top5_accuracy"),
        "macro_f1": results["macro_f1"],
        "per_class_accuracy": results["per_class_accuracy"],
    }

    summary_path = OUTPUT_DIR / f"{file_stem}_summary.json"
    per_class_csv_path = OUTPUT_DIR / f"{file_stem}_per_class_accuracy.csv"
    confusion_csv_path = OUTPUT_DIR / f"{file_stem}_top_confused_pairs.csv"
    search_csv_path = OUTPUT_DIR / f"{file_stem}_c_search.csv"

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
    save_csv(
        search_csv_path,
        fieldnames=["C", "overall_accuracy", "macro_f1"],
        rows=search_rows,
    )

    return {
        "summary_path": summary_path,
        "per_class_csv_path": per_class_csv_path,
        "confusion_csv_path": confusion_csv_path,
        "search_csv_path": search_csv_path,
    }


def print_results(results, class_names, best_result, training_info, saved_paths):
    per_class_row_map = {row["class_name"]: row for row in results["per_class_rows"]}

    print("\n========================")
    print("EuroSAT Linear Probe")
    print("========================")
    print(f"Best C on val:     {best_result['C']}")
    print(f"Evaluation split:  {EVAL_SPLIT}")
    print(f"Final train split: {training_info['training_split_name']}")
    print(f"Train sample size: {training_info['training_sample_count']}")
    print(f"Overall Accuracy:  {results['overall_accuracy']:.4f}")
    if "top3_accuracy" in results["top_k_accuracy"]:
        print(f"Top-3 Accuracy:    {results['top_k_accuracy']['top3_accuracy']:.4f}")
    if "top5_accuracy" in results["top_k_accuracy"]:
        print(f"Top-5 Accuracy:    {results['top_k_accuracy']['top5_accuracy']:.4f}")
    print(f"Macro F1:          {results['macro_f1']:.4f}")
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
    print(f" - C search CSV: {saved_paths['search_csv_path']}")


def main():
    set_random_seed(LINEAR_PROBE_SEED)
    model, preprocess, _ = load_model()
    class_names, loaders, split_metadata = load_split_loaders(preprocess)

    train_features, train_labels = extract_features(model, loaders["train"], "train")
    val_features, val_labels = extract_features(model, loaders["val"], "val")
    test_features = None
    test_labels = None
    if EVAL_SPLIT == "test":
        test_features, test_labels = extract_features(model, loaders["test"], "test")

    print("\nSearching for the best C on the validation split:")
    best_result, search_rows = search_best_c(
        train_features,
        train_labels,
        val_features,
        val_labels,
        class_names,
    )

    print("\nTraining final classifier:")
    final_classifier, training_split_name, training_sample_count = fit_final_classifier(
        best_result["C"],
        train_features,
        train_labels,
        val_features,
        val_labels,
    )

    if EVAL_SPLIT == "test":
        results = evaluate_classifier(final_classifier, test_features, test_labels, class_names)
    elif EVAL_SPLIT == "val":
        results = evaluate_classifier(final_classifier, val_features, val_labels, class_names)
    else:
        raise ValueError(f"Unsupported EVAL_SPLIT: {EVAL_SPLIT}")

    training_info = {
        "training_split_name": training_split_name,
        "training_sample_count": training_sample_count,
    }
    saved_paths = save_results(
        results,
        class_names,
        split_metadata,
        best_result,
        search_rows,
        training_info,
    )
    print_results(results, class_names, best_result, training_info, saved_paths)


if __name__ == "__main__":
    main()
