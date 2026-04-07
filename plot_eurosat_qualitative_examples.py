import argparse
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import EuroSAT

matplotlib.use("Agg")

from eurosat_linear_probe import (
    BATCH_SIZE,
    LINEAR_PROBE_SEED,
    NUM_WORKERS,
    PIN_MEMORY,
    build_classifier,
    extract_features,
    set_random_seed,
)
from eurosat_zeroshot import (
    DATA_ROOT,
    DEVICE,
    build_prompt_groups,
    encode_text_features,
    load_model,
    load_or_create_split_indices,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "eurosat_qualitative_examples"
LINEAR_PROBE_OUTPUT_DIR = BASE_DIR / "outputs" / "eurosat_linear_probe"
PROMPT_SEARCH_OUTPUT_DIR = BASE_DIR / "outputs" / "eurosat_prompt_optimize_val"


def find_latest_file(directory, pattern):
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No file matched {pattern} in {directory}")
    return candidates[-1]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt_groups(class_names, prompt_summary_path=None):
    if prompt_summary_path is None:
        prompt_summary_path = find_latest_file(
            PROMPT_SEARCH_OUTPUT_DIR,
            "eurosat_prompt_search_*_summary.json",
        )
    prompt_summary = load_json(prompt_summary_path)
    best_prompt_groups = prompt_summary.get("best_prompt_groups")
    if best_prompt_groups:
        return best_prompt_groups
    return build_prompt_groups(class_names)


def load_split_datasets(preprocess, split_name):
    raw_dataset = EuroSAT(
        root=DATA_ROOT,
        download=True,
        transform=None,
    )
    feature_dataset = EuroSAT(
        root=DATA_ROOT,
        download=True,
        transform=preprocess,
    )
    split_indices, split_path = load_or_create_split_indices(feature_dataset)
    if split_name not in split_indices:
        raise ValueError(f"Unsupported split: {split_name}")

    selected_indices = split_indices[split_name]
    feature_subset = Subset(feature_dataset, selected_indices)
    loader = DataLoader(
        feature_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
    )
    return raw_dataset, feature_dataset.classes, loader, selected_indices, split_path


def predict_zero_shot(model, tokenizer, loader, class_names, prompt_groups):
    text_features = encode_text_features(model, tokenizer, class_names, prompt_groups)

    predictions = []
    confidences = []

    for images, _labels in loader:
        images = images.to(DEVICE, non_blocking=PIN_MEMORY)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ text_features.T
            probs = torch.softmax(logits, dim=1)
            pred_indices = probs.argmax(dim=1)
            pred_confidence = probs.max(dim=1).values

        predictions.extend(pred_indices.cpu().tolist())
        confidences.extend(pred_confidence.cpu().tolist())

    return np.array(predictions), np.array(confidences)


def train_linear_probe_and_predict(model, loaders_by_split, split_name, class_names, linear_probe_summary):
    train_features, train_labels = extract_features(model, loaders_by_split["train"], "train")
    eval_features, eval_labels = extract_features(model, loaders_by_split[split_name], split_name)

    training_split_name = linear_probe_summary.get("final_training_split", "train")
    best_c = linear_probe_summary["best_validation_result"]["C"]

    if training_split_name == "train+val":
        val_features, val_labels = extract_features(model, loaders_by_split["val"], "val")
        train_features = np.concatenate([train_features, val_features], axis=0)
        train_labels = np.concatenate([train_labels, val_labels], axis=0)

    classifier = build_classifier(best_c)
    classifier.fit(train_features, train_labels)

    probabilities = classifier.predict_proba(eval_features)
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)

    if len(class_names) != probabilities.shape[1]:
        raise ValueError("Class count mismatch while generating linear-probe predictions.")

    return np.array(eval_labels), np.array(predictions), np.array(confidences)


def build_loaders_by_split(feature_dataset):
    split_indices, _ = load_or_create_split_indices(feature_dataset)
    loaders = {}
    for split_name in ("train", "val", "test"):
        subset = Subset(feature_dataset, split_indices[split_name])
        loaders[split_name] = DataLoader(
            subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=NUM_WORKERS > 0,
        )
    return loaders


def choose_example_rows(true_labels, zero_shot_preds, linear_probe_preds, linear_probe_confidence, corrected_count, failed_count):
    corrected_rows = []
    failed_rows = []

    corrected_candidates = [
        index
        for index, (true_label, zero_pred, linear_pred) in enumerate(
            zip(true_labels, zero_shot_preds, linear_probe_preds)
        )
        if zero_pred != true_label and linear_pred == true_label
    ]
    corrected_candidates.sort(key=lambda idx: linear_probe_confidence[idx], reverse=True)

    failed_candidates = [
        index
        for index, (true_label, linear_pred) in enumerate(zip(true_labels, linear_probe_preds))
        if linear_pred != true_label
    ]
    failed_candidates.sort(key=lambda idx: linear_probe_confidence[idx], reverse=True)

    corrected_rows = corrected_candidates[:corrected_count]
    failed_rows = failed_candidates[:failed_count]
    return corrected_rows, failed_rows


def save_metadata_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "example_type",
                "split",
                "dataset_index",
                "ground_truth",
                "zero_shot_prediction",
                "linear_probe_prediction",
                "linear_probe_confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_examples(raw_dataset, selected_indices, class_names, split_name, corrected_rows, failed_rows, zero_shot_preds, linear_probe_preds, linear_probe_confidence, output_path):
    row_specs = [
        ("Corrected By Linear Probe", corrected_rows),
        ("Still Wrong After Linear Probe", failed_rows),
    ]
    num_cols = max(max(len(rows) for _, rows in row_specs), 1)
    num_rows = len(row_specs)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4.5 * num_rows))
    if num_rows == 1:
        axes = np.array([axes])
    if num_cols == 1:
        axes = axes.reshape(num_rows, 1)

    metadata_rows = []

    for row_index, (row_title, row_indices) in enumerate(row_specs):
        for col_index in range(num_cols):
            ax = axes[row_index, col_index]
            ax.axis("off")

            if col_index >= len(row_indices):
                continue

            local_index = row_indices[col_index]
            dataset_index = selected_indices[local_index]
            image, true_label = raw_dataset[dataset_index]

            zero_pred = zero_shot_preds[local_index]
            linear_pred = linear_probe_preds[local_index]
            confidence = linear_probe_confidence[local_index]

            ax.imshow(image)
            ax.set_title(
                f"{row_title}\n"
                f"GT: {class_names[true_label]}\n"
                f"ZS: {class_names[zero_pred]}\n"
                f"LP: {class_names[linear_pred]} ({confidence:.2f})",
                fontsize=9,
            )

            metadata_rows.append(
                {
                    "example_type": row_title,
                    "split": split_name,
                    "dataset_index": dataset_index,
                    "ground_truth": class_names[true_label],
                    "zero_shot_prediction": class_names[zero_pred],
                    "linear_probe_prediction": class_names[linear_pred],
                    "linear_probe_confidence": confidence,
                }
            )

    fig.suptitle("EuroSAT Qualitative Examples", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return metadata_rows


def main():
    parser = argparse.ArgumentParser(
        description="Create qualitative EuroSAT examples comparing zero-shot and linear probe."
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Split to visualize. Default: val",
    )
    parser.add_argument(
        "--corrected-count",
        type=int,
        default=3,
        help="How many corrected examples to show. Default: 3",
    )
    parser.add_argument(
        "--failed-count",
        type=int,
        default=3,
        help="How many still-failed examples to show. Default: 3",
    )
    parser.add_argument(
        "--linear-probe-summary",
        type=Path,
        default=None,
        help="Path to a linear-probe summary JSON. Defaults to the latest one.",
    )
    parser.add_argument(
        "--prompt-summary",
        type=Path,
        default=None,
        help="Path to a prompt-search summary JSON. Defaults to the latest one.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to outputs/eurosat_qualitative_examples/.",
    )
    args = parser.parse_args()

    set_random_seed(LINEAR_PROBE_SEED)
    linear_probe_summary_path = args.linear_probe_summary or find_latest_file(
        LINEAR_PROBE_OUTPUT_DIR,
        "eurosat_linear_probe_*_summary.json",
    )
    linear_probe_summary = load_json(linear_probe_summary_path)

    model, preprocess, tokenizer = load_model()
    raw_dataset, class_names, eval_loader, selected_indices, _split_path = load_split_datasets(
        preprocess,
        args.split,
    )
    feature_dataset = EuroSAT(
        root=DATA_ROOT,
        download=True,
        transform=preprocess,
    )
    loaders_by_split = build_loaders_by_split(feature_dataset)

    prompt_groups = load_prompt_groups(class_names, args.prompt_summary)
    zero_shot_preds, _zero_shot_confidence = predict_zero_shot(
        model,
        tokenizer,
        eval_loader,
        class_names,
        prompt_groups,
    )
    true_labels, linear_probe_preds, linear_probe_confidence = train_linear_probe_and_predict(
        model,
        loaders_by_split,
        args.split,
        class_names,
        linear_probe_summary,
    )

    corrected_rows, failed_rows = choose_example_rows(
        true_labels,
        zero_shot_preds,
        linear_probe_preds,
        linear_probe_confidence,
        corrected_count=args.corrected_count,
        failed_count=args.failed_count,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or OUTPUT_DIR / f"eurosat_qualitative_examples_{args.split}.png"
    metadata_path = output_path.with_suffix(".csv")

    metadata_rows = plot_examples(
        raw_dataset,
        selected_indices,
        class_names,
        args.split,
        corrected_rows,
        failed_rows,
        zero_shot_preds,
        linear_probe_preds,
        linear_probe_confidence,
        output_path,
    )
    save_metadata_csv(metadata_path, metadata_rows)

    print(f"Saved qualitative example figure to: {output_path}")
    print(f"Saved qualitative example metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
