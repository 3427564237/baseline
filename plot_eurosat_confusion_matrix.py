import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parent
LINEAR_PROBE_OUTPUT_DIR = BASE_DIR / "outputs" / "eurosat_linear_probe"


def find_latest_confusion_matrix_csv():
    candidates = sorted(
        LINEAR_PROBE_OUTPUT_DIR.glob("eurosat_linear_probe_*_confusion_matrix.csv")
    )
    if not candidates:
        raise FileNotFoundError(
            "No confusion_matrix.csv file was found in outputs/eurosat_linear_probe."
        )
    return candidates[-1]


def load_confusion_matrix_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        raise ValueError(f"Confusion matrix CSV is empty or invalid: {path}")

    class_names = rows[0][1:]
    matrix = []
    row_labels = []

    for row in rows[1:]:
        row_labels.append(row[0])
        matrix.append([int(value) for value in row[1:]])

    if row_labels != class_names:
        raise ValueError(
            "The confusion matrix CSV format is unexpected. "
            "Expected row labels to match the class-name header."
        )

    return class_names, np.array(matrix, dtype=np.float32)


def row_normalize(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def plot_heatmap(matrix, class_names, output_path, normalized):
    if normalized:
        data = row_normalize(matrix)
        color_label = "Row-normalized accuracy"
        title = "EuroSAT Confusion Matrix (Normalized)"
        value_text = "proportion"
    else:
        data = matrix
        color_label = "Sample count"
        title = "EuroSAT Confusion Matrix"
        value_text = "count"

    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = ax.imshow(data, cmap="Blues", aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label(color_label)

    # Keep the plot readable for a report by only annotating the diagonal.
    for index, class_name in enumerate(class_names):
        if normalized:
            label = f"{data[index, index]:.2f}"
        else:
            label = str(int(data[index, index]))
        ax.text(index, index, label, ha="center", va="center", color="black", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {value_text} heatmap to: {output_path}")


def build_default_output_path(input_path, normalized):
    suffix = "_normalized_heatmap.png" if normalized else "_heatmap.png"
    return input_path.with_name(input_path.stem + suffix)


def main():
    parser = argparse.ArgumentParser(
        description="Plot a heatmap from a saved EuroSAT confusion_matrix.csv file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to a confusion_matrix.csv file. Defaults to the latest linear probe output.",
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Plot a row-normalized heatmap instead of raw counts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to the same folder as the input CSV.",
    )
    args = parser.parse_args()

    input_path = args.input if args.input is not None else find_latest_confusion_matrix_csv()
    output_path = args.output if args.output is not None else build_default_output_path(
        input_path,
        normalized=args.normalized,
    )

    class_names, matrix = load_confusion_matrix_csv(input_path)
    plot_heatmap(matrix, class_names, output_path, normalized=args.normalized)


if __name__ == "__main__":
    main()
