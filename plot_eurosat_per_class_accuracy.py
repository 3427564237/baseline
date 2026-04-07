import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parent
LINEAR_PROBE_OUTPUT_DIR = BASE_DIR / "outputs" / "eurosat_linear_probe"


def find_latest_per_class_csv():
    candidates = sorted(
        LINEAR_PROBE_OUTPUT_DIR.glob("eurosat_linear_probe_*_per_class_accuracy.csv")
    )
    if not candidates:
        raise FileNotFoundError(
            "No per_class_accuracy.csv file was found in outputs/eurosat_linear_probe."
        )
    return candidates[-1]


def load_per_class_accuracy_csv(path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "class_name": row["class_name"],
                    "per_class_accuracy": float(row["per_class_accuracy"]),
                    "sample_count": int(row["sample_count"]),
                    "correct_count": int(row["correct_count"]),
                }
            )
    if not rows:
        raise ValueError(f"Per-class CSV is empty or invalid: {path}")
    return rows


def build_default_output_path(input_path):
    return input_path.with_name(input_path.stem + "_bar_chart.png")


def plot_per_class_accuracy(rows, output_path):
    sorted_rows = sorted(rows, key=lambda row: row["per_class_accuracy"], reverse=True)
    class_names = [row["class_name"] for row in sorted_rows]
    accuracies = [row["per_class_accuracy"] for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(class_names, accuracies, color="#4C78A8")

    ax.set_title("EuroSAT Per-Class Accuracy")
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")

    for bar, accuracy in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 0.015,
            f"{accuracy:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved per-class accuracy bar chart to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a bar chart from a saved EuroSAT per_class_accuracy.csv file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to a per_class_accuracy.csv file. Defaults to the latest linear probe output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to the same folder as the input CSV.",
    )
    args = parser.parse_args()

    input_path = args.input if args.input is not None else find_latest_per_class_csv()
    output_path = args.output if args.output is not None else build_default_output_path(input_path)

    rows = load_per_class_accuracy_csv(input_path)
    plot_per_class_accuracy(rows, output_path)


if __name__ == "__main__":
    main()
