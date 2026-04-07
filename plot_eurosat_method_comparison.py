import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
LINEAR_PROBE_OUTPUT_DIR = OUTPUT_DIR / "eurosat_linear_probe"
BASELINE_ZERO_SHOT_OUTPUT_DIR = OUTPUT_DIR / "eurosat_311_test"
TUNED_ZERO_SHOT_OUTPUT_DIR = OUTPUT_DIR / "eurosat_tuned_311_test"


def find_latest_file(directory, pattern):
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No file matched {pattern} in {directory}")
    return candidates[-1]


def find_latest_summary(directory, pattern, split_field=None, split_value=None):
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No file matched {pattern} in {directory}")

    if split_field is None:
        return candidates[-1]

    matching_candidates = []
    for path in candidates:
        payload = load_json(path)
        if payload.get(split_field) == split_value:
            matching_candidates.append(path)

    if matching_candidates:
        return matching_candidates[-1]

    raise FileNotFoundError(
        f"No file matched {pattern} in {directory} with {split_field}={split_value!r}"
    )


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_method_rows(zero_shot_baseline_path, tuned_zero_shot_path, linear_probe_path):
    zero_shot_baseline = load_json(zero_shot_baseline_path)
    tuned_zero_shot = load_json(tuned_zero_shot_path)
    linear_probe = load_json(linear_probe_path)

    rows = [
        {
            "method": "Zero-shot baseline",
            "split": zero_shot_baseline.get("dataset_split", "unknown"),
            "overall_accuracy": zero_shot_baseline["overall_accuracy"],
            "macro_f1": zero_shot_baseline["macro_f1"],
        },
        {
            "method": "Tuned zero-shot",
            "split": tuned_zero_shot.get("dataset_split", "unknown"),
            "overall_accuracy": tuned_zero_shot["overall_accuracy"],
            "macro_f1": tuned_zero_shot["macro_f1"],
        },
        {
            "method": "Linear probe",
            "split": linear_probe.get("evaluation_split", "unknown"),
            "overall_accuracy": linear_probe["overall_accuracy"],
            "macro_f1": linear_probe["macro_f1"],
        },
    ]
    return rows


def build_default_output_path(linear_probe_path):
    return linear_probe_path.with_name("eurosat_method_comparison.png")


def plot_method_comparison(rows, output_path):
    methods = [f"{row['method']}\n({row['split']})" for row in rows]
    accuracy_values = [row["overall_accuracy"] for row in rows]
    macro_f1_values = [row["macro_f1"] for row in rows]

    x = np.arange(len(methods))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 6))
    accuracy_bars = ax.bar(x - width / 2, accuracy_values, width, label="Overall Accuracy")
    macro_f1_bars = ax.bar(x + width / 2, macro_f1_values, width, label="Macro F1")

    ax.set_title("EuroSAT Method Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    for bars in (accuracy_bars, macro_f1_bars):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved method comparison chart to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a simple comparison chart for EuroSAT methods."
    )
    parser.add_argument(
        "--zero-shot",
        type=Path,
        default=None,
        help="Path to a baseline zero-shot summary JSON. Defaults to the latest file in outputs/eurosat_311_test.",
    )
    parser.add_argument(
        "--tuned-zero-shot",
        type=Path,
        default=None,
        help="Path to a tuned zero-shot summary JSON. Defaults to the latest file in outputs/eurosat_tuned_311_test.",
    )
    parser.add_argument(
        "--linear-probe",
        type=Path,
        default=None,
        help="Path to a linear-probe summary JSON. Defaults to the latest file in outputs/eurosat_linear_probe.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to outputs/eurosat_linear_probe/eurosat_method_comparison.png.",
    )
    args = parser.parse_args()

    zero_shot_path = args.zero_shot or find_latest_summary(
        BASELINE_ZERO_SHOT_OUTPUT_DIR,
        "eurosat_zeroshot_*_summary.json",
        split_field="dataset_split",
        split_value="test",
    )
    tuned_zero_shot_path = args.tuned_zero_shot or find_latest_summary(
        TUNED_ZERO_SHOT_OUTPUT_DIR,
        "eurosat_zeroshot_*_summary.json",
        split_field="dataset_split",
        split_value="test",
    )
    linear_probe_path = args.linear_probe or find_latest_summary(
        LINEAR_PROBE_OUTPUT_DIR,
        "eurosat_linear_probe_*_summary.json",
        split_field="evaluation_split",
        split_value="test",
    )
    output_path = args.output or build_default_output_path(linear_probe_path)

    rows = collect_method_rows(zero_shot_path, tuned_zero_shot_path, linear_probe_path)
    plot_method_comparison(rows, output_path)


if __name__ == "__main__":
    main()
