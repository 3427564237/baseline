# CLIP Evaluation Baselines for Food-101 and EuroSAT

This folder contains the code used to evaluate OpenAI CLIP on two image-classification scenarios for the assignment project:

- `Food-101`: food image recognition
- `EuroSAT`: remote sensing / land-use recognition

The codebase focuses on reproducible evaluation rather than full model training. It includes:

- zero-shot CLIP baselines
- validation-only prompt search
- a linear probe on frozen CLIP image features for EuroSAT
- plotting scripts for per-class accuracy, confusion matrices, and method comparison

## Project Structure

```text
baseline/
├── food101_zeroshot.py
├── food101_subset_zeroshot.py
├── food101_prompt_search.py
├── eurosat_zeroshot.py
├── eurosat_prompt_search.py
├── eurosat_linear_probe.py
├── plot_*.py
├── splits/
├── outputs/
└── test/
```

## Main Scripts

### Food-101

- `food101_zeroshot.py`
  Runs a zero-shot baseline on the official Food-101 `test` split using all 101 classes.

- `food101_subset_zeroshot.py`
  Runs zero-shot evaluation on a fixed 50-class subset of Food-101. The subset is deterministic:
  - `seed = 42`
  - `50` selected classes
  - `250` validation samples per class taken from the official training split

- `food101_prompt_search.py`
  Searches prompt templates and class-name aliases on the fixed `val` split only. This keeps the `test` split untouched until final evaluation.

### EuroSAT

- `eurosat_zeroshot.py`
  Runs zero-shot evaluation on EuroSAT using a fixed stratified `3:1:1` split:
  - `train`
  - `val`
  - `test`

- `eurosat_prompt_search.py`
  Searches prompt combinations on the EuroSAT `val` split only.

- `eurosat_linear_probe.py`
  Trains a multinomial logistic regression classifier on frozen CLIP image features. This is a linear probe, not end-to-end CLIP fine-tuning.

### Plotting

- `plot_food101_per_class_accuracy.py`
- `plot_food101_confusion_matrix.py`
- `plot_eurosat_per_class_accuracy.py`
- `plot_eurosat_confusion_matrix.py`
- `plot_eurosat_method_comparison.py`
- `plot_eurosat_qualitative_examples.py`

These scripts generate figures used for analysis and reporting.

## Environment Setup

Run commands from this folder:

```bash
cd baseline
```

### Option A: `micromamba` / `conda`

```bash
micromamba create -f environment.yml
micromamba activate clip-baselines
```

Notes:

- The provided `environment.yml` is a good default CPU-safe setup.
- If you want CUDA acceleration, install the matching PyTorch CUDA package for your machine after creating the environment.

### Option B: `venv` + `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
```

Notes:

- `torch` and `torchvision` are left as a separate install step because the correct wheel depends on your platform and CUDA version.
- If you need a specific CUDA build, use the selector from the official PyTorch installation page, then run `pip install -r requirements.txt`.

## Dependency Summary

The project uses the following external Python packages:

- `torch`
- `torchvision`
- `open-clip-torch`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tqdm`

Standard-library modules such as `argparse`, `csv`, `json`, `pathlib`, `datetime`, `random`, and `inspect` are also used.

## Reproducing the Experiments

### 1. Food-101 full zero-shot baseline

```bash
python food101_zeroshot.py
```

### 2. Food-101 50-class subset workflow

Search on validation:

```bash
python food101_prompt_search.py --search-split val
```

Run final evaluation on test using the selected prompt setup:

```bash
python food101_subset_zeroshot.py \
  --data-split test \
  --class-name-set default \
  --prompt-template "a food photo of {}"
```

Generate plots:

```bash
python plot_food101_per_class_accuracy.py
python plot_food101_confusion_matrix.py
python plot_food101_confusion_matrix.py --normalized
```

### 3. EuroSAT zero-shot workflow

Run the baseline:

```bash
python eurosat_zeroshot.py \
  --data-split test \
  --class-name-set default \
  --prompt-template "a satellite image of {}"
```

Search better prompts on validation:

```bash
python eurosat_prompt_search.py --search-split val
```

Run the tuned zero-shot setting on test:

```bash
python eurosat_zeroshot.py \
  --data-split test \
  --class-name-set student_v1 \
  --prompt-template "a remote sensing image of {}" \
  --prompt-template "an overhead view of {}"
```

### 4. EuroSAT linear probe

Validation run for selecting `C`:

```bash
python eurosat_linear_probe.py --eval-split val
```

Final test run:

```bash
python eurosat_linear_probe.py --eval-split test
```

Generate plots:

```bash
python plot_eurosat_per_class_accuracy.py
python plot_eurosat_confusion_matrix.py
python plot_eurosat_confusion_matrix.py --normalized
python plot_eurosat_method_comparison.py
python plot_eurosat_qualitative_examples.py
```

## Outputs

Generated files are saved under `outputs/` and typically include:

- `*_summary.json`
- `*_per_class_accuracy.csv`
- `*_top_confused_pairs.csv`
- `*_confusion_matrix.csv`
- `*.png` plot files

Fixed split definitions are stored under `splits/`.

## Included Result Snapshots

The repository already contains saved outputs from previous runs, including:

- Food-101 50-class tuned zero-shot test:
  - overall accuracy: about `0.9043`
  - macro F1: about `0.9038`

- EuroSAT zero-shot baseline test:
  - overall accuracy: about `0.3556`

- EuroSAT tuned zero-shot test:
  - overall accuracy: about `0.4630`

- EuroSAT linear probe test:
  - overall accuracy: about `0.9561`
  - macro F1: about `0.9546`

These numbers come from the saved JSON summaries under `outputs/`.

## Reproducibility Notes

- Food-101 prompt search is performed on the fixed subset `val` split only.
- EuroSAT prompt search is performed on the fixed `val` split only.
- Final test results are kept separate from validation-time prompt selection.
- The Food-101 subset result should be reported as `Food-101 50-class subset`, not as a full Food-101 benchmark.
- The EuroSAT linear probe uses frozen CLIP features plus logistic regression. It should be described as a linear probe or lightweight adaptation, not as full CLIP fine-tuning.

## About `test/`

The `test/` folder currently contains lightweight environment and exploration scripts rather than a formal automated test suite. It is useful for quick checks, but the main reproducibility evidence comes from the experiment scripts, fixed split files, and saved outputs.
