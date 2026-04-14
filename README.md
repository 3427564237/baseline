# CLIP Evaluation Baselines for Food-101 and EuroSAT

## English Version

This folder contains the code used to evaluate OpenAI CLIP on two image-classification scenarios for the assignment project:

- `Food-101`: food image recognition
- `EuroSAT`: remote sensing and land-use recognition

The codebase focuses on reproducible evaluation rather than full model training. It includes:

- zero-shot CLIP baselines
- validation-only prompt search
- a linear probe on frozen CLIP image features for EuroSAT
- plotting scripts for per-class accuracy, confusion matrices, and method comparison

### Project Structure

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

### Main Scripts

#### Food-101

- `food101_zeroshot.py`  
  Runs a zero-shot baseline on the official Food-101 `test` split using all 101 classes.

- `food101_subset_zeroshot.py`  
  Runs zero-shot evaluation on a fixed 50-class subset of Food-101. The subset is deterministic:
  - `seed = 42`
  - `50` selected classes
  - `250` validation samples per class taken from the official training split

- `food101_prompt_search.py`  
  Searches prompt templates and class-name aliases on the fixed `val` split only. This keeps the `test` split untouched until final evaluation.

#### EuroSAT

- `eurosat_zeroshot.py`  
  Runs zero-shot evaluation on EuroSAT using a fixed stratified `3:1:1` split:
  - `train`
  - `val`
  - `test`

- `eurosat_prompt_search.py`  
  Searches prompt combinations on the EuroSAT `val` split only.

- `eurosat_linear_probe.py`  
  Trains a multinomial logistic regression classifier on frozen CLIP image features. This is a linear probe, not end-to-end CLIP fine-tuning.

#### Plotting

- `plot_food101_per_class_accuracy.py`
- `plot_food101_confusion_matrix.py`
- `plot_eurosat_per_class_accuracy.py`
- `plot_eurosat_confusion_matrix.py`
- `plot_eurosat_method_comparison.py`
- `plot_eurosat_qualitative_examples.py`

These scripts generate figures used for analysis and reporting.

### Environment Setup

Run commands from this folder:

```bash
cd baseline
```

#### Option A: `micromamba` / `conda`

```bash
micromamba create -f environment.yml
micromamba activate clip-baselines
```

Notes:

- The provided `environment.yml` is a good default CPU-safe setup.
- If you want CUDA acceleration, install the matching PyTorch CUDA package for your machine after creating the environment.

#### Option B: `venv` + `pip`

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

### Dependency Summary

The project uses the following external Python packages:

- `torch`
- `torchvision`
- `open-clip-torch`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tqdm`

Standard-library modules such as `argparse`, `csv`, `json`, `pathlib`, `datetime`, `random`, and `inspect` are also used.

### Reproducing the Experiments

#### 1. Food-101 full zero-shot baseline

```bash
python food101_zeroshot.py
```

#### 2. Food-101 50-class subset workflow

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

#### 3. EuroSAT zero-shot workflow

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

#### 4. EuroSAT linear probe

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

### Outputs

Generated files are saved under `outputs/` and typically include:

- `*_summary.json`
- `*_per_class_accuracy.csv`
- `*_top_confused_pairs.csv`
- `*_confusion_matrix.csv`
- `*.png` plot files

Fixed split definitions are stored under `splits/`.

### Included Result Snapshots

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

### Reproducibility Notes

- Food-101 prompt search is performed on the fixed subset `val` split only.
- EuroSAT prompt search is performed on the fixed `val` split only.
- Final test results are kept separate from validation-time prompt selection.
- The Food-101 subset result should be reported as `Food-101 50-class subset`, not as a full Food-101 benchmark.
- The EuroSAT linear probe uses frozen CLIP features plus logistic regression. It should be described as a linear probe or lightweight adaptation, not as full CLIP fine-tuning.

### About `test/`

The `test/` folder currently contains lightweight environment and exploration scripts rather than a formal automated test suite. It is useful for quick checks, but the main reproducibility evidence comes from the experiment scripts, fixed split files, and saved outputs.

---

## 中文版

本目录包含本次作业项目中用于评估 OpenAI CLIP 的代码，覆盖两个图像分类场景：

- `Food-101`：食品图像识别
- `EuroSAT`：遥感与土地利用识别

代码库重点在于可复现的评估流程，而不是完整模型训练。当前内容包括：

- CLIP 零样本基线
- 仅在验证集上进行的提示词搜索
- 基于冻结 CLIP 图像特征的 EuroSAT 线性探针
- 用于绘制分类准确率、混淆矩阵和方法对比图的脚本

### 项目结构

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

### 主要脚本

#### Food-101

- `food101_zeroshot.py`  
  在官方 Food-101 `test` 划分上使用全部 101 个类别运行零样本基线。

- `food101_subset_zeroshot.py`  
  在固定的 50 类 Food-101 子集上进行零样本评估。该子集是确定性的：
  - `seed = 42`
  - 选取 `50` 个类别
  - 从官方训练集中每类抽取 `250` 个验证样本

- `food101_prompt_search.py`  
  只在固定的 `val` 划分上搜索提示模板和类别别名，确保 `test` 集在最终评估前保持未使用状态。

#### EuroSAT

- `eurosat_zeroshot.py`  
  在固定的分层 `3:1:1` 划分上运行 EuroSAT 零样本评估，包括：
  - `train`
  - `val`
  - `test`

- `eurosat_prompt_search.py`  
  仅在 EuroSAT 的 `val` 划分上搜索更优提示组合。

- `eurosat_linear_probe.py`  
  在冻结的 CLIP 图像特征上训练多项逻辑回归分类器。这属于线性探针，而不是端到端 CLIP 微调。

#### 绘图脚本

- `plot_food101_per_class_accuracy.py`
- `plot_food101_confusion_matrix.py`
- `plot_eurosat_per_class_accuracy.py`
- `plot_eurosat_confusion_matrix.py`
- `plot_eurosat_method_comparison.py`
- `plot_eurosat_qualitative_examples.py`

这些脚本用于生成分析与报告中使用的图表。

### 环境配置

请在本目录下执行命令：

```bash
cd baseline
```

#### 方案 A：`micromamba` / `conda`

```bash
micromamba create -f environment.yml
micromamba activate clip-baselines
```

说明：

- 提供的 `environment.yml` 适合作为默认的 CPU 安全环境配置。
- 如果你需要 CUDA 加速，请在环境创建完成后，根据本机配置安装匹配版本的 PyTorch CUDA 包。

#### 方案 B：`venv` + `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
```

说明：

- 将 `torch` 和 `torchvision` 单独安装，是因为正确的安装包依赖你的平台和 CUDA 版本。
- 如果你需要指定 CUDA 版本，请先参考 PyTorch 官方安装页面选择合适命令，再执行 `pip install -r requirements.txt`。

### 依赖概览

项目使用的主要外部 Python 包如下：

- `torch`
- `torchvision`
- `open-clip-torch`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tqdm`

此外还使用了若干 Python 标准库模块，例如 `argparse`、`csv`、`json`、`pathlib`、`datetime`、`random` 和 `inspect`。

### 复现实验

#### 1. Food-101 全量零样本基线

```bash
python food101_zeroshot.py
```

#### 2. Food-101 50 类子集流程

先在验证集上搜索：

```bash
python food101_prompt_search.py --search-split val
```

再使用选定的提示配置在测试集上进行最终评估：

```bash
python food101_subset_zeroshot.py \
  --data-split test \
  --class-name-set default \
  --prompt-template "a food photo of {}"
```

生成图表：

```bash
python plot_food101_per_class_accuracy.py
python plot_food101_confusion_matrix.py
python plot_food101_confusion_matrix.py --normalized
```

#### 3. EuroSAT 零样本流程

运行基线：

```bash
python eurosat_zeroshot.py \
  --data-split test \
  --class-name-set default \
  --prompt-template "a satellite image of {}"
```

在验证集上搜索更优提示：

```bash
python eurosat_prompt_search.py --search-split val
```

使用调优后的零样本设置在测试集上评估：

```bash
python eurosat_zeroshot.py \
  --data-split test \
  --class-name-set student_v1 \
  --prompt-template "a remote sensing image of {}" \
  --prompt-template "an overhead view of {}"
```

#### 4. EuroSAT 线性探针

用于选择 `C` 的验证运行：

```bash
python eurosat_linear_probe.py --eval-split val
```

最终测试运行：

```bash
python eurosat_linear_probe.py --eval-split test
```

生成图表：

```bash
python plot_eurosat_per_class_accuracy.py
python plot_eurosat_confusion_matrix.py
python plot_eurosat_confusion_matrix.py --normalized
python plot_eurosat_method_comparison.py
python plot_eurosat_qualitative_examples.py
```

### 输出文件

生成的文件保存在 `outputs/` 下，通常包括：

- `*_summary.json`
- `*_per_class_accuracy.csv`
- `*_top_confused_pairs.csv`
- `*_confusion_matrix.csv`
- `*.png` 图像文件

固定划分定义保存在 `splits/` 中。

### 已包含的结果快照

仓库中已经包含若干历史运行结果，例如：

- Food-101 50 类调优零样本测试：
  - 总体准确率约为 `0.9043`
  - 宏平均 F1 约为 `0.9038`

- EuroSAT 零样本基线测试：
  - 总体准确率约为 `0.3556`

- EuroSAT 调优零样本测试：
  - 总体准确率约为 `0.4630`

- EuroSAT 线性探针测试：
  - 总体准确率约为 `0.9561`
  - 宏平均 F1 约为 `0.9546`

这些数值来自 `outputs/` 目录下已保存的 JSON 汇总文件。

### 可复现性说明

- Food-101 的提示词搜索仅在固定子集的 `val` 划分上进行。
- EuroSAT 的提示词搜索仅在固定的 `val` 划分上进行。
- 最终测试结果与验证阶段的提示选择严格分离。
- Food-101 子集结果应表述为 `Food-101 50-class subset`，不应当作完整 Food-101 基准。
- EuroSAT 线性探针使用的是冻结 CLIP 特征加逻辑回归，应描述为线性探针或轻量适配，而非完整 CLIP 微调。

### 关于 `test/`

`test/` 目录当前包含的是轻量级环境检查与探索脚本，而不是正式的自动化测试套件。它适合做快速验证，但主要的可复现性证据仍然来自实验脚本、固定划分文件和已保存的输出结果。
