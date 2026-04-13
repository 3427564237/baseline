# CLIP Baselines for Food-101 and EuroSAT

## 项目概述

本项目使用 OpenAI 的 CLIP 模型进行零样本学习（Zero-Shot Learning）分类，当前包含三个实验方向：

- `Food-101`：使用官方 `test` split 全量样本跑 baseline
- `Food-101 50-class subset`：固定 seed 选 50 类，从官方 `train` 切出平衡 `val` 做 prompt search
- `EuroSAT`：使用固定 seed 的 `3:1:1` 分层切分，默认在 `test` split 上跑 baseline

CLIP 是一个视觉-语言模型，无需微调即可直接进行分类任务，后续也可以在固定切分基础上继续做 prompt 改进或轻量微调。

## 文件说明

### 主要脚本

- **`food101_zeroshot.py`** ⭐ **主要版本（全量数据）**
  - 使用 Food-101 测试集的全部 25,250 张图像
  - 使用官方 `test` split 作为正式 baseline
  - 全部 101 个类别
  - 无随机抽样，无额外的 label 映射
  - 优化的 DataLoader 参数（batch_size=64, num_workers, pin_memory）
  - **推荐用于最终报告和结果统计**

- **`food101_subset_zeroshot.py`** ⭐ **Food-101 50 类子集评估脚本**
  - 固定 `seed=42` 随机保留 50 个类别
  - 从官方 `train` split 中为每类切出 `250` 张做 `val`
  - 默认运行对应 50 类子集的 `test` split
  - 支持多模板 prompt 和类名 alias set
  - 会输出 summary、per-class CSV、top confused pairs CSV、full confusion matrix CSV

- **`food101_prompt_search.py`** ⭐ **Food-101 子集 prompt 搜索脚本**
  - 只允许在 `train` 或 `val` 上搜索 prompt，默认使用 `val`
  - 支持多模板组合（multi-prompt）和类名 alias set
  - 会输出 prompt 排名表和最佳配置，方便回填到评估脚本

- **`eurosat_zeroshot.py`** ⭐ **主要版本（固定切分）**
  - 自动生成并复用固定 seed 的 `3:1:1` 分层切分
  - 默认运行 `test` split，适合做正式 baseline
  - 切分索引会保存到 `splits/`，方便复现实验
  - 如果需要训练或调参，可把 `DATA_SPLIT` 改成 `train` 或 `val`

- **`eurosat_prompt_search.py`** ⭐ **EuroSAT prompt 搜索脚本**
  - 只在固定 `val` split 上搜索 prompt，避免测试集泄露
  - 支持多模板组合（multi-prompt）和类名重命名
  - 会输出 prompt 排名表和最佳配置，方便后续写报告

- **`food101_zeroshot_subset.py`** 原始版本（子集版本）
  - 支持选择特定类别数量
  - 保留了随机抽样逻辑
  - 用于对比实验或学习参考

### 辅助文件

- `EXPERIMENT_LOG.md` —— 详细的实验参数和结果记录
- `test/` —— 单元测试文件

## 环境要求

```bash
# 依赖
torch >= 2.0
torchvision
open-clip-torch
tqdm

# 推荐环境
Python 3.10+
CUDA 11.8+ (NVIDIA GPU)
```

## 使用方法

### 基本运行

```bash
python food101_zeroshot.py
python food101_subset_zeroshot.py
python food101_prompt_search.py
python eurosat_zeroshot.py
python eurosat_prompt_search.py
```

### 首次运行

第一次运行会自动下载对应数据集到 `./data/` 目录。

### 主要参数

在脚本开头可调整以下参数：

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 64 | 批大小（显存充足可增加到 128-256） |
| `NUM_WORKERS` | `min(8, cpu_count())` | 数据加载进程数 |
| `PIN_MEMORY` | `True` (GPU 时) | 固定 CPU 内存以加速 GPU 搬运 |
| `MODEL_NAME` | "ViT-B-16" | CLIP 模型大小 |
| `PRETRAINED` | "openai" | 预训练权重来源 |
| `DATA_SPLIT` | `test` (EuroSAT) / `test` (Food-101) | 评估使用的数据划分 |
| `SPLIT_SEED` | 42 (EuroSAT) | 固定随机种子，保证切分可复现 |

## 输出说明

脚本会输出：

1. **使用设备** —— "cuda" 或 "cpu"
2. **类别列表** —— 101 个食物类别的名称和编号
3. **Prompt 示例** —— 生成的文本提示（一部分）
4. **整体准确率** —— Overall Accuracy (xx.xx%)
5. **每类准确率** —— 按类别统计的准确率、正确数和总数
6. **切分信息** —— 记录固定切分文件、seed 和各 split 大小
7. **Food-101 子集附加输出** —— full confusion matrix CSV，可直接用于绘图

## Food-101 50-Class Subset 实验建议流程

1. 先运行 `food101_prompt_search.py`，只在固定 `val` 上搜索 prompt
2. 选好 prompt 后，把最佳 `--class-name-set` 和 `--prompt-template` 回填到 `food101_subset_zeroshot.py`
3. 最终只在对应 50 类的 `test` 子集上评估一次
4. 再运行绘图脚本输出图片，用于报告展示

推荐命令：

```bash
python food101_prompt_search.py --search-split val

python food101_subset_zeroshot.py \
  --data-split test \
  --class-name-set student_v1 \
  --prompt-template "a photo of a dish of {}"

python plot_food101_per_class_accuracy.py
python plot_food101_confusion_matrix.py
python plot_food101_confusion_matrix.py --normalized
```

这样做的好处是：

- `val` 用来选 prompt
- `test` 用来做最终汇报
- 固定 50 类和固定切分，实验可复现
- 不会因为反复看 `test` 结果而产生数据泄露

注意：

- 这个实验必须写成 `Food-101 50-class subset`
- 不能把它和标准 `Food-101 101 classes` 的结果直接横向比较
- 图片输出包括：per-class accuracy bar chart、raw confusion matrix heatmap、normalized confusion matrix heatmap

## EuroSAT Prompt 实验建议流程

1. 先运行 `eurosat_zeroshot.py`，得到固定 `test` baseline
2. 再运行 `eurosat_prompt_search.py`，只在 `val` 上挑选更好的 prompt
3. 选好 prompt 后，再把最终方案放回 `eurosat_zeroshot.py` 或新脚本中，只在 `test` 上评估一次

这样做的好处是：

- `val` 用来选 prompt
- `test` 用来做最终汇报
- 不会因为反复看 `test` 结果而产生数据泄露

## 实验记录

详见 `EXPERIMENT_LOG.md` 中的结果对比表。

## 参考

- 原始 Food-101 论文：https://data.vision.ee.ethz.ch/datasets/food-101/
- CLIP 模型：https://github.com/openai/CLIP
