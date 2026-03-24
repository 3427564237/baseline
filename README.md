# Food-101 Zero-Shot Classification with CLIP

## 项目概述

本项目使用 OpenAI 的 CLIP 模型进行 Food-101 数据集的零样本学习（Zero-Shot Learning）分类。CLIP 是一个视觉-语言模型，无需微调即可直接进行分类任务。

当前项目包含两条主线：

1. 使用 `food101_prompt_search.py` 搜索更好的文字提示（prompt）组合
2. 使用 `food101_zeroshot.py` 在 Food-101 全量测试集上运行最终零样本分类，并输出详细结果

目前搜索到的最佳 Prompt 组合为：

- `a restaurant style photo of {}`
- `a professional food photography shot of {}`

## 文件说明

### 主要脚本

- **`food101_zeroshot.py`** ⭐ **主要版本（全量数据最终评估）**
  - 使用 Food-101 测试集的全部 25,250 张图像
  - 全部 101 个类别
  - 当前版本使用最佳 Prompt 组合做文本特征平均
  - 输出整体准确率、Top-3 / Top-5、Macro F1、每类准确率、易混淆类别对
  - **推荐用于最终报告和结果统计**

- **`food101_prompt_search.py`** ⭐ **Prompt 搜索脚本**
  - 在一组候选 prompt template 中搜索表现最好的组合
  - 默认搜索大小为 1 到 4 的模板组合
  - 输出组合排名 CSV 和最佳组合 summary JSON
  - 用于确定最终 `food101_zeroshot.py` 中应采用的 prompt 方案

- **`food101_zeroshot_subset.py`** 原始版本（子集版本）
  - 支持选择特定类别数量
  - 保留了随机抽样逻辑
  - 用于对比实验或学习参考

### 辅助文件

- `EXPERIMENT_LOG.md` —— 详细的实验参数和结果记录
- `outputs/` —— 按实验版本整理的输出结果
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
```

### 首次运行

第一次运行会自动下载 Food-101 数据集到 `./data/` 目录（约 8GB）。

### Prompt 搜索

```bash
python food101_prompt_search.py
```

可选参数示例：

```bash
python food101_prompt_search.py --max-set-size 4 --top-n 20
```

### 最终零样本评估

```bash
python food101_zeroshot.py
```


### 主要参数

在 `food101_zeroshot.py` 开头可调整以下参数：

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 64 | 批大小（显存充足可增加到 128-256） |
| `NUM_WORKERS` | `min(8, cpu_count())` | 数据加载进程数 |
| `PIN_MEMORY` | `True` (GPU 时) | 固定 CPU 内存以加速 GPU 搬运 |
| `MODEL_NAME` | "ViT-B-16" | CLIP 模型大小 |
| `PRETRAINED` | "openai" | 预训练权重来源 |
| `PROMPT_TEMPLATES` | 2 个最佳模板 | 当前最终实验使用的 Prompt 组合 |
| `EXPERIMENT_TAG` | `"best_prompt_pair"` | 当前输出目录标签 |

## 输出说明

### `food101_zeroshot.py` 输出

脚本会输出并保存：

1. **使用设备** —— `cuda` 或 `cpu`
2. **整体指标** —— Overall Accuracy、Top-3、Top-5、Macro F1
3. **每类准确率** —— 每个类别的准确率、正确数和样本数
4. **易混淆类别对** —— 最常被混淆的类别组合
5. **Summary JSON** —— 记录模型、prompt、设备、整体指标
6. **Per-class CSV** —— 每类准确率统计
7. **Top confused pairs CSV** —— 易混淆类别对统计

### `food101_prompt_search.py` 输出

脚本会输出并保存：

1. **Prompt 组合排名 CSV**
2. **最佳 Prompt 组合 summary JSON**

## 输出目录结构

当前输出按实验版本分目录保存：

- `outputs/food101_baseline/` —— 单模板基线实验结果
- `outputs/food101_best_prompt_pair/` —— 最佳双模板组合实验结果
- `outputs/food101_prompt_search/` —— Prompt 搜索结果

## 实验记录

详见 `EXPERIMENT_LOG.md` 中的结果对比表。目前已经记录：

- 实验 1：基线单模板配置
- 实验 2：最佳 Prompt 组合配置

## 参考

- 原始 Food-101 论文：https://data.vision.ee.ethz.ch/datasets/food-101/
- CLIP 模型：https://github.com/openai/CLIP
