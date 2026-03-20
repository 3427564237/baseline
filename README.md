# Food-101 Zero-Shot Classification with CLIP

## 项目概述

本项目使用 OpenAI 的 CLIP 模型进行 Food-101 数据集的零样本学习（Zero-Shot Learning）分类。CLIP 是一个视觉-语言模型，无需微调即可直接进行分类任务。

## 文件说明

### 主要脚本

- **`food101_zeroshot.py`** ⭐ **主要版本（全量数据）**
  - 使用 Food-101 测试集的全部 25,250 张图像
  - 全部 101 个类别
  - 无随机抽样，无额外的 label 映射
  - 优化的 DataLoader 参数（batch_size=64, num_workers, pin_memory）
  - **推荐用于最终报告和结果统计**

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
```

### 首次运行

第一次运行会自动下载 Food-101 数据集到 `./data/` 目录（约 8GB）。

### 主要参数

在 `food101_zeroshot.py` 开头可调整以下参数：

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 64 | 批大小（显存充足可增加到 128-256） |
| `NUM_WORKERS` | `min(8, cpu_count())` | 数据加载进程数 |
| `PIN_MEMORY` | `True` (GPU 时) | 固定 CPU 内存以加速 GPU 搬运 |
| `MODEL_NAME` | "ViT-B-16" | CLIP 模型大小 |
| `PRETRAINED` | "openai" | 预训练权重来源 |

## 输出说明

脚本会输出：

1. **使用设备** —— "cuda" 或 "cpu"
2. **类别列表** —— 101 个食物类别的名称和编号
3. **Prompt 示例** —— 生成的文本提示（一部分）
4. **整体准确率** —— Overall Accuracy (xx.xx%)
5. **每类准确率** —— 按类别统计的准确率、正确数和总数

## 实验记录

详见 `EXPERIMENT_LOG.md` 中的结果对比表。

## 参考

- 原始 Food-101 论文：https://data.vision.ee.ethz.ch/datasets/food-101/
- CLIP 模型：https://github.com/openai/CLIP
