## 实验 1：Baseline（全量 101 类）

**运行日期**：2026-03-20  
**脚本**：`food101_zeroshot.py`

### 配置参数

| 参数 | 数值 |
|------|------|
| BATCH_SIZE | 64 |
| NUM_WORKERS | min(8, cpu_count()) |
| PIN_MEMORY | True (on GPU) |
| PREFETCH_FACTOR | 默认 (未设置) |
| torch.no_grad() | ✓ 使用 |
| torch.inference_mode() | ✗ 未使用 |
| non_blocking (labels) | ✗ 未使用 |

### 结果

- **Overall Accuracy**: 0.8410 (84.10%)
- **GPU 显存占用**: 1.9 / 8.0 GB（占用率 ~24%）
- **CPU 占用**: 某个核心 100%
- **推导时间**: 2:30

---

## 实验 2：最佳 Prompt 组合（全量 101 类）

**运行日期**：2026-03-24  
**脚本**：`food101_zeroshot.py`

### 配置参数

| 参数 | 数值 |
|------|------|
| EXPERIMENT_TAG | best_prompt_pair |
| PROMPT_TEMPLATES | `["a restaurant style photo of {}", "a professional food photography shot of {}"]` |
| BATCH_SIZE | 64 |
| NUM_WORKERS | min(8, cpu_count()) |
| PIN_MEMORY | True (on GPU) |
| PREFETCH_FACTOR | 默认 (未设置) |
| torch.no_grad() | ✓ 使用 |
| torch.inference_mode() | ✗ 未使用 |
| non_blocking (labels) | ✓ 使用 |

### 结果

- **Overall Accuracy**: 0.8537 (85.37%)
- **Macro F1**: 0.8531
- **Top-3 Accuracy**: 0.9588
- **Top-5 Accuracy**: 0.9777
- **相对实验 1 提升**: Accuracy +1.28%，Macro F1 +1.35%
- **最容易混淆的类别对**: `steak -> prime_rib (70)`、`cup_cakes -> red_velvet_cake (47)`、`filet_mignon -> steak (47)`
- **输出目录**: `outputs/food101_best_prompt_pair/`
