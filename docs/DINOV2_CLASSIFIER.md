# 🦖 DINOv2 交通标志分类器

基于 Meta DINOv2 视觉大模型的交通标志分类方案，相比 CLIP 具有更强的细粒度特征提取能力。

## 📋 概述

DINOv2 是 Meta 开发的自监督视觉模型，通过大规模自监督学习获得了出色的视觉特征表示能力。本分类器使用 DINOv2 提取交通标志图像特征，并通过 Chroma 向量数据库进行相似度检索匹配。

### 优势

- **更强的视觉特征**：DINOv2 在多种下游视觉任务上超越 CLIP
- **更好的细粒度区分**：对形状、颜色、细节的感知更精确
- **对背景干扰更鲁棒**：自监督训练使其对复杂背景更不敏感
- **无需文本编码**：纯视觉匹配，不依赖文本描述质量

### 技术架构

```
实拍图片 → VLM检测bbox → 裁剪标志区域 → DINOv2特征提取 → Chroma向量检索 → 返回最相似标志
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision timm chromadb
```

### 2. 建立向量数据库（首次使用）

```bash
# 使用全部 188 个交通标志建库
python scripts/dinov2_classifier.py --build --all-signs

# 或只使用 69 个核心标志（默认）
python scripts/dinov2_classifier.py --build
```

### 3. 测试分类

```bash
# 单张图片测试
python scripts/dinov2_classifier.py --test path/to/sign.jpg

# 指定 bbox 测试
python scripts/dinov2_classifier.py --test path/to/image.jpg --bbox 100,200,150,250

# 查看 Top-5 结果
python scripts/dinov2_classifier.py --test path/to/sign.jpg --top-k 5
```

## 📖 使用方式

### 方式一：测试脚本（推荐调试）

完整 Pipeline 测试：VLM 检测 + DINOv2 分类

```bash
export ZAI_API_KEY="your_api_key"

# 使用 DINOv2 分类器（默认）
python scripts/test_clip_pipeline.py path/to/image.jpg --classifier dinov2

# 对比其他分类器
python scripts/test_clip_pipeline.py path/to/image.jpg --classifier clip
python scripts/test_clip_pipeline.py path/to/image.jpg --classifier gtsrb
```

### 方式二：完整视频标注管道

```bash
export ZAI_API_KEY="your_api_key"

# 使用 DINOv2 进行交通标志细粒度分类
python scripts/video_to_dataset_async.py \
    --video raw_data/videos/clips/D1/D1_000.mp4 \
    --sign-classifier dinov2 \
    --workers 15

# 可选分类器：dinov2, clip, vlm
```

### 方式三：作为 Python 模块导入

```python
from scripts.dinov2_classifier import DINOv2SignClassifier

# 初始化分类器
classifier = DINOv2SignClassifier(use_69_signs=False)  # 使用全部 188 个标志

# 确保数据库已建立
if classifier.db.collection.count() == 0:
    classifier.build_database()

# 分类交通标志
label, score = classifier.classify("image.jpg", bbox=[100, 200, 150, 250])
print(f"分类结果: {label} (相似度: {score:.4f})")

# 获取 Top-K 详细结果
results = classifier.classify_with_details("image.jpg", bbox=[100, 200, 150, 250], top_k=5)
for r in results:
    print(f"  {r['label']}: {r['score']:.4f}")
```

## ⚙️ 配置选项

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--build` | 建立向量数据库 | - |
| `--rebuild` | 重建数据库（清空现有数据）| - |
| `--test` | 测试图片路径 | - |
| `--bbox` | 边界框 `x1,y1,x2,y2` | 整张图片 |
| `--all-signs` | 使用全部 188 个标志 | 69 个核心标志 |
| `--top-k` | 返回前 K 个结果 | 5 |
| `--model` | DINOv2 模型大小 | `dinov2_vitl14` |

### 可用模型

| 模型 | 参数量 | 特点 |
|------|--------|------|
| `dinov2_vits14` | 21M | 最快，适合实时应用 |
| `dinov2_vitb14` | 86M | 平衡速度与效果 |
| `dinov2_vitl14` | 300M | **推荐**，效果好 |
| `dinov2_vitg14` | 1.1B | 最强，需要大显存 |

## 📁 文件结构

```
GLM_Labeling/
├── scripts/
│   ├── dinov2_classifier.py      # DINOv2 分类器实现
│   ├── test_clip_pipeline.py     # 测试脚本（支持多种分类器）
│   └── video_to_dataset_async.py # 完整标注管道
├── raw_data/
│   ├── signs/                    # 188 个参考标志图片
│   └── chroma_dinov2_db/         # DINOv2 向量数据库
└── docs/
    └── DINOV2_CLASSIFIER.md      # 本文档
```

## 🔍 结果解读

### 相似度分数

- **> 0.5**：高置信度匹配
- **0.3 - 0.5**：中等置信度，可能需要人工确认
- **< 0.3**：低置信度，可能是未知标志或方向牌

### 典型输出

```
🏆 Top-5 结果:
🥇 1. [███████░░░░░░░░░░░░░] 0.3521
      标签: No_entry_for_all_vehicles
🥈 2. [██████░░░░░░░░░░░░░░] 0.3102
      标签: No_motor_vehicles
🥉 3. [█████░░░░░░░░░░░░░░░] 0.2845
      标签: No_stopping_at_any_time
```

## ⚠️ 注意事项

### 网络问题

首次加载 DINOv2 模型需要从 PyTorch Hub 下载权重。如果网络不通，会自动回退到 timm 本地模型：

```
⚠️ 从 hub 加载失败: It looks like there is no internet connection
   尝试使用本地 timm 模型...
   使用 timm 模型: vit_large_patch14_dinov2.lvd142m
   ✅ DINOv2 模型加载完成 (输入: 518x518)
```

### 已知限制

1. **方向牌/导航牌**：这些不在 188 个标准交通标志库中，会返回低分匹配
2. **裁剪质量**：VLM 检测的 bbox 可能包含较多背景，影响匹配精度
3. **域差异**：实拍图（复杂背景、光照、角度）与参考图（干净背景）存在差异

### 改进方向

参考 [traffic_sign_detection_methods.md](./traffic_sign_detection_methods.md) 中的方案：

1. **方案 2**：使用 SAM2/Grounding DINO 精确裁剪标志区域
2. **方案 4**：训练小型分类头，提高准确率

## 📊 性能对比

| 分类器 | 特点 | 速度 | 准确率 |
|--------|------|------|--------|
| VLM (原版) | 依赖模型预训练知识 | 慢 | 低 |
| CLIP | 通用图像编码 | 中 | 中 |
| **DINOv2** | 细粒度视觉特征 | 中 | **较高** |
| GTSRB | 德国标志专用模型 | 快 | 德国标志高 |

## 🔗 相关文档

- [traffic_sign_detection_methods.md](./traffic_sign_detection_methods.md) - 技术方案总览
- [RAG_IMPLEMENTATION.md](./RAG_IMPLEMENTATION.md) - RAG 实现细节
- [WORKFLOW_MANUAL.md](./WORKFLOW_MANUAL.md) - 完整工作流程




