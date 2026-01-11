# Traffic-AutoLabel 工作手册

> 基于 GLM-4.6V 的交通场景自动标注系统 - 最优流程版本

---

## 📋 目录

1. [流程概述](#流程概述)
2. [环境准备](#环境准备)
3. [操作流程](#操作流程)
4. [目录结构](#目录结构)
5. [产物说明](#产物说明)
6. [常用命令速查](#常用命令速查)
7. [故障排除](#故障排除)

---

## 流程概述

```
原始视频 → 视频分割 → 异步标注 → 整合打包 → 最终数据集
   │          │          │          │          │
   │          │          │          │          └── Dx_dataset.zip
   │          │          │          └── generate_dataset_info.py
   │          │          └── video_to_dataset_async.py (优化版：直接输出)
   │          └── split_video.py (FFmpeg 流复制)
   └── raw_data/videos/raw_videos/
```

### 关键优化点

| 优化项 | 说明 |
|--------|------|
| 直接输出 | 帧/标注/可视化直接写入 dataset 目录，无中间复制 |
| 断点续传 | 自动跳过已处理的帧，支持中断恢复 |
| 并行处理 | 2进程×4workers=8并发，平衡速度与API限流 |
| RAG分类 | 188种交通标志细粒度分类 |

---

## 环境准备

### 1. 激活虚拟环境

```bash
cd /Users/justin/Desktop/GLM_Labeling
source venv/bin/activate
```

### 2. 设置 API Key

```bash
export ZAI_API_KEY="your_api_key_here"
```

### 3. 验证环境

```bash
python -c "import httpx, PIL; print('环境正常')"
ffmpeg -version | head -1
```

---

## 操作流程

### Step 1: 视频分割

将原始视频按 ~33秒/段 分割（对应 3FPS 下约 100 帧）。

**单个视频：**
```bash
python scripts/split_video.py raw_data/videos/raw_videos/D1.mp4
```

**批量分割（外部硬盘）：**
```bash
for f in /Volumes/硬盘名/视频目录/*.mp4; do
  python scripts/split_video.py "$f"
done
```

**参数说明：**
- `--segment-time 33.33` - 每段时长（秒），默认33.33秒
- `--min-duration 10.0` - 末段最小时长，低于此值自动删除
- `--prefix D1` - 输出文件前缀

**输出：**
```
raw_data/videos/clips/D1/
├── D1_000.mp4
├── D1_001.mp4
└── ...
```

---

### Step 2: 异步标注

对分割后的视频片段进行标注。

**单个片段：**
```bash
python scripts/video_to_dataset_async.py \
  --video raw_data/videos/clips/D1/D1_000.mp4 \
  --workers 4 \
  --rag
```

**批量标注（推荐）：**
```bash
find raw_data/videos/clips/D1 -name "*.mp4" | sort | \
  xargs -P 2 -I {} python scripts/video_to_dataset_async.py \
  --video "{}" --workers 4 --rag
```

**参数说明：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video` | 视频文件路径 | 必填 |
| `--workers` | 并发数 | 15 (建议4) |
| `--fps` | 抽帧率 | 3 |
| `--rag` | 启用RAG细粒度分类 | 默认启用 |
| `--no-rag` | 禁用RAG | - |
| `--skip-visualize` | 跳过可视化 | - |

**并发建议：**
- `xargs -P 2` × `--workers 4` = 8 并发（推荐）
- API 并发限制约 10，超过会触发 429 限流

**输出（直接生成）：**
```
dataset_output/D1_000_dataset/
├── frames/         # 抽取的帧
├── annotations/    # JSON 标注
├── visualized/     # 可视化图片
├── video/          # 源视频
├── SUMMARY.md      # 标注报告
└── stats.json      # 统计数据
```

---

### Step 3: 整合打包

将多个片段整合为完整数据集并生成压缩包。

```bash
python scripts/generate_dataset_info.py D1 --consolidate --zip
```

**参数说明：**
- `--consolidate` - 整合分散的片段到统一目录
- `--zip` - 生成压缩包

**输出：**
```
dataset_output/
├── D1_dataset/           # 整合后的数据集
│   ├── D1_000_dataset/
│   ├── D1_001_dataset/
│   └── D1_dataset_info.txt
└── D1_dataset.zip        # 压缩包
```

---

### 一键批量处理

处理多个视频的完整流程：

```bash
# 设置环境
cd /Users/justin/Desktop/GLM_Labeling
source venv/bin/activate
export ZAI_API_KEY="your_key"

# 批量处理 D3-D6
for v in D3 D4 D5 D6; do
  echo "=== 处理 $v ==="
  
  # 1. 分割
  python scripts/split_video.py raw_data/videos/raw_videos/$v.mp4
  
  # 2. 标注
  find raw_data/videos/clips/$v -name "*.mp4" | sort | \
    xargs -P 2 -I {} python scripts/video_to_dataset_async.py \
    --video "{}" --workers 4 --rag
  
  # 3. 整合打包
  python scripts/generate_dataset_info.py $v --consolidate --zip
done
```

---

## 目录结构

```
GLM_Labeling/
├── raw_data/
│   ├── videos/
│   │   ├── raw_videos/      # 原始视频
│   │   │   ├── D1.mp4
│   │   │   └── ...
│   │   ├── clips/           # 分割后的片段
│   │   │   ├── D1/
│   │   │   │   ├── D1_000.mp4
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── archive/         # 归档的原始视频
│   └── signs/               # 188种交通标志图片（RAG用）
│
├── dataset_output/          # 最终输出
│   ├── D1_dataset/          # 整合后的数据集
│   ├── D1_dataset.zip       # 压缩包
│   └── ...
│
├── scripts/                 # 核心脚本
│   ├── split_video.py       # 视频分割
│   ├── video_to_dataset_async.py  # 异步标注（优化版）
│   └── generate_dataset_info.py   # 整合打包
│
└── venv/                    # Python 虚拟环境
```

---

## 产物说明

### 中间产物

| 产物 | 位置 | 说明 | 可删除？ |
|------|------|------|----------|
| 视频片段 | `raw_data/videos/clips/` | 分割后的~33秒片段 | 打包后可删 |
| 分散的dataset | `dataset_output/Dx_xxx_dataset/` | 每个片段的标注 | 整合后可删 |

### 最终产物

| 产物 | 位置 | 说明 |
|------|------|------|
| 整合数据集 | `dataset_output/Dx_dataset/` | 包含所有片段 |
| 压缩包 | `dataset_output/Dx_dataset.zip` | 可分发的完整数据集 |

### 数据集结构

```
Dx_dataset/
├── Dx_000_dataset/
│   ├── video/
│   │   └── Dx_000.mp4
│   ├── frames/
│   │   └── Dx_000_000001.jpg ...
│   ├── annotations/
│   │   └── Dx_000_000001.json ...
│   ├── visualized/
│   │   └── Dx_000_000001_vis.jpg ...
│   ├── SUMMARY.md
│   └── stats.json
├── Dx_001_dataset/
│   └── ...
└── Dx_dataset_info.txt      # 整体报告
```

---

## 常用命令速查

### 环境设置
```bash
cd /Users/justin/Desktop/GLM_Labeling
source venv/bin/activate
export ZAI_API_KEY="your_key"
```

### 视频分割
```bash
# 单个
python scripts/split_video.py raw_data/videos/raw_videos/D1.mp4

# 外部硬盘批量
for f in /Volumes/LQ1000/DJI_1080p/*.mp4; do
  python scripts/split_video.py "$f" --prefix "$(basename "$f" _1080.mp4)"
done
```

### 标注
```bash
# 单个片段
python scripts/video_to_dataset_async.py --video path/to/clip.mp4 --workers 4 --rag

# 批量（2进程并行）
find raw_data/videos/clips/D1 -name "*.mp4" | sort | \
  xargs -P 2 -I {} python scripts/video_to_dataset_async.py --video "{}" --workers 4 --rag
```

### 整合打包
```bash
python scripts/generate_dataset_info.py D1 --consolidate --zip
```

### 检查进度
```bash
# 查看完成数量
ls -d dataset_output/D1_*_dataset 2>/dev/null | wc -l

# 查看磁盘空间
df -h /
```

### 清理空间
```bash
# 删除临时文件（优化后的脚本已不再生成）
rm -rf temp_frames output

# 删除已打包的clips（可选）
rm -rf raw_data/videos/clips/D1
```

---

## 故障排除

### 1. API 限流 (429 Too Many Requests)

**症状：** 大量 Timeout 和 JSON parse error

**解决：** 降低并发
```bash
# 改为串行处理
xargs -P 1 -I {} ... --workers 4

# 或减少 workers
xargs -P 2 -I {} ... --workers 3
```

### 2. 磁盘空间不足 (Errno 28)

**症状：** No space left on device

**解决：**
```bash
# 1. 清空废纸篓
# 2. 删除临时文件
rm -rf temp_frames output
# 3. 删除已打包的原始clips
rm -rf raw_data/videos/clips/D1
```

### 3. 断点续传

脚本自动支持断点续传：
- 已抽取的帧会跳过
- 已标注的帧会跳过

如需完全重跑：
```bash
rm -rf dataset_output/D1_000_dataset
python scripts/video_to_dataset_async.py --video ... 
```

### 4. 外部硬盘视频处理

```bash
# 直接从外部硬盘读取，clips输出到本地
python scripts/split_video.py /Volumes/硬盘/视频.mp4 -o raw_data/videos/clips/视频名
```

---

## 性能参考

| 指标 | 数值 |
|------|------|
| 处理速度 | 视频时长 × 7-8 倍 |
| 每片段耗时 | 4-6 分钟 (100帧) |
| API成本 | ~¥10/视频 (按D1-N8平均) |
| 推荐并发 | 2进程 × 4workers = 8 |

---

*最后更新: 2024-12-17*
*版本: v2.0 (优化版)*






