# Gemini 视频扫描器使用指南

> 使用 Gemini 多模态大模型直接分析视频，检测行人横穿马路事件

## 功能概述

- **视频直接输入**：无需抽帧，直接上传视频给 Gemini 分析
- **精确时间定位**：返回事件发生的精确时间戳（精确到 0.1 秒）
- **风险等级分类**：区分高风险（闯红灯/不走斑马线）和低风险（正常过马路）
- **自动截帧**：根据时间戳自动提取关键帧并保存
- **双坐标输出**：同时提供归一化坐标和像素坐标
- **长视频支持**：自动切片处理，支持数小时的长视频

## 环境准备

### 1. 安装依赖

```bash
pip install google-genai opencv-python
```

### 2. 获取 API Key

1. 访问 [Google AI Studio](https://aistudio.google.com/)
2. 登录并点击 "Get API key"
3. 创建 API Key 并复制

### 3. 设置环境变量

```bash
export GOOGLE_API_KEY="你的API密钥"
```

## 基本用法

### 扫描单个视频

```bash
python scripts/gemini_video_scanner.py --video raw_data/videos/DJI_0020.MP4
```

### 指定输出目录

```bash
python scripts/gemini_video_scanner.py --video DJI_0020.MP4 --output my_results
```

### 使用不同模型

```bash
# 快速模型（便宜，适合测试）
python scripts/gemini_video_scanner.py --video video.mp4 --model gemini-2.0-flash

# 推荐模型（平衡速度和精度）
python scripts/gemini_video_scanner.py --video video.mp4 --model gemini-2.5-flash

# 最强模型（最精确，最贵）
python scripts/gemini_video_scanner.py --video video.mp4 --model gemini-2.5-pro
```

### 处理长视频

对于超过 30 分钟的长视频，建议使用自动切片功能：

```bash
# 每 10 分钟切一段
python scripts/gemini_video_scanner.py --video 8hour_video.mp4 --segment-minutes 10

# 每 20 分钟切一段（减少 API 调用次数）
python scripts/gemini_video_scanner.py --video long_video.mp4 --segment-minutes 20
```

## 输出结果

### 目录结构

```
pedestrian_crossing_results/
└── DJI_0020/
    ├── scan_result.json    # 完整检测结果
    ├── images/             # 截取的关键帧（带标注框）
    │   ├── high_05_23_5_001.jpg
    │   └── low_12_45_0_002.jpg
    └── crops/              # 裁剪出的行人区域
        ├── high_05_23_5_001_crop.jpg
        └── low_12_45_0_002_crop.jpg
```

### JSON 结果格式

```json
{
  "video": "raw_data/videos/DJI_0020.MP4",
  "video_info": {
    "width": 3840,
    "height": 2160,
    "fps": 29.97,
    "duration_sec": 1500.5
  },
  "model": "gemini-2.0-flash",
  "scan_time": "2025-12-24T10:30:00",
  "total_detections": 15,
  "high_risk_count": 5,
  "low_risk_count": 10,
  "detections": [
    {
      "timestamp": "05:23.5",
      "label": "pedestrian_crossing",
      "risk_level": "high",
      "description": "行人在无斑马线处突然横穿马路",
      "normalized_box": [300, 450, 600, 550],
      "pixel_box": [648, 1728, 1296, 2112],
      "resolution": {"width": 3840, "height": 2160},
      "frame_number": 9678,
      "saved_frame": "images/high_05_23_5_000.jpg",
      "saved_crop": "crops/high_05_23_5_000_crop.jpg"
    }
  ]
}
```

### 坐标说明

| 字段 | 格式 | 说明 |
|------|------|------|
| `normalized_box` | `[ymin, xmin, ymax, xmax]` | 归一化坐标 (0-1000)，适合训练模型 |
| `pixel_box` | `[ymin, xmin, ymax, xmax]` | 像素坐标，可直接用于绘图/裁剪 |
| `frame_number` | `int` | 帧序号，可用于精确定位 |

## 成本估算

### 视频处理成本（2025 年 12 月）

| 模型 | 价格 (每百万 Token) | 1 分钟视频 | 8 小时视频 |
|------|---------------------|------------|------------|
| gemini-2.0-flash | ~$0.15 | ~$0.005 | ~$2.5 |
| gemini-2.5-flash | ~$0.50 | ~$0.02 | ~$8 |
| gemini-2.5-pro | ~$2.00 | ~$0.08 | ~$30 |

**建议**：先用 `gemini-2.0-flash` 测试效果，满意后再考虑 Pro 版本。

## 常见问题

### Q: 视频太大上传失败？

使用 `--segment-minutes` 参数自动切片：

```bash
python scripts/gemini_video_scanner.py --video large.mp4 --segment-minutes 10
```

### Q: 时间戳不准确？

1. 在 Prompt 中强调时间精度要求
2. 尝试使用 `gemini-2.5-pro` 模型
3. 缩短视频切片时长（如 5 分钟）

### Q: 如何修改检测目标？

编辑 `scripts/gemini_video_scanner.py` 中的 `DETECTION_PROMPT` 变量，修改为你需要检测的内容。

### Q: 如何批量处理多个视频？

```bash
# 简单循环
for video in raw_data/videos/DJI_*.MP4; do
    python scripts/gemini_video_scanner.py --video "$video"
done
```

## 风险等级定义

### 高风险 (high)
- 行人闯红灯横穿马路
- 行人不走斑马线，直接穿越车行道
- 行人突然从路边冲出横穿
- 行人在车流中穿行

### 低风险 (low)
- 行人走斑马线过马路（即使没有红绿灯）
- 行人在车辆较少时正常过马路
- 行人在路口等待后通过

## 相关工具

- [video_to_dataset_async.py](../scripts/video_to_dataset_async.py) - 异步视频标注流水线
- [split_video.py](../scripts/split_video.py) - 视频切片工具
