#!/usr/bin/env python3
"""
交通标志标注 UI

用于对筛选后的 valid 图片进行细粒度标注：
- 左侧显示待标注图片
- 右侧显示 188 种标准交通标志参考图（可搜索）
- 点击参考图完成标注

快捷键：
- Z: 撤销上一个
- Q: 保存并退出
- 左/右箭头: 上一张/下一张
- /: 聚焦搜索框
- Esc: 清空搜索
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    from flask import Flask, render_template_string, jsonify, request, send_file
except ImportError:
    print("请安装 Flask: pip install flask")
    sys.exit(1)


# ============ 配置 ============

EXTRACTED_SIGNS_DIR = Path(__file__).parent.parent / "extracted_signs"
REFERENCE_SIGNS_DIR = Path(__file__).parent.parent / "raw_data" / "signs"

# Batch 配置
BATCH_CONFIG = {
    "batch1": {
        "signs_dir": EXTRACTED_SIGNS_DIR / "DJI_batch1",
        "filter_file": EXTRACTED_SIGNS_DIR / "DJI_batch1" / "filter_results_batch1.json",
        "output_file": EXTRACTED_SIGNS_DIR / "DJI_batch1" / "label_results_batch1.json",
    },
    "batch2": {
        "signs_dir": EXTRACTED_SIGNS_DIR / "DJI_batch2",
        "filter_file": EXTRACTED_SIGNS_DIR / "DJI_batch2" / "filter_results_batch2.json",
        "output_file": EXTRACTED_SIGNS_DIR / "DJI_batch2" / "label_results_batch2.json",
    },
    "batch3": {
        "signs_dir": EXTRACTED_SIGNS_DIR / "DJI_batch3",
        "filter_file": None,  # 直接标注，不需要筛选
        "output_file": EXTRACTED_SIGNS_DIR / "DJI_batch3" / "label_results_batch3.json",
    },
    "batch4": {
        "signs_dir": EXTRACTED_SIGNS_DIR / "DJI_batch4",
        "filter_file": None,
        "output_file": EXTRACTED_SIGNS_DIR / "DJI_batch4" / "label_results_batch4.json",
    },
    "batch5": {
        "signs_dir": EXTRACTED_SIGNS_DIR / "DJI_batch5",
        "filter_file": None,
        "output_file": EXTRACTED_SIGNS_DIR / "DJI_batch5" / "label_results_batch5.json",
    },
}

# 默认使用 batch2
DEFAULT_BATCH = "batch2"

# 历史标注频率文件 (用于排序参考标志)
FREQUENCY_FILE = EXTRACTED_SIGNS_DIR / "DJI_batch1" / "label_results_batch1.json"


# ============ 数据管理 ============

class LabelManager:
    def __init__(
        self,
        signs_dir: Path,
        reference_dir: Path,
        filter_file: Path,
        output_file: Path,
        frequency_file: Path = None
    ):
        self.signs_dir = signs_dir
        self.reference_dir = reference_dir
        self.filter_file = filter_file
        self.output_file = output_file
        self.frequency_file = frequency_file

        self.valid_images: list[Path] = []
        self.reference_signs: list[dict] = []
        self.labels: dict[str, str] = {}  # filename -> sign_name
        self.image_sources: dict[str, dict] = {}  # filename -> {source_video, clip}
        self.current_index = 0
        self.history: list[str] = []
        self.sign_frequency: dict[str, int] = {}  # sign_name -> count

        self._load_frequency()
        self._load_reference_signs()
        self._load_valid_images()
        self._load_labels()

    def _load_frequency(self):
        """加载历史标注频率"""
        if self.frequency_file and self.frequency_file.exists():
            try:
                with open(self.frequency_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.sign_frequency = data.get("label_statistics", {})
                    print(f"加载历史频率: {len(self.sign_frequency)} 种标志")
            except Exception as e:
                print(f"警告: 无法加载频率文件: {e}")

    def _load_reference_signs(self):
        """加载 188 种标准标志，按历史频率排序，开头添加特殊筛选选项"""
        sign_files = list(self.reference_dir.glob("*.png"))

        signs_data = []
        for f in sign_files:
            name = f.stem
            display_name = name.replace("_", " ")
            freq = self.sign_frequency.get(name, 0)
            signs_data.append({
                "name": name,
                "display_name": display_name,
                "path": str(f.relative_to(self.reference_dir.parent.parent)),
                "frequency": freq,
                "is_special": False
            })

        # 按频率降序排序，频率为 0 的按字母排序
        sorted_signs = sorted(
            signs_data,
            key=lambda x: (-x["frequency"], x["name"])
        )

        # 在开头插入特殊选项（用于筛选）
        # 注意：frequency 使用大数字而非 inf，因为 JSON 不支持 infinity
        special_options = [
            {
                "name": "lowlight",
                "display_name": "低光照 Lowlight",
                "path": None,
                "frequency": 999999,
                "is_special": True,
                "icon": "🌙"  # 月亮表示夜间/低光照
            },
            {
                "name": "blur",
                "display_name": "模糊 Blur",
                "path": None,
                "frequency": 999999,
                "is_special": True,
                "icon": "💨"  # 表示运动模糊
            },
            {
                "name": "glare",
                "display_name": "眩光 Glare",
                "path": None,
                "frequency": 999999,
                "is_special": True,
                "icon": "☀️"  # 太阳表示眩光
            },
            {
                "name": "not_sign",
                "display_name": "非交通标志",
                "path": None,
                "frequency": 999999,
                "is_special": True,
                "icon": "❌"  # 叉号表示无效
            },
            {
                "name": "other",
                "display_name": "未包含在图例中",
                "path": None,
                "frequency": 999999,
                "is_special": True,
                "icon": "❓"  # 问号表示未知标志
            }
        ]

        self.reference_signs = special_options + sorted_signs

        # 统计有历史记录的标志数
        with_freq = sum(1 for s in sorted_signs if s["frequency"] > 0)
        print(f"加载 {len(sorted_signs)} 种标准标志 ({with_freq} 种有历史记录)")
        print(f"  + 5 个特殊筛选选项: 低光照, 模糊, 眩光, 非交通标志, 未包含在图例中")
        if with_freq > 0:
            top3 = [f"{s['name']}({s['frequency']})" for s in sorted_signs[:3]]
            print(f"  TOP 3: {', '.join(top3)}")

    def _load_valid_images(self):
        """加载筛选结果中的 valid 图片"""
        if self.filter_file is None or not self.filter_file.exists():
            if self.filter_file is not None:
                print(f"警告: 筛选结果文件不存在 {self.filter_file}")
            print("将加载所有图片（实时模式）")
            self.valid_images = sorted(self.signs_dir.rglob("*.png"))
        else:
            with open(self.filter_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            results = data.get("results", {})
            valid_names = {k for k, v in results.items() if v == "valid"}

            # 找到对应的图片文件
            all_images = list(self.signs_dir.rglob("*.png"))
            self.valid_images = sorted(
                [img for img in all_images if img.name in valid_names],
                key=lambda p: p.name
            )
            print(f"筛选结果中有 {len(valid_names)} 个 valid 标记")

        # 解析图片来源信息
        for img in self.valid_images:
            self._parse_image_source(img)

        print(f"待标注图片: {len(self.valid_images)} 张")

    def refresh_images(self) -> dict:
        """刷新图片列表，返回新增图片数量（用于实时标注模式）"""
        old_count = len(self.valid_images)
        old_names = {img.name for img in self.valid_images}

        # 重新扫描目录
        if self.filter_file is None or not self.filter_file.exists():
            all_images = sorted(self.signs_dir.rglob("*.png"))
        else:
            with open(self.filter_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            results = data.get("results", {})
            valid_names = {k for k, v in results.items() if v == "valid"}
            all_images = sorted(
                [img for img in self.signs_dir.rglob("*.png") if img.name in valid_names],
                key=lambda p: p.name
            )

        # 找出新增的图片
        new_images = [img for img in all_images if img.name not in old_names]

        if new_images:
            # 添加新图片到列表
            self.valid_images = sorted(list(self.valid_images) + new_images, key=lambda p: p.name)
            # 解析新图片的来源信息
            for img in new_images:
                self._parse_image_source(img)

        return {
            "old_count": old_count,
            "new_count": len(self.valid_images),
            "added": len(new_images),
            "new_images": [img.name for img in new_images[:10]]  # 只返回前10个新图片名
        }

    def _parse_image_source(self, img_path: Path):
        """解析图片的来源视频和片段信息

        路径格式: DJI/DJI_20250430193505_0014_D/DJI_0014_4K_005/f00022_sign3_xxx.png
        - source_video: DJI_0014 (从 DJI_20250430193505_0014_D 提取)
        - clip: DJI_0014_4K_005
        """
        try:
            rel_path = img_path.relative_to(self.signs_dir)
            parts = rel_path.parts

            if len(parts) >= 3:
                source_dir = parts[0]  # DJI_20250430193505_0014_D
                clip = parts[1]        # DJI_0014_4K_005

                # 从目录名提取源视频编号 (例如从 DJI_20250430193505_0014_D 提取 DJI_0014)
                source_parts = source_dir.split('_')
                if len(source_parts) >= 3:
                    source_video = f"DJI_{source_parts[-2]}"  # DJI_0014
                else:
                    source_video = source_dir

                self.image_sources[img_path.name] = {
                    "source_video": source_video,
                    "source_dir": source_dir,
                    "clip": clip
                }
        except Exception:
            pass

    def _load_labels(self):
        """加载已有的标注结果"""
        if self.output_file.exists():
            with open(self.output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.labels = data.get("labels", {})
                print(f"已加载 {len(self.labels)} 条标注记录")

        # 跳到第一个未标注的图片
        for i, img in enumerate(self.valid_images):
            if img.name not in self.labels:
                self.current_index = i
                break

    def save_labels(self):
        """保存标注结果，包含来源信息和统计"""
        # 计算标签统计
        label_counts = {}
        for label in self.labels.values():
            label_counts[label] = label_counts.get(label, 0) + 1

        # 按数量降序排序
        sorted_counts = dict(sorted(label_counts.items(), key=lambda x: -x[1]))

        # 计算视频来源统计
        video_counts = {}
        clip_counts = {}
        for filename in self.labels:
            source = self.image_sources.get(filename, {})
            video = source.get("source_video", "unknown")
            clip = source.get("clip", "unknown")
            video_counts[video] = video_counts.get(video, 0) + 1
            clip_counts[clip] = clip_counts.get(clip, 0) + 1

        # 构建详细标注数据（包含来源信息）
        detailed_labels = {}
        for filename, label in self.labels.items():
            source = self.image_sources.get(filename, {})
            detailed_labels[filename] = {
                "label": label,
                "source_video": source.get("source_video", "unknown"),
                "clip": source.get("clip", "unknown")
            }

        data = {
            "updated_at": datetime.now().isoformat(),
            "total_images": len(self.valid_images),
            "labeled_count": len(self.labels),
            "label_statistics": sorted_counts,
            "video_statistics": dict(sorted(video_counts.items())),
            "clip_statistics": dict(sorted(clip_counts.items())),
            "labels": self.labels,  # 简单格式：filename -> label
            "detailed_labels": detailed_labels  # 详细格式：包含来源
        }
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(self.labels)} 条标注到 {self.output_file}")

    def get_current_image(self) -> Optional[Path]:
        """获取当前图片"""
        if 0 <= self.current_index < len(self.valid_images):
            return self.valid_images[self.current_index]
        return None

    def label_current(self, sign_name: str) -> bool:
        """标注当前图片"""
        img = self.get_current_image()
        if img is None:
            return False

        self.labels[img.name] = sign_name
        self.history.append(img.name)

        # 前进到下一个未标注的
        self._move_to_next_unlabeled()
        return True

    def _move_to_next_unlabeled(self):
        """移动到下一个未标注的图片"""
        start = self.current_index + 1
        for i in range(start, len(self.valid_images)):
            if self.valid_images[i].name not in self.labels:
                self.current_index = i
                return
        for i in range(0, start):
            if self.valid_images[i].name not in self.labels:
                self.current_index = i
                return
        self.current_index = len(self.valid_images)

    def undo(self) -> bool:
        """撤销上一个标注"""
        if not self.history:
            return False

        last_name = self.history.pop()
        if last_name in self.labels:
            del self.labels[last_name]

        for i, img in enumerate(self.valid_images):
            if img.name == last_name:
                self.current_index = i
                break

        return True

    def go_prev(self):
        if self.current_index > 0:
            self.current_index -= 1

    def go_next(self):
        if self.current_index < len(self.valid_images) - 1:
            self.current_index += 1

    def get_stats(self) -> dict:
        # 计算标签统计
        label_counts = {}
        for label in self.labels.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        sorted_counts = dict(sorted(label_counts.items(), key=lambda x: -x[1]))

        return {
            "total": len(self.valid_images),
            "labeled": len(self.labels),
            "remaining": len(self.valid_images) - len(self.labels),
            "current_index": self.current_index,
            "label_counts": sorted_counts,
            "unique_labels": len(label_counts),
        }

    def get_image_source(self, filename: str) -> dict:
        """获取图片的来源信息"""
        return self.image_sources.get(filename, {})


# ============ Flask App ============

app = Flask(__name__)
manager: Optional[LabelManager] = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>香港交通标志标注</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #ffffff;
            color: #1a1a1a;
            height: 100vh;
            overflow: hidden;
        }
        .header {
            padding: 12px 24px;
            background: #f8f9fa;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #e0e0e0;
        }
        .header h1 { font-size: 1.3rem; font-weight: 600; color: #1a1a1a; }
        .header h1 span { color: #0066cc; }
        .stats { display: flex; gap: 20px; font-size: 0.9rem; color: #666; }
        .stat-highlight { color: #0066cc; font-weight: 600; }
        .stat-new { color: #28a745; font-weight: 600; animation: pulse 1s ease-in-out; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .live-indicator {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            background: rgba(40,167,69,0.1);
            border-radius: 12px;
            font-size: 0.8rem;
            color: #28a745;
        }
        .live-dot {
            width: 8px;
            height: 8px;
            background: #28a745;
            border-radius: 50%;
            animation: blink 1.5s ease-in-out infinite;
        }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        .main {
            display: flex;
            height: calc(100vh - 90px);
        }

        /* 左侧：待标注图片 */
        .left-panel {
            width: 40%;
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
            border-right: 1px solid #e0e0e0;
            background: #fafafa;
        }
        .current-image-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f0f0f0;
            border-radius: 12px;
            width: 100%;
            margin-bottom: 16px;
            border: 1px solid #e0e0e0;
        }
        .current-image-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 8px;
        }
        .image-info {
            color: #666;
            font-size: 0.9rem;
            text-align: center;
            margin-bottom: 12px;
        }
        .source-info {
            color: #888;
            font-size: 0.8rem;
            text-align: center;
            margin-bottom: 8px;
        }
        .source-info span {
            background: rgba(0,102,204,0.1);
            padding: 2px 8px;
            border-radius: 4px;
            margin: 0 4px;
        }
        .current-label-display {
            padding: 8px 16px;
            background: #28a745;
            color: #fff;
            border-radius: 8px;
            font-size: 0.85rem;
            max-width: 100%;
            word-break: break-all;
            text-align: center;
        }
        .current-label-display.special-label {
            background: #dc3545;
            color: #fff;
        }
        .nav-buttons {
            display: flex;
            gap: 10px;
            margin-top: 16px;
            flex-wrap: wrap;
            justify-content: center;
        }
        .btn {
            padding: 10px 18px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .btn:active { transform: translateY(0); }
        .btn-nav {
            background: #f0f0f0;
            border: 1px solid #d0d0d0;
            color: #333;
        }
        .btn-nav:hover {
            background: #e8e8e8;
            border-color: #0066cc;
            color: #0066cc;
        }
        .btn-save {
            background: #0066cc;
            color: #fff;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(0,102,204,0.3);
        }
        .btn-save:hover {
            background: #0052a3;
            box-shadow: 0 4px 16px rgba(0,102,204,0.4);
        }
        .btn-unknown {
            background: #fd7e14;
            color: #fff;
            font-weight: 500;
            box-shadow: 0 2px 8px rgba(253,126,20,0.3);
        }
        .btn-unknown:hover {
            background: #e06c0a;
            box-shadow: 0 4px 16px rgba(253,126,20,0.4);
        }
        .btn-stats {
            background: rgba(40,167,69,0.1);
            border: 1px solid rgba(40,167,69,0.3);
            color: #28a745;
        }
        .btn-stats:hover {
            background: rgba(40,167,69,0.2);
            border-color: rgba(40,167,69,0.5);
        }

        /* 右侧：参考标志 */
        .right-panel {
            width: 60%;
            display: flex;
            flex-direction: column;
            background: #ffffff;
        }
        .search-bar {
            padding: 16px 24px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
        }
        .search-input {
            width: 100%;
            padding: 12px 16px;
            background: #ffffff;
            border: 1px solid #d0d0d0;
            border-radius: 8px;
            color: #1a1a1a;
            font-size: 1rem;
        }
        .search-input:focus {
            outline: none;
            border-color: #0066cc;
        }
        .search-input::placeholder { color: #999; }

        .reference-grid {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 12px;
            align-content: start;
            background: #ffffff;
        }
        .reference-item {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 8px;
            cursor: pointer;
            border: 2px solid #e0e0e0;
            transition: border-color 0.2s, transform 0.1s;
        }
        .reference-item:hover {
            border-color: #0066cc;
            transform: translateY(-2px);
        }
        .reference-item.selected {
            border-color: #28a745;
        }
        .reference-item img {
            width: 100%;
            aspect-ratio: 1;
            object-fit: contain;
            background: #ffffff;
            border-radius: 4px;
        }
        .reference-item .name {
            margin-top: 6px;
            font-size: 0.7rem;
            color: #666;
            text-align: center;
            line-height: 1.3;
            max-height: 3.9em;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
        }
        .reference-item.special-item {
            background: #fff5f5;
            border: 2px solid #fca5a5;
        }
        .reference-item.special-item:hover {
            border-color: #dc3545;
            background: #fee2e2;
        }
        .special-icon {
            width: 100%;
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 3rem;
            background: #fef2f2;
            border-radius: 4px;
        }

        .shortcuts-hint {
            padding: 8px 24px;
            background: #f8f9fa;
            color: #666;
            font-size: 0.8rem;
            border-top: 1px solid #e0e0e0;
        }
        .shortcuts-hint kbd {
            background: #e9ecef;
            padding: 2px 6px;
            border-radius: 4px;
            margin: 0 2px;
            border: 1px solid #d0d0d0;
        }

        .done-message {
            text-align: center;
            padding: 60px;
        }
        .done-message h2 { color: #28a745; margin-bottom: 16px; }

        .progress-bar {
            height: 3px;
            background: #e0e0e0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #0066cc);
            transition: width 0.3s;
        }
        .config-bar {
            padding: 6px 24px;
            background: #f0f4f8;
            border-bottom: 1px solid #e0e0e0;
            font-size: 0.75rem;
            color: #666;
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
        }
        .config-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .config-label {
            color: #999;
        }
        .config-value {
            color: #333;
            font-family: 'SF Mono', Monaco, monospace;
            background: rgba(0,0,0,0.05);
            padding: 2px 6px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1><span>香港交通标志</span> 标注工具</h1>
        <div class="stats">
            <div class="live-indicator" id="live-indicator" style="display: none;">
                <div class="live-dot"></div>
                <span>实时模式</span>
                <span id="new-count"></span>
            </div>
            <span>来源: <span id="stat-source" style="color: #0066cc;">-</span></span>
            <span>已标注: <span class="stat-highlight" id="stat-labeled">0</span></span>
            <span>剩余: <span id="stat-remaining">0</span></span>
            <span>总计: <span id="stat-total">0</span></span>
        </div>
    </div>
    <div class="progress-bar">
        <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
    </div>
    <div class="config-bar">
        <div class="config-item">
            <span class="config-label">图片目录:</span>
            <span class="config-value" id="config-signs-dir" title="">-</span>
        </div>
        <div class="config-item">
            <span class="config-label">标注输出:</span>
            <span class="config-value" id="config-output-file" title="">-</span>
        </div>
    </div>

    <div class="main">
        <div class="left-panel">
            <div class="current-image-container">
                <img id="current-image" src="" alt="待标注图片">
            </div>
            <div class="image-info">
                <span id="image-name">加载中...</span>
                <span id="image-index"></span>
            </div>
            <div class="source-info" id="source-info"></div>
            <div id="current-label-container"></div>
            <div class="nav-buttons">
                <button class="btn btn-nav" onclick="navigate('prev')">◀ 上一张</button>
                <button class="btn btn-nav" onclick="undo()">↶ 撤销</button>
                <button class="btn btn-nav" onclick="navigate('next')">下一张 ▶</button>
                <button class="btn btn-save" onclick="saveAndQuit()">💾 保存</button>
            </div>
        </div>

        <div class="right-panel">
            <div class="search-bar">
                <input type="text" class="search-input" id="search-input"
                       placeholder="搜索标志名称... (按 / 聚焦, Esc 清空)">
            </div>
            <div class="reference-grid" id="reference-grid">
                <!-- 参考标志将在这里动态生成 -->
            </div>
            <div class="shortcuts-hint">
                快捷键: <kbd>/</kbd> 搜索 | <kbd>Esc</kbd> 清空 | <kbd>←</kbd><kbd>→</kbd> 导航 | <kbd>Z</kbd> 撤销 | <kbd>R</kbd> 刷新 | <kbd>Q</kbd> 保存
            </div>
        </div>
    </div>

    <script>
        let currentData = null;
        let referenceSignsData = [];

        async function loadReferenceSigns() {
            const res = await fetch('/api/references');
            const data = await res.json();
            referenceSignsData = data.references;
            renderReferences(referenceSignsData);
        }

        function renderReferences(signs) {
            const grid = document.getElementById('reference-grid');
            grid.innerHTML = signs.map(sign => {
                if (sign.is_special) {
                    // 特殊选项使用图标
                    return `
                        <div class="reference-item special-item" data-name="${sign.name}" onclick="labelAs('${sign.name}')">
                            <div class="special-icon">${sign.icon}</div>
                            <div class="name">${sign.display_name}</div>
                        </div>
                    `;
                } else {
                    // 普通标志使用图片
                    return `
                        <div class="reference-item" data-name="${sign.name}" onclick="labelAs('${sign.name}')">
                            <img src="/api/reference-image/${sign.path}" alt="${sign.display_name}">
                            <div class="name">${sign.display_name}</div>
                        </div>
                    `;
                }
            }).join('');
        }

        async function loadCurrent() {
            const res = await fetch('/api/current');
            const data = await res.json();
            currentData = data;

            if (data.done) {
                document.querySelector('.left-panel').innerHTML = `
                    <div class="done-message">
                        <h2>🎉 全部标注完成！</h2>
                        <p>共标注 ${data.stats.labeled} 张图片</p>
                        <p>覆盖 ${data.stats.unique_labels} 种标签</p>
                    </div>
                `;
                return;
            }

            document.getElementById('current-image').src = '/api/image/' + data.image_path;
            document.getElementById('image-name').textContent = data.image_name;
            document.getElementById('image-index').textContent = ` (${data.stats.current_index + 1}/${data.stats.total})`;

            // 显示来源信息
            const sourceInfo = document.getElementById('source-info');
            if (data.source_video || data.clip) {
                sourceInfo.innerHTML = `来源: <span>${data.source_video}</span> / <span>${data.clip}</span>`;
            } else {
                sourceInfo.innerHTML = '';
            }

            const labelContainer = document.getElementById('current-label-container');
            if (data.current_label) {
                // 处理特殊标签的显示名称
                let displayName;
                const specialLabels = ['lowlight', 'blur', 'glare', 'not_sign', 'other'];
                if (data.current_label === 'lowlight') {
                    displayName = '低光照 Lowlight';
                } else if (data.current_label === 'blur') {
                    displayName = '模糊 Blur';
                } else if (data.current_label === 'glare') {
                    displayName = '眩光 Glare';
                } else if (data.current_label === 'not_sign') {
                    displayName = '非交通标志';
                } else if (data.current_label === 'other') {
                    displayName = '未包含在图例中';
                } else if (data.current_label === 'unclear') {
                    // 兼容旧数据
                    displayName = '图片不清晰 (旧标签)';
                } else {
                    displayName = data.current_label.replace(/_/g, ' ');
                }

                // 特殊标签使用不同的样式
                const isSpecial = specialLabels.includes(data.current_label);
                const labelClass = isSpecial ? 'current-label-display special-label' : 'current-label-display';
                labelContainer.innerHTML = `<div class="${labelClass}">${displayName}</div>`;
                // 高亮对应的参考图
                document.querySelectorAll('.reference-item').forEach(item => {
                    item.classList.toggle('selected', item.dataset.name === data.current_label);
                });
            } else {
                labelContainer.innerHTML = '';
                document.querySelectorAll('.reference-item').forEach(item => {
                    item.classList.remove('selected');
                });
            }

            updateStats(data.stats);
        }

        function updateStats(stats) {
            document.getElementById('stat-labeled').textContent = stats.labeled;
            document.getElementById('stat-remaining').textContent = stats.remaining;
            document.getElementById('stat-total').textContent = stats.total;
            const progress = stats.total > 0 ? (stats.labeled / stats.total) * 100 : 0;
            document.getElementById('progress-fill').style.width = progress + '%';
        }

        async function labelAs(signName) {
            await fetch('/api/label', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({sign_name: signName})
            });
            loadCurrent();
        }

        async function navigate(direction) {
            await fetch('/api/navigate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({direction})
            });
            loadCurrent();
        }

        async function undo() {
            await fetch('/api/undo', {method: 'POST'});
            loadCurrent();
        }

        async function saveAndQuit() {
            await fetch('/api/save', {method: 'POST'});
            alert('已保存！');
        }

        async function showStatistics() {
            const res = await fetch('/api/statistics');
            const stats = await res.json();

            let labelList = Object.entries(stats.label_counts)
                .map(([label, count]) => `${label.replace(/_/g, ' ')}: ${count}`)
                .join('\\n');

            let videoList = Object.entries(stats.video_counts)
                .map(([video, count]) => `${video}: ${count}`)
                .join('\\n');

            const msg = `=== 标注统计 ===
总数: ${stats.total}
已标注: ${stats.labeled}
剩余: ${stats.remaining}
标签种类: ${stats.unique_labels}

=== 各标签数量 ===
${labelList}

=== 各视频来源 ===
${videoList}`;

            alert(msg);
        }

        // 搜索功能
        const searchInput = document.getElementById('search-input');
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            if (!query) {
                renderReferences(referenceSignsData);
                return;
            }
            const filtered = referenceSignsData.filter(sign =>
                sign.display_name.toLowerCase().includes(query) ||
                sign.name.toLowerCase().includes(query)
            );
            renderReferences(filtered);
        });

        // 键盘快捷键
        document.addEventListener('keydown', (e) => {
            // 如果在搜索框中，只处理 Esc
            if (e.target === searchInput) {
                if (e.key === 'Escape') {
                    searchInput.value = '';
                    searchInput.blur();
                    renderReferences(referenceSignsData);
                }
                return;
            }

            switch(e.key) {
                case '/':
                    e.preventDefault();
                    searchInput.focus();
                    break;
                case 'z':
                case 'Z':
                    undo();
                    break;
                case 'q':
                case 'Q':
                    saveAndQuit();
                    break;
                case 'ArrowLeft':
                    navigate('prev');
                    break;
                case 'ArrowRight':
                    navigate('next');
                    break;
                case 'Escape':
                    searchInput.value = '';
                    renderReferences(referenceSignsData);
                    break;
            }
        });

        // 加载配置信息
        async function loadConfig() {
            const res = await fetch('/api/config');
            const config = await res.json();

            const signsDir = document.getElementById('config-signs-dir');
            const outputFile = document.getElementById('config-output-file');
            const statSource = document.getElementById('stat-source');

            signsDir.textContent = config.signs_dir;
            signsDir.title = config.signs_dir;

            outputFile.textContent = config.output_file;
            outputFile.title = config.output_file;

            // 从 signs_dir 提取数据来源名称（最后一个目录名）
            const sourceName = config.signs_dir.split('/').filter(p => p).pop() || '-';
            statSource.textContent = sourceName;
            statSource.title = config.signs_dir;
        }

        // 初始加载
        loadConfig();
        loadReferenceSigns();
        loadCurrent();

        // 实时刷新功能 - 每 10 秒检查新图片
        let liveMode = false;
        let lastTotal = 0;

        async function refreshImages() {
            try {
                const res = await fetch('/api/refresh', {method: 'POST'});
                const data = await res.json();

                if (data.added > 0) {
                    // 有新图片，显示提示
                    const newCountEl = document.getElementById('new-count');
                    newCountEl.textContent = `+${data.added}`;
                    newCountEl.classList.add('stat-new');
                    setTimeout(() => newCountEl.classList.remove('stat-new'), 2000);

                    // 更新统计
                    updateStats(data.stats);
                    console.log(`发现 ${data.added} 张新图片，总计: ${data.new_count}`);
                }

                lastTotal = data.new_count;
            } catch (e) {
                console.error('刷新失败:', e);
            }
        }

        function enableLiveMode() {
            liveMode = true;
            document.getElementById('live-indicator').style.display = 'inline-flex';
            // 立即刷新一次
            refreshImages();
            // 设置定时刷新
            setInterval(refreshImages, 10000);
            console.log('实时模式已启用，每 10 秒刷新');
        }

        // 检查是否需要启用实时模式（总数为 0 或手动触发）
        setTimeout(() => {
            if (currentData && currentData.stats.total === 0) {
                enableLiveMode();
            }
        }, 1000);

        // 手动刷新快捷键 R
        document.addEventListener('keydown', (e) => {
            if (e.target === searchInput) return;
            if (e.key === 'r' || e.key === 'R') {
                if (!liveMode) enableLiveMode();
                refreshImages();
            }
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/references")
def get_references():
    return jsonify({"references": manager.reference_signs})


@app.route("/api/config")
def get_config():
    """返回配置信息：数据来源目录和输出文件"""
    return jsonify({
        "signs_dir": str(manager.signs_dir),
        "output_file": str(manager.output_file),
        "reference_dir": str(manager.reference_dir),
    })


@app.route("/api/current")
def get_current():
    img = manager.get_current_image()
    stats = manager.get_stats()

    if img is None or manager.current_index >= len(manager.valid_images):
        return jsonify({"done": True, "stats": stats})

    rel_path = img.relative_to(manager.signs_dir)
    source = manager.get_image_source(img.name)

    return jsonify({
        "done": False,
        "image_name": img.name,
        "image_path": str(rel_path),
        "current_label": manager.labels.get(img.name),
        "source_video": source.get("source_video", ""),
        "clip": source.get("clip", ""),
        "stats": stats
    })


@app.route("/api/image/<path:image_path>")
def get_image(image_path):
    full_path = manager.signs_dir / image_path
    if full_path.exists():
        return send_file(full_path)
    return "Not found", 404


@app.route("/api/reference-image/<path:image_path>")
def get_reference_image(image_path):
    full_path = manager.reference_dir.parent.parent / image_path
    if full_path.exists():
        return send_file(full_path)
    return "Not found", 404


@app.route("/api/label", methods=["POST"])
def label_image():
    data = request.json
    sign_name = data.get("sign_name")
    if not sign_name:
        return jsonify({"error": "Missing sign_name"}), 400

    manager.label_current(sign_name)
    return jsonify({"success": True})


@app.route("/api/navigate", methods=["POST"])
def navigate():
    data = request.json
    direction = data.get("direction")
    if direction == "prev":
        manager.go_prev()
    elif direction == "next":
        manager.go_next()
    return jsonify({"success": True})


@app.route("/api/undo", methods=["POST"])
def undo():
    manager.undo()
    return jsonify({"success": True})


@app.route("/api/save", methods=["POST"])
def save():
    manager.save_labels()
    return jsonify({"success": True})


@app.route("/api/statistics")
def get_statistics():
    """返回标签统计信息"""
    stats = manager.get_stats()

    # 计算视频来源统计
    video_counts = {}
    for filename in manager.labels:
        source = manager.image_sources.get(filename, {})
        video = source.get("source_video", "unknown")
        video_counts[video] = video_counts.get(video, 0) + 1

    return jsonify({
        "total": stats["total"],
        "labeled": stats["labeled"],
        "remaining": stats["remaining"],
        "unique_labels": stats["unique_labels"],
        "label_counts": stats["label_counts"],
        "video_counts": dict(sorted(video_counts.items()))
    })


@app.route("/api/refresh", methods=["POST"])
def refresh_images():
    """刷新图片列表，用于实时标注模式"""
    result = manager.refresh_images()
    stats = manager.get_stats()
    result["stats"] = stats
    return jsonify(result)


# ============ 主函数 ============

def main():
    global manager

    import argparse
    parser = argparse.ArgumentParser(description="交通标志标注 UI")
    parser.add_argument("--port", type=int, default=8082, help="端口号")
    parser.add_argument("--batch", type=str, default=DEFAULT_BATCH,
                        choices=list(BATCH_CONFIG.keys()),
                        help=f"选择批次 (默认: {DEFAULT_BATCH})")
    parser.add_argument("--signs-dir", type=str, default=None, help="待标注图片目录 (覆盖批次配置)")
    parser.add_argument("--reference-dir", type=str, default=str(REFERENCE_SIGNS_DIR), help="参考标志目录")
    parser.add_argument("--filter-file", type=str, default=None, help="筛选结果文件 (覆盖批次配置)")
    parser.add_argument("--output", type=str, default=None, help="标注输出文件 (覆盖批次配置)")
    parser.add_argument("--frequency-file", type=str, default=str(FREQUENCY_FILE), help="历史频率文件 (用于排序)")
    args = parser.parse_args()

    # 使用批次配置或自定义路径
    batch_config = BATCH_CONFIG[args.batch]
    signs_dir = Path(args.signs_dir) if args.signs_dir else batch_config["signs_dir"]
    # filter_file 可以为 None（实时模式）
    if args.filter_file:
        filter_file = Path(args.filter_file)
    else:
        filter_file = batch_config["filter_file"]
        if filter_file is not None:
            filter_file = Path(filter_file) if isinstance(filter_file, str) else filter_file
    output_file = Path(args.output) if args.output else batch_config["output_file"]
    reference_dir = Path(args.reference_dir)
    frequency_file = Path(args.frequency_file) if args.frequency_file else None

    if not signs_dir.exists():
        # 对于实��模式，目录可能还不存在，创建它
        signs_dir.mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {signs_dir}")

    if not reference_dir.exists():
        print(f"错误: 参考标志目录不存在 {reference_dir}")
        sys.exit(1)

    manager = LabelManager(signs_dir, reference_dir, filter_file, output_file, frequency_file)

    # 判断是否为实时模式
    is_live_mode = filter_file is None
    mode_str = "实时模式 (边提取边标注)" if is_live_mode else "标准模式"

    print(f"\n{'='*50}")
    print(f"交通标志标注 UI - {args.batch.upper()}")
    print(f"模式: {mode_str}")
    print(f"{'='*50}")
    print(f"待标注图片: {signs_dir}")
    print(f"参考标志: {reference_dir}")
    print(f"筛选结果: {filter_file or '无 (直接标注所有图片)'}")
    print(f"标注输出: {output_file}")
    print(f"待标注数量: {len(manager.valid_images)}")
    print(f"已标注数量: {len(manager.labels)}")
    print(f"参考标志数: {len(manager.reference_signs)}")
    if is_live_mode:
        print(f"\n提示: 实时模式下，UI 会每 10 秒自动刷新检测新图片")
        print(f"      也可以按 R 键手动刷新")
    print(f"{'='*50}")
    print(f"\n打开浏览器访问: http://localhost:{args.port}\n")

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
