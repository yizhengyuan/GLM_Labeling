#!/usr/bin/env python3
"""
创建数据集
- valid 图片：使用标注工具的细粒度标签
- unclear 图片：标注为 unclear（图片太模糊）
- not_sign 图片：标注为 not_sign（非交通标志）

筛选工具 (sign_filter_ui.py) 与标注工具 (sign_labeling_ui.py) 标签映射：
- valid -> 具体标签名（如 Direction_sign）
- unclear -> unclear
- not_sign -> not_sign
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

# 路径配置
BASE_DIR = Path(__file__).parent.parent
EXTRACTED_SIGNS_DIR = BASE_DIR / "extracted_signs"
DJI_SIGNS_DIR = EXTRACTED_SIGNS_DIR / "DJI"
FILTER_RESULTS_FILE = EXTRACTED_SIGNS_DIR / "filter_results.json"
LABEL_RESULTS_FILE = EXTRACTED_SIGNS_DIR / "label_results.json"
OUTPUT_DIR = EXTRACTED_SIGNS_DIR / "labeled_data"

def main():
    # 加载 filter 结果
    with open(FILTER_RESULTS_FILE, "r", encoding="utf-8") as f:
        filter_data = json.load(f)

    # 加载 label 结果
    with open(LABEL_RESULTS_FILE, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    # 分离 valid, unclear, not_sign 图片
    valid_images = []
    unclear_images = []
    not_sign_images = []

    for filename, status in filter_data["results"].items():
        if status == "valid":
            valid_images.append(filename)
        elif status == "unclear":
            unclear_images.append(filename)
        elif status == "not_sign":
            not_sign_images.append(filename)

    print(f"Valid 图片: {len(valid_images)} 张")
    print(f"Unclear 图片: {len(unclear_images)} 张")
    print(f"Not Sign 图片: {len(not_sign_images)} 张")

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 查找所有图片文件
    all_image_files = {f.name: f for f in DJI_SIGNS_DIR.rglob("*.png")}
    print(f"DJI 目录中共有 {len(all_image_files)} 张图片")

    # 复制 valid 图片
    copied_valid = 0
    for filename in valid_images:
        if filename in all_image_files:
            src = all_image_files[filename]
            dst = OUTPUT_DIR / filename
            shutil.copy2(src, dst)
            copied_valid += 1
    print(f"已复制 {copied_valid} 张 valid 图片")

    # 复制 unclear 图片
    copied_unclear = 0
    for filename in unclear_images:
        if filename in all_image_files:
            src = all_image_files[filename]
            dst = OUTPUT_DIR / filename
            shutil.copy2(src, dst)
            copied_unclear += 1
    print(f"已复制 {copied_unclear} 张 unclear 图片")

    # 复制 not_sign 图片
    copied_not_sign = 0
    for filename in not_sign_images:
        if filename in all_image_files:
            src = all_image_files[filename]
            dst = OUTPUT_DIR / filename
            shutil.copy2(src, dst)
            copied_not_sign += 1
    print(f"已复制 {copied_not_sign} 张 not_sign 图片")

    # 构建新的数据集 JSON
    labels = label_data.get("labels", {})
    detailed_labels = label_data.get("detailed_labels", {})

    # 添加 unclear 图片的标签
    new_labels = dict(labels)
    new_detailed_labels = dict(detailed_labels)

    for filename in unclear_images:
        if filename in all_image_files:
            new_labels[filename] = "unclear"
            new_detailed_labels[filename] = {
                "label": "unclear",
                "source_video": "unknown",
                "clip": "unknown"
            }

    # 添加 not_sign 图片的标签
    for filename in not_sign_images:
        if filename in all_image_files:
            new_labels[filename] = "not_sign"
            new_detailed_labels[filename] = {
                "label": "not_sign",
                "source_video": "unknown",
                "clip": "unknown"
            }

    # 计算统计信息
    label_counts = {}
    for label in new_labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    sorted_counts = dict(sorted(label_counts.items(), key=lambda x: -x[1]))

    # 构建输出数据
    total_count = copied_valid + copied_unclear + copied_not_sign
    dataset = {
        "created_at": datetime.now().isoformat(),
        "description": f"{total_count}张香港交通标志数据集 ({copied_valid} valid + {copied_unclear} unclear + {copied_not_sign} not_sign)",
        "total_images": len(new_labels),
        "label_statistics": sorted_counts,
        "unique_labels": len(label_counts),
        "source_breakdown": {
            "valid_labeled": copied_valid,
            "unclear": copied_unclear,
            "not_sign": copied_not_sign
        },
        "labels": new_labels,
        "detailed_labels": new_detailed_labels
    }

    # 保存数据集 JSON
    output_json = OUTPUT_DIR / "dataset.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"数据集创建完成！")
    print(f"{'='*50}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"总图片数: {len(new_labels)}")
    print(f"标签种类: {len(label_counts)}")
    print(f"数据文件: {output_json}")

if __name__ == "__main__":
    main()
