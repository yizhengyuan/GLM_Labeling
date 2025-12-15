#!/usr/bin/env python3
"""
通用可视化标注结果脚本
支持命令行参数指定图片前缀

用法:
    python3 visualize_universal.py --prefix D2
    python3 visualize_universal.py --prefix D1
"""

import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# 类别颜色定义 (RGB)
CATEGORY_COLORS = {
    "pedestrian": (255, 0, 0),      # 红色 - 行人
    "vehicle": (0, 255, 0),         # 绿色 - 车辆
    "traffic_sign": (0, 100, 255),  # 蓝色 - 交通标志
    "construction": (255, 165, 0),  # 橙色 - 施工标志
    "unknown": (128, 128, 128),     # 灰色 - 未知
}


def visualize_annotations(image_path: str, json_path: str, output_path: str):
    """在图片上绘制标注框"""
    
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    with open(json_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)
    
    for shape in annotation.get("shapes", []):
        label = shape["label"]
        points = shape["points"]
        category = shape.get("flags", {}).get("category", "unknown")
        
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        color = CATEGORY_COLORS.get(category, (128, 128, 128))
        
        # 绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 绘制标签背景
        label_text = label
        text_bbox = draw.textbbox((x1, y1 - 18), label_text)
        draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=color)
        draw.text((x1, y1 - 18), label_text, fill=(255, 255, 255))
    
    img.save(output_path)
    print(f"  💾 Saved: {Path(output_path).name}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Annotations")
    parser.add_argument("--prefix", type=str, required=True, help="Image prefix (e.g., D1, D2)")
    parser.add_argument("--images-dir", type=str, default="test_images/extracted_frames", help="Images directory")
    args = parser.parse_args()
    
    images_dir = Path(args.images_dir)
    annotations_dir = Path(f"output/{args.prefix.lower()}_annotations")
    output_dir = Path(f"output/{args.prefix.lower()}_visualized")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"🎨 Visualize Annotations - {args.prefix} series")
    print("=" * 70)
    
    json_files = list(annotations_dir.glob("*.json"))
    
    if not json_files:
        print(f"❌ No annotation files found in {annotations_dir}")
        return
    
    count = 0
    for json_path in sorted(json_files):
        image_name = json_path.stem + ".jpg"
        image_path = images_dir / image_name
        
        if image_path.exists():
            output_path = output_dir / f"{json_path.stem}_annotated.jpg"
            print(f"\n📷 {image_name}")
            visualize_annotations(str(image_path), str(json_path), str(output_path))
            count += 1
    
    print("\n" + "=" * 70)
    print(f"✅ Done! {count} images visualized")
    print(f"📁 Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
