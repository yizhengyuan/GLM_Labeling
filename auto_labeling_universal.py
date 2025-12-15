#!/usr/bin/env python3
"""
GLM-4.6V 通用数据标注脚本
支持命令行参数指定图片前缀

用法:
    python3 auto_labeling_universal.py --prefix D2
    python3 auto_labeling_universal.py --prefix D4 --limit 50
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path
from PIL import Image
from zai import ZaiClient

# ============================================================================
# 配置
# ============================================================================
API_KEY = os.getenv("ZAI_API_KEY", "")

# ============================================================================
# 标签体系（与 v2 相同）
# ============================================================================

def get_category(label: str) -> str:
    """根据标签获取粗颗粒度类别"""
    label_lower = label.lower().replace(" ", "_").replace("-", "_")
    
    if any(p in label_lower for p in ["pedestrian", "person", "people", "child", "cyclist", "crowd"]):
        return "pedestrian"
    if any(v in label_lower for v in ["car", "truck", "bus", "motorcycle", "bicycle", "van", "suv", "taxi", "vehicle"]):
        return "vehicle"
    if any(c in label_lower for c in ["cone", "construction", "barrier", "road_work", "detour"]):
        return "construction"
    if any(s in label_lower for s in ["sign", "speed", "limit", "no_", "traffic", "light", "stop", "give_way", "direction", "exit", "lane"]):
        return "traffic_sign"
    return "unknown"


def normalize_label(label: str) -> str:
    """标准化标签为英文格式"""
    label = label.strip().lower()
    
    mapping = {
        "行人": "pedestrian", "人": "pedestrian", "路人": "pedestrian",
        "骑车人": "cyclist", "骑自行车的人": "cyclist",
        "儿童": "child", "小孩": "child",
        "车": "car", "汽车": "car", "轿车": "car", "小汽车": "car",
        "货车": "truck", "卡车": "truck",
        "公交车": "bus", "巴士": "bus",
        "摩托车": "motorcycle", "电动车": "motorcycle",
        "自行车": "bicycle",
        "面包车": "van", "越野车": "suv", "出租车": "taxi",
        "车辆": "car",
        "限速": "speed_limit", "限速牌": "speed_limit",
        "限速70": "speed_limit_70", "限速60": "speed_limit_60",
        "限速80": "speed_limit_80", "限速100": "speed_limit_100",
        "禁止停车": "no_parking", "禁止驶入": "no_entry",
        "红绿灯": "traffic_light", "交通灯": "traffic_light",
        "指示牌": "direction_sign", "路牌": "street_sign",
        "交通标志": "traffic_sign", "标志": "traffic_sign",
        "锥桶": "traffic_cone", "路锥": "traffic_cone",
        "施工": "construction",
    }
    
    if label in mapping:
        return mapping[label]
    for cn, en in mapping.items():
        if cn in label:
            return en
    return label.replace(" ", "_").replace("-", "_")


# ============================================================================
# 工具函数
# ============================================================================

def image_to_base64_url(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    ext = Path(image_path).suffix.lower()
    mime_type = {'jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png'}.get(ext, 'image/jpeg')
    return f"data:{mime_type};base64,{image_data}"


def get_image_size(image_path: str) -> tuple:
    with Image.open(image_path) as img:
        return img.width, img.height


# ============================================================================
# 检测函数
# ============================================================================

def detect_objects(client: ZaiClient, image_path: str, max_retries: int = 3) -> list:
    base64_url = image_to_base64_url(image_path)
    width, height = get_image_size(image_path)
    
    prompt = """请检测图片中的以下4类物体，返回JSON格式。

## 检测类别（使用英文标签）：
1. 行人：pedestrian, cyclist, child
   - 如果行人很多（超过5人），可以用 crowd 标签框住整个人群区域
2. 车辆：car, truck, bus, motorcycle, bicycle, van, taxi（不要标注第一人称摩托车/自行车）
3. 交通标志：speed_limit_30/50/60/70/80, no_entry, no_parking, stop, traffic_light, direction_sign 等
4. 施工标志：traffic_cone, construction_barrier

## 返回格式：
[{"label": "car", "bbox_2d": [xmin, ymin, xmax, ymax]}, {"label": "crowd", "bbox_2d": [x1, y1, x2, y2]}]

如果没有目标，返回 []

重要：只返回JSON数组，不要其他文字！"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="glm-4.6v",
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": base64_url}},
                    {"type": "text", "text": prompt}
                ]}]
            )
            
            result_text = response.choices[0].message.content
            
            # 如果返回为空，重试
            if not result_text or result_text.strip() == "":
                if attempt < max_retries - 1:
                    print(f"  ⚠️ Empty response, retrying ({attempt + 2}/{max_retries})...")
                    continue
                else:
                    print(f"  ⚠️ Empty response after {max_retries} attempts")
                    return []
            
            # 解析 JSON
            if "```json" in result_text:
                json_str = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                json_str = result_text.split("```")[1].split("```")[0].strip()
            elif "[" in result_text:
                json_str = result_text[result_text.find("["):result_text.rfind("]")+1]
            else:
                json_str = result_text.strip()
            
            # 处理空数组情况
            if json_str == "[]" or json_str == "":
                return []
            
            # 修复被截断的 JSON（如果最后一个元素不完整）
            if json_str and not json_str.endswith("]"):
                # 找到最后一个完整的对象
                last_complete = json_str.rfind("},")
                if last_complete > 0:
                    json_str = json_str[:last_complete+1] + "]"
                    print(f"  ⚠️ JSON truncated, recovered {json_str.count('label')} objects")
                else:
                    # 尝试补全
                    json_str = json_str.rstrip(",") + "]"
            
            detections = json.loads(json_str)
            processed = []
            
            for det in detections:
                if "label" not in det or "bbox_2d" not in det:
                    continue
                raw_bbox = det["bbox_2d"]
                bbox = [
                    int(round(raw_bbox[0] / 1000 * width)),
                    int(round(raw_bbox[1] / 1000 * height)),
                    int(round(raw_bbox[2] / 1000 * width)),
                    int(round(raw_bbox[3] / 1000 * height))
                ]
                bbox = [max(0, min(bbox[0], width)), max(0, min(bbox[1], height)),
                        max(0, min(bbox[2], width)), max(0, min(bbox[3], height))]
                
                label = normalize_label(det["label"])
                processed.append({
                    "label": label,
                    "category": get_category(label),
                    "bbox": bbox,
                    "original": det["label"]
                })
            return processed
            
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"  ⚠️ JSON parse error, retrying ({attempt + 2}/{max_retries})...")
                continue
            else:
                print(f"  ⚠️ JSON parse error after {max_retries} attempts: {e}")
                return []
        except Exception as e:
            print(f"  ⚠️ Unexpected error: {e}")
            return []
    
    return []


def to_xanylabeling_format(detections: list, image_path: str) -> dict:
    width, height = get_image_size(image_path)
    shapes = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        shapes.append({
            "label": det["label"],
            "text": det.get("original", ""),
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {"category": det["category"]}
        })
    return {
        "version": "0.4.1", "flags": {}, "shapes": shapes,
        "imagePath": Path(image_path).name, "imageData": None,
        "imageHeight": height, "imageWidth": width
    }


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GLM-4.6V Auto Labeling")
    parser.add_argument("--prefix", type=str, required=True, help="Image prefix (e.g., D1, D2, D4)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all)")
    parser.add_argument("--images-dir", type=str, default="test_images/extracted_frames", help="Images directory")
    args = parser.parse_args()
    
    if not API_KEY:
        print("❌ Please set ZAI_API_KEY environment variable")
        sys.exit(1)
    
    images_dir = Path(args.images_dir)
    pattern = f"{args.prefix}_*.jpg"
    images = sorted(list(images_dir.glob(pattern)))
    
    if args.limit > 0:
        images = images[:args.limit]
    
    if not images:
        print(f"❌ No images found matching pattern: {pattern}")
        sys.exit(1)
    
    # 获取第一张图片的尺寸用于显示
    sample_w, sample_h = get_image_size(str(images[0]))
    
    print("=" * 70)
    print(f"🏷️  GLM-4.6V Auto Labeling - {args.prefix} series")
    print(f"   Images: {len(images)} | Resolution: {sample_w}x{sample_h}")
    print("=" * 70)
    
    client = ZaiClient(api_key=API_KEY)
    
    output_dir = Path(f"output/{args.prefix.lower()}_annotations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {"pedestrian": 0, "vehicle": 0, "traffic_sign": 0, "construction": 0}
    
    for i, img_path in enumerate(images, 1):
        print(f"\n📷 [{i}/{len(images)}] {img_path.name}")
        print("-" * 50)
        
        try:
            detections = detect_objects(client, str(img_path))
            
            print(f"  ✅ Detected {len(detections)} objects:")
            for det in detections:
                cat = det["category"]
                stats[cat] = stats.get(cat, 0) + 1
                emoji = {"pedestrian": "🔴", "vehicle": "🟢", "traffic_sign": "🔵", "construction": "🟠"}.get(cat, "⚪")
                print(f"     {emoji} {det['label']} [{cat}] {det['bbox']}")
            
            annotation = to_xanylabeling_format(detections, str(img_path))
            json_path = output_dir / f"{img_path.stem}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 Statistics:")
    for cat, count in stats.items():
        print(f"   {cat}: {count}")
    print(f"📁 Annotations saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
