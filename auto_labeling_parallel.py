#!/usr/bin/env python3
"""
并行版自动标注脚本

使用 concurrent.futures 实现多线程并行处理，大幅提升批量标注速度。
原来 100 张图需要 ~15 分钟，现在 ~3 分钟。

用法:
    python3 auto_labeling_parallel.py --prefix D2 --limit 50 --workers 5
"""

import os
import json
import base64
import argparse
import time
import uuid  # 用于生成唯一文件名，避免多线程冲突
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from zai import ZaiClient


# ============================================================================
# 配置
# ============================================================================

SIGNS_DIR = Path("examples/signs/highres/png2560px")

def load_sign_candidates():
    """从标志图片目录动态加载所有标志名称"""
    if not SIGNS_DIR.exists():
        return []
    return [f.stem for f in sorted(SIGNS_DIR.glob("*.png"))]

ALL_SIGN_CANDIDATES = load_sign_candidates()


# ============================================================================
# 辅助函数
# ============================================================================

def get_category(label: str) -> str:
    """根据标签获取粗粒度类别"""
    label_lower = label.lower().replace(" ", "_").replace("-", "_")
    
    if any(p in label_lower for p in ["pedestrian", "person", "people", "child", "cyclist", "crowd"]):
        return "pedestrian"
    if any(v in label_lower for v in ["car", "truck", "bus", "motorcycle", "bicycle", "van", "suv", "taxi", "vehicle"]):
        return "vehicle"
    if any(c in label_lower for c in ["cone", "construction", "barrier", "road_work", "detour"]):
        return "construction"
    if any(s in label_lower for s in ["sign", "speed", "limit", "no_", "traffic", "light", "stop", "give_way", "direction", "exit", "lane", "countdown"]):
        return "traffic_sign"
    return "unknown"


def image_to_base64_url(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    ext = Path(image_path).suffix.lower()
    mime_type = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png'}.get(ext, 'image/jpeg')
    return f"data:{mime_type};base64,{image_data}"


def get_image_size(image_path: str) -> tuple:
    with Image.open(image_path) as img:
        return img.width, img.height


# ============================================================================
# 单张图片处理函数（用于并行）
# ============================================================================

def classify_sign_rag(client, image_path: str, bbox: list) -> str:
    """RAG 二阶段交通标志精排（线程安全版）"""
    import re
    
    temp_path = None  # 确保 finally 块可以访问
    
    try:
        img = Image.open(image_path)
        padding = 10
        x1 = max(0, bbox[0] - padding)
        y1 = max(0, bbox[1] - padding)
        x2 = min(img.width, bbox[2] + padding)
        y2 = min(img.height, bbox[3] + padding)
        
        sign_crop = img.crop((x1, y1, x2, y2))
        # 使用 uuid 生成唯一文件名，避免多线程冲突
        unique_id = uuid.uuid4()
        temp_path = f"/tmp/sign_crop_{unique_id}.jpg"
        sign_crop.save(temp_path, "JPEG")
        
        with open(temp_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
        
        # 阶段1：判断类型
        type_prompt = """请判断这是什么类型的交通标志：
1. 限速标志（红圈白底，中间有数字）
2. 禁止标志（红圈）
3. 警告标志（三角形）
4. 指示/方向标志（蓝色或绿色）
5. 其他

只返回数字（1-5）。"""
        
        response1 = client.chat.completions.create(
            model="glm-4.6v",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}},
                    {"type": "text", "text": type_prompt}
                ]
            }],
            temperature=0.1
        )
        
        type_response = response1.choices[0].message.content.strip()
        type_match = re.search(r'[1-5]', type_response)
        
        if not type_match:
            return "traffic_sign"
        
        sign_type = type_match.group()
        
        # 阶段2：细节识别
        if sign_type == "1":  # 限速
            detail_prompt = "请识别这个限速标志上的数字。只返回数字。"
            response2 = client.chat.completions.create(
                model="glm-4.6v",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}},
                        {"type": "text", "text": detail_prompt}
                    ]
                }],
                temperature=0.1
            )
            numbers = re.findall(r'\d+', response2.choices[0].message.content)
            if numbers:
                return f"Speed_limit_{numbers[0]}_km_h"
            return "Speed_limit"
        
        elif sign_type == "4":  # 方向/指示
            # 检测是否为距离牌
            detail_prompt = """这是一个指示或方向标志。请判断：
1. 方向指示牌
2. 高速公路标志
3. 倒计时距离牌（100m/200m/300m斜条）
4. 其他

只返回数字（1-4）。"""
            response2 = client.chat.completions.create(
                model="glm-4.6v",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}},
                        {"type": "text", "text": detail_prompt}
                    ]
                }],
                temperature=0.1
            )
            detail = re.search(r'[1-4]', response2.choices[0].message.content)
            if detail:
                label_map = {
                    "1": "Direction_sign",
                    "2": "Expressway_sign",
                    "3": "100m_Countdown_markers",
                    "4": "Direction_other"
                }
                return label_map.get(detail.group(), "Direction_sign")
            return "Direction_sign"
        
        type_labels = {
            "2": "Prohibition_sign",
            "3": "Warning_sign",
            "5": "traffic_sign"
        }
        return type_labels.get(sign_type, "traffic_sign")
        
    except Exception as e:
        return "traffic_sign"
    
    finally:
        # ✅ 清理临时文件，防止磁盘空间泄漏
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass  # 忽略删除失败


def process_single_image(args_tuple):
    """
    处理单张图片（线程安全）
    
    Args:
        args_tuple: (image_path, api_key, max_retries, use_rag)
    
    Returns:
        (image_path, detections, error)
    """
    image_path, api_key, max_retries, use_rag = args_tuple
    
    try:
        # 每个线程创建自己的 client
        client = ZaiClient(api_key=api_key)
        
        base64_url = image_to_base64_url(image_path)
        width, height = get_image_size(image_path)
        
        prompt = """请检测图片中的以下4类物体，返回JSON格式。

## 检测类别（使用英文标签）：
1. 行人：pedestrian, cyclist, child, crowd
2. 车辆：car, truck, bus, motorcycle, bicycle, van, taxi（不要标注第一人称）
3. 交通标志：traffic_sign
4. 施工标志：traffic_cone, construction_barrier

## 返回格式：
[{"label": "car", "bbox_2d": [xmin, ymin, xmax, ymax]}]

如果没有目标，返回 []
只返回JSON数组！"""

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="glm-4.6v",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": base64_url}},
                            {"type": "text", "text": prompt}
                        ]
                    }]
                )
                
                result_text = response.choices[0].message.content.strip()
                
                if not result_text:
                    if attempt < max_retries - 1:
                        continue
                    return (image_path, [], None)
                
                # 解析 JSON
                if "```json" in result_text:
                    json_str = result_text.split("```json")[1].split("```")[0].strip()
                elif "```" in result_text:
                    json_str = result_text.split("```")[1].split("```")[0].strip()
                elif "[" in result_text:
                    json_str = result_text[result_text.find("["):result_text.rfind("]")+1]
                else:
                    json_str = result_text.strip()
                
                if json_str == "[]" or json_str == "":
                    return (image_path, [], None)
                
                # 修复截断 JSON
                if json_str and not json_str.endswith("]"):
                    last_complete = json_str.rfind("},")
                    if last_complete > 0:
                        json_str = json_str[:last_complete+1] + "]"
                
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
                    
                    label = det["label"].lower().replace(" ", "_").replace("-", "_")
                    category = get_category(label)
                    
                    # RAG 细粒度分类（仅交通标志）
                    if use_rag and category == "traffic_sign" and label in ["traffic_sign", "sign"]:
                        label = classify_sign_rag(client, image_path, bbox)
                        category = "traffic_sign"
                    
                    processed.append({
                        "label": label,
                        "category": category,
                        "bbox": bbox
                    })
                
                return (image_path, processed, None)
                
            except json.JSONDecodeError:
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))  # 指数退避
                    continue
                return (image_path, [], "JSON parse error")
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))  # 指数退避，避免 429 错误
                    continue
                return (image_path, [], str(e))
        
        return (image_path, [], "Max retries exceeded")
    
    except Exception as e:
        return (image_path, [], str(e))


# ============================================================================
# 输出函数
# ============================================================================

def to_xanylabeling_format(detections: list, image_path: str) -> dict:
    """转换为 X-AnyLabeling 格式"""
    width, height = get_image_size(image_path)
    
    shapes = []
    for det in detections:
        shapes.append({
            "label": det["label"],
            "text": det["label"],
            "points": [
                [det["bbox"][0], det["bbox"][1]],
                [det["bbox"][2], det["bbox"][3]]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {"category": det["category"]}
        })
    
    return {
        "version": "0.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": Path(image_path).name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="并行版自动标注脚本")
    parser.add_argument("--prefix", type=str, required=True, help="图片前缀 (如 D1, D2)")
    parser.add_argument("--limit", type=int, default=None, help="限制处理数量")
    parser.add_argument("--workers", type=int, default=5, help="并行线程数 (默认 5)")
    parser.add_argument("--rag", action="store_true", help="启用 RAG 细粒度分类")
    parser.add_argument("--images-dir", type=str, default="test_images/extracted_frames")
    args = parser.parse_args()
    
    # 初始化
    api_key = os.getenv("ZAI_API_KEY")
    if not api_key:
        print("❌ 请设置 ZAI_API_KEY 环境变量")
        return
    
    # 获取图片列表
    images_dir = Path(args.images_dir)
    image_files = sorted(images_dir.glob(f"{args.prefix}_*.jpg"))
    
    if args.limit:
        image_files = image_files[:args.limit]
    
    if not image_files:
        print(f"❌ 没有找到 {args.prefix} 开头的图片")
        return
    
    # 创建输出目录
    rag_suffix = "_rag" if args.rag else ""
    output_dir = Path(f"output/{args.prefix.lower()}_annotations{rag_suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备参数
    width, height = get_image_size(str(image_files[0]))
    task_args = [(str(img), api_key, 3, args.rag) for img in image_files]
    
    print("=" * 70)
    print(f"🚀 并行自动标注 - {args.prefix} series")
    print(f"   📁 Images: {len(image_files)} | Resolution: {width}x{height}")
    print(f"   🔧 Workers: {args.workers} 个并行线程")
    print(f"   🔍 RAG Mode: {'✅ Enabled' if args.rag else '❌ Disabled'}")
    print("=" * 70)
    
    start_time = time.time()
    
    # 统计
    stats = {"pedestrian": 0, "vehicle": 0, "traffic_sign": 0, "construction": 0}
    success_count = 0
    error_count = 0
    
    # 并行处理
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_image, arg): arg[0] for arg in task_args}
        
        for i, future in enumerate(as_completed(futures)):
            image_path = futures[future]
            image_name = Path(image_path).name
            
            try:
                _, detections, error = future.result()
                
                if error:
                    print(f"  ⚠️ [{i+1}/{len(image_files)}] {image_name}: {error}")
                    error_count += 1
                else:
                    # 统计
                    for det in detections:
                        stats[det["category"]] = stats.get(det["category"], 0) + 1
                    
                    # 保存
                    annotation = to_xanylabeling_format(detections, image_path)
                    output_path = output_dir / f"{Path(image_path).stem}.json"
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(annotation, f, ensure_ascii=False, indent=2)
                    
                    emoji = "✅" if detections else "⚪"
                    print(f"  {emoji} [{i+1}/{len(image_files)}] {image_name}: {len(detections)} objects")
                    success_count += 1
                    
            except Exception as e:
                print(f"  ❌ [{i+1}/{len(image_files)}] {image_name}: {e}")
                error_count += 1
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("📊 Statistics:")
    for cat, count in stats.items():
        print(f"   {cat}: {count}")
    print(f"\n⏱️ Time: {elapsed:.1f}s ({elapsed/len(image_files):.2f}s per image)")
    print(f"✅ Success: {success_count} | ⚠️ Errors: {error_count}")
    print(f"📁 Saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
