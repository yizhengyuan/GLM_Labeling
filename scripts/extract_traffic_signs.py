#!/usr/bin/env python3
"""
🎯 交通标志提取脚本

使用 VLM 目标检测，从一批图片中提取所有交通标志，
裁剪保存为单独的图片，方便手动标注创建训练数据。

用法:
    # 从单个图片提取
    python scripts/extract_traffic_signs.py path/to/image.jpg
    
    # 从目录批量提取
    python scripts/extract_traffic_signs.py path/to/frames/ --output extracted_signs/
    
    # 指定模型
    python scripts/extract_traffic_signs.py path/to/frames/ --model glm
"""

import os
import sys
import argparse
import asyncio
import base64
import io
from pathlib import Path
from typing import List, Tuple
import json

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import httpx

from scripts.video_to_dataset_async import (
    MODEL_CONFIGS, 
    DETECTION_PROMPT,
    get_image_size,
    convert_coords,
    image_to_base64_url,
)
from glm_labeling.utils.labels import get_category, normalize_vehicle_label
from glm_labeling.utils.json_utils import parse_llm_json


# ============================================================================
# 检测函数
# ============================================================================

async def detect_image(image_path: str, api_key: str, model_type: str = "glm") -> List[dict]:
    """
    检测单张图片中的目标（针对 4K 进行优化：缩放检测 + 原始裁剪）
    
    Returns:
        检测结果列表
    """
    width, height = get_image_size(image_path)
    model_config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["glm"])
    
    # ⭐ 针对 4K 进行优化：如果长边超过 2048，缩放后再发给 API，避免 400 Bad Request
    MAX_DETECTION_SIZE = 2048
    if width > MAX_DETECTION_SIZE or height > MAX_DETECTION_SIZE:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((MAX_DETECTION_SIZE, MAX_DETECTION_SIZE))
        # 转换缩放后的图为 Base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        base64_url = f"data:image/jpeg;base64,{img_base64}"
        img_data_for_gemini = buffered.getvalue()
    else:
        # 普通图片直接转换
        base64_url = image_to_base64_url(image_path)
        img_data_for_gemini = None
    
    if model_config.get("type") == "gemini":
        # Gemini API
        from google import genai
        from google.genai import types
        
        client = genai.Client()
        
        if img_data_for_gemini:
            image_data = img_data_for_gemini
            mime_type = "image/jpeg"
        else:
            with open(image_path, "rb") as f:
                image_data = f.read()
            ext = Path(image_path).suffix.lower()
            mime_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
            mime_type = mime_types.get(ext, "image/jpeg")
        
        response = client.models.generate_content(
            model=model_config["name"],
            contents=[
                types.Part.from_bytes(data=image_data, mime_type=mime_type),
                DETECTION_PROMPT
            ]
        )
        content = response.text
    else:
        # GLM API
        async with httpx.AsyncClient(
            base_url=model_config["api_base"],
            timeout=httpx.Timeout(60.0, connect=10.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        ) as client:
            payload = {
                "model": model_config["name"],
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": base64_url}},
                        {"type": "text", "text": DETECTION_PROMPT}
                    ]
                }]
            }
            
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
    
    # 解析结果
    detections = parse_llm_json(content)
    
    if detections is None:
        return []
    
    # 后处理，只保留交通标志
    signs = []
    for det in detections:
        if "label" not in det or "bbox_2d" not in det:
            continue
        
        # ⭐ 即使检测用的是缩略图，只要坐标是归一化的 (0-1000)，
        # 我们用原始尺寸 (width, height) 还原，结果就是准确的 4K 坐标。
        bbox = convert_coords(det["bbox_2d"], width, height)
        label = det["label"].lower().replace(" ", "_").replace("-", "_")
        category = get_category(label)
        
        # 只保留交通标志
        if category == "traffic_sign":
            signs.append({
                "label": label,
                "bbox": bbox,
                "bbox_normalized": det["bbox_2d"]
            })
    
    return signs


def crop_and_save(
    image_path: str, 
    bbox: List[int], 
    output_path: str, 
    padding: int = 10,
    lossless: bool = False
) -> str:
    """
    裁剪并保存交通标志
    
    Args:
        image_path: 原图路径
        bbox: 边界框 [x1, y1, x2, y2]
        output_path: 输出路径
        padding: 边界扩展像素
        lossless: 是否无损保存（PNG格式）
    
    Returns:
        保存的文件路径
    """
    image = Image.open(image_path).convert("RGB")
    
    # 添加 padding
    x1 = max(0, bbox[0] - padding)
    y1 = max(0, bbox[1] - padding)
    x2 = min(image.width, bbox[2] + padding)
    y2 = min(image.height, bbox[3] + padding)
    
    # 裁剪
    cropped = image.crop((x1, y1, x2, y2))
    
    # 保存
    if lossless:
        # PNG 无损保存
        output_path = output_path.replace('.jpg', '.png')
        cropped.save(output_path, format='PNG')
    else:
        # JPEG 高质量保存
        cropped.save(output_path, quality=100, subsampling=0)
    
    return output_path


# ============================================================================
# 批量处理
# ============================================================================

async def process_batch(
    image_paths: List[Path],
    output_dir: Path,
    api_key: str,
    model_type: str,
    padding: int,
    max_concurrent: int,
    lossless: bool,
    metadata: list,
    stats: dict,
    retry_round: int = 0
) -> List[dict]:
    """
    处理一批图片，返回失败的帧列表
    """
    failed_frames = []
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(image_paths)

    async def process_single(image_path: Path, idx: int):
        """处理单张图片"""
        async with semaphore:
            try:
                prefix = f"[重试{retry_round}]" if retry_round > 0 else ""
                print(f"{prefix}[{idx+1}/{total}] 处理: {image_path.name}")

                # 检测
                signs = await detect_image(str(image_path), api_key, model_type)

                if not signs:
                    print(f"   ⚠️ 未检测到交通标志")
                    stats["processed_images"] += 1
                    stats["signs_per_image"].append(0)
                    return

                print(f"   ✅ 检测到 {len(signs)} 个交通标志")

                # 裁剪并保存每个标志
                for sign_idx, sign in enumerate(signs):
                    bbox = sign["bbox"]

                    # 生成输出文件名
                    ext = ".png" if lossless else ".jpg"
                    output_name = f"{image_path.stem}_sign{sign_idx+1}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}{ext}"
                    output_path = output_dir / output_name

                    # 裁剪保存
                    crop_and_save(str(image_path), bbox, str(output_path), padding, lossless)

                    stats["total_signs"] += 1
                    stats["extracted_files"].append(str(output_path))

                    # 记录元数据
                    metadata.append({
                        "source_image": str(image_path),
                        "output_file": output_name,
                        "bbox": bbox,
                        "vlm_label": sign["label"],
                        "manual_label": ""
                    })

                    print(f"      📍 标志 #{sign_idx+1}: {output_name}")

                stats["processed_images"] += 1
                stats["signs_per_image"].append(len(signs))

            except Exception as e:
                print(f"   ❌ 处理失败: {e}")
                stats["failed_images"] += 1
                failed_frames.append({
                    "frame": str(image_path),
                    "error": str(e)
                })

    # 并发处理
    tasks = [process_single(p, i) for i, p in enumerate(image_paths)]
    await asyncio.gather(*tasks)

    return failed_frames


async def process_images(
    input_path: str,
    output_dir: str,
    api_key: str,
    model_type: str = "glm",
    padding: int = 10,
    max_concurrent: int = 5,
    lossless: bool = False,
    max_retries: int = 3,
    retry_delay: int = 10
) -> dict:
    """
    批量处理图片，提取交通标志（带自动重试）

    Args:
        input_path: 输入路径（图片或目录）
        output_dir: 输出目录
        api_key: API Key
        model_type: 模型类型
        padding: 裁剪时的边界扩展
        max_concurrent: 最大并发数
        lossless: 是否无损保存（PNG格式）
        max_retries: 最大重试次数
        retry_delay: 重试前等待秒数

    Returns:
        统计信息
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图片
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = sorted(
            list(input_path.glob("*.jpg")) +
            list(input_path.glob("*.jpeg")) +
            list(input_path.glob("*.png"))
        )

    print(f"\n{'='*60}")
    print(f"🎯 交通标志提取")
    print(f"{'='*60}")
    print(f"   输入: {input_path}")
    print(f"   输出: {output_dir}")
    print(f"   图片数: {len(image_paths)}")
    print(f"   模型: {model_type}")
    print(f"   并发数: {max_concurrent}")
    print(f"   保存格式: {'PNG (无损)' if lossless else 'JPEG (quality=100)'}")
    print(f"   自动重试: 最多 {max_retries} 次")
    print(f"{'='*60}\n")

    # 统计
    stats = {
        "total_images": len(image_paths),
        "processed_images": 0,
        "failed_images": 0,
        "total_signs": 0,
        "signs_per_image": [],
        "extracted_files": []
    }

    # 记录元数据
    metadata = []

    # 第一轮处理
    failed_frames = await process_batch(
        image_paths, output_dir, api_key, model_type,
        padding, max_concurrent, lossless, metadata, stats
    )

    # 自动重试失败的帧
    retry_round = 0
    while failed_frames and retry_round < max_retries:
        retry_round += 1
        print(f"\n{'='*60}")
        print(f"🔄 自动重试 (第 {retry_round}/{max_retries} 轮)")
        print(f"   失败帧数: {len(failed_frames)}")
        print(f"   等待 {retry_delay} 秒后开始...")
        print(f"{'='*60}\n")

        await asyncio.sleep(retry_delay)

        # 重置失败计数（因为要重新统计）
        stats["failed_images"] = 0

        # 获取失败帧的路径
        retry_paths = [Path(f["frame"]) for f in failed_frames]

        # 重试
        failed_frames = await process_batch(
            retry_paths, output_dir, api_key, model_type,
            padding, max_concurrent, lossless, metadata, stats,
            retry_round=retry_round
        )

    # 保存元数据
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 创建标注模板
    labels_template_path = output_dir / "labels_template.csv"
    with open(labels_template_path, "w", encoding="utf-8") as f:
        f.write("filename,label\n")
        for item in metadata:
            f.write(f"{item['output_file']},\n")

    # 保存最终失败帧列表（如果还有）
    if failed_frames:
        failed_frames_path = output_dir / "failed_frames.json"
        with open(failed_frames_path, "w", encoding="utf-8") as f:
            json.dump(failed_frames, f, indent=2, ensure_ascii=False)

        failed_frames_txt = output_dir / "failed_frames.txt"
        with open(failed_frames_txt, "w", encoding="utf-8") as f:
            for item in failed_frames:
                f.write(Path(item["frame"]).name + "\n")
    else:
        # 如果全部成功，删除之前可能存在的失败列表文件
        failed_frames_path = output_dir / "failed_frames.json"
        failed_frames_txt = output_dir / "failed_frames.txt"
        if failed_frames_path.exists():
            failed_frames_path.unlink()
        if failed_frames_txt.exists():
            failed_frames_txt.unlink()

    # 打印统计
    print(f"\n{'='*60}")
    print(f"📊 提取完成统计")
    print(f"{'='*60}")
    print(f"   处理图片: {stats['processed_images']}/{stats['total_images']}")
    print(f"   失败图片: {len(failed_frames)}")
    print(f"   提取标志: {stats['total_signs']} 个")
    print(f"   重试轮数: {retry_round}")
    if stats['signs_per_image']:
        avg = sum(stats['signs_per_image']) / len(stats['signs_per_image'])
        print(f"   平均每图: {avg:.1f} 个标志")
    print(f"\n📁 输出文件:")
    print(f"   标志图片: {output_dir}/")
    print(f"   元数据:   {metadata_path}")
    print(f"   标注模板: {labels_template_path}")
    if failed_frames:
        print(f"   失败列表: {output_dir}/failed_frames.txt ({len(failed_frames)} 帧)")
    print(f"\n💡 下一步:")
    print(f"   1. 查看提取的标志图片")
    print(f"   2. 编辑 labels_template.csv，填写每张图片的标签")
    print(f"   3. 标签应该使用 raw_data/signs/ 目录下的文件名（不含 .png）")
    print(f"{'='*60}")

    return stats


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="从图片中提取交通标志")
    parser.add_argument("input", type=str, help="输入路径（图片或目录）")
    parser.add_argument("--output", "-o", type=str, default="extracted_signs",
                        help="输出目录 (默认: extracted_signs)")
    parser.add_argument("--model", type=str, default="glm",
                        choices=["glm", "gemini", "gemini-2.5-flash", "gemini-2.0-flash"],
                        help="检测模型 (默认: glm)")
    parser.add_argument("--padding", type=int, default=10,
                        help="裁剪时的边界扩展像素 (默认: 10)")
    parser.add_argument("--concurrent", type=int, default=5,
                        help="最大并发数 (默认: 5)")
    parser.add_argument("--lossless", action="store_true",
                        help="无损保存（PNG格式），默认为JPEG高质量")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="失败帧最大重试次数 (默认: 3)")
    parser.add_argument("--retry-delay", type=int, default=10,
                        help="重试前等待秒数 (默认: 10)")
    args = parser.parse_args()

    # 检查输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 输入路径不存在: {input_path}")
        return

    # 获取 API Key
    model_config = MODEL_CONFIGS.get(args.model, MODEL_CONFIGS["glm"])
    api_key_env = model_config["api_key_env"]
    api_key = os.getenv(api_key_env)

    if not api_key:
        print(f"❌ 请设置 {api_key_env} 环境变量")
        return

    # 运行
    asyncio.run(process_images(
        input_path=args.input,
        output_dir=args.output,
        api_key=api_key,
        model_type=args.model,
        padding=args.padding,
        max_concurrent=args.concurrent,
        lossless=args.lossless,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay
    ))


if __name__ == "__main__":
    main()

