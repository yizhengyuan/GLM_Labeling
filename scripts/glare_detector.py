#!/usr/bin/env python3
"""
炫光/反光检测器 - 检测交通标志图片中的过曝问题

用于过滤因强光/反光导致标志不清晰的图片。

用法:
    # 检测单张图片
    python scripts/glare_detector.py --image sign.png

    # 检测目录下所有图片
    python scripts/glare_detector.py --dir extracted_signs/DJI_batch1/

    # 批量检测并移动问题图片
    python scripts/glare_detector.py --dir extracted_signs/ --move-glare glare_images/

    # 调整阈值
    python scripts/glare_detector.py --dir signs/ --threshold 0.2 --brightness 240
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


def detect_glare(image_path: str,
                 brightness_threshold: int = 250,
                 area_threshold: float = 0.15) -> Dict:
    """检测图片是否存在炫光/过曝问题

    Args:
        image_path: 图片路径
        brightness_threshold: 亮度阈值 (0-255)，高于此值视为过曝
        area_threshold: 面积阈值 (0-1)，过曝区域占比超过此值判定为炫光

    Returns:
        检测结果字典
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        return {
            "path": image_path,
            "error": "无法读取图片",
            "has_glare": False
        }

    # 转换到 HSV 空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]  # 亮度通道

    # 计算过曝像素
    overexposed = v_channel > brightness_threshold
    overexposed_ratio = np.sum(overexposed) / overexposed.size

    # 计算亮度统计
    mean_brightness = np.mean(v_channel)
    max_brightness = np.max(v_channel)

    # 判定是否有炫光
    has_glare = overexposed_ratio > area_threshold

    # 计算过曝区域的连通性（大块过曝更可能是炫光）
    overexposed_uint8 = overexposed.astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(overexposed_uint8)

    # 找最大的过曝区域
    max_area = 0
    if num_labels > 1:  # 排除背景
        max_area = max(stats[1:, cv2.CC_STAT_AREA]) if len(stats) > 1 else 0

    total_pixels = v_channel.size
    max_area_ratio = max_area / total_pixels

    return {
        "path": image_path,
        "has_glare": has_glare,
        "overexposed_ratio": round(overexposed_ratio, 4),
        "mean_brightness": round(float(mean_brightness), 2),
        "max_brightness": int(max_brightness),
        "max_overexposed_area_ratio": round(max_area_ratio, 4),
        "image_size": list(image.shape[:2]),
        "thresholds": {
            "brightness": brightness_threshold,
            "area": area_threshold
        }
    }


def detect_glare_batch(image_dir: str,
                       brightness_threshold: int = 250,
                       area_threshold: float = 0.15,
                       extensions: List[str] = None) -> List[Dict]:
    """批量检测目录下的图片

    Args:
        image_dir: 图片目录
        brightness_threshold: 亮度阈值
        area_threshold: 面积阈值
        extensions: 支持的扩展名

    Returns:
        检测结果列表
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']

    image_dir = Path(image_dir)
    results = []

    # 递归查找所有图片
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.rglob(f"*{ext}"))

    image_files = sorted(image_files)
    total = len(image_files)

    print(f"找到 {total} 张图片")

    glare_count = 0
    for i, img_path in enumerate(image_files, 1):
        result = detect_glare(str(img_path), brightness_threshold, area_threshold)
        results.append(result)

        if result.get("has_glare"):
            glare_count += 1

        # 进度显示
        if i % 100 == 0 or i == total:
            print(f"   处理进度: {i}/{total} ({i*100//total}%) | 炫光: {glare_count}")

    return results


def print_summary(results: List[Dict]):
    """打印检测汇总"""
    total = len(results)
    glare_images = [r for r in results if r.get("has_glare")]
    glare_count = len(glare_images)

    print(f"\n{'='*50}")
    print(f"检测汇总")
    print(f"{'='*50}")
    print(f"   总图片数: {total}")
    print(f"   炫光图片: {glare_count} ({glare_count*100//total if total > 0 else 0}%)")
    print(f"   正常图片: {total - glare_count}")

    if glare_images:
        # 按过曝比例排序，显示最严重的几张
        sorted_glare = sorted(glare_images, key=lambda x: x.get("overexposed_ratio", 0), reverse=True)
        print(f"\n   最严重的炫光图片 (Top 5):")
        for r in sorted_glare[:5]:
            ratio = r.get("overexposed_ratio", 0) * 100
            print(f"      - {Path(r['path']).name}: {ratio:.1f}% 过曝")

    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="炫光/反光检测器 - 检测交通标志图片中的过曝问题",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", "-i", type=str, help="单张图片路径")
    input_group.add_argument("--dir", "-d", type=str, help="图片目录路径")

    # 检测参数
    parser.add_argument("--threshold", "-t", type=float, default=0.15,
                        help="过曝面积阈值 (0-1)，默认 0.15")
    parser.add_argument("--brightness", "-b", type=int, default=250,
                        help="亮度阈值 (0-255)，默认 250")

    # 输出选项
    parser.add_argument("--output", "-o", type=str, help="保存检测结果 JSON")
    parser.add_argument("--move-glare", type=str, help="将炫光图片移动到指定目录")
    parser.add_argument("--copy-glare", type=str, help="将炫光图片复制到指定目录")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")

    args = parser.parse_args()

    if args.image:
        # 单张图片检测
        result = detect_glare(args.image, args.brightness, args.threshold)

        print(f"\n检测结果: {args.image}")
        print(f"   炫光: {'是' if result['has_glare'] else '否'}")
        print(f"   过曝比例: {result['overexposed_ratio']*100:.1f}%")
        print(f"   平均亮度: {result['mean_brightness']}")
        print(f"   最大亮度: {result['max_brightness']}")

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存: {args.output}")

    else:
        # 批量检测
        print(f"\n开始批量检测: {args.dir}")
        print(f"   亮度阈值: {args.brightness}")
        print(f"   面积阈值: {args.threshold}")

        results = detect_glare_batch(args.dir, args.brightness, args.threshold)
        print_summary(results)

        # 保存结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存: {args.output}")

        # 移动/复制炫光图片
        glare_images = [r for r in results if r.get("has_glare")]

        if args.move_glare and glare_images:
            dest_dir = Path(args.move_glare)
            dest_dir.mkdir(parents=True, exist_ok=True)

            for r in glare_images:
                src = Path(r['path'])
                dst = dest_dir / src.name
                shutil.move(str(src), str(dst))

            print(f"\n已移动 {len(glare_images)} 张炫光图片到: {dest_dir}")

        elif args.copy_glare and glare_images:
            dest_dir = Path(args.copy_glare)
            dest_dir.mkdir(parents=True, exist_ok=True)

            for r in glare_images:
                src = Path(r['path'])
                dst = dest_dir / src.name
                shutil.copy2(str(src), str(dst))

            print(f"\n已复制 {len(glare_images)} 张炫光图片到: {dest_dir}")


if __name__ == "__main__":
    main()
