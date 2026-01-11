#!/usr/bin/env python3
"""
从 labelme 格式的 annotations 裁剪交通标志图片 (4K 无损)

用法:
    python scripts/crop_signs_from_annotations.py
"""

import json
from pathlib import Path
from PIL import Image


def crop_and_save(
    image_path: str,
    bbox: list,
    output_path: str,
    padding: int = 10
) -> str:
    """
    裁剪并保存交通标志 (PNG 无损)

    Args:
        image_path: 原图路径 (4K)
        bbox: 边界框 [x1, y1, x2, y2]
        output_path: 输出路径
        padding: 边界扩展像素

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

    # PNG 无损保存
    cropped.save(output_path, format='PNG')

    return output_path


def extract_signs_from_dataset(dataset_dir: Path, output_dir: Path, padding: int = 10) -> int:
    """从单个 dataset 目录提取 traffic_sign 类别的标志"""
    annotations_dir = dataset_dir / "annotations"
    frames_dir = dataset_dir / "frames"

    if not annotations_dir.exists() or not frames_dir.exists():
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for json_file in sorted(annotations_dir.glob("*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # 找对应的帧图片
            frame_name = data.get("imagePath", json_file.stem + ".jpg")
            frame_path = frames_dir / frame_name

            if not frame_path.exists():
                continue

            # 提取 traffic_sign 类别的标志
            shapes = data.get("shapes", [])
            sign_idx = 0
            for shape in shapes:
                category = shape.get("flags", {}).get("category", "")
                if category != "traffic_sign":
                    continue

                # 获取 bbox
                points = shape.get("points", [])
                if len(points) != 2:
                    continue

                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])

                # 确保坐标有效
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue

                sign_idx += 1

                # 生成输出文件名: 原图名_signX_x1_y1_x2_y2.png
                out_name = f"{json_file.stem}_sign{sign_idx}_{x1}_{y1}_{x2}_{y2}.png"
                out_path = output_dir / out_name

                # 裁剪并保存
                crop_and_save(str(frame_path), [x1, y1, x2, y2], str(out_path), padding)
                count += 1

        except Exception as e:
            print(f"  错误: {json_file.name}: {e}")

    return count


def main():
    dataset_output = Path("/Users/justin/Desktop/GLM_Labeling/dataset_output")
    batch2_output = Path("/Users/justin/Desktop/GLM_Labeling/extracted_signs/DJI_batch2")
    padding = 10

    # 找所有 DJI_00xx 开头的 dataset
    videos = {}
    for d in sorted(dataset_output.glob("DJI_00*_dataset")):
        # DJI_0016_000_dataset -> DJI_0016
        parts = d.name.split("_")
        video_name = f"{parts[0]}_{parts[1]}"  # DJI_0016
        if video_name not in videos:
            videos[video_name] = []
        videos[video_name].append(d)

    print(f"找到 {len(videos)} 个视频的数据")
    print(f"输出目录: {batch2_output}")
    print(f"Padding: {padding}px")
    print("=" * 50)

    total = 0
    for video_name, datasets in sorted(videos.items()):
        output_dir = batch2_output / video_name
        print(f"\n处理 {video_name} ({len(datasets)} 个片段)...")

        video_count = 0
        for dataset_dir in datasets:
            count = extract_signs_from_dataset(dataset_dir, output_dir, padding)
            video_count += count

        print(f"  提取: {video_count} 张标志")
        total += video_count

    print("\n" + "=" * 50)
    print(f"总计提取: {total} 张标志图片")
    print(f"输出目录: {batch2_output}")


if __name__ == "__main__":
    main()
