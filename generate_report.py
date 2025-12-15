#!/usr/bin/env python3
"""
生成数据标注总结报告

用法:
    python3 generate_report.py --prefix D1
    python3 generate_report.py --prefix D2
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime


def generate_report(prefix: str):
    """生成指定数据集的标注总结报告"""
    
    # 路径配置
    annotations_dir = Path(f'output/{prefix.lower()}_annotations')
    if not annotations_dir.exists():
        annotations_dir = Path(f'output/{prefix.lower()}_annotations_v2')
    
    if not annotations_dir.exists():
        print(f"❌ 找不到标注目录: {annotations_dir}")
        return
    
    json_files = sorted(annotations_dir.glob('*.json'))
    if not json_files:
        print(f"❌ 目录中没有 JSON 文件")
        return
    
    # 统计数据
    total_objects = 0
    empty_frames = 0
    label_counter = Counter()
    category_counter = Counter()
    
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        shapes = data.get('shapes', [])
        total_objects += len(shapes)
        if len(shapes) == 0:
            empty_frames += 1
        for shape in shapes:
            label_counter[shape['label']] += 1
            category_counter[shape.get('flags', {}).get('category', 'unknown')] += 1
    
    # 分类标签
    vehicles = []
    pedestrians = []
    signs = []
    construction = []
    others = []
    
    for label, count in label_counter.items():
        label_lower = label.lower()
        if label in ['car', 'van', 'suv', 'truck', 'bus', 'taxi', 'motorcycle', 'bicycle']:
            vehicles.append((label, count))
        elif label in ['pedestrian', 'cyclist', 'child', 'crowd']:
            pedestrians.append((label, count))
        elif 'sign' in label_lower or 'limit' in label_lower or 'light' in label_lower or label in ['ahead_only', 'turn_right', 'stop', 'no_entry']:
            signs.append((label, count))
        elif 'cone' in label_lower or 'barrier' in label_lower or 'construction' in label_lower:
            construction.append((label, count))
        else:
            others.append((label, count))
    
    # 查找源视频
    video_path = f"traffic_sign_data/videos/raw_clips/{prefix}.mp4"
    if not Path(video_path).exists():
        video_path = "(未找到)"
    
    # 生成报告内容
    report_lines = [
        "=" * 80,
        "                        数据标注总结报告",
        "=" * 80,
        "",
        "【基本信息】",
        f"数据集名称：{prefix}",
        f"源视频路径：{video_path}",
        f"图片来源：test_images/extracted_frames/{prefix}_*.jpg",
        f"标注时间：{datetime.now().strftime('%Y-%m-%d')}",
        f"标注工具：GLM-4.6V + auto_labeling_universal.py",
        f"输出目录：{annotations_dir}/",
        f"可视化目录：output/{prefix.lower()}_visualized/",
        "",
        "=" * 80,
        "【数据概览】",
        "=" * 80,
        f"总帧数：{len(json_files)}",
        f"总检测目标数：{total_objects}",
        f"空帧数：{empty_frames} ({empty_frames/len(json_files)*100:.1f}%)",
        f"平均每帧目标数：{total_objects/len(json_files):.2f}",
        "",
        "=" * 80,
        "【类别统计】",
        "=" * 80,
    ]
    
    for cat, count in category_counter.most_common():
        report_lines.append(f"{cat}: {count}")
    
    report_lines.extend([
        "",
        "=" * 80,
        "【标签详情】",
        "=" * 80,
    ])
    
    # 各类别详情
    if vehicles:
        report_lines.append(f"\n--- 车辆类 ({len(vehicles)} 种) ---")
        for label, count in sorted(vehicles, key=lambda x: -x[1]):
            report_lines.append(f"{label:20}: {count}")
    
    if pedestrians:
        report_lines.append(f"\n--- 行人类 ({len(pedestrians)} 种) ---")
        for label, count in sorted(pedestrians, key=lambda x: -x[1]):
            report_lines.append(f"{label:20}: {count}")
    
    if signs:
        report_lines.append(f"\n--- 交通标志类 ({len(signs)} 种) ---")
        for label, count in sorted(signs, key=lambda x: -x[1]):
            report_lines.append(f"{label:20}: {count}")
    
    if construction:
        report_lines.append(f"\n--- 施工标志类 ({len(construction)} 种) ---")
        for label, count in sorted(construction, key=lambda x: -x[1]):
            report_lines.append(f"{label:20}: {count}")
    
    if others:
        report_lines.append(f"\n--- 其他 ({len(others)} 种) ---")
        for label, count in sorted(others, key=lambda x: -x[1]):
            report_lines.append(f"{label:20}: {count}")
    
    report_lines.extend([
        "",
        "=" * 80,
        "【质量指标】",
        "=" * 80,
        f"标注覆盖率：{(len(json_files)-empty_frames)/len(json_files)*100:.1f}% ({len(json_files)-empty_frames}/{len(json_files)} 帧有标注)",
        f"标注格式：X-AnyLabeling JSON",
        "",
        "=" * 80,
        "【备注】",
        "=" * 80,
        f"- 数据集 {prefix} 标注完成",
        "- 标注使用 GLM-4.6V 多模态模型",
        "- 部分帧可能因模型响应问题为空",
        "",
        "=" * 80,
        "                          报告结束",
        "=" * 80,
    ])
    
    report = "\n".join(report_lines)
    
    # 保存报告
    output_path = Path(f'output/{prefix}_annotation_report.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 报告已保存到: {output_path}")
    print("\n" + "=" * 60)
    print("报告预览:")
    print("=" * 60)
    print(report)


def main():
    parser = argparse.ArgumentParser(description="生成数据标注总结报告")
    parser.add_argument("--prefix", type=str, required=True, help="数据集前缀 (如 D1, D2)")
    args = parser.parse_args()
    
    generate_report(args.prefix)


if __name__ == "__main__":
    main()
