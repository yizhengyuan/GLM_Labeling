#!/usr/bin/env python3
"""
将 DJI_0020_2 的 labels.json 格式转换为 DJI_0019_labels.json 的格式
"""
import json
from pathlib import Path
from collections import Counter


def convert_format(input_path: Path) -> dict:
    """
    将 DJI_0020_2 格式转换为 DJI_0019 格式
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        old_data = json.load(f)

    images = old_data.get('images', {})
    total = len(images)

    # 构建 labels 字典 (filename -> label)
    labels = {filename: img_data['label'] for filename, img_data in images.items()}

    # 计算 label_statistics
    label_counts = Counter(labels.values())
    label_statistics = dict(sorted(label_counts.items(), key=lambda x: -x[1]))

    # 计算 ai_evaluation
    correct_count = sum(1 for img_data in images.values() if img_data.get('is_correct', False))
    incorrect_count = total - correct_count
    error_rate = incorrect_count / total if total > 0 else 0

    # 计算 most_error_prone_labels (AI 预测错误最多的标签)
    error_labels = []
    for filename, img_data in images.items():
        if not img_data.get('is_correct', False):
            error_labels.append(img_data.get('ai_predicted', ''))
    most_error_prone = dict(sorted(Counter(error_labels).items(), key=lambda x: -x[1])[:10])

    # 计算 common_confusions (AI预测 -> 人工标注)
    confusions = []
    for filename, img_data in images.items():
        if not img_data.get('is_correct', False):
            ai_pred = img_data.get('ai_predicted', '')
            human_label = img_data.get('label', '')
            if ai_pred and human_label:
                confusions.append(f"{ai_pred} -> {human_label}")
    common_confusions = dict(sorted(Counter(confusions).items(), key=lambda x: -x[1])[:15])

    # 构建新格式
    new_data = {
        "total": total,
        "ai_evaluation": {
            "total_evaluated": total,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "error_rate": round(error_rate, 4),
            "error_rate_percent": f"{error_rate * 100:.2f}%",
            "most_error_prone_labels": most_error_prone,
            "common_confusions": common_confusions
        },
        "label_statistics": label_statistics,
        "labels": labels
    }

    return new_data


def main():
    input_path = Path("/Users/justin/Desktop/GLM_Labeling/extracted_signs/auto_label_tests/DJI_0020_2_evaluation_results_new/labels.json")

    print("正在转换格式...")
    new_data = convert_format(input_path)

    # 保存
    with open(input_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"转换完成！")
    print(f"  总数: {new_data['total']}")
    print(f"  正确数: {new_data['ai_evaluation']['correct_count']}")
    print(f"  错误数: {new_data['ai_evaluation']['incorrect_count']}")
    print(f"  错误率: {new_data['ai_evaluation']['error_rate_percent']}")
    print(f"\n标签统计 (前10):")
    for i, (label, count) in enumerate(list(new_data['label_statistics'].items())[:10]):
        print(f"  {label}: {count}")


if __name__ == '__main__':
    main()
