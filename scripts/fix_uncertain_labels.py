#!/usr/bin/env python3
"""
修改 uncertain 开头的标签：去掉 uncertain_ 前缀
同时更新统计信息
"""
import json
from pathlib import Path
from collections import Counter


def fix_uncertain_labels(data: dict, file_type: str) -> dict:
    """
    修复 uncertain_ 前缀的标签
    file_type: 'dji_0019' 或 'dji_0020_2'
    """
    changes_made = []

    if file_type == 'dji_0019':
        # DJI_0019 格式: labels 是一个 {filename: label} 的字典
        labels = data.get('labels', {})
        new_labels = {}

        for filename, label in labels.items():
            if label.startswith('uncertain_'):
                new_label = label[len('uncertain_'):]
                changes_made.append((filename, label, new_label))
                new_labels[filename] = new_label
            else:
                new_labels[filename] = label

        data['labels'] = new_labels

        # 重新计算 label_statistics
        label_counts = Counter(new_labels.values())
        data['label_statistics'] = dict(sorted(label_counts.items(), key=lambda x: -x[1]))

        # 更新 ai_evaluation 中的 most_error_prone_labels
        if 'ai_evaluation' in data:
            # 移除 most_error_prone_labels 中以 uncertain_ 开头的项
            old_most_error = data['ai_evaluation'].get('most_error_prone_labels', {})
            new_most_error = {}
            for label, count in old_most_error.items():
                if label.startswith('uncertain_'):
                    new_label = label[len('uncertain_'):]
                    if new_label in new_most_error:
                        new_most_error[new_label] += count
                    else:
                        new_most_error[new_label] = count
                else:
                    if label in new_most_error:
                        new_most_error[label] += count
                    else:
                        new_most_error[label] = count
            data['ai_evaluation']['most_error_prone_labels'] = dict(
                sorted(new_most_error.items(), key=lambda x: -x[1])
            )

            # 更新 common_confusions
            old_confusions = data['ai_evaluation'].get('common_confusions', {})
            new_confusions = {}
            for confusion, count in old_confusions.items():
                # 格式: "label1 -> label2"
                parts = confusion.split(' -> ')
                if len(parts) == 2:
                    from_label, to_label = parts
                    if from_label.startswith('uncertain_'):
                        from_label = from_label[len('uncertain_'):]
                    if to_label.startswith('uncertain_'):
                        to_label = to_label[len('uncertain_'):]
                    new_key = f"{from_label} -> {to_label}"
                    if new_key in new_confusions:
                        new_confusions[new_key] += count
                    else:
                        new_confusions[new_key] = count
                else:
                    new_confusions[confusion] = count
            data['ai_evaluation']['common_confusions'] = dict(
                sorted(new_confusions.items(), key=lambda x: -x[1])
            )

    elif file_type == 'dji_0020_2':
        # DJI_0020_2 格式: images 是一个 {filename: {label, ai_predicted, ...}} 的字典
        images = data.get('images', {})

        for filename, img_data in images.items():
            label = img_data.get('label', '')
            if label.startswith('uncertain_'):
                new_label = label[len('uncertain_'):]
                changes_made.append((filename, label, new_label))
                img_data['label'] = new_label

        # 重新计算 label_statistics
        label_counts = Counter(img_data.get('label', '') for img_data in images.values())
        data['label_statistics'] = dict(sorted(label_counts.items(), key=lambda x: -x[1]))

    return data, changes_made


def main():
    # 文件路径
    dji_0019_path = Path("/Users/justin/Desktop/GLM_Labeling/extracted_signs/auto_label_tests/DJI_0019/DJI_0019_labels.json")
    dji_0020_2_path = Path("/Users/justin/Desktop/GLM_Labeling/extracted_signs/auto_label_tests/DJI_0020_2_evaluation_results_new/labels.json")

    # 处理 DJI_0019
    print("=" * 60)
    print("处理 DJI_0019_labels.json")
    print("=" * 60)

    with open(dji_0019_path, 'r', encoding='utf-8') as f:
        data_0019 = json.load(f)

    data_0019, changes_0019 = fix_uncertain_labels(data_0019, 'dji_0019')

    print(f"共修改了 {len(changes_0019)} 个标签:")
    for filename, old_label, new_label in changes_0019:
        print(f"  {filename}: {old_label} -> {new_label}")

    # 保存修改后的文件
    with open(dji_0019_path, 'w', encoding='utf-8') as f:
        json.dump(data_0019, f, indent=2, ensure_ascii=False)
    print(f"\n已保存到 {dji_0019_path}")

    # 处理 DJI_0020_2
    print("\n" + "=" * 60)
    print("处理 DJI_0020_2 labels.json")
    print("=" * 60)

    with open(dji_0020_2_path, 'r', encoding='utf-8') as f:
        data_0020_2 = json.load(f)

    data_0020_2, changes_0020_2 = fix_uncertain_labels(data_0020_2, 'dji_0020_2')

    print(f"共修改了 {len(changes_0020_2)} 个标签:")
    for filename, old_label, new_label in changes_0020_2:
        print(f"  {filename}: {old_label} -> {new_label}")

    # 保存修改后的文件
    with open(dji_0020_2_path, 'w', encoding='utf-8') as f:
        json.dump(data_0020_2, f, indent=2, ensure_ascii=False)
    print(f"\n已保存到 {dji_0020_2_path}")

    # 输出统计
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)
    print(f"DJI_0019: 修改了 {len(changes_0019)} 个 uncertain 标签")
    print(f"DJI_0020_2: 修改了 {len(changes_0020_2)} 个 uncertain 标签")


if __name__ == '__main__':
    main()
