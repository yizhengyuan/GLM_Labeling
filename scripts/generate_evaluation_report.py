#!/usr/bin/env python3
"""Generate evaluation report from annotations.json and predictions.json"""
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

# 设置路径
data_dir = Path('extracted_signs/auto_label_tests/DJI_0020_2')
predictions_file = data_dir / 'predictions.json'
annotations_file = data_dir / 'annotations.json'

# 读取数据
with open(predictions_file) as f:
    predictions_data = json.load(f)
    predictions = predictions_data.get('predictions', predictions_data)  # 处理嵌套结构
with open(annotations_file) as f:
    annotations = json.load(f)

# 统计
total = len(annotations)
correct = sum(1 for a in annotations.values() if a.get('is_correct', False))
incorrect = total - correct
accuracy = correct / total * 100 if total > 0 else 0

# 按标签统计
label_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
error_distribution = Counter()  # 正确标签 -> 错误次数
confusion_pairs = Counter()  # (AI预测, 正确标签) -> 次数

for img_name, annotation in annotations.items():
    predicted_label = predictions.get(img_name, {}).get('label', 'unknown')
    is_correct = annotation.get('is_correct', False)
    correct_label = annotation.get('correct_label')

    if is_correct:
        label_stats[predicted_label]['correct'] += 1
        label_stats[predicted_label]['total'] += 1
    else:
        if correct_label:
            error_distribution[correct_label] += 1
            confusion_pairs[(predicted_label, correct_label)] += 1
            label_stats[correct_label]['total'] += 1
            # 预测标签也算一次（但这次是错的）
            label_stats[predicted_label]['total'] += 1
        else:
            # 没有正确标签的，算作预测标签的一次错误
            label_stats[predicted_label]['total'] += 1

# 置信度区间统计
confidence_buckets = defaultdict(lambda: {'total': 0, 'correct': 0})
for img_name, annotation in annotations.items():
    confidence = predictions.get(img_name, {}).get('confidence', 0)
    is_correct = annotation.get('is_correct', False)

    if confidence >= 0.9:
        bucket = '0.9-1.0'
    elif confidence >= 0.8:
        bucket = '0.8-0.9'
    elif confidence >= 0.7:
        bucket = '0.7-0.8'
    elif confidence >= 0.6:
        bucket = '0.6-0.7'
    elif confidence >= 0.5:
        bucket = '0.5-0.6'
    else:
        bucket = '<0.5'

    confidence_buckets[bucket]['total'] += 1
    if is_correct:
        confidence_buckets[bucket]['correct'] += 1

# 生成 evaluation_report.json
report_json = {
    'dataset': 'DJI_0020_2',
    'total_samples': total,
    'correct': correct,
    'incorrect': incorrect,
    'accuracy': round(accuracy, 2),
    'timestamp': datetime.now().isoformat()
}

# 按标签准确率
label_accuracy = []
for label, stats in sorted(label_stats.items(), key=lambda x: x[1]['total'], reverse=True):
    if stats['total'] > 0:
        acc = stats['correct'] / stats['total'] * 100
        label_accuracy.append({
            'label': label,
            'total': stats['total'],
            'correct': stats['correct'],
            'accuracy': round(acc, 1)
        })

report_json['label_accuracy'] = label_accuracy

# 置信度区间
confidence_report = []
for bucket in ['0.9-1.0', '0.8-0.9', '0.7-0.8', '0.6-0.7', '0.5-0.6', '<0.5']:
    if bucket in confidence_buckets:
        stats = confidence_buckets[bucket]
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        confidence_report.append({
            'bucket': bucket,
            'total': stats['total'],
            'correct': stats['correct'],
            'accuracy': round(acc, 1)
        })

report_json['confidence_accuracy'] = confidence_report

# 保存 JSON 报告
with open(data_dir / 'evaluation_report.json', 'w') as f:
    json.dump(report_json, f, indent=2, ensure_ascii=False)

print(f'JSON report saved: accuracy = {accuracy:.2f}%')

# 生成 evaluation_report.md
md_lines = [
    '# 模型分类评估报告',
    '',
    f'**数据集**: DJI_0020_2',
    f'**总样本数**: {total}',
    '',
    '## 整体统计',
    '',
    '| 指标 | 数值 |',
    '|------|------|',
    f'| 总样本数 | {total} |',
    f'| 正确数 | {correct} |',
    f'| 错误数 | {incorrect} |',
    f'| **准确率** | **{accuracy:.2f}%** |',
    '',
    '## 各标签准确率',
    '',
    '| 标签 | 总数 | 正确 | 准确率 |',
    '|------|------|------|--------|'
]

# 按准确率排序标签
sorted_labels = sorted(label_stats.items(), key=lambda x: x[1]['correct'] / x[1]['total'] if x[1]['total'] > 0 else 0)
for label, stats in sorted_labels:
    if stats['total'] > 0:
        acc = stats['correct'] / stats['total'] * 100
        md_lines.append(f'| {label} | {stats["total"]} | {stats["correct"]} | {acc:.1f}% |')

# 错误分布
md_lines.extend([
    '',
    '## 错误分布（实际应标注的标签）',
    '',
    '| 正确标签 | 错误次数 |',
    '|----------|----------|'
])

for correct_label, count in error_distribution.most_common():
    md_lines.append(f'| {correct_label} | {count} |')

# 常见混淆对
md_lines.extend([
    '',
    '## 常见混淆对（AI预测 → 正确标签）',
    '',
    '| AI预测 | 正确标签 | 次数 |',
    '|--------|----------|------|'
])

for (ai_pred, correct_lbl), count in confusion_pairs.most_common():
    md_lines.append(f'| {ai_pred} | {correct_lbl} | {count} |')

# 置信度区间
md_lines.extend([
    '',
    '## 按置信度区间的准确率',
    '',
    '| 置信度区间 | 样本数 | 正确数 | 准确率 |',
    '|------------|--------|--------|--------|'
])

for bucket in ['0.9-1.0', '0.8-0.9', '0.7-0.8', '0.6-0.7', '0.5-0.6', '<0.5']:
    if bucket in confidence_buckets:
        stats = confidence_buckets[bucket]
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        md_lines.append(f'| {bucket} | {stats["total"]} | {stats["correct"]} | {acc:.1f}% |')

# 结论
md_lines.extend([
    '',
    '## 结论',
    '',
    f'模型在 DJI_0020_2 数据集上的整体准确率为 **{accuracy:.2f}%**。',
    ''
])

if accuracy >= 85:
    md_lines.append('✅ 模型表现优秀！')
elif accuracy >= 70:
    md_lines.append('⚠️ 模型表现良好，但仍有改进空间。')
else:
    md_lines.append('❌ 模型准确率较低，需要进一步优化。')

# 建议
if incorrect > 0:
    md_lines.extend([
        '',
        '### 建议',
        '',
        '1. 重点关注准确率较低的标签'
    ])

    low_acc_labels = [label for label, stats in sorted_labels[:5] if stats['total'] >= 2 and stats['correct'] / stats['total'] < 0.7]
    if low_acc_labels:
        labels_str = ', '.join(low_acc_labels[:3])
        md_lines.append(f'   - {labels_str}')

    if confusion_pairs:
        top_confusion = confusion_pairs.most_common(3)
        md_lines.append('2. 针对以下混淆对增加训练样本:')
        for (ai_pred, correct_lbl), count in top_confusion:
            md_lines.append(f'   - `{ai_pred}` → `{correct_lbl}` (出现 {count} 次)')

# 保存 MD 报告
with open(data_dir / 'evaluation_report.md', 'w') as f:
    f.write('\n'.join(md_lines))

print(f'MD report saved')
print(f'Total: {total}, Correct: {correct}, Incorrect: {incorrect}, Accuracy: {accuracy:.2f}%')
