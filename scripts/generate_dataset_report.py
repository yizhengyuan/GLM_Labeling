#!/usr/bin/env python3
"""
生成交通标志数据集报告

输出：
1. dataset_report.md - 数据集概述报告
2. hk_traffic_signs_dataset_v1.zip - 包含所有有效标志图片和标签的压缩包
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import zipfile

# 配置
EXTRACTED_SIGNS_DIR = Path(__file__).parent.parent / "extracted_signs"
OUTPUT_DIR = Path(__file__).parent.parent / "dataset_release"

BATCH_CONFIG = {
    "batch1": {
        "signs_dir": EXTRACTED_SIGNS_DIR / "DJI_batch1",
        "label_file": EXTRACTED_SIGNS_DIR / "DJI_batch1" / "label_results_batch1.json",
        "filter_file": EXTRACTED_SIGNS_DIR / "DJI_batch1" / "filter_results_batch1.json",
    },
    "batch2": {
        "signs_dir": EXTRACTED_SIGNS_DIR / "DJI_batch2",
        "label_file": EXTRACTED_SIGNS_DIR / "DJI_batch2" / "label_results_batch2.json",
        "filter_file": None,  # batch2 没有 filter 文件
    },
}

# 特殊标签（不计入有效数据）
SPECIAL_LABELS = {"__invalid__", "__blur__", "__unlisted__", "other"}


def load_batch_data(batch_name: str) -> dict:
    """加载批次数据"""
    config = BATCH_CONFIG[batch_name]

    with open(config["label_file"], "r", encoding="utf-8") as f:
        label_data = json.load(f)

    # 获取 valid 图片列表
    valid_images = set()
    if config["filter_file"] and config["filter_file"].exists():
        with open(config["filter_file"], "r", encoding="utf-8") as f:
            filter_data = json.load(f)
        valid_images = {k for k, v in filter_data.get("results", {}).items() if v == "valid"}
    else:
        # 如果没有 filter 文件，使用所有已标注的图片
        valid_images = set(label_data.get("labels", {}).keys())

    return {
        "signs_dir": config["signs_dir"],
        "label_data": label_data,
        "valid_images": valid_images,
    }


def collect_valid_signs(batch_data: dict) -> list:
    """收集有效的标志数据"""
    valid_signs = []
    labels = batch_data["label_data"].get("labels", {})
    signs_dir = batch_data["signs_dir"]

    for filename, label in labels.items():
        if filename not in batch_data["valid_images"]:
            continue
        if label in SPECIAL_LABELS:
            continue

        # 查找图片文件
        img_path = None
        for p in signs_dir.rglob(filename):
            img_path = p
            break

        if img_path and img_path.exists():
            valid_signs.append({
                "filename": filename,
                "label": label,
                "path": img_path,
            })

    return valid_signs


def generate_report(all_signs: list, output_path: Path):
    """生成 Markdown 报告"""
    # 统计
    label_counts = defaultdict(int)
    for sign in all_signs:
        label_counts[sign["label"]] += 1

    sorted_labels = sorted(label_counts.items(), key=lambda x: -x[1])
    total_count = len(all_signs)
    unique_labels = len(label_counts)

    # 按类别分组
    category_groups = {
        "禁令标志": [],
        "警告标志": [],
        "指示标志": [],
        "方向标志": [],
        "停车标志": [],
        "其他标志": [],
    }

    for label, count in sorted_labels:
        if "No_" in label or "Stop" in label or "Give_way" in label:
            category_groups["禁令标志"].append((label, count))
        elif "ahead" in label.lower() or "Risk" in label or "Bend" in label or "Road_" in label:
            category_groups["警告标志"].append((label, count))
        elif "Direction" in label or "Street" in label:
            category_groups["方向标志"].append((label, count))
        elif "Parking" in label:
            category_groups["停车标志"].append((label, count))
        elif "Keep" in label or "One_way" in label or "Reduce" in label:
            category_groups["指示标志"].append((label, count))
        else:
            category_groups["其他标志"].append((label, count))

    # 生成报告
    report = f"""# 香港交通标志数据集报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 数据集概述

| 指标 | 数值 |
|------|------|
| **有效标志图片总数** | {total_count:,} |
| **标志类别数** | {unique_labels} |
| **数据来源** | Batch 1 + Batch 2 |
| **图片格式** | PNG (无损) |
| **标注方式** | 人工标注 |

---

## 数据来源

- **Batch 1**: 4 个视频 (DJI_0004, DJI_0013, DJI_0014, DJI_0015)
- **Batch 2**: 2 个视频 (DJI_0016, DJI_0017)

视频采集设备：DJI 行车记录仪 4K
采集地点：香港特别行政区

---

## 标签分布

### 总体分布 (Top 20)

| 排名 | 标志名称 | 数量 | 占比 |
|------|----------|------|------|
"""

    for i, (label, count) in enumerate(sorted_labels[:20], 1):
        pct = count / total_count * 100
        display_name = label.replace("_", " ")
        report += f"| {i} | {display_name} | {count} | {pct:.1f}% |\n"

    report += f"""
### 按类别统计

"""

    for category, items in category_groups.items():
        if items:
            cat_total = sum(c for _, c in items)
            report += f"#### {category} ({cat_total} 张, {cat_total/total_count*100:.1f}%)\n\n"
            report += "| 标志名称 | 数量 |\n|----------|------|\n"
            for label, count in items:
                display_name = label.replace("_", " ")
                report += f"| {display_name} | {count} |\n"
            report += "\n"

    report += f"""---

## 完整标签列表

共 {unique_labels} 种标签：

| 标签名称 | 数量 | 占比 |
|----------|------|------|
"""

    for label, count in sorted_labels:
        pct = count / total_count * 100
        display_name = label.replace("_", " ")
        report += f"| {display_name} | {count} | {pct:.1f}% |\n"

    report += """
---

## 文件结构

```
hk_traffic_signs_dataset_v1/
├── images/              # 所有标志图片
│   ├── xxx_sign1.png
│   └── ...
├── labels.json          # 标签数据
└── README.md            # 数据集说明
```

---

## 标签数据格式

`labels.json` 文件格式：

```json
{
  "metadata": {
    "version": "1.0",
    "created_at": "2025-12-23",
    "total_images": 1234,
    "unique_labels": 56
  },
  "labels": {
    "filename.png": "Label_Name",
    ...
  },
  "label_statistics": {
    "Label_Name": 123,
    ...
  }
}
```

---

## 使用说明

1. 解压 `hk_traffic_signs_dataset_v1.zip`
2. 图片位于 `images/` 目录
3. 标签数据位于 `labels.json`

---

*由 GLM Labeling 工具自动生成*
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"报告已生成: {output_path}")
    return sorted_labels


def create_dataset_zip(all_signs: list, label_stats: list, output_path: Path):
    """创建数据集压缩包"""
    temp_dir = output_path.parent / "temp_dataset"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    images_dir = temp_dir / "images"
    images_dir.mkdir(parents=True)

    # 复制图片并构建标签数据
    labels_data = {
        "metadata": {
            "version": "1.0",
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "total_images": len(all_signs),
            "unique_labels": len(label_stats),
            "source": "Hong Kong Traffic Signs from DJI Dashcam 4K",
            "batches": ["batch1", "batch2"],
        },
        "labels": {},
        "label_statistics": dict(label_stats),
    }

    print(f"复制 {len(all_signs)} 张图片...")
    for i, sign in enumerate(all_signs):
        # 复制图片
        dst_path = images_dir / sign["filename"]
        shutil.copy2(sign["path"], dst_path)

        # 记录标签
        labels_data["labels"][sign["filename"]] = sign["label"]

        if (i + 1) % 500 == 0:
            print(f"  已处理 {i + 1}/{len(all_signs)}")

    # 保存标签文件
    labels_path = temp_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, ensure_ascii=False, indent=2)

    # 创建 README
    readme_path = temp_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""# 香港交通标志数据集 v1.0

## 概述

- **图片数量**: {len(all_signs)}
- **标签类别**: {len(label_stats)}
- **创建日期**: {datetime.now().strftime("%Y-%m-%d")}

## 文件说明

- `images/` - 标志图片 (PNG 格式)
- `labels.json` - 标签数据

## 使用方法

```python
import json
from pathlib import Path

# 加载标签
with open("labels.json", "r") as f:
    data = json.load(f)

# 遍历数据
for filename, label in data["labels"].items():
    img_path = Path("images") / filename
    print(f"{{img_path}}: {{label}}")
```

## 许可

仅供学术研究使用。
""")

    # 创建 ZIP
    print(f"创建压缩包: {output_path}")
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in temp_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(temp_dir)
                zf.write(file_path, arcname)

    # 清理临时目录
    shutil.rmtree(temp_dir)

    # 显示文件大小
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"压缩包已创建: {output_path} ({size_mb:.1f} MB)")


def main():
    print("=" * 50)
    print("香港交通标志数据集报告生成器")
    print("=" * 50)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 收集所有批次的有效标志
    all_signs = []

    for batch_name in ["batch1", "batch2"]:
        print(f"\n处理 {batch_name}...")
        batch_data = load_batch_data(batch_name)
        signs = collect_valid_signs(batch_data)
        print(f"  有效标志: {len(signs)} 张")
        all_signs.extend(signs)

    print(f"\n总计有效标志: {len(all_signs)} 张")

    # 生成报告
    report_path = OUTPUT_DIR / "dataset_report.md"
    label_stats = generate_report(all_signs, report_path)

    # 创建压缩包
    zip_path = OUTPUT_DIR / "hk_traffic_signs_dataset_v1.zip"
    create_dataset_zip(all_signs, label_stats, zip_path)

    print("\n" + "=" * 50)
    print("完成！输出文件:")
    print(f"  - {report_path}")
    print(f"  - {zip_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
