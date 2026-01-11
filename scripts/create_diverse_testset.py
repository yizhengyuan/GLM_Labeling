#!/usr/bin/env python3
"""
从手标数据中选择多样化的测试集图片。

核心去重策略（两阶段）：
1. 阶段一：基于帧间隔+位置的快速去重（同一片段内）
2. 阶段二：基于感知哈希的跨视频去重（全局）

这样可以识别：
- 同一视频连续帧中的相同标志
- 不同视频/不同时间拍摄的相同标志（如车辆多次经过同一地点）
"""

import json
import os
import re
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from PIL import Image
import imagehash

# 配置
BASE_DIR = Path("/Users/justin/Desktop/GLM_Labeling")
BATCH1_DIR = BASE_DIR / "extracted_signs/DJI_batch1/labeled_data_batch1"
BATCH2_DIR = BASE_DIR / "extracted_signs/DJI_batch2/labeled_data_batch2"
BATCH3_DIR = BASE_DIR / "extracted_signs/DJI_batch3/DJI_0020"
OUTPUT_DIR = BASE_DIR / "dataset_release/test_set_1200"

# 无效标签列表（排除这些标签的图片）
INVALID_LABELS = {
    "not_sign",  # 不是交通标志
    "unclear",   # 不清晰/无法辨认
    "other"      # 其他/未列出的标志
}

# 阶段一参数：帧间隔去重
MAX_FRAME_GAP = 30
MAX_CENTER_DISTANCE = 150

# 阶段二参数：感知哈希去重
# 汉明距离阈值 - 越小越严格（通常 0-10 表示相似，>15 表示不同）
# 设置为 12 来捕获更多视觉相似的图片
PHASH_THRESHOLD = 12

# 目标图片数量
TARGET_COUNT = 1200


def parse_filename_full(filename: str) -> Dict:
    """解析文件名，提取视频ID、片段ID、帧号和边界框坐标。"""
    name = filename.replace(".png", "")

    if name.startswith("f"):
        match = re.match(r'f(\d+)_sign(\d+)_(\d+)_(\d+)_(\d+)_(\d+)', name)
        if match:
            return {
                "video_id": "batch1",
                "clip_id": "clip0",
                "frame_num": int(match.group(1)),
                "sign_num": int(match.group(2)),
                "bbox": (int(match.group(3)), int(match.group(4)),
                        int(match.group(5)), int(match.group(6)))
            }

    match = re.match(r'DJI_(\d+)_(\d+)_(\d+)_sign(\d+)_(\d+)_(\d+)_(\d+)_(\d+)', name)
    if match:
        return {
            "video_id": f"DJI_{match.group(1)}",
            "clip_id": f"clip_{match.group(2)}",
            "frame_num": int(match.group(3)),
            "sign_num": int(match.group(4)),
            "bbox": (int(match.group(5)), int(match.group(6)),
                    int(match.group(7)), int(match.group(8)))
        }

    return {"video_id": "unknown", "clip_id": "unknown", "frame_num": 0,
            "sign_num": 0, "bbox": (0, 0, 0, 0)}


def bbox_center(bbox): return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
def bbox_size(bbox): return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
def center_distance(b1, b2):
    c1, c2 = bbox_center(b1), bbox_center(b2)
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5


def load_labels() -> Dict[str, Dict]:
    """加载所有批次的标签数据"""
    all_labels = {}

    batch1_json = BATCH1_DIR / "dataset.json"
    if batch1_json.exists():
        with open(batch1_json, "r") as f:
            data = json.load(f)
            for filename, label in data.get("labels", {}).items():
                if label not in INVALID_LABELS:
                    parsed = parse_filename_full(filename)
                    all_labels[filename] = {
                        "label": label, "source": "batch1",
                        "path": BATCH1_DIR / filename, **parsed
                    }
    print(f"Batch1: 加载 {len([k for k,v in all_labels.items() if v['source']=='batch1'])} 张有效图片")

    batch2_json = BASE_DIR / "extracted_signs/DJI_batch2/label_results_batch2.json"
    if batch2_json.exists():
        with open(batch2_json, "r") as f:
            data = json.load(f)
            for filename, label in data.get("labels", {}).items():
                if label not in INVALID_LABELS:
                    parsed = parse_filename_full(filename)
                    all_labels[filename] = {
                        "label": label, "source": "batch2",
                        "path": BATCH2_DIR / filename, **parsed
                    }
    print(f"Batch2: 加载 {len([k for k,v in all_labels.items() if v['source']=='batch2'])} 张有效图片")

    return all_labels


class UnionFind:
    """并查集"""
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return
        if self.rank[px] < self.rank[py]: px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]: self.rank[px] += 1


def stage1_frame_dedup(labels: Dict[str, Dict]) -> List[str]:
    """
    阶段一：基于帧间隔和位置的快速去重（同一片段内）
    """
    by_context = defaultdict(list)
    for filename, info in labels.items():
        key = (info["video_id"], info["clip_id"], info["label"])
        by_context[key].append((filename, info))

    uf = UnionFind()

    for key, items in by_context.items():
        items.sort(key=lambda x: x[1]["frame_num"])
        for i, (fn1, info1) in enumerate(items):
            for j in range(i + 1, len(items)):
                fn2, info2 = items[j]
                if info2["frame_num"] - info1["frame_num"] > MAX_FRAME_GAP:
                    break
                if center_distance(info1["bbox"], info2["bbox"]) <= MAX_CENTER_DISTANCE:
                    uf.union(fn1, fn2)

    # 从每组选最大的
    groups = defaultdict(list)
    for filename in labels:
        groups[uf.find(filename)].append(filename)

    unique = []
    for group in groups.values():
        best = max(group, key=lambda fn: bbox_size(labels[fn]["bbox"]))
        unique.append(best)

    return unique


def compute_phash(path: Path) -> Optional[imagehash.ImageHash]:
    """计算图片的感知哈希"""
    try:
        img = Image.open(path)
        return imagehash.phash(img)
    except Exception as e:
        return None


def stage2_phash_dedup(filenames: List[str], labels: Dict[str, Dict]) -> List[str]:
    """
    阶段二：基于感知哈希的跨视频去重
    使用 pHash 检测视觉上相似的图片
    """
    print("  计算图片哈希...")

    # 按标签分组（只在同标签内比较哈希，提高效率）
    by_label = defaultdict(list)
    for fn in filenames:
        by_label[labels[fn]["label"]].append(fn)

    uf = UnionFind()
    total_comparisons = 0
    duplicates_found = 0

    for label, fns in by_label.items():
        if len(fns) < 2:
            continue

        # 计算这组图片的哈希
        hashes = {}
        for fn in fns:
            h = compute_phash(labels[fn]["path"])
            if h is not None:
                hashes[fn] = h

        # 比较哈希
        fn_list = list(hashes.keys())
        for i in range(len(fn_list)):
            for j in range(i + 1, len(fn_list)):
                fn1, fn2 = fn_list[i], fn_list[j]
                total_comparisons += 1
                distance = hashes[fn1] - hashes[fn2]
                if distance <= PHASH_THRESHOLD:
                    uf.union(fn1, fn2)
                    duplicates_found += 1

    print(f"  哈希比较: {total_comparisons} 次, 发现 {duplicates_found} 对相似图片")

    # 从每组选最大的
    groups = defaultdict(list)
    for fn in filenames:
        groups[uf.find(fn)].append(fn)

    unique = []
    for group in groups.values():
        best = max(group, key=lambda fn: bbox_size(labels[fn]["bbox"]))
        unique.append(best)

    return unique


def select_diverse_samples(labels: Dict[str, Dict], target_count: int = TARGET_COUNT) -> List[str]:
    """选择多样化的样本（两阶段去重）"""

    print("阶段一：帧间隔去重...")
    stage1_result = stage1_frame_dedup(labels)
    print(f"  {len(labels)} -> {len(stage1_result)} 张图片")

    print("\n阶段二：感知哈希去重...")
    stage2_result = stage2_phash_dedup(stage1_result, labels)
    print(f"  {len(stage1_result)} -> {len(stage2_result)} 张图片")

    unique_signs = stage2_result

    # 如果去重后数量仍然超过目标，进行均衡采样
    if len(unique_signs) > target_count:
        by_label = defaultdict(list)
        for fn in unique_signs:
            by_label[labels[fn]["label"]].append(fn)

        num_labels = len(by_label)
        base_quota = target_count // num_labels
        remainder = target_count % num_labels

        sorted_labels = sorted(by_label.keys(), key=lambda l: len(by_label[l]))

        selected = []
        extra_pool = []

        for i, label in enumerate(sorted_labels):
            items = by_label[label]
            quota = base_quota + (1 if i < remainder else 0)

            if len(items) <= quota:
                selected.extend(items)
            else:
                # 按视频来源多样性选择
                by_video = defaultdict(list)
                for fn in items:
                    by_video[labels[fn]["video_id"]].append(fn)

                video_list = list(by_video.keys())
                random.shuffle(video_list)
                chosen = []
                while len(chosen) < quota:
                    for vid in video_list:
                        if by_video[vid] and len(chosen) < quota:
                            chosen.append(by_video[vid].pop(0))

                selected.extend(chosen)
                for vid in by_video:
                    extra_pool.extend(by_video[vid])

        if len(selected) < target_count:
            random.shuffle(extra_pool)
            selected.extend(extra_pool[:target_count - len(selected)])

        unique_signs = selected
        print(f"\n均衡采样后: {len(unique_signs)} 张图片")

    if len(unique_signs) > target_count:
        random.shuffle(unique_signs)
        unique_signs = unique_signs[:target_count]

    if len(unique_signs) < target_count:
        print(f"\n注意: 只有 {len(unique_signs)} 张独立标志，少于目标 {target_count}")

    return unique_signs


def copy_selected_images(selected: List[str], labels: Dict[str, Dict], output_dir: Path):
    """复制选中的图片到输出目录，重命名为随机序号"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 清空旧文件
    for old_file in output_dir.glob("*.png"):
        old_file.unlink()
    for old_file in output_dir.glob("*.json"):
        old_file.unlink()
    for old_file in output_dir.glob("*.txt"):
        old_file.unlink()

    # 打乱顺序
    shuffled = list(selected)
    random.shuffle(shuffled)

    labels_dict = {}
    copied_count = 0

    # 用序号重命名文件
    for idx, filename in enumerate(shuffled):
        info = labels[filename]
        if info["path"].exists():
            new_name = f"{idx+1:04d}.png"  # 0001.png, 0002.png, ...
            shutil.copy2(info["path"], output_dir / new_name)
            labels_dict[new_name] = {
                "original_filename": filename,
                "label": info["label"],
                "source_video": info["video_id"],
                "source_clip": info["clip_id"],
                "frame_num": info["frame_num"],
                "bbox": info["bbox"]
            }
            copied_count += 1

    # 统计
    label_stats = defaultdict(int)
    for v in labels_dict.values():
        label_stats[v["label"]] += 1

    with open(output_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({
            "description": f"多样化测试集 - {copied_count} 张手标交通标志图片（两阶段去重+打乱重命名）",
            "total_images": copied_count,
            "selection_criteria": {
                "stage1": {"max_frame_gap": MAX_FRAME_GAP, "max_center_distance": MAX_CENTER_DISTANCE},
                "stage2": {"phash_threshold": PHASH_THRESHOLD},
                "excluded_labels": list(INVALID_LABELS),
                "shuffled_and_renamed": True
            },
            "label_statistics": dict(sorted(label_stats.items(), key=lambda x: -x[1])),
            "labels": labels_dict
        }, f, ensure_ascii=False, indent=2)

    print(f"\n完成! 复制 {copied_count} 张图片到 {output_dir}")
    print(f"  文件已打乱并重命名为 0001.png ~ {copied_count:04d}.png")
    return copied_count


def analyze_selection(selected: List[str], labels: Dict[str, Dict]):
    """分析选择结果"""
    print("\n=== 选择结果分析 ===")

    video_counts = defaultdict(int)
    label_counts = defaultdict(int)

    for fn in selected:
        info = labels[fn]
        video_counts[info["video_id"]] += 1
        label_counts[info["label"]] += 1

    print(f"\n视频来源分布:")
    for vid, cnt in sorted(video_counts.items(), key=lambda x: -x[1]):
        print(f"  {vid}: {cnt} 张")

    print(f"\n类别分布 (前15个):")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {lbl}: {cnt} 张")

    print(f"\n总计: {len(video_counts)} 个视频源, {len(label_counts)} 个类别, {len(selected)} 张图片")


def main():
    print("=" * 60)
    print("创建多样化测试集（两阶段去重）")
    print("=" * 60)

    random.seed(42)

    print("\n1. 加载标签数据...")
    labels = load_labels()
    print(f"   总计 {len(labels)} 张有效标注图片")

    print("\n2. 两阶段去重选择...")
    selected = select_diverse_samples(labels, TARGET_COUNT)

    analyze_selection(selected, labels)

    print("\n3. 复制图片...")
    copy_selected_images(selected, labels, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"测试集位置: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
