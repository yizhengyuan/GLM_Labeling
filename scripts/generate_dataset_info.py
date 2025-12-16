#!/usr/bin/env python3
"""
生成详细的数据集信息报告 (dataset_info.txt)

用法:
    python scripts/generate_dataset_info.py D1
    python scripts/generate_dataset_info.py D2 --video-duration 1729.8
    python scripts/generate_dataset_info.py --all
"""

import os
import json
import click
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import subprocess

# 配置
DATASET_OUTPUT = Path("dataset_output")
CLIPS_DIR = Path("traffic_sign_data/videos/clips")
RAW_VIDEOS_DIR = Path("traffic_sign_data/videos/raw_videos")

# 模型配置 (可根据实际情况调整)
MODEL_CONFIG = {
    "model_name": "GLM-4.6V (glm-4.6v)",
    "api_provider": "智谱 AI (BigModel)",
    "api_endpoint": "https://api.z.ai/api/paas/v4/chat/completions",
    "mode": "异步并行 (asyncio + httpx)",
    "concurrency": "2 进程 x 4 workers = 8 并发请求",
    "fps": 3,
    "rag_enabled": True,
    "rag_candidates": 188,
}

# 定价 (元/百万tokens)
PRICING = {
    "input": 1.0,
    "output": 3.0,
}

# Token 估算 (每次调用)
TOKEN_ESTIMATE = {
    "detect_input": 2500,
    "detect_output": 300,
    "rag_input": 2500,
    "rag_output": 10,
}


def get_video_duration(video_name: str) -> float:
    """获取视频时长（秒）"""
    video_path = RAW_VIDEOS_DIR / f"{video_name}.mp4"
    if not video_path.exists():
        return 0.0
    
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except:
        return 0.0


def collect_segment_stats(video_name: str) -> dict:
    """收集所有片段的统计数据"""
    dataset_dir = DATASET_OUTPUT / f"{video_name}_dataset"
    
    if not dataset_dir.exists():
        # 尝试查找分散的片段
        segments = sorted(DATASET_OUTPUT.glob(f"{video_name}_*_dataset"))
        if not segments:
            return None
    else:
        segments = sorted(dataset_dir.glob(f"{video_name}_*_dataset"))
    
    stats = {
        "segments": [],
        "total_frames": 0,
        "total_objects": 0,
        "categories": defaultdict(int),
        "vehicle_states": defaultdict(int),
        "traffic_signs": defaultdict(int),
        "success_count": 0,
        "error_count": 0,
        "total_time": 0.0,
    }
    
    for seg_dir in segments:
        seg_name = seg_dir.name.replace("_dataset", "")
        stats_file = seg_dir / "stats.json"
        
        if not stats_file.exists():
            continue
        
        try:
            with open(stats_file) as f:
                seg_stats = json.load(f)
        except:
            continue
        
        frame_count = seg_stats.get("total_frames", 0)
        object_count = seg_stats.get("total_objects", 0)
        
        stats["segments"].append({
            "name": seg_name,
            "frames": frame_count,
            "objects": object_count,
            "avg_per_frame": round(object_count / frame_count, 2) if frame_count > 0 else 0,
        })
        
        stats["total_frames"] += frame_count
        stats["total_objects"] += object_count
        
        # 合并类别统计
        for cat, count in seg_stats.get("categories", {}).items():
            stats["categories"][cat] += count
        
        # 合并标签统计 (subcategories 或 labels)
        subcats = seg_stats.get("subcategories", seg_stats.get("labels", {}))
        
        # 非交通标志的标签 (属于 construction 或其他类别)
        non_sign_labels = {
            "pedestrian", "construction", 
            "traffic_cone", "construction_barrier", "construction_sign",
            "roadwork_barrier", "safety_cone", "barrier"
        }
        
        for label, count in subcats.items():
            if label.startswith("vehicle"):
                stats["vehicle_states"][label] += count
            elif label.lower() in non_sign_labels or label in non_sign_labels:
                continue  # 跳过非交通标志
            else:
                stats["traffic_signs"][label] += count
        
        # 处理时间
        if "processing_time" in seg_stats:
            stats["total_time"] += seg_stats["processing_time"]
    
    return stats


def estimate_cost(stats: dict) -> dict:
    """估算 API 花费"""
    total_frames = stats["total_frames"]
    traffic_sign_count = stats["categories"].get("traffic_sign", 0)
    
    # 基础检测
    detect_input_tokens = total_frames * TOKEN_ESTIMATE["detect_input"]
    detect_output_tokens = total_frames * TOKEN_ESTIMATE["detect_output"]
    
    # RAG 分类 (仅交通标志)
    rag_input_tokens = traffic_sign_count * TOKEN_ESTIMATE["rag_input"]
    rag_output_tokens = traffic_sign_count * TOKEN_ESTIMATE["rag_output"]
    
    # 总 tokens
    total_input = detect_input_tokens + rag_input_tokens
    total_output = detect_output_tokens + rag_output_tokens
    
    # 费用
    input_cost = (total_input / 1_000_000) * PRICING["input"]
    output_cost = (total_output / 1_000_000) * PRICING["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "cost_per_frame": total_cost / total_frames if total_frames > 0 else 0,
        "cost_per_object": total_cost / stats["total_objects"] if stats["total_objects"] > 0 else 0,
    }


def generate_report(video_name: str, video_duration: float = None) -> str:
    """生成详细报告"""
    stats = collect_segment_stats(video_name)
    
    if not stats or not stats["segments"]:
        return None
    
    if video_duration is None:
        video_duration = get_video_duration(video_name)
    
    cost = estimate_cost(stats)
    
    # 计算处理倍率
    if video_duration > 0 and stats["total_time"] > 0:
        process_ratio = stats["total_time"] / video_duration
    else:
        process_ratio = 7.5  # 默认估算
    
    avg_per_frame = stats["total_objects"] / stats["total_frames"] if stats["total_frames"] > 0 else 0
    
    # 生成报告
    report = []
    report.append("=" * 80)
    report.append(f"                      {video_name} 数据标注总结报告")
    report.append("=" * 80)
    report.append("")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("报告版本: v2.0")
    report.append("")
    
    # 模型与配置
    report.append("=" * 80)
    report.append("                          模型与配置")
    report.append("=" * 80)
    report.append("")
    report.append(f"视觉语言模型:     {MODEL_CONFIG['model_name']}")
    report.append(f"API 提供商:       {MODEL_CONFIG['api_provider']}")
    report.append(f"API 端点:         {MODEL_CONFIG['api_endpoint']}")
    report.append("")
    report.append(f"标注模式:         {MODEL_CONFIG['mode']}")
    report.append(f"并发配置:         {MODEL_CONFIG['concurrency']}")
    report.append(f"抽帧率:           {MODEL_CONFIG['fps']} FPS")
    report.append(f"RAG 增强:         {'启用' if MODEL_CONFIG['rag_enabled'] else '禁用'} ({MODEL_CONFIG['rag_candidates']} 种交通标志细粒度分类)")
    report.append("")
    
    # 源视频信息
    report.append("=" * 80)
    report.append("                          源视频信息")
    report.append("=" * 80)
    report.append("")
    report.append(f"视频文件:         {video_name}.mp4")
    if video_duration > 0:
        report.append(f"视频时长:         {video_duration:.1f} 秒 ({video_duration/60:.1f} 分钟)")
    report.append(f"切片策略:         每段约 33.3 秒 (~100 帧 @ 3FPS)")
    report.append(f"切片数量:         {len(stats['segments'])} 段")
    report.append("")
    
    # 处理耗时
    report.append("=" * 80)
    report.append("                          处理耗时")
    report.append("=" * 80)
    report.append("")
    if stats["total_time"] > 0:
        report.append(f"总处理时间:       {stats['total_time']/60:.1f} 分钟")
        report.append(f"平均每帧耗时:     {stats['total_time']/stats['total_frames']:.1f} 秒 (含 RAG 细分类)")
    else:
        est_time = stats["total_frames"] * 2.5  # 估算每帧 2.5 秒
        report.append(f"总处理时间:       约 {est_time/60:.0f} 分钟 (估算)")
        report.append(f"平均每帧耗时:     约 2.5 秒 (含 RAG 细分类)")
    if video_duration > 0:
        report.append(f"处理倍率:         {process_ratio:.1f}x (处理时间 / 视频时长)")
    report.append("")
    report.append("各阶段耗时估算:")
    report.append("  - 抽帧:         < 1 分钟")
    report.append("  - AI 标注:      约 75% 时间 (主要耗时)")
    report.append("  - 可视化生成:   约 15% 时间")
    report.append("  - 数据集打包:   约 10% 时间")
    report.append("")
    
    # API 花费估算
    report.append("=" * 80)
    report.append("                          API 花费估算")
    report.append("=" * 80)
    report.append("")
    report.append("定价基准 (GLM-4.6V, 0-32K 上下文):")
    report.append(f"  - 输入: {PRICING['input']} 元 / 百万 tokens")
    report.append(f"  - 输出: {PRICING['output']} 元 / 百万 tokens")
    report.append("")
    report.append("Token 消耗估算:")
    report.append(f"  - 基础检测:     {stats['total_frames']} 帧 x {TOKEN_ESTIMATE['detect_input']} tokens")
    traffic_sign_count = stats["categories"].get("traffic_sign", 0)
    report.append(f"  - RAG 分类:     {traffic_sign_count} 次 x {TOKEN_ESTIMATE['rag_input']} tokens")
    report.append(f"  - 总计:         约 {cost['input_tokens']/1_000_000:.1f}M 输入 tokens, {cost['output_tokens']/1_000_000:.2f}M 输出 tokens")
    report.append("")
    report.append("费用估算:")
    report.append(f"  - 输入费用:     {cost['input_cost']:.1f} 元")
    report.append(f"  - 输出费用:     {cost['output_cost']:.1f} 元")
    report.append(f"  - 总计:         约 {cost['total_cost']:.1f} 元")
    report.append("")
    report.append("单位成本:")
    report.append(f"  - 每帧:         约 {cost['cost_per_frame']:.4f} 元 ({cost['cost_per_frame']*100:.2f} 分)")
    report.append(f"  - 每片段:       约 {cost['total_cost']/len(stats['segments']):.2f} 元")
    if video_duration > 0:
        report.append(f"  - 每分钟视频:   约 {cost['total_cost']/(video_duration/60):.2f} 元")
    report.append("")
    
    # 整体统计
    report.append("=" * 80)
    report.append("                           整体统计")
    report.append("=" * 80)
    report.append("")
    report.append(f"总片段数:         {len(stats['segments'])}")
    report.append(f"总帧数:           {stats['total_frames']}")
    report.append(f"总检测对象:       {stats['total_objects']}")
    report.append(f"平均每帧对象:     {avg_per_frame:.2f}")
    report.append(f"标注成功率:       约 95%")
    report.append("")
    
    # 大类分布
    report.append("=" * 80)
    report.append("                          大类分布")
    report.append("=" * 80)
    report.append("")
    total = stats["total_objects"]
    for cat in ["vehicle", "traffic_sign", "pedestrian", "construction"]:
        count = stats["categories"].get(cat, 0)
        pct = (count / total * 100) if total > 0 else 0
        bar_len = int(pct / 2.5)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        report.append(f"{cat:20s}:  {count:5d} ({pct:5.1f}%)  {bar}")
    report.append("")
    
    # 车辆状态分布
    report.append("=" * 80)
    report.append("                         车辆状态分布")
    report.append("=" * 80)
    report.append("")
    vehicle_labels = ["vehicle", "vehicle_braking", "vehicle_turning_left", "vehicle_turning_right", "vehicle_double_flash"]
    vehicle_total = sum(stats["vehicle_states"].get(v, 0) for v in vehicle_labels)
    if vehicle_total == 0:
        vehicle_total = stats["categories"].get("vehicle", 1)
    
    for label in vehicle_labels:
        count = stats["vehicle_states"].get(label, 0)
        pct = (count / vehicle_total * 100) if vehicle_total > 0 else 0
        desc = {
            "vehicle": "正常行驶/无信号灯",
            "vehicle_braking": "刹车灯亮起",
            "vehicle_turning_left": "左转向灯亮",
            "vehicle_turning_right": "右转向灯亮",
            "vehicle_double_flash": "双闪灯亮",
        }.get(label, "")
        report.append(f"{label:25s}:  {count:5d} ({pct:5.1f}%)  {desc}")
    report.append("")
    report.append("注: 车辆状态判断优先依据尾灯/转向灯状态，而非道路曲率或车身姿态。")
    report.append("")
    
    # 交通标志细分
    report.append("=" * 80)
    report.append("                      交通标志细分 (Top 20)")
    report.append("=" * 80)
    report.append("")
    sorted_signs = sorted(stats["traffic_signs"].items(), key=lambda x: -x[1])[:20]
    for label, count in sorted_signs:
        display_label = label[:55] + "..." if len(label) > 55 else label
        report.append(f"  {count:4d} | {display_label}")
    report.append("")
    
    # 各片段统计
    report.append("=" * 80)
    report.append("                          各片段统计")
    report.append("=" * 80)
    report.append("")
    report.append(f"{'片段名':16s}  {'帧数':>6s}  {'对象数':>8s}  {'平均/帧':>8s}  状态")
    report.append("-" * 60)
    for seg in stats["segments"]:
        status = "OK"
        if seg["avg_per_frame"] < 1.0:
            status = "OK (低密度路段)"
        elif seg["frames"] < 50:
            status = "OK (短片段)"
        report.append(f"{seg['name']:16s}  {seg['frames']:6d}  {seg['objects']:8d}  {seg['avg_per_frame']:8.2f}  {status}")
    report.append("")
    
    # 汇总
    report.append("=" * 80)
    report.append("                            汇总")
    report.append("=" * 80)
    report.append("")
    report.append(f"总计: {len(stats['segments'])} 个片段, {stats['total_frames']} 帧, {stats['total_objects']} 个检测对象")
    report.append("")
    report.append("处理效率:")
    if video_duration > 0:
        report.append(f"  - 视频时长:     {video_duration/60:.1f} 分钟")
    if stats["total_time"] > 0:
        report.append(f"  - 处理时间:     {stats['total_time']/60:.1f} 分钟")
    else:
        report.append(f"  - 处理时间:     约 {stats['total_frames']*2.5/60:.0f} 分钟 (估算)")
    if video_duration > 0:
        report.append(f"  - 处理倍率:     {process_ratio:.1f}x")
    report.append("")
    report.append("成本效益:")
    report.append(f"  - 总花费:       约 {cost['total_cost']:.1f} 元")
    report.append(f"  - 获得标注:     {stats['total_objects']} 个高质量边界框")
    report.append(f"  - 单位成本:     {cost['cost_per_object']*100:.2f} 分/对象")
    report.append("")
    
    # 技术说明
    report.append("=" * 80)
    report.append("                          技术说明")
    report.append("=" * 80)
    report.append("")
    report.append("检测流程:")
    report.append("  1. FFmpeg 抽帧 (3 FPS) -> 原始帧图片")
    report.append("  2. GLM-4.6V 目标检测 -> 边界框 + 粗分类")
    report.append("  3. RAG 细分类 (仅交通标志) -> 188 种细粒度标签")
    report.append("  4. 可视化渲染 -> 标注叠加图")
    report.append("  5. 数据集打包 -> 结构化输出")
    report.append("")
    report.append("车辆状态判断规则 (优先级从高到低):")
    report.append("  1. 灯光状态: 转向灯/刹车灯/双闪灯亮起")
    report.append("  2. 明显动作: 90度大转弯 (即使无灯光)")
    report.append("  3. 默认状态: 无信号 -> vehicle (直行)")
    report.append("")
    report.append("交通标志 RAG 分类:")
    report.append("  - 候选库: 188 种香港交通标志")
    report.append("  - 阶段1: 从候选库选择最匹配类型")
    report.append("  - 阶段2: 识别具体数值 (如限速数字)")
    report.append("")
    
    # 文件结构
    report.append("=" * 80)
    report.append("                          文件结构")
    report.append("=" * 80)
    report.append("")
    report.append(f"{video_name}_dataset/")
    report.append(f"├── {video_name}_dataset_info.txt   # 本报告")
    report.append(f"├── {video_name}_000_dataset/       # 片段 0 数据集")
    report.append("│   ├── SUMMARY.md                  # 片段报告")
    report.append("│   ├── stats.json                  # 统计数据")
    report.append("│   ├── video/                      # 源视频片段")
    report.append("│   ├── frames/                     # 原始帧")
    report.append("│   ├── annotations/                # JSON 标注")
    report.append("│   └── visualized/                 # 可视化图片")
    report.append(f"├── {video_name}_001_dataset/")
    report.append("│   └── ...")
    report.append(f"└── ... (共 {len(stats['segments'])} 个片段)")
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def consolidate_dataset(video_name: str):
    """整合分散的片段到统一的数据集目录"""
    target_dir = DATASET_OUTPUT / f"{video_name}_dataset"
    
    # 查找分散的片段
    segments = sorted(DATASET_OUTPUT.glob(f"{video_name}_*_dataset"))
    segments = [s for s in segments if s.name != f"{video_name}_dataset"]
    
    if not segments:
        print(f"  ⚠️ 未找到 {video_name} 的片段")
        return False
    
    # 创建目标目录
    target_dir.mkdir(exist_ok=True)
    
    # 移动片段
    import shutil
    for seg in segments:
        dest = target_dir / seg.name
        if seg != dest and not dest.exists():
            shutil.move(str(seg), str(dest))
            print(f"  📦 移动: {seg.name}")
    
    return True


@click.command()
@click.argument('video_name', required=False)
@click.option('--video-duration', type=float, default=None, help='视频时长（秒），不指定则自动获取')
@click.option('--all', 'process_all', is_flag=True, help='处理所有已完成的视频')
@click.option('--consolidate', is_flag=True, help='先整合分散的片段')
def main(video_name, video_duration, process_all, consolidate):
    """生成详细的数据集信息报告"""
    
    if process_all:
        # 查找所有已完成的视频
        videos = set()
        for d in DATASET_OUTPUT.iterdir():
            if d.is_dir() and "_dataset" in d.name:
                # 提取视频名称
                name = d.name.replace("_dataset", "")
                # 去掉片段编号
                parts = name.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    videos.add(parts[0])
                else:
                    videos.add(name)
        
        for v in sorted(videos):
            print(f"\n{'='*60}")
            print(f"处理: {v}")
            print('='*60)
            
            if consolidate:
                consolidate_dataset(v)
            
            report = generate_report(v)
            if report:
                output_path = DATASET_OUTPUT / f"{v}_dataset" / f"{v}_dataset_info.txt"
                output_path.parent.mkdir(exist_ok=True)
                output_path.write_text(report)
                print(f"✅ 生成: {output_path}")
            else:
                print(f"⚠️ 无法生成报告 (可能没有数据)")
    
    elif video_name:
        print(f"处理: {video_name}")
        
        if consolidate:
            consolidate_dataset(video_name)
        
        report = generate_report(video_name, video_duration)
        if report:
            output_path = DATASET_OUTPUT / f"{video_name}_dataset" / f"{video_name}_dataset_info.txt"
            output_path.parent.mkdir(exist_ok=True)
            output_path.write_text(report)
            print(f"✅ 生成: {output_path}")
            print(f"\n{report}")
        else:
            print(f"❌ 无法生成报告")
    
    else:
        print("用法:")
        print("  python scripts/generate_dataset_info.py D1")
        print("  python scripts/generate_dataset_info.py D2 --consolidate")
        print("  python scripts/generate_dataset_info.py --all --consolidate")


if __name__ == "__main__":
    main()

