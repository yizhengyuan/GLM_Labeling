#!/usr/bin/env python3
"""
批量标注视频并自动整合打包

用法:
    python scripts/batch_label_videos.py D3 D4 D5 D6
    python scripts/batch_label_videos.py D3 D4 D5 D6 --workers 4 --parallel 2
    python scripts/batch_label_videos.py --all-pending
"""

import os
import sys
import click
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置
CLIPS_DIR = Path("raw_data/videos/clips")
DATASET_OUTPUT = Path("dataset_output")


def get_video_clips(video_name: str) -> list:
    """获取视频的所有片段"""
    clips_path = CLIPS_DIR / video_name
    if not clips_path.exists():
        return []
    return sorted(clips_path.glob("*.mp4"))


def label_clip(clip_path: str, workers: int, use_rag: bool, api_key: str) -> tuple:
    """标注单个片段"""
    cmd = [
        sys.executable, "scripts/video_to_dataset_async.py",
        "--video", clip_path,
        "--workers", str(workers),
    ]
    if use_rag:
        cmd.append("--rag")
    
    env = os.environ.copy()
    env["ZAI_API_KEY"] = api_key
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return (clip_path, result.returncode == 0, result.stdout, result.stderr)
    except Exception as e:
        return (clip_path, False, "", str(e))


def consolidate_and_zip(video_name: str) -> bool:
    """整合并打包"""
    cmd = [
        sys.executable, "scripts/generate_dataset_info.py",
        video_name, "--consolidate", "--zip"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"  ❌ 整合失败: {e}")
        return False


@click.command()
@click.argument('videos', nargs=-1)
@click.option('--workers', type=int, default=4, help='每个进程的并发数 (默认 4)')
@click.option('--parallel', type=int, default=2, help='并行进程数 (默认 2)')
@click.option('--rag/--no-rag', default=True, help='是否启用 RAG (默认启用)')
@click.option('--all-pending', is_flag=True, help='处理所有待处理的视频')
@click.option('--skip-labeled', is_flag=True, default=True, help='跳过已标注的片段 (默认启用)')
def main(videos, workers, parallel, rag, all_pending, skip_labeled):
    """批量标注视频并自动整合打包"""
    
    # 获取 API Key
    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        print("❌ 请先设置 ZAI_API_KEY 环境变量")
        print("   export ZAI_API_KEY='your_key_here'")
        return
    
    # 确定要处理的视频
    if all_pending:
        # 查找所有有 clips 但没有完整 dataset 的视频
        videos = []
        for d in CLIPS_DIR.iterdir():
            if d.is_dir():
                dataset_dir = DATASET_OUTPUT / f"{d.name}_dataset"
                if not dataset_dir.exists() or not (dataset_dir / f"{d.name}_dataset_info.txt").exists():
                    videos.append(d.name)
        videos = sorted(videos)
    
    if not videos:
        print("❌ 请指定要处理的视频，如: D3 D4 D5 D6")
        return
    
    print("=" * 60)
    print("🚀 批量视频标注")
    print("=" * 60)
    print(f"视频: {', '.join(videos)}")
    print(f"并行: {parallel} 进程 x {workers} workers = {parallel * workers} 并发")
    print(f"RAG: {'启用' if rag else '禁用'}")
    print()
    
    for video_name in videos:
        print(f"\n{'='*60}")
        print(f"📹 处理: {video_name}")
        print("=" * 60)
        
        # 获取片段
        clips = get_video_clips(video_name)
        if not clips:
            print(f"  ⚠️ 未找到 {video_name} 的片段")
            continue
        
        print(f"  📁 共 {len(clips)} 个片段")
        
        # 检查已完成的
        if skip_labeled:
            pending_clips = []
            for clip in clips:
                clip_name = clip.stem
                dataset_dir = DATASET_OUTPUT / f"{clip_name}_dataset"
                if not dataset_dir.exists():
                    pending_clips.append(clip)
            
            if len(pending_clips) < len(clips):
                print(f"  ⏭️ 跳过 {len(clips) - len(pending_clips)} 个已完成")
            clips = pending_clips
        
        if not clips:
            print(f"  ✅ 所有片段已完成，跳过标注")
        else:
            # 并行标注
            print(f"  🏷️ 开始标注 {len(clips)} 个片段...")
            
            success = 0
            failed = 0
            
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {
                    executor.submit(label_clip, str(clip), workers, rag, api_key): clip
                    for clip in clips
                }
                
                for future in as_completed(futures):
                    clip = futures[future]
                    clip_path, ok, stdout, stderr = future.result()
                    clip_name = Path(clip_path).stem
                    
                    if ok:
                        success += 1
                        print(f"    ✅ [{success + failed}/{len(clips)}] {clip_name}")
                    else:
                        failed += 1
                        print(f"    ❌ [{success + failed}/{len(clips)}] {clip_name}")
            
            print(f"  📊 标注完成: {success} 成功, {failed} 失败")
        
        # 整合并打包
        print(f"  📦 整合并打包...")
        if consolidate_and_zip(video_name):
            print(f"  ✅ {video_name}_dataset 已生成")
        else:
            print(f"  ⚠️ 整合失败")
    
    print(f"\n{'='*60}")
    print("🎉 全部完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

