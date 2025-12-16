#!/usr/bin/env python3
"""
极速视频切分脚本 - 使用 FFmpeg 流复制模式
按时间切分视频，保持原始帧率和画质，用于后续标注流程
"""
import subprocess
import os
import click
from pathlib import Path
import shutil


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-o', '--output-dir', type=click.Path(), default=None, help='输出目录 (默认: raw_data/videos/clips/<VideoName>)')
@click.option('--segment-time', type=float, default=33.33, help='每个切片的时长（秒），默认 33.33 秒 (对应 3FPS 标注约 100 帧)')
@click.option('--min-ratio', type=float, default=0.3, help='最后一个片段的最小时长比例，低于此值自动删除 (默认: 0.3 即 30%)')
@click.option('--prefix', type=str, default=None, help='输出文件前缀 (默认: 原文件名)')
def split_video(input_path, output_dir, segment_time, min_ratio, prefix):
    """
    [极速版] 使用 FFmpeg 流复制模式切分视频
    
    按时间切分，保持原始帧率和画质，不重新编码。
    
    示例: python scripts/split_video.py raw_data/videos/raw_videos/D1.mp4
    """
    input_path = Path(input_path)
    
    if output_dir is None:
        output_dir = Path("raw_data/videos/clips") / input_path.stem
    else:
        output_dir = Path(output_dir)
    
    # 清空或创建目录
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if prefix is None:
        prefix = input_path.stem

    print(f"🚀 启动 FFmpeg 极速切分（流复制模式）...")
    print(f"输入: {input_path}")
    print(f"切片时长: {segment_time:.2f} 秒 (标注 3FPS 时约 {int(segment_time * 3)} 帧)")
    print(f"输出: {output_dir}")

    # FFmpeg 命令 - 流复制模式，极速
    output_pattern = str(output_dir / f"{prefix}_%03d.mp4")
    
    cmd = [
        'ffmpeg',
        '-y',                       # 覆盖输出文件不询问
        '-i', str(input_path),      # 输入文件
        '-c', 'copy',               # 流复制，不重新编码（极速）
        '-an',                      # 去除音频
        '-f', 'segment',            # 分段模式
        '-segment_time', str(segment_time),  # 切片时长
        '-reset_timestamps', '1',   # 每个切片时间戳归零
        output_pattern              # 输出文件名模板
    ]

    try:
        # 执行命令
        print("\n" + "=" * 50)
        subprocess.run(cmd, check=True)
        print("=" * 50)
        
        # 后处理：清理过短的尾巴文件
        generated_files = sorted(output_dir.glob("*.mp4"))
        total_segments = len(generated_files)
        
        if generated_files and len(generated_files) > 1:
            last_file = generated_files[-1]
            
            # 用 ffprobe 检查最后一个文件时长
            check_cmd = [
                'ffprobe', 
                '-v', 'error', 
                '-show_entries', 'format=duration', 
                '-of', 'default=noprint_wrappers=1:nokey=1', 
                str(last_file)
            ]
            result = subprocess.run(check_cmd, capture_output=True, text=True)
            
            try:
                duration = float(result.stdout.strip())
                min_duration = segment_time * min_ratio
                # 如果时长小于预期时长的 min_ratio，则删除
                if duration < min_duration:
                    os.remove(last_file)
                    print(f"\n⚠️ 已删除过短的末尾片段: {last_file.name} (时长 {duration:.2f}s < {min_duration:.2f}s)")
                    total_segments -= 1
            except ValueError:
                pass
        
        print(f"\n✅ 完成！共生成 {total_segments} 个有效片段，保存在: {output_dir}")
        
        # 列出生成的文件
        final_files = sorted(output_dir.glob("*.mp4"))
        if final_files:
            print(f"\n生成的片段:")
            for f in final_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  📁 {f.name} ({size_mb:.1f} MB)")

    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg 执行出错: {e}")
    except FileNotFoundError:
        print("❌ 错误: 未找到 ffmpeg。请先安装:")
        print("   Mac: brew install ffmpeg")
        print("   Linux: sudo apt install ffmpeg")


if __name__ == '__main__':
    split_video()
