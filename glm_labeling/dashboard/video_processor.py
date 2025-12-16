"""
视频处理模块

提供视频抽帧、打包等功能。
"""

import subprocess
import shutil
import json
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime
from collections import defaultdict


class VideoProcessor:
    """视频处理器"""
    
    # 默认视频目录
    DEFAULT_VIDEO_DIR = Path("raw_data/videos/raw_videos")
    
    def __init__(self, video_dir: Optional[str] = None):
        self.video_dir = Path(video_dir) if video_dir else self.DEFAULT_VIDEO_DIR
    
    def list_videos(self) -> list:
        """列出可用的视频文件"""
        videos = []
        if self.video_dir.exists():
            for ext in ['*.mp4', '*.MP4', '*.avi', '*.mov']:
                videos.extend(self.video_dir.glob(ext))
        return sorted([v.stem for v in videos])
    
    def get_video_path(self, video_name: str) -> Optional[Path]:
        """获取视频文件路径"""
        for ext in ['.mp4', '.MP4', '.avi', '.mov']:
            path = self.video_dir / f"{video_name}{ext}"
            if path.exists():
                return path
        return None
    
    def extract_frames(
        self,
        video_name: str,
        output_dir: Path,
        fps: int = 3,
        output_prefix: str = None,
        on_progress: Callable[[float], None] = None
    ) -> tuple:
        """
        从视频抽帧
        
        Args:
            video_name: 视频名称（不含扩展名）
            output_dir: 输出目录
            fps: 抽帧率
            output_prefix: 输出文件名前缀（默认与视频同名）
            on_progress: 进度回调 (0.0 ~ 1.0)
            
        Returns:
            (frames_dir, frame_count) 或 (None, 0) 失败时
        """
        video_path = self.get_video_path(video_name)
        if not video_path:
            return None, 0
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 清空旧帧
        for old_frame in output_dir.glob("*.jpg"):
            old_frame.unlink()
        
        prefix = output_prefix or video_name
        output_pattern = str(output_dir / f"{prefix}_%06d.jpg")
        
        # 获取视频时长（用于计算进度）
        duration = self._get_video_duration(video_path)
        
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            output_pattern,
            "-y",
            "-progress", "pipe:1"
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 解析进度
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                
                if line.startswith("out_time_ms="):
                    try:
                        current_ms = int(line.split("=")[1])
                        if duration > 0 and on_progress:
                            progress = min(current_ms / (duration * 1000000), 1.0)
                            on_progress(progress)
                    except:
                        pass
            
            process.wait()
            
            if on_progress:
                on_progress(1.0)
            
            frame_count = len(list(output_dir.glob("*.jpg")))
            return output_dir, frame_count
            
        except Exception as e:
            print(f"ffmpeg error: {e}")
            return None, 0
    
    def _get_video_duration(self, video_path: Path) -> float:
        """获取视频时长（秒）"""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return float(result.stdout.strip())
        except:
            return 0
    
    def create_dataset(
        self,
        video_name: str,
        output_name: str,
        frames_dir: Path,
        annotations_dir: Path,
        visualized_dir: Optional[Path],
        fps: int = 3,
        use_rag: bool = False,
        on_progress: Callable[[float], None] = None
    ) -> Optional[Path]:
        """
        创建数据集文件夹
        
        Args:
            video_name: 原始视频名称
            output_name: 输出数据集名称
            frames_dir: 帧目录
            annotations_dir: 标注目录
            visualized_dir: 可视化目录
            fps: 抽帧率
            use_rag: 是否使用 RAG
            on_progress: 进度回调
            
        Returns:
            数据集目录路径
        """
        output_base = Path("dataset_output")
        output_base.mkdir(parents=True, exist_ok=True)
        dataset_dir = output_base / f"{output_name}_dataset"
        
        # 创建目录结构
        (dataset_dir / "video").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "frames").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "annotations").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "visualized").mkdir(parents=True, exist_ok=True)
        
        total_steps = 5
        current_step = 0
        
        def update_progress():
            nonlocal current_step
            current_step += 1
            if on_progress:
                on_progress(current_step / total_steps)
        
        # 1. 复制视频
        video_path = self.get_video_path(video_name)
        if video_path and video_path.exists():
            shutil.copy(video_path, dataset_dir / "video" / video_path.name)
        update_progress()
        
        # 2. 复制帧
        frames_dir = Path(frames_dir)
        for frame in frames_dir.glob("*.jpg"):
            shutil.copy(frame, dataset_dir / "frames" / frame.name)
        update_progress()
        
        # 3. 复制标注
        annotations_dir = Path(annotations_dir)
        for ann in annotations_dir.glob("*.json"):
            shutil.copy(ann, dataset_dir / "annotations" / ann.name)
        update_progress()
        
        # 4. 复制可视化
        if visualized_dir:
            visualized_dir = Path(visualized_dir)
            if visualized_dir.exists():
                for vis in visualized_dir.glob("*.jpg"):
                    shutil.copy(vis, dataset_dir / "visualized" / vis.name)
        update_progress()
        
        # 5. 生成总结文档
        frame_count = len(list(frames_dir.glob("*.jpg")))
        stats = self._generate_summary(annotations_dir, output_name, frame_count, fps)
        summary_md = self._create_summary_markdown(stats, output_name, fps, use_rag)
        
        with open(dataset_dir / "SUMMARY.md", "w", encoding="utf-8") as f:
            f.write(summary_md)
        
        # 保存 JSON 统计
        stats_json = {
            "video_name": video_name,
            "output_name": output_name,
            "total_frames": stats["total_frames"],
            "annotated_frames": stats["annotated_frames"],
            "total_objects": stats["total_objects"],
            "categories": dict(stats["categories"]),
            "fps": fps,
            "use_rag": use_rag,
            "created_at": datetime.now().isoformat()
        }
        with open(dataset_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats_json, f, ensure_ascii=False, indent=2)
        
        # 创建压缩包
        zip_path = output_base / f"{output_name}_dataset.zip"
        shutil.make_archive(str(dataset_dir), 'zip', str(output_base), f"{output_name}_dataset")
        
        update_progress()
        
        return dataset_dir
    
    def _generate_summary(self, annotations_dir: Path, name: str, frame_count: int, fps: int) -> dict:
        """分析标注数据"""
        stats = {
            "total_frames": frame_count,
            "annotated_frames": 0,
            "total_objects": 0,
            "categories": defaultdict(int),
            "subcategories": defaultdict(int),
        }
        
        for json_path in annotations_dir.glob("*.json"):
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
                
                shapes = data.get("shapes", [])
                if shapes:
                    stats["annotated_frames"] += 1
                
                for shape in shapes:
                    stats["total_objects"] += 1
                    category = shape.get("flags", {}).get("category", "unknown")
                    stats["categories"][category] += 1
                    label = shape.get("label", "")
                    if label:
                        stats["subcategories"][label] += 1
            except:
                pass
        
        return stats
    
    def _create_summary_markdown(self, stats: dict, name: str, fps: int, use_rag: bool) -> str:
        """生成 Markdown 总结"""
        lines = []
        lines.append(f"# 📊 数据标注总结 - {name}")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**抽帧率**: {fps} FPS")
        lines.append(f"**RAG增强**: {'✅ 启用' if use_rag else '❌ 未启用'}")
        lines.append("")
        
        lines.append("## 📈 概览统计")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| 总帧数 | {stats['total_frames']} |")
        lines.append(f"| 有标注的帧 | {stats['annotated_frames']} |")
        lines.append(f"| 总检测对象 | {stats['total_objects']} |")
        lines.append("")
        
        lines.append("## 🏷️ 类别分布")
        lines.append("")
        lines.append("| 类别 | 数量 | 占比 |")
        lines.append("|------|------|------|")
        total = stats['total_objects'] or 1
        for cat, count in sorted(stats['categories'].items(), key=lambda x: -x[1]):
            pct = count / total * 100
            lines.append(f"| {cat} | {count} | {pct:.1f}% |")
        lines.append("")
        
        lines.append("---")
        lines.append("*由 GLM Labeling Dashboard 自动生成*")
        
        return "\n".join(lines)

