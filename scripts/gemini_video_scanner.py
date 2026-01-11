#!/usr/bin/env python3
"""
🎬 Gemini 视频扫描器 - 行人横穿马路检测

直接将视频上传到 Gemini，让 AI 分析并返回精确时间戳。
支持长视频自动切片、自动截帧、坐标转换。

用法:
    # 扫描单个视频
    python scripts/gemini_video_scanner.py --video raw_data/videos/DJI_0020.MP4

    # 扫描并指定输出目录
    python scripts/gemini_video_scanner.py --video DJI_0020.MP4 --output pedestrian_results

    # 使用 Pro 模型（更精确，更贵）
    python scripts/gemini_video_scanner.py --video DJI_0020.MP4 --model gemini-2.0-flash

    # 处理长视频（自动切片）
    python scripts/gemini_video_scanner.py --video 8hour_video.mp4 --segment-minutes 10

环境变量:
    GOOGLE_API_KEY: Gemini API 密钥
"""

import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import cv2

# ============================================================================
# 进度显示工具
# ============================================================================

class ProgressTracker:
    """进度追踪器 - 显示详细的处理进度"""

    def __init__(self, total: int, desc: str = "处理中"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = time.time()
        self.detections_count = 0

    def update(self, n: int = 1, detections: int = 0):
        """更新进度"""
        self.current += n
        self.detections_count += detections
        self._display()

    def set_detections(self, count: int):
        """设置检测数"""
        self.detections_count = count

    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    def _display(self):
        """显示进度"""
        elapsed = time.time() - self.start_time
        percent = (self.current / self.total * 100) if self.total > 0 else 0

        # 计算预估剩余时间
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = self._format_time(eta)
        else:
            eta_str = "?"

        # 进度条
        bar_width = 30
        filled = int(bar_width * self.current / self.total) if self.total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)

        # 输出
        print(f"\r   [{bar}] {self.current}/{self.total} ({percent:.0f}%) | "
              f"⏱️ {self._format_time(elapsed)} | ETA: {eta_str} | "
              f"🚶 检测: {self.detections_count}", end="", flush=True)

    def finish(self, message: str = None):
        """完成进度"""
        elapsed = time.time() - self.start_time
        print()  # 换行
        if message:
            print(f"   ✅ {message} ({self._format_time(elapsed)})")


def print_stage(stage: str, current: int = None, total: int = None):
    """打印当前阶段"""
    if current is not None and total is not None:
        print(f"\n   📍 {stage} [{current}/{total}]")
    else:
        print(f"\n   📍 {stage}")


# ============================================================================
# 重试装饰器
# ============================================================================

def retry_with_backoff(max_retries: int = 3, base_delay: float = 5.0, max_delay: float = 60.0):
    """带指数退避的重试装饰器

    Args:
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        print(f"   ⚠️ 失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                        print(f"   ⏳ {delay:.1f} 秒后重试...")
                        time.sleep(delay)
                    else:
                        print(f"   ❌ 已达最大重试次数 ({max_retries + 1})")
            raise last_exception
        return wrapper
    return decorator


# ============================================================================
# 配置
# ============================================================================

# 支持的 Gemini 模型
MODEL_OPTIONS = {
    "gemini-2.0-flash": "gemini-2.0-flash",      # 快速，便宜
    "gemini-2.5-flash": "gemini-2.5-flash",      # 更强，推荐
    "gemini-2.5-pro": "gemini-2.5-pro",          # 最强，最贵
    "gemini-3-flash": "gemini-3-flash-preview",  # 最新 (preview)
    "gemini-1.5-flash": "gemini-1.5-flash",      # 旧版快速
    "gemini-1.5-pro": "gemini-1.5-pro",          # 旧版 Pro
}

DEFAULT_MODEL = "gemini-3-flash"
DEFAULT_OUTPUT_DIR = Path("pedestrian_crossing_results")

# 检测提示词 - 行人横穿马路（严格版）
DETECTION_PROMPT = """请仔细分析这段骑行视频，找出所有【行人横穿马路】的情况。

## 严格判断标准

### 必须同时满足以下条件才能标注：
1. **必须是行人**（不是骑车的人、不是摩托车）
2. **必须正在穿越车行道**（行人的身体正在车道上移动）
3. **必须能清晰看到行人在画面中**（不是模糊的远景或遮挡物）
4. **行人必须正在移动穿越**（不是站立等待）

### 绝对不要标注：
- 行人在人行道上行走
- 行人站在路边等待
- 行人已经走完横穿、站在对面路边
- 远处模糊看不清的人影
- 骑自行车或摩托车的人
- 只是路过画面边缘的行人

## 去重要求（非常重要）

- 每个行人横穿事件只报告一次，报告行人**开始进入车道的时刻**
- 如果同一个行人持续在画面中穿越，不要重复报告
- 如果多个行人同时横穿，可以分别报告，但每人只报告一次

## 输出格式

请严格按以下 JSON 列表格式输出（不要输出任何多余解释文字）：

[
  {
    "timestamp": "MM:SS.s",
    "label": "pedestrian_crossing",
    "box_2d": [xmin, ymin, xmax, ymax],
    "description": "简短描述（如：一名穿红衣的女性从左向右横穿马路）"
  }
]

## 关于 box_2d 边界框

- 格式为 [xmin, ymin, xmax, ymax]，数值范围 0-1000
- 这是归一化坐标：0 表示最左/最上，1000 表示最右/最下
- **必须准确框选正在横穿马路的行人本身**

## 注意

- timestamp 精确到 0.1 秒
- 宁可漏报，不要误报（只标注非常确定的情况）
- 如果没有发现符合条件的行人横穿，返回空数组 []
"""


# ============================================================================
# Gemini API 封装
# ============================================================================

class GeminiVideoScanner:
    """Gemini 视频扫描器"""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = MODEL_OPTIONS.get(model, model)
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化 Gemini 客户端"""
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            print(f"   ✅ Gemini 客户端已初始化 ({self.model})")
        except ImportError:
            raise ImportError("请安装 google-genai: pip install google-genai")

    def _do_upload(self, video_path: str) -> Any:
        """执行上传操作（内部方法，用于重试）"""
        video_file = self.client.files.upload(file=video_path)

        # 等待处理完成
        while video_file.state.name == "PROCESSING":
            print("   ⏳ 视频处理中...", end="\r")
            time.sleep(3)
            video_file = self.client.files.get(name=video_file.name)

        return video_file

    def upload_video(self, video_path: str, max_retries: int = 3) -> Any:
        """上传视频到 Gemini（带重试）"""
        print(f"   📤 正在上传视频: {Path(video_path).name}")
        start_time = time.time()

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                video_file = self._do_upload(video_path)
                elapsed = time.time() - start_time
                print(f"   ✅ 上传完成 ({elapsed:.1f}s)")
                return video_file
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    delay = min(5.0 * (2 ** attempt), 60.0)
                    print(f"   ⚠️ 上传失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"   ⏳ {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                else:
                    print(f"   ❌ 上传已达最大重试次数 ({max_retries + 1})")

        raise last_exception

    def _do_analyze(self, video_file: Any, prompt: str) -> str:
        """执行分析操作（内部方法，用于重试）"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=[video_file, prompt]
        )
        return response.text

    def analyze(self, video_file: Any, prompt: str = DETECTION_PROMPT, max_retries: int = 3) -> str:
        """分析视频内容（带重试）"""
        print(f"   🔍 正在分析视频...")
        start_time = time.time()

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                result = self._do_analyze(video_file, prompt)
                elapsed = time.time() - start_time
                print(f"   ✅ 分析完成 ({elapsed:.1f}s)")
                return result
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    delay = min(5.0 * (2 ** attempt), 60.0)
                    print(f"   ⚠️ 分析失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"   ⏳ {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                else:
                    print(f"   ❌ 分析已达最大重试次数 ({max_retries + 1})")

        raise last_exception

    def cleanup(self, video_file: Any):
        """清理云端文件"""
        try:
            self.client.files.delete(name=video_file.name)
            print(f"   🗑️ 已清理云端临时文件")
        except Exception as e:
            print(f"   ⚠️ 清理失败: {e}")


# ============================================================================
# 视频处理工具
# ============================================================================

def get_video_info(video_path: str) -> Dict[str, Any]:
    """获取视频信息"""
    cap = cv2.VideoCapture(video_path)

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_sec": 0
    }

    if info["fps"] > 0:
        info["duration_sec"] = info["frame_count"] / info["fps"]

    cap.release()
    return info


def split_video(video_path: str, segment_minutes: float, output_dir: Path) -> List[Path]:
    """将长视频切分成多个片段"""
    segment_seconds = segment_minutes * 60

    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem
    output_pattern = str(output_dir / f"{video_name}_seg_%03d.mp4")

    print(f"\n📹 切分视频 (每段 {segment_minutes} 分钟)")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(segment_seconds),
        "-reset_timestamps", "1",
        output_pattern
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        segments = sorted(output_dir.glob(f"{video_name}_seg_*.mp4"))
        print(f"   ✅ 切分为 {len(segments)} 个片段")
        return segments
    except subprocess.CalledProcessError as e:
        print(f"   ❌ 切分失败: {e}")
        return []


def parse_timestamp(timestamp: str) -> float:
    """解析时间戳为毫秒"""
    try:
        parts = timestamp.replace(",", ".").split(":")
        if len(parts) == 2:
            # MM:SS.s 格式
            minutes = int(parts[0])
            seconds = float(parts[1])
            return (minutes * 60 + seconds) * 1000
        elif len(parts) == 3:
            # HH:MM:SS.s 格式
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return (hours * 3600 + minutes * 60 + seconds) * 1000
        else:
            return float(timestamp) * 1000
    except Exception:
        return 0


def extract_frame(video_path: str, timestamp_ms: float, output_path: str,
                  box: List[int] = None, video_info: Dict = None) -> bool:
    """从视频中提取指定时间点的帧"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)

    success, frame = cap.read()
    cap.release()

    if not success:
        return False

    # 如果有边界框，绘制标注（暂时禁用，Gemini 定位不够精确）
    # if box and video_info:
    #     h, w = video_info["height"], video_info["width"]
    #     # box 格式: [xmin, ymin, xmax, ymax] (归一化 0-1000)
    #     xmin = int(box[0] * w / 1000)
    #     ymin = int(box[1] * h / 1000)
    #     xmax = int(box[2] * w / 1000)
    #     ymax = int(box[3] * h / 1000)
    #
    #     # 绘制矩形框（红色，用于高风险场景）
    #     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
    #
    #     # 添加标签
    #     label = "Pedestrian Crossing"
    #     cv2.putText(frame, label, (xmin, ymin - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imwrite(output_path, frame)
    return True


def crop_region(video_path: str, timestamp_ms: float, box: List[int],
                output_path: str, video_info: Dict) -> bool:
    """从视频中裁剪指定区域"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)

    success, frame = cap.read()
    cap.release()

    if not success:
        return False

    h, w = video_info["height"], video_info["width"]
    # box 格式: [xmin, ymin, xmax, ymax] (归一化 0-1000)
    xmin = int(box[0] * w / 1000)
    ymin = int(box[1] * h / 1000)
    xmax = int(box[2] * w / 1000)
    ymax = int(box[3] * h / 1000)

    # 裁剪
    crop = frame[ymin:ymax, xmin:xmax]

    if crop.size > 0:
        cv2.imwrite(output_path, crop)
        return True
    return False


def extract_clip(video_path: str, timestamp_ms: float, output_path: str,
                 duration_before: float = 2.0, duration_after: float = 2.0) -> bool:
    """从视频中提取指定时间点前后的片段

    Args:
        video_path: 视频文件路径
        timestamp_ms: 检测时间戳（毫秒）
        output_path: 输出视频片段路径
        duration_before: 时间戳前的秒数（默认2秒）
        duration_after: 时间戳后的秒数（默认2秒）

    Returns:
        是否成功提取
    """
    timestamp_sec = timestamp_ms / 1000
    start_sec = max(0, timestamp_sec - duration_before)
    duration = duration_before + duration_after

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path)
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️ 视频片段提取失败: {e}")
        return False


# ============================================================================
# 结果处理
# ============================================================================

def parse_ai_response(response_text: str) -> List[Dict]:
    """解析 AI 返回的 JSON"""
    try:
        # 提取 JSON 部分
        start = response_text.find("[")
        end = response_text.rfind("]") + 1

        if start == -1 or end == 0:
            print(f"   ⚠️ 未找到 JSON 数组")
            return []

        json_str = response_text[start:end]
        detections = json.loads(json_str)
        return detections

    except json.JSONDecodeError as e:
        print(f"   ⚠️ JSON 解析失败: {e}")
        print(f"   原始输出: {response_text[:500]}...")
        return []


def deduplicate_detections(detections: List[Dict], min_interval_sec: float = 4.0) -> List[Dict]:
    """去重：移除时间间隔小于 min_interval_sec 的重复检测

    Args:
        detections: 检测结果列表
        min_interval_sec: 最小时间间隔（秒），默认4秒

    Returns:
        去重后的检测结果
    """
    if not detections:
        return []

    # 解析时间戳并排序
    for det in detections:
        det["_timestamp_ms"] = parse_timestamp(det.get("timestamp", "00:00"))

    sorted_dets = sorted(detections, key=lambda x: x["_timestamp_ms"])

    # 去重
    result = []
    last_timestamp_ms = -float('inf')

    for det in sorted_dets:
        current_ms = det["_timestamp_ms"]
        if (current_ms - last_timestamp_ms) >= min_interval_sec * 1000:
            # 清理临时字段
            det_clean = {k: v for k, v in det.items() if not k.startswith("_")}
            result.append(det_clean)
            last_timestamp_ms = current_ms
        else:
            print(f"   🔄 去重: 跳过 {det.get('timestamp')} (距上一个检测 < {min_interval_sec}s)")

    return result


def convert_coordinates(detection: Dict, video_info: Dict) -> Dict:
    """转换坐标：添加像素坐标"""
    result = detection.copy()

    norm_box = detection.get("box_2d")
    if norm_box and len(norm_box) == 4:
        h, w = video_info["height"], video_info["width"]

        # norm_box 格式: [xmin, ymin, xmax, ymax] (归一化 0-1000)
        result["normalized_box"] = norm_box
        result["pixel_box"] = [
            int(norm_box[0] * w / 1000),  # xmin
            int(norm_box[1] * h / 1000),  # ymin
            int(norm_box[2] * w / 1000),  # xmax
            int(norm_box[3] * h / 1000),  # ymax
        ]
        result["resolution"] = {"width": w, "height": h}

    return result


def process_detections(detections: List[Dict], video_path: str,
                       output_dir: Path, video_info: Dict,
                       segment_offset_sec: float = 0) -> List[Dict]:
    """处理检测结果：截帧、视频片段提取、坐标转换、保存"""
    images_dir = output_dir / "images"
    clips_dir = output_dir / "clips"
    images_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    processed = []

    for i, det in enumerate(detections):
        # 解析时间戳
        timestamp = det.get("timestamp", "00:00")
        timestamp_ms = parse_timestamp(timestamp)

        # 如果是分段视频，需要加上偏移
        if segment_offset_sec > 0:
            total_sec = timestamp_ms / 1000 + segment_offset_sec
            minutes = int(total_sec // 60)
            seconds = total_sec % 60
            det["original_timestamp"] = timestamp
            det["timestamp"] = f"{minutes:02d}:{seconds:05.2f}"
            timestamp_ms = total_sec * 1000

        # 转换坐标
        result = convert_coordinates(det, video_info)

        # 计算帧序号
        if video_info["fps"] > 0:
            result["frame_number"] = int(timestamp_ms / 1000 * video_info["fps"])

        # 生成文件名
        safe_timestamp = timestamp.replace(":", "_").replace(".", "_")
        filename = f"ped_{safe_timestamp}_{i:03d}"

        # 截取帧
        frame_path = images_dir / f"{filename}.jpg"
        box = det.get("box_2d")
        if extract_frame(video_path, timestamp_ms, str(frame_path), box, video_info):
            result["saved_frame"] = str(frame_path)

        # 提取视频片段（前后各2秒，共4秒）
        clip_path = clips_dir / f"{filename}.mp4"
        if extract_clip(video_path, timestamp_ms, str(clip_path)):
            result["saved_clip"] = str(clip_path)

        processed.append(result)

        # 输出进度
        print(f"   🚶 [{result['timestamp']}] {det.get('description', '')[:40]}")

    return processed


# ============================================================================
# 断点续传支持
# ============================================================================

def load_progress(output_dir: Path) -> Dict:
    """加载处理进度"""
    progress_file = output_dir / ".progress.json"
    if progress_file.exists():
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_progress(output_dir: Path, progress: Dict):
    """保存处理进度"""
    progress_file = output_dir / ".progress.json"
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def clear_progress(output_dir: Path):
    """清除进度文件（处理完成后）"""
    progress_file = output_dir / ".progress.json"
    if progress_file.exists():
        progress_file.unlink()


# ============================================================================
# 主流程
# ============================================================================

def scan_video(video_path: str, output_dir: Path, model: str = DEFAULT_MODEL,
               segment_minutes: float = None, resume: bool = True) -> Dict:
    """扫描单个视频

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        model: 使用的模型
        segment_minutes: 切片时长（分钟）
        resume: 是否启用断点续传（默认启用）

    Returns:
        扫描结果
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("请设置 GOOGLE_API_KEY 环境变量")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")

    # 获取视频信息
    video_info = get_video_info(str(video_path))
    duration_min = video_info["duration_sec"] / 60

    print(f"\n{'='*60}")
    print(f"🎬 Gemini 视频扫描器 - 行人横穿检测")
    print(f"{'='*60}")
    print(f"   视频: {video_path.name}")
    print(f"   时长: {duration_min:.1f} 分钟")
    print(f"   分辨率: {video_info['width']}x{video_info['height']}")
    print(f"   模型: {model}")
    print(f"   输出: {output_dir}")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_detections = []

    # 加载之前的进度（断点续传）
    progress = load_progress(output_dir) if resume else {}
    completed_segments = set(progress.get("completed_segments", []))
    if completed_segments and resume:
        all_detections = progress.get("detections", [])
        print(f"   📂 发现之前的进度：已完成 {len(completed_segments)} 个片段，{len(all_detections)} 个检测")

    # 判断是否需要切片
    if segment_minutes and duration_min > segment_minutes:
        # 长视频：切片处理
        temp_dir = Path(tempfile.mkdtemp())
        segments = split_video(str(video_path), segment_minutes, temp_dir)

        scanner = GeminiVideoScanner(api_key, model)
        segment_duration_sec = segment_minutes * 60

        # 创建分段进度追踪器
        segment_progress = ProgressTracker(len(segments), "分段处理")
        print(f"\n   📊 共 {len(segments)} 个片段待处理")

        for idx, segment_path in enumerate(segments):
            segment_key = f"seg_{idx}"

            # 检查是否已处理（断点续传）
            if segment_key in completed_segments:
                print(f"\n   ⏭️ 跳过已处理片段 [{idx+1}/{len(segments)}]: {segment_path.name}")
                segment_progress.update(1)
                continue

            print(f"\n   📹 片段 [{idx+1}/{len(segments)}]: {segment_path.name}")

            try:
                print_stage("上传视频")
                video_file = scanner.upload_video(str(segment_path))

                print_stage("AI 分析")
                response = scanner.analyze(video_file)
                scanner.cleanup(video_file)

                detections = parse_ai_response(response)
                detections = deduplicate_detections(detections)  # 去重
                offset_sec = idx * segment_duration_sec

                print_stage("提取截图和片段")
                processed = process_detections(
                    detections, str(video_path), output_dir,
                    video_info, segment_offset_sec=offset_sec
                )
                all_detections.extend(processed)

                # 更新进度
                segment_progress.update(1, len(processed))
                print(f"   ✅ 片段完成: 检测到 {len(processed)} 个行人横穿")

                # 保存进度
                completed_segments.add(segment_key)
                save_progress(output_dir, {
                    "video": str(video_path),
                    "completed_segments": list(completed_segments),
                    "detections": all_detections,
                    "last_update": datetime.now().isoformat()
                })

            except Exception as e:
                print(f"   ❌ 片段处理失败: {e}")
                # 保存当前进度，以便下次续传
                save_progress(output_dir, {
                    "video": str(video_path),
                    "completed_segments": list(completed_segments),
                    "detections": all_detections,
                    "last_update": datetime.now().isoformat(),
                    "last_error": str(e)
                })

        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)

    else:
        # 短视频：直接处理
        scanner = GeminiVideoScanner(api_key, model)

        try:
            print_stage("上传视频")
            video_file = scanner.upload_video(str(video_path))

            print_stage("AI 分析")
            response = scanner.analyze(video_file)
            scanner.cleanup(video_file)

            detections = parse_ai_response(response)
            detections = deduplicate_detections(detections)  # 去重

            print_stage("提取截图和片段")
            all_detections = process_detections(
                detections, str(video_path), output_dir, video_info
            )

        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            raise

    # 保存结果
    result = {
        "video": str(video_path),
        "video_info": video_info,
        "model": model,
        "scan_time": datetime.now().isoformat(),
        "total_detections": len(all_detections),
        "detections": all_detections
    }

    # 保存 JSON 报告
    report_path = output_dir / "scan_result.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 清除进度文件（处理完成）
    clear_progress(output_dir)

    # 打印统计
    print(f"\n{'='*60}")
    print(f"📊 扫描完成")
    print(f"{'='*60}")
    print(f"   🚶 行人横穿: {result['total_detections']} 次")
    print(f"   📁 结果保存: {output_dir}")
    print(f"{'='*60}")

    return result


def scan_videos_batch(video_paths: List[str], output_base_dir: Path,
                      model: str = DEFAULT_MODEL, segment_minutes: float = None,
                      auto_segment_threshold_gb: float = 1.5, resume: bool = True) -> Dict:
    """批量扫描多个视频

    Args:
        video_paths: 视频文件路径列表
        output_base_dir: 输出基础目录
        model: 使用的模型
        segment_minutes: 切片时长（分钟），None 表示自动检测
        auto_segment_threshold_gb: 自动切片的文件大小阈值（GB）
        resume: 是否启用断点续传

    Returns:
        批量处理结果汇总
    """
    total_videos = len(video_paths)
    batch_start_time = time.time()

    print(f"\n{'='*60}")
    print(f"🎬 批量视频扫描")
    print(f"{'='*60}")
    print(f"   📁 视频数量: {total_videos}")
    print(f"   🤖 模型: {model}")
    print(f"   📂 输出目录: {output_base_dir}")
    print(f"{'='*60}")

    # 列出所有待处理视频
    print(f"\n   📋 待处理视频列表:")
    for i, vp in enumerate(video_paths, 1):
        vp = Path(vp)
        if vp.exists():
            size_mb = vp.stat().st_size / (1024 ** 2)
            print(f"      {i:2d}. {vp.name} ({size_mb:.0f} MB)")
        else:
            print(f"      {i:2d}. {vp.name} (不存在)")

    results = []
    failed = []
    total_detections = 0

    # 创建进度追踪器
    progress = ProgressTracker(total_videos, "批量处理")

    for idx, video_path in enumerate(video_paths, 1):
        video_path = Path(video_path)

        print(f"\n{'─'*60}")
        print(f"📹 视频 [{idx}/{total_videos}]: {video_path.name}")
        print(f"{'─'*60}")

        if not video_path.exists():
            print(f"   ⚠️ 跳过: 文件不存在")
            failed.append({"video": str(video_path), "error": "文件不存在"})
            progress.update(1)
            continue

        # 显示文件信息
        file_size_gb = video_path.stat().st_size / (1024 ** 3)
        print(f"   📏 文件大小: {file_size_gb:.2f} GB")

        # 确定输出目录
        output_dir = output_base_dir / video_path.stem

        # 自动检测是否需要分段
        actual_segment_minutes = segment_minutes
        if segment_minutes is None:
            if file_size_gb > auto_segment_threshold_gb:
                actual_segment_minutes = 5.0  # 默认 5 分钟切片
                print(f"   ✂️ 自动启用 {actual_segment_minutes} 分钟切片 (文件 > {auto_segment_threshold_gb} GB)")

        try:
            result = scan_video(str(video_path), output_dir, model, actual_segment_minutes, resume=resume)
            results.append(result)
            video_detections = result.get("total_detections", 0)
            total_detections += video_detections
            progress.update(1, video_detections)

            # 显示单视频结果
            print(f"\n   📊 本视频检测: {video_detections} 个")

        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            failed.append({"video": str(video_path), "error": str(e)})
            progress.update(1)

        # 显示累计进度
        elapsed = time.time() - batch_start_time
        print(f"\n   📈 累计进度: {idx}/{total_videos} 视频 | {total_detections} 检测 | 耗时 {elapsed/60:.1f} 分钟")

    # 完成
    progress.finish("批量处理完成")
    total_elapsed = time.time() - batch_start_time

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"📊 批量处理汇总")
    print(f"{'='*60}")
    print(f"   ✅ 成功: {len(results)}/{total_videos} 个视频")
    print(f"   🚶 总检测数: {total_detections}")
    print(f"   ⏱️ 总耗时: {total_elapsed/60:.1f} 分钟")
    if total_videos > 0:
        print(f"   📊 平均: {total_elapsed/60/total_videos:.1f} 分钟/视频")
    if failed:
        print(f"   ❌ 失败: {len(failed)} 个视频")
        for f in failed:
            print(f"      - {Path(f['video']).name}: {f['error'][:50]}")
    print(f"   📁 结果目录: {output_base_dir}")
    print(f"{'='*60}")

    # 保存批量处理汇总
    batch_result = {
        "batch_time": datetime.now().isoformat(),
        "total_videos": len(video_paths),
        "successful": len(results),
        "failed": len(failed),
        "total_detections": total_detections,
        "model": model,
        "results": results,
        "failures": failed
    }

    summary_path = output_base_dir / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(batch_result, f, ensure_ascii=False, indent=2)

    return batch_result


def find_videos_in_dir(directory: str, extensions: List[str] = None) -> List[Path]:
    """在目录中查找视频文件"""
    if extensions is None:
        extensions = [".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV"]

    directory = Path(directory)
    videos = []

    for ext in extensions:
        videos.extend(directory.glob(f"*{ext}"))

    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(
        description="Gemini 视频扫描器 - 检测行人横穿马路",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 扫描单个视频
  python scripts/gemini_video_scanner.py --video raw_data/videos/DJI_0020.MP4

  # 批量扫描多个视频
  python scripts/gemini_video_scanner.py --videos video1.mp4 video2.mp4 video3.mp4

  # 扫描整个目录
  python scripts/gemini_video_scanner.py --video-dir /path/to/videos/

  # 使用更精确的模型
  python scripts/gemini_video_scanner.py --video video.mp4 --model gemini-2.5-pro

  # 处理长视频（自动切片，每段10分钟）
  python scripts/gemini_video_scanner.py --video long_video.mp4 --segment-minutes 10

  # 自动检测大文件并切片（默认 > 1.5GB）
  python scripts/gemini_video_scanner.py --video-dir /videos/ --auto-segment

环境变量:
  GOOGLE_API_KEY: Gemini API 密钥（必需）
        """
    )

    # 输入选项（互斥）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", "-v", type=str,
                             help="单个视频文件路径")
    input_group.add_argument("--videos", nargs="+", type=str,
                             help="多个视频文件路径")
    input_group.add_argument("--video-dir", type=str,
                             help="视频目录路径（扫描目录下所有视频）")

    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出目录 (默认: pedestrian_crossing_results/<视频名>)")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        choices=list(MODEL_OPTIONS.keys()),
                        help=f"Gemini 模型 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--segment-minutes", type=float, default=None,
                        help="长视频切片时长（分钟），不指定则不切片")
    parser.add_argument("--auto-segment", action="store_true",
                        help="自动检测大文件并切片（> 1.5GB）")
    parser.add_argument("--auto-segment-threshold", type=float, default=1.5,
                        help="自动切片的文件大小阈值（GB，默认 1.5）")
    parser.add_argument("--no-resume", action="store_true",
                        help="禁用断点续传（从头开始处理）")

    args = parser.parse_args()

    # 确定输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    try:
        resume = not args.no_resume

        if args.video:
            # 单个视频模式
            video_output = output_dir / Path(args.video).stem if not args.output else output_dir
            segment_minutes = args.segment_minutes
            if args.auto_segment and segment_minutes is None:
                file_size_gb = Path(args.video).stat().st_size / (1024 ** 3)
                if file_size_gb > args.auto_segment_threshold:
                    segment_minutes = 5.0
                    print(f"📏 文件大小 {file_size_gb:.2f} GB，自动启用 5 分钟切片")
            scan_video(args.video, video_output, args.model, segment_minutes, resume=resume)

        elif args.videos:
            # 多个视频模式
            output_dir.mkdir(parents=True, exist_ok=True)
            segment_minutes = args.segment_minutes if not args.auto_segment else None
            scan_videos_batch(
                args.videos, output_dir, args.model, segment_minutes,
                auto_segment_threshold_gb=args.auto_segment_threshold,
                resume=resume
            )

        elif args.video_dir:
            # 目录扫描模式
            videos = find_videos_in_dir(args.video_dir)
            if not videos:
                print(f"❌ 目录中没有找到视频文件: {args.video_dir}")
                sys.exit(1)
            print(f"📁 在目录中找到 {len(videos)} 个视频文件")
            output_dir.mkdir(parents=True, exist_ok=True)
            segment_minutes = args.segment_minutes if not args.auto_segment else None
            scan_videos_batch(
                [str(v) for v in videos], output_dir, args.model, segment_minutes,
                auto_segment_threshold_gb=args.auto_segment_threshold,
                resume=resume
            )

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
