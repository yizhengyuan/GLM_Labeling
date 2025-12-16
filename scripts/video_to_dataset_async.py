#!/usr/bin/env python3
"""
🚀 异步版本 - 视频到数据集流水线

使用 asyncio + httpx 实现真正的并发 API 请求，速度更快。

用法:
    python3 scripts/video_to_dataset_async.py --video D4.1 --workers 15
"""

import os
import sys
import json
import argparse
import subprocess
import time
import asyncio
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional

import httpx
from PIL import Image

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from glm_labeling.utils.labels import get_category, normalize_vehicle_label
from glm_labeling.utils.json_utils import parse_llm_json


# ============================================================================
# 配置
# ============================================================================

API_BASE_URL = "https://api.z.ai/api/paas/v4"
MODEL_NAME = "glm-4.6v"
COORD_BASE = 1000  # GLM 输出坐标基数

VIDEO_DIR = Path("raw_data/videos/clips")  # 默认查找切分后的片段
TEMP_FRAMES_DIR = Path("temp_frames")
OUTPUT_BASE = Path("output")

# 188 种交通标志候选库
SIGNS_DIR = Path("raw_data/images/signs/highres/png2560px")

def load_sign_candidates():
    """从标志图片目录动态加载所有标志名称（188种）"""
    if not SIGNS_DIR.exists():
        print(f"⚠️ 找不到标志目录: {SIGNS_DIR}")
        return []
    return [f.stem for f in sorted(SIGNS_DIR.glob("*.png"))]

ALL_SIGN_CANDIDATES = load_sign_candidates()

COLORS = {
    'pedestrian': (255, 0, 0),
    'vehicle': (0, 255, 0),
    'traffic_sign': (0, 100, 255),
    'construction': (255, 165, 0),
}

DETECTION_PROMPT = """请检测图片中的以下4类物体，返回JSON格式。

## 重要排除规则：
⛔ 不要标注第一人称视角下自己骑的车（摩托车/电动车/自行车的车把、仪表盘、手臂等）！

## 检测类别与细粒度要求：

### 1. 行人类 (pedestrian) - 2种标签
- pedestrian: 单个或少量行人
- crowd: 人群（多人聚集）

### 2. 车辆类 (vehicle) - 5种标签
统一使用 vehicle，只区分行驶状态：

**🚨 状态判断规则（核心：关注尾灯！按优先级）：**
1. **刹车状态**: 尾灯明显变亮、红色刹车灯亮起 → `vehicle_braking`
2. **双闪状态**: 左右两侧转向灯同时亮起/闪烁 → `vehicle_double_flash`
3. **右转状态**: 右侧转向灯亮（黄色/琥珀色）或明显右转弯 → `vehicle_turning_right`
4. **左转状态**: 左侧转向灯亮（黄色/琥珀色）或明显左转弯 → `vehicle_turning_left`
5. **正常状态**: 直行或无灯光信号 → `vehicle`

⚠️ 注意：仅道路弯曲但车辆正常行驶、没有打灯 → 标为 `vehicle`（直行）

### 3. 交通标志类 (traffic_sign)
traffic_sign

### 4. 施工标志类 (construction)
traffic_cone, construction_barrier

## 返回格式示例：
[
  {"label": "vehicle_braking", "bbox_2d": [100, 200, 300, 400]},
  {"label": "vehicle_double_flash", "bbox_2d": [400, 300, 600, 500]},
  {"label": "traffic_sign", "bbox_2d": [50, 50, 80, 80]}
]

如果没有目标，返回 []
只返回JSON数组！"""


# ============================================================================
# 工具函数
# ============================================================================

import re
import uuid


def image_to_base64_url(image_path: str) -> str:
    """将图片转为 base64 data URL"""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    
    ext = Path(image_path).suffix.lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}
    mime_type = mime.get(ext.lstrip("."), "jpeg")
    
    return f"data:image/{mime_type};base64,{data}"


def get_image_size(image_path: str) -> tuple:
    """获取图片尺寸"""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def convert_coords(bbox: List[int], width: int, height: int) -> List[int]:
    """将 GLM 归一化坐标 (0-1000) 转为像素坐标"""
    x1, y1, x2, y2 = bbox
    return [
        int(x1 * width / COORD_BASE),
        int(y1 * height / COORD_BASE),
        int(x2 * width / COORD_BASE),
        int(y2 * height / COORD_BASE)
    ]


def to_xanylabeling_format(detections: List[Dict], image_path: str) -> Dict:
    """转换为 X-AnyLabeling 格式"""
    width, height = get_image_size(image_path)
    
    shapes = []
    for det in detections:
        bbox = det["bbox"]
        shapes.append({
            "label": det["label"],
            "points": [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
            "shape_type": "rectangle",
            "flags": {"category": det["category"]}
        })
    
    return {
        "version": "0.4.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": Path(image_path).name,
        "imageHeight": height,
        "imageWidth": width
    }


# ============================================================================
# 异步 API 调用
# ============================================================================

class AsyncDetector:
    """异步目标检测器"""
    
    def __init__(self, api_key: str, max_concurrent: int = 12, timeout: float = 45.0):
        self.api_key = api_key
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            base_url=API_BASE_URL,
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=15)
        )
        return self
    
    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()
    
    async def detect(self, image_path: str, retry: int = 3) -> tuple:
        """
        异步检测单张图片
        
        Returns:
            (detections, error)
        """
        async with self.semaphore:  # 控制并发
            return await self._detect_with_retry(image_path, retry)
    
    async def _detect_with_retry(self, image_path: str, max_retry: int) -> tuple:
        """带重试的检测"""
        image_name = Path(image_path).name
        last_error = None
        
        for attempt in range(max_retry):
            try:
                base64_url = image_to_base64_url(image_path)
                width, height = get_image_size(image_path)
                
                payload = {
                    "model": MODEL_NAME,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": base64_url}},
                            {"type": "text", "text": DETECTION_PROMPT}
                        ]
                    }]
                }
                
                response = await self.client.post("/chat/completions", json=payload)
                
                # 处理 429 限流
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", 3))
                    await asyncio.sleep(retry_after * (attempt + 1))
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                content = data["choices"][0]["message"]["content"]
                detections = parse_llm_json(content)
                
                if detections is None:
                    # JSON 解析失败，重试
                    last_error = "JSON parse error"
                    await asyncio.sleep(1)
                    continue
                
                if not detections:
                    return [], None
                
                # 后处理
                processed = []
                for det in detections:
                    if "label" not in det or "bbox_2d" not in det:
                        continue
                    
                    bbox = convert_coords(det["bbox_2d"], width, height)
                    label = det["label"].lower().replace(" ", "_").replace("-", "_")
                    category = get_category(label)
                    
                    if category == "vehicle":
                        label = normalize_vehicle_label(label)
                    
                    processed.append({
                        "label": label,
                        "category": category,
                        "bbox": bbox
                    })
                
                return processed, None
                
            except httpx.TimeoutException:
                last_error = "Timeout"
                await asyncio.sleep(2 * (attempt + 1))
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}"
                if e.response.status_code == 429:
                    await asyncio.sleep(3 * (attempt + 1))
                else:
                    await asyncio.sleep(2)
        
        return [], last_error
    
    async def classify_sign_rag(self, image_path: str, bbox: list) -> str:
        """
        RAG 交通标志精排（异步版）- 支持 188 种细粒度分类
        
        Args:
            image_path: 原图路径
            bbox: 交通标志的边界框 [x1, y1, x2, y2]
        
        Returns:
            细粒度标签，如 Speed_limit_70_km_h, No_stopping_at_any_time 等
        """
        if not ALL_SIGN_CANDIDATES:
            return "traffic_sign"
        
        temp_path = None
        
        try:
            img = Image.open(image_path)
            padding = 10
            x1 = max(0, bbox[0] - padding)
            y1 = max(0, bbox[1] - padding)
            x2 = min(img.width, bbox[2] + padding)
            y2 = min(img.height, bbox[3] + padding)
            
            sign_crop = img.crop((x1, y1, x2, y2))
            unique_id = uuid.uuid4()
            temp_path = f"/tmp/sign_crop_{unique_id}.jpg"
            sign_crop.save(temp_path, "JPEG")
            
            with open(temp_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            
            base64_url = f"data:image/jpeg;base64,{img_data}"
            
            # ================================================================
            # 阶段1：从 188 种候选中选择最匹配的标志
            # ================================================================
            candidate_list = "\n".join([f"{i+1}. {c}" for i, c in enumerate(ALL_SIGN_CANDIDATES)])
            
            select_prompt = f"""请仔细观察这个交通标志，从以下选项中选择最匹配的：

{candidate_list}

规则：
1. 观察标志的颜色、形状、文字、数字
2. 如果是限速标志，选择 "Speed_limit_(in_km_h)"
3. 如果都不匹配，返回 0

请只返回选项编号（如 1、2、3），不要其他解释。"""
            
            payload = {
                "model": MODEL_NAME,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": base64_url}},
                        {"type": "text", "text": select_prompt}
                    ]
                }],
                "temperature": 0.1
            }
            
            response = await self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            choice = response.json()["choices"][0]["message"]["content"].strip()
            
            # 解析选择
            base_label = "traffic_sign"
            numbers = re.findall(r'\d+', choice)
            if numbers:
                idx = int(numbers[0]) - 1
                if 0 <= idx < len(ALL_SIGN_CANDIDATES):
                    base_label = ALL_SIGN_CANDIDATES[idx]
            
            # ================================================================
            # 阶段2：对通用标志进一步识别具体数字
            # ================================================================
            generic_signs = {
                "Speed_limit_(in_km_h)": {
                    "question": "请识别这个限速标志上显示的具体数字（如 20, 30, 50, 70, 100）。只返回数字。",
                    "format": "Speed_limit_{}_km_h"
                },
                "Variable_speed_limit_(in_km_h)": {
                    "question": "请识别这个可变限速标志上显示的数字。只返回数字。",
                    "format": "Variable_speed_limit_{}_km_h"
                },
                "Distance_as_shown_to_hazard": {
                    "question": "请识别标志上显示的距离数字（单位：米）。只返回数字。",
                    "format": "Distance_{}_m_to_hazard"
                },
                "Maximum_height_as_shown_(in_metres)": {
                    "question": "请识别标志上显示的最大高度限制（单位：米）。只返回数字。",
                    "format": "Maximum_height_{}_m"
                },
                "Maximum_payload_as_shown_(in_tonnes)": {
                    "question": "请识别标志上显示的最大载重限制（单位：吨）。只返回数字。",
                    "format": "Maximum_payload_{}_tonnes"
                }
            }
            
            if base_label in generic_signs:
                detail_info = generic_signs[base_label]
                
                payload["messages"][0]["content"][1]["text"] = detail_info["question"]
                
                response2 = await self.client.post("/chat/completions", json=payload)
                response2.raise_for_status()
                detail_text = response2.json()["choices"][0]["message"]["content"].strip()
                
                detail_numbers = re.findall(r'\d+', detail_text)
                if detail_numbers:
                    specific_value = detail_numbers[0]
                    return detail_info["format"].format(specific_value)
            
            return base_label
            
        except Exception as e:
            return "traffic_sign"
        
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


# ============================================================================
# Step 1: 抽帧
# ============================================================================

def extract_frames(video_path: str, output_name: str, fps: int = 3) -> tuple:
    """从视频抽帧
    
    Args:
        video_path: 视频文件路径
        output_name: 输出名称（用于命名帧和目录）
        fps: 抽帧率
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"❌ 视频不存在: {video_path}")
        return None, 0
    
    frames_dir = TEMP_FRAMES_DIR / output_name
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # 清空旧帧
    for old_frame in frames_dir.glob("*.jpg"):
        old_frame.unlink()
    
    print(f"\n📹 Step 1: 抽帧 ({fps} FPS)")
    print(f"   视频: {video_path}")
    
    output_pattern = str(frames_dir / f"{output_name}_%06d.jpg")
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        output_pattern,
        "-y"
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        frame_count = len(list(frames_dir.glob("*.jpg")))
        print(f"   ✅ 抽取 {frame_count} 帧")
        return frames_dir, frame_count
    except Exception as e:
        print(f"   ❌ ffmpeg 错误: {e}")
        return None, 0


# ============================================================================
# Step 2: 异步标注
# ============================================================================

async def run_labeling_async(
    frames_dir: Path, 
    video_name: str, 
    workers: int,
    api_key: str,
    use_rag: bool = True
) -> Path:
    """异步运行标注"""
    rag_status = "✅ 启用" if use_rag else "❌ 禁用"
    print(f"\n🏷️ Step 2: 异步标注")
    print(f"   并发数: {workers} | 模式: asyncio + httpx | RAG: {rag_status}")
    
    image_files = sorted(frames_dir.glob("*.jpg"))
    if not image_files:
        print("   ❌ 没有找到帧")
        return None
    
    output_dir = OUTPUT_BASE / f"{video_name.lower()}_annotations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 过滤已处理的（断点续传）
    todo_files = []
    for img in image_files:
        json_path = output_dir / f"{img.stem}.json"
        if not json_path.exists():
            todo_files.append(img)
    
    skipped = len(image_files) - len(todo_files)
    if skipped > 0:
        print(f"   📌 断点续传: 跳过 {skipped} 张已处理")
    
    if not todo_files:
        print("   ✅ 所有帧已处理完成")
        return output_dir
    
    print(f"   📝 待处理: {len(todo_files)} 帧")
    
    start_time = time.time()
    stats = {"pedestrian": 0, "vehicle": 0, "traffic_sign": 0, "construction": 0}
    success = 0
    errors = 0
    
    async with AsyncDetector(api_key, max_concurrent=workers) as detector:
        # 创建任务并记录对应的文件
        tasks = {
            asyncio.create_task(
                detect_and_save(detector, str(img), output_dir, stats, use_rag=use_rag)
            ): img for img in todo_files
        }
        
        total = len(image_files)
        completed = 0
        
        # 实时输出：任务完成一个就输出一个
        for coro in asyncio.as_completed(tasks.keys()):
            completed += 1
            idx = skipped + completed
            
            try:
                result = await coro
                
                if result[1]:  # error
                    print(f"  ⚠️ [{idx}/{total}] {result[1]}", flush=True)
                    errors += 1
                else:
                    count = result[0]
                    emoji = "✅" if count > 0 else "⚪"
                    print(f"  {emoji} [{idx}/{total}] {count} objects", flush=True)
                    success += 1
                    
            except Exception as e:
                print(f"  ❌ [{idx}/{total}] {e}", flush=True)
                errors += 1
    
    elapsed = time.time() - start_time
    print(f"\n   📊 统计: {dict(stats)}")
    print(f"   ⏱️ 耗时: {elapsed:.1f}s ({elapsed/len(todo_files):.2f}s/帧)")
    print(f"   ✅ 成功: {success} | ❌ 错误: {errors}")
    
    return output_dir


async def detect_and_save(
    detector: AsyncDetector,
    image_path: str,
    output_dir: Path,
    stats: dict,
    use_rag: bool = True
) -> tuple:
    """检测并保存结果"""
    detections, error = await detector.detect(image_path)
    
    if error:
        return (0, error)
    
    # RAG 细粒度分类（交通标志）
    if use_rag:
        for det in detections:
            if det.get("category") == "traffic_sign" and det.get("label") in ["traffic_sign", "sign"]:
                fine_label = await detector.classify_sign_rag(image_path, det["bbox"])
                det["label"] = fine_label
    
    # 更新统计
    for det in detections:
        cat = det.get("category", "unknown")
        if cat in stats:
            stats[cat] += 1
    
    # 保存
    annotation = to_xanylabeling_format(detections, image_path)
    out_path = output_dir / f"{Path(image_path).stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)
    
    return (len(detections), None)


# ============================================================================
# Step 3: 可视化
# ============================================================================

def generate_visualizations(frames_dir: Path, annotations_dir: Path, video_name: str) -> Path:
    """生成可视化图片"""
    from PIL import ImageDraw
    
    print(f"\n🎨 Step 3: 生成可视化")
    
    vis_dir = OUTPUT_BASE / f"{video_name.lower()}_visualized"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for json_path in sorted(annotations_dir.glob("*.json")):
        frame_name = json_path.stem + ".jpg"
        frame_path = frames_dir / frame_name
        
        if not frame_path.exists():
            continue
        
        img = Image.open(frame_path)
        draw = ImageDraw.Draw(img)
        
        with open(json_path) as f:
            data = json.load(f)
        
        for shape in data.get("shapes", []):
            pts = shape["points"]
            cat = shape.get("flags", {}).get("category", "unknown")
            label = shape["label"]
            
            color = COLORS.get(cat, (128, 128, 128))
            draw.rectangle([pts[0][0], pts[0][1], pts[1][0], pts[1][1]], 
                          outline=color, width=3)
            
            short_label = label[:20] + "..." if len(label) > 20 else label
            draw.text((pts[0][0], pts[0][1] - 15), short_label, fill=color)
        
        out_path = vis_dir / f"{json_path.stem}_vis.jpg"
        img.save(out_path)
        count += 1
        
        if count % 50 == 0:
            print(f"   已处理 {count} 张...")
    
    print(f"   ✅ 生成 {count} 张可视化图片")
    return vis_dir


# ============================================================================
# Step 4: 打包 Dataset
# ============================================================================

def generate_summary(annotations_dir: Path, video_name: str, frame_count: int) -> dict:
    """分析标注数据并生成统计信息"""
    from collections import defaultdict
    
    stats = {
        "total_frames": frame_count,
        "annotated_frames": 0,
        "total_objects": 0,
        "categories": defaultdict(int),
        "subcategories": defaultdict(int),
    }
    
    for json_path in sorted(annotations_dir.glob("*.json")):
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
    
    return stats


def create_summary_markdown(stats: dict, video_name: str, fps: int, elapsed_time: float = None) -> str:
    """生成 Markdown 格式的总结文档"""
    from datetime import datetime
    
    lines = []
    lines.append(f"# 📊 数据标注总结 - {video_name}")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**抽帧率**: {fps} FPS")
    lines.append(f"**标注方式**: 异步并行 (asyncio + httpx)")
    if elapsed_time:
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        lines.append(f"**处理耗时**: {minutes}分{seconds:.1f}秒")
    lines.append("")
    
    lines.append("## 📈 概览统计")
    lines.append("")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 总帧数 | {stats['total_frames']} |")
    lines.append(f"| 有标注的帧 | {stats['annotated_frames']} |")
    lines.append(f"| 空帧（无检测） | {stats['total_frames'] - stats['annotated_frames']} |")
    lines.append(f"| 总检测对象 | {stats['total_objects']} |")
    if stats['annotated_frames'] > 0:
        lines.append(f"| 平均每帧对象数 | {stats['total_objects'] / stats['annotated_frames']:.2f} |")
    lines.append("")
    
    lines.append("## 🏷️ 主类别分布")
    lines.append("")
    lines.append(f"| 类别 | 数量 | 占比 |")
    lines.append(f"|------|------|------|")
    total = stats['total_objects'] or 1
    for cat, count in sorted(stats['categories'].items(), key=lambda x: -x[1]):
        percentage = count / total * 100
        lines.append(f"| {cat} | {count} | {percentage:.1f}% |")
    lines.append("")
    
    if stats['subcategories']:
        lines.append("## 🔍 细分类别 Top 20")
        lines.append("")
        lines.append(f"| 标签 | 数量 |")
        lines.append(f"|------|------|")
        for label, count in sorted(stats['subcategories'].items(), key=lambda x: -x[1])[:20]:
            display_label = label[:50] + "..." if len(label) > 50 else label
            lines.append(f"| {display_label} | {count} |")
        lines.append("")
    
    lines.append("---")
    lines.append(f"*此报告由 video_to_dataset_async.py 自动生成*")
    
    return "\n".join(lines)


def create_dataset(video_name: str, video_path: str, frames_dir: Path, annotations_dir: Path, vis_dir: Path, fps: int = 3, elapsed_time: float = None) -> Path:
    """创建 Dataset 文件夹"""
    import shutil
    
    print(f"\n📦 Step 4: 创建 Dataset")
    
    output_base = Path("dataset_output")
    output_base.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_base / f"{video_name}_dataset"
    
    # 清理旧的
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    
    (dataset_dir / "video").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "frames").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "visualized").mkdir(parents=True, exist_ok=True)
    
    # 复制视频
    video_src = Path(video_path)
    if video_src.exists():
        shutil.copy(video_src, dataset_dir / "video" / video_src.name)
        print(f"   ✅ 复制视频")
    
    # 复制帧
    frame_count = 0
    for frame in frames_dir.glob("*.jpg"):
        shutil.copy(frame, dataset_dir / "frames" / frame.name)
        frame_count += 1
    print(f"   ✅ 复制 {frame_count} 帧")
    
    # 复制标注
    ann_count = 0
    for ann in annotations_dir.glob("*.json"):
        shutil.copy(ann, dataset_dir / "annotations" / ann.name)
        ann_count += 1
    print(f"   ✅ 复制 {ann_count} 标注")
    
    # 复制可视化
    if vis_dir and vis_dir.exists():
        vis_count = 0
        for vis in vis_dir.glob("*.jpg"):
            shutil.copy(vis, dataset_dir / "visualized" / vis.name)
            vis_count += 1
        print(f"   ✅ 复制 {vis_count} 可视化")
    
    # 生成总结报告
    print(f"   📝 生成标注总结文档...")
    stats = generate_summary(dataset_dir / "annotations", video_name, frame_count)
    summary_md = create_summary_markdown(stats, video_name, fps, elapsed_time)
    
    summary_path = dataset_dir / "SUMMARY.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"   ✅ 生成 SUMMARY.md")
    
    # 保存 JSON 格式的统计数据
    stats_json = {
        "video_name": video_name,
        "total_frames": stats["total_frames"],
        "annotated_frames": stats["annotated_frames"],
        "total_objects": stats["total_objects"],
        "categories": dict(stats["categories"]),
        "subcategories": dict(stats["subcategories"]),
        "fps": fps
    }
    stats_path = dataset_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_json, f, ensure_ascii=False, indent=2)
    print(f"   ✅ 生成 stats.json")
    
    # 注: 不再为每个片段单独生成 zip，统一在最后整合时生成
    # 如需单独压缩，可运行: zip -r {video_name}_dataset.zip {video_name}_dataset
    
    return dataset_dir


# ============================================================================
# 主函数
# ============================================================================

async def main_async():
    parser = argparse.ArgumentParser(description="异步视频到数据集流水线")
    parser.add_argument("--video", type=str, required=True, help="视频文件路径 (如 raw_data/videos/clips/D1/D1_000.mp4)")
    parser.add_argument("--name", type=str, default=None, help="输出名称 (默认使用视频文件名)")
    parser.add_argument("--fps", type=int, default=3, help="抽帧率 (默认 3)")
    parser.add_argument("--workers", type=int, default=15, help="并发数 (默认 15)")
    parser.add_argument("--skip-extract", action="store_true", help="跳过抽帧")
    parser.add_argument("--skip-visualize", action="store_true", help="跳过可视化")
    parser.add_argument("--rag", action="store_true", default=True, help="启用 RAG 交通标志细粒度分类 (默认启用)")
    parser.add_argument("--no-rag", dest="rag", action="store_false", help="禁用 RAG 交通标志细粒度分类")
    args = parser.parse_args()
    
    video_path = Path(args.video)
    
    # 自动确定输出名称
    if args.name:
        output_name = args.name
    else:
        output_name = video_path.stem  # 如 D1_000
    
    api_key = os.getenv("ZAI_API_KEY")
    
    if not api_key:
        print("❌ 请设置 ZAI_API_KEY 环境变量")
        return
    
    if not video_path.exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    print("=" * 70)
    print(f"🚀 异步视频标注流水线 - {output_name}")
    print(f"   视频: {video_path}")
    print(f"   FPS: {args.fps} | 并发: {args.workers} | 模式: asyncio")
    print("=" * 70)
    
    start_time = time.time()
    
    # Step 1
    if args.skip_extract:
        frames_dir = TEMP_FRAMES_DIR / output_name
        print(f"\n⏭️ 跳过抽帧，使用: {frames_dir}")
    else:
        frames_dir, _ = extract_frames(str(video_path), output_name, args.fps)
        if not frames_dir:
            return
    
    # Step 2
    annotations_dir = await run_labeling_async(frames_dir, output_name, args.workers, api_key, use_rag=args.rag)
    if not annotations_dir:
        return
    
    # Step 3
    if args.skip_visualize:
        vis_dir = None
        print(f"\n⏭️ 跳过可视化")
    else:
        vis_dir = generate_visualizations(frames_dir, annotations_dir, output_name)
    
    # Step 4
    total_time = time.time() - start_time
    dataset_dir = create_dataset(output_name, str(video_path), frames_dir, annotations_dir, vis_dir, fps=args.fps, elapsed_time=total_time)
    
    print("\n" + "=" * 70)
    print(f"🎉 完成！总耗时: {total_time/60:.1f} 分钟 ({total_time:.1f}秒)")
    print(f"📁 Dataset: {dataset_dir}/")
    print("=" * 70)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

