#!/usr/bin/env python3
"""
🚀 异步版本 - 视频到数据集流水线

使用 asyncio + httpx 实现真正的并发 API 请求，速度更快。

用法:
    # 基本用法 (默认 PNG 无损抽帧)
    python3 scripts/video_to_dataset_async.py --video D4.1 --workers 15

    # 使用 JPEG 抽帧（省空间）
    python3 scripts/video_to_dataset_async.py --video D4.1 --jpeg-frames

    # 使用 Gemini 模型
    python3 scripts/video_to_dataset_async.py --video D4.1 --model gemini-2.5-flash
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
from glm_labeling.core.sign_classifier_v2 import SignClassifierV2

# DINOv2 分类器（推荐，最强特征提取）
try:
    from scripts.dinov2_classifier import DINOv2SignClassifier
    DINOV2_AVAILABLE = True
except ImportError:
    DINOV2_AVAILABLE = False

# CLIP 分类器（可选）
try:
    from scripts.clip_rag_classifier import CLIPSignClassifier
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


# ============================================================================
# 配置
# ============================================================================

# 多模型配置
MODEL_CONFIGS = {
    # GLM 系列
    "glm": {
        "name": "glm-4.6v",
        "api_base": "https://api.z.ai/api/paas/v4",
        "api_key_env": "ZAI_API_KEY",
        "coord_base": 1000,
        "type": "glm",
    },
    # Gemini 系列 - 推荐
    "gemini-2.5-flash": {
        "name": "gemini-2.5-flash",
        "api_base": None,
        "api_key_env": "GOOGLE_API_KEY",
        "coord_base": 1000,
        "type": "gemini",
    },
    "gemini-2.0-flash": {
        "name": "gemini-2.0-flash",
        "api_base": None,
        "api_key_env": "GOOGLE_API_KEY",
        "coord_base": 1000,
        "type": "gemini",
    },
    "gemini-2.5-pro": {
        "name": "gemini-2.5-pro",
        "api_base": None,
        "api_key_env": "GOOGLE_API_KEY",
        "coord_base": 1000,
        "type": "gemini",
    },
    "gemini-3-pro": {
        "name": "gemini-3-pro-preview",
        "api_base": None,
        "api_key_env": "GOOGLE_API_KEY",
        "coord_base": 1000,
        "type": "gemini",
    },
    # 兼容旧参数
    "gemini": {
        "name": "gemini-2.5-flash",  # 默认使用 2.5-flash
        "api_base": None,
        "api_key_env": "GOOGLE_API_KEY",
        "coord_base": 1000,
        "type": "gemini",
    },
}

# 默认模型
DEFAULT_MODEL = "glm"
COORD_BASE = 1000

VIDEO_DIR = Path("raw_data/videos/clips")  # 默认查找切分后的片段
DATASET_OUTPUT = Path("dataset_output")  # 直接输出到最终目录

# 188 种交通标志候选库
SIGNS_DIR = Path("raw_data/signs")

def load_sign_candidates():
    """从标志图片目录动态加载所有标志名称（188种）"""
    if not SIGNS_DIR.exists():
        print(f"⚠️ 找不到标志目录: {SIGNS_DIR}")
        return []
    return [f.stem for f in sorted(SIGNS_DIR.glob("*.png"))]

ALL_SIGN_CANDIDATES = load_sign_candidates()

COLORS = {
    'traffic_sign': (0, 100, 255),    # ID 0: 蓝色
    'pedestrian': (255, 0, 0),        # ID 1: 红色
    'vehicle': (0, 255, 0),           # ID 2: 绿色
    'small_obstacle': (255, 165, 0),  # ID 3: 橙色
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

### 4. 小型障碍物类 (small_obstacle)
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
    """异步目标检测器 - 支持多模型 (GLM / Gemini) 和多种 RAG 分类器"""
    
    def __init__(self, api_key: str, max_concurrent: int = 12, timeout: float = 45.0, model_type: str = "glm", 
                 use_clip_rag: bool = False, sign_classifier_type: str = None):
        """
        Args:
            api_key: API 密钥
            max_concurrent: 最大并发数
            timeout: 超时时间
            model_type: VLM 检测模型类型
            use_clip_rag: (旧参数，兼容) 是否使用 CLIP 分类
            sign_classifier_type: 交通标志分类器类型 ("dinov2", "clip", "vlm", None)
                - None/vlm: 使用 VLM 文字选项（默认）
                - dinov2: 使用 DINOv2 向量检索（推荐，最强）
                - clip: 使用 CLIP 向量检索
        """
        self.api_key = api_key
        self.timeout = timeout
        self.model_type = model_type
        self.model_config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["glm"])
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client: Optional[httpx.AsyncClient] = None
        self.gemini_client = None
        
        # 确定分类器类型（兼容旧参数）
        if sign_classifier_type:
            self.sign_classifier_type = sign_classifier_type
        elif use_clip_rag:
            self.sign_classifier_type = "clip"
        else:
            self.sign_classifier_type = "vlm"
        
        # 交通标志分类器初始化
        self.vector_classifier = None
        self.sign_classifier = None
        
        if self.sign_classifier_type == "dinov2":
            if DINOV2_AVAILABLE:
                print("   🦖 使用 DINOv2 向量检索进行交通标志分类（推荐）")
                self.vector_classifier = DINOv2SignClassifier(use_69_signs=False)
            else:
                print("   ⚠️ DINOv2 不可用，回退到 VLM 方式")
                self.sign_classifier_type = "vlm"
        elif self.sign_classifier_type == "clip":
            if CLIP_AVAILABLE:
                print("   📎 使用 CLIP 向量检索进行交通标志分类")
                self.vector_classifier = CLIPSignClassifier(use_69_signs=False)
            else:
                print("   ⚠️ CLIP 不可用，回退到 VLM 方式")
                self.sign_classifier_type = "vlm"
        
        if self.sign_classifier_type == "vlm":
            # 使用原版 VLM 分类器（69个核心标志 + other）
            self.sign_classifier = SignClassifierV2(
                api_key=api_key, 
                timeout=timeout,
                model_choice=model_type
            )
    
    async def __aenter__(self):
        model_type = self.model_config.get("type", self.model_type)
        if model_type == "gemini":
            # 使用 google-genai SDK
            try:
                from google import genai
                self.gemini_client = genai.Client()
                print(f"   🔌 Gemini 客户端已初始化 ({self.model_config['name']})")
            except ImportError:
                raise ImportError("请安装 google-genai: pip install google-genai")
        else:
            # 使用 httpx 客户端 (GLM)
            self.client = httpx.AsyncClient(
                base_url=self.model_config["api_base"],
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
        """带重试的检测 - 支持多模型"""
        image_name = Path(image_path).name
        last_error = None
        width, height = get_image_size(image_path)
        
        for attempt in range(max_retry):
            try:
                model_type = self.model_config.get("type", self.model_type)
                if model_type == "gemini":
                    # ============ Gemini API ============
                    content = await self._detect_gemini(image_path)
                else:
                    # ============ GLM API ============
                    content = await self._detect_glm(image_path)
                
                # 解析 JSON
                detections = parse_llm_json(content)
                
                if detections is None:
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
                
            except Exception as e:
                last_error = str(e)[:50]
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    await asyncio.sleep(3 * (attempt + 1))
                else:
                    await asyncio.sleep(2 * (attempt + 1))
        
        return [], last_error
    
    async def _detect_glm(self, image_path: str) -> str:
        """GLM API 检测"""
        base64_url = image_to_base64_url(image_path)
        
        payload = {
            "model": self.model_config["name"],
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": base64_url}},
                    {"type": "text", "text": DETECTION_PROMPT}
                ]
            }]
        }
        
        response = await self.client.post("/chat/completions", json=payload)
        
        if response.status_code == 429:
            raise Exception("429 Rate Limited")
        
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def _detect_gemini(self, image_path: str) -> str:
        """Gemini API 检测"""
        from google.genai import types
        
        # 读取图片
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        ext = Path(image_path).suffix.lower()
        mime_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
        mime_type = mime_types.get(ext, "image/jpeg")
        
        # 调用 Gemini (同步调用，在线程池中执行)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.gemini_client.models.generate_content(
                model=self.model_config["name"],
                contents=[
                    types.Part.from_bytes(data=image_data, mime_type=mime_type),
                    DETECTION_PROMPT
                ]
            )
        )
        
        return response.text
    
    async def classify_sign_rag(self, image_path: str, bbox: list) -> str:
        """
        RAG 交通标志精排（异步版）- 支持多种分类器
        
        DINOv2 模式（推荐）：
        - 使用 DINOv2 强大的视觉特征 + Chroma 向量数据库
        - 细粒度特征提取，对复杂场景更鲁棒
        
        CLIP 模式：
        - 使用 CLIP 图像向量 + Chroma 向量数据库
        - 通过余弦相似度检索最相似的标志
        
        VLM 模式（原版）：
        - 使用 VLM + 文字选项列表
        - 依赖模型预训练知识
        
        Args:
            image_path: 原图路径
            bbox: 交通标志的边界框 [x1, y1, x2, y2]
        
        Returns:
            细粒度标签，或 "other"（导航/方向牌等）
        """
        try:
            if self.vector_classifier:
                # 向量检索模式（DINOv2 或 CLIP）
                loop = asyncio.get_event_loop()
                label, score = await loop.run_in_executor(
                    None,
                    lambda: self.vector_classifier.classify(image_path, bbox)
                )
                return label
            else:
                # VLM 文字选项模式（原版）
                label, description, raw = await self.sign_classifier.classify(image_path, bbox)
                return label
        except Exception as e:
            return "traffic_sign"


# ============================================================================
# Step 1: 抽帧
# ============================================================================

def extract_frames(video_path: str, output_name: str, dataset_dir: Path, fps: int = 3, lossless: bool = False, keyframes_only: bool = True) -> tuple:
    """从视频抽帧，直接输出到 dataset 目录

    Args:
        video_path: 视频文件路径
        output_name: 输出名称（用于命名帧）
        dataset_dir: 目标 dataset 目录
        fps: 抽帧率
        lossless: 是否无损输出 (PNG 格式)
        keyframes_only: 是否只提取 I-frame 关键帧（默认 True，画质最高）
    """
    video_path = Path(video_path)

    if not video_path.exists():
        print(f"❌ 视频不存在: {video_path}")
        return None, 0

    frames_dir = dataset_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已有帧（断点续传，支持 jpg 和 png）
    existing_frames = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
    if existing_frames:
        print(f"\n📹 Step 1: 抽帧 (已存在 {len(existing_frames)} 帧，跳过)")
        return frames_dir, len(existing_frames)

    format_desc = "PNG 无损" if lossless else "JPEG q=2"
    keyframe_desc = "仅关键帧(I-frame)" if keyframes_only else "所有帧"
    print(f"\n📹 Step 1: 抽帧 ({fps} FPS, {format_desc}, {keyframe_desc})")
    print(f"   视频: {video_path}")
    print(f"   目标: {frames_dir}")

    # 构建视频滤镜
    if keyframes_only:
        # 只提取 I-frame 关键帧，再用 fps 过滤
        vf_filter = f"select='eq(pict_type,I)',fps={fps}"
    else:
        # 提取所有帧，按 fps 采样
        vf_filter = f"fps={fps}"

    if lossless:
        # PNG 无损输出
        output_pattern = str(frames_dir / f"{output_name}_%06d.png")
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", vf_filter,
            "-vsync", "vfr",
            output_pattern,
            "-y"
        ]
    else:
        # JPEG 高质量输出 (q=2)
        output_pattern = str(frames_dir / f"{output_name}_%06d.jpg")
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", vf_filter,
            "-vsync", "vfr",
            "-q:v", "2",
            output_pattern,
            "-y"
        ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        frame_count = len(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
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
    dataset_dir: Path,
    video_name: str, 
    workers: int,
    api_key: str,
    use_rag: bool = True,
    model_type: str = "glm",
    use_clip_rag: bool = False,
    sign_classifier_type: str = None
) -> Path:
    """异步运行标注，直接输出到 dataset 目录 - 支持多模型和多种分类器"""
    # 确定分类器显示名称
    if sign_classifier_type == "dinov2":
        rag_status = "🦖 DINOv2"
    elif sign_classifier_type == "clip" or use_clip_rag:
        rag_status = "📎 CLIP"
    elif use_rag:
        rag_status = "✅ VLM"
    else:
        rag_status = "❌ 禁用"
    
    model_name = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["glm"])["name"]
    print(f"\n🏷️ Step 2: 异步标注")
    print(f"   并发数: {workers} | 模型: {model_name} | RAG: {rag_status}")
    
    # 支持 jpg 和 png 两种格式
    image_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    if not image_files:
        print("   ❌ 没有找到帧")
        return None
    
    output_dir = dataset_dir / "annotations"
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
    stats = {"traffic_sign": 0, "pedestrian": 0, "vehicle": 0, "small_obstacle": 0}
    success = 0
    errors = 0
    
    async with AsyncDetector(api_key, max_concurrent=workers, model_type=model_type, 
                             use_clip_rag=use_clip_rag, sign_classifier_type=sign_classifier_type) as detector:
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

def generate_visualizations(frames_dir: Path, annotations_dir: Path, dataset_dir: Path) -> Path:
    """生成可视化图片，直接输出到 dataset 目录"""
    from PIL import ImageDraw
    
    print(f"\n🎨 Step 3: 生成可视化")
    
    vis_dir = dataset_dir / "visualized"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for json_path in sorted(annotations_dir.glob("*.json")):
        # 支持 jpg 和 png 两种格式
        frame_path = None
        for ext in [".jpg", ".png"]:
            candidate = frames_dir / (json_path.stem + ext)
            if candidate.exists():
                frame_path = candidate
                break

        if not frame_path:
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


def finalize_dataset(video_name: str, video_path: str, dataset_dir: Path, fps: int = 3, elapsed_time: float = None) -> Path:
    """生成报告并完成 Dataset（文件已直接生成到目录，无需复制）"""
    import shutil
    
    print(f"\n📦 Step 4: 完成 Dataset")
    
    # 复制视频到 video 子目录
    video_subdir = dataset_dir / "video"
    video_subdir.mkdir(parents=True, exist_ok=True)
    video_src = Path(video_path)
    video_dest = video_subdir / video_src.name
    if video_src.exists() and not video_dest.exists():
        shutil.copy(video_src, video_dest)
        print(f"   ✅ 复制视频")
    
    # 统计已有文件（支持 jpg 和 png）
    frames_dir = dataset_dir / "frames"
    annotations_dir = dataset_dir / "annotations"
    vis_dir = dataset_dir / "visualized"

    frame_count = len(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))) if frames_dir.exists() else 0
    ann_count = len(list(annotations_dir.glob("*.json"))) if annotations_dir.exists() else 0
    vis_count = len(list(vis_dir.glob("*.jpg"))) if vis_dir.exists() else 0
    
    print(f"   📊 帧: {frame_count} | 标注: {ann_count} | 可视化: {vis_count}")
    
    # 生成总结报告
    print(f"   📝 生成标注总结文档...")
    stats = generate_summary(annotations_dir, video_name, frame_count)
    stats["processing_time"] = elapsed_time  # 记录处理时间
    summary_md = create_summary_markdown(stats, video_name, fps, elapsed_time)
    
    summary_path = dataset_dir / f"{video_name}_dataset_info.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"   ✅ 生成 {video_name}_dataset_info.txt")
    
    # 保存 JSON 格式的统计数据
    stats_json = {
        "video_name": video_name,
        "total_frames": stats["total_frames"],
        "annotated_frames": stats["annotated_frames"],
        "total_objects": stats["total_objects"],
        "categories": dict(stats["categories"]),
        "subcategories": dict(stats["subcategories"]),
        "fps": fps,
        "processing_time": elapsed_time
    }
    stats_path = dataset_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_json, f, ensure_ascii=False, indent=2)
    print(f"   ✅ 生成 stats.json")
    
    return dataset_dir


# ============================================================================
# 主函数
# ============================================================================

async def main_async():
    parser = argparse.ArgumentParser(description="异步视频到数据集流水线（支持多模型：GLM / Gemini）")
    parser.add_argument("--video", type=str, required=True, help="视频文件路径 (如 raw_data/videos/clips/D1/D1_000.mp4)")
    parser.add_argument("--name", type=str, default=None, help="输出名称 (默认使用视频文件名)")
    parser.add_argument("--output", type=str, default=None, help="输出目录 (默认 dataset_output)")
    parser.add_argument("--fps", type=int, default=1, help="抽帧率 (默认 1)")
    parser.add_argument("--jpeg-frames", action="store_true", help="使用 JPEG q=2 抽帧（省空间），默认为 PNG 无损")
    parser.add_argument("--all-frames", action="store_true", help="提取所有帧（默认只提取 I-frame 关键帧，画质最高）")
    parser.add_argument("--workers", type=int, default=15, help="并发数 (默认 15)")
    parser.add_argument("--skip-visualize", action="store_true", help="跳过可视化")
    parser.add_argument("--rag", action="store_true", default=True, help="启用 RAG 交通标志细粒度分类 (默认启用)")
    parser.add_argument("--no-rag", dest="rag", action="store_false", help="禁用 RAG 交通标志细粒度分类")
    parser.add_argument("--clip-rag", action="store_true", help="(旧参数) 使用 CLIP 向量检索")
    parser.add_argument("--sign-classifier", type=str, default=None,
                        choices=["dinov2", "clip", "vlm"],
                        help="交通标志分类器: dinov2(推荐,最强), clip, vlm(默认)")
    parser.add_argument("--model", type=str, default="glm", 
                        choices=["glm", "gemini", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro", "gemini-3-pro"], 
                        help="选择模型: glm, gemini-2.5-flash(推荐), gemini-2.0-flash, gemini-2.5-pro, gemini-3-pro (默认 glm)")
    args = parser.parse_args()
    
    video_path = Path(args.video)
    model_type = args.model
    model_config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["glm"])
    
    # 自动确定输出名称
    if args.name:
        output_name = args.name
    else:
        output_name = video_path.stem  # 如 D1_000
    
    # 根据模型获取对应的 API Key
    api_key_env = model_config["api_key_env"]
    api_key = os.getenv(api_key_env)
    
    if not api_key:
        print(f"❌ 请设置 {api_key_env} 环境变量")
        print(f"   export {api_key_env}='your_api_key'")
        return
    
    if not video_path.exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    # 创建 dataset 目录（所有文件直接输出到这里）
    output_base = Path(args.output) if args.output else DATASET_OUTPUT
    dataset_dir = output_base / f"{output_name}_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    model_display = f"🔷 {model_config['name']}" if model_type == "glm" else f"🔶 {model_config['name']}"
    rag_display = "🎯 CLIP" if args.clip_rag else ("✅ VLM" if args.rag else "❌ 禁用")
    frame_format = "JPEG q=2" if args.jpeg_frames else "PNG 无损"

    print("=" * 70)
    print(f"🚀 异步视频标注流水线 - {output_name}")
    print(f"   视频: {video_path}")
    print(f"   输出: {dataset_dir}")
    print(f"   模型: {model_display} | 标志分类: {rag_display}")
    print(f"   FPS: {args.fps} | 帧格式: {frame_format} | 并发: {args.workers}")
    print("=" * 70)
    
    start_time = time.time()
    
    # Step 1: 抽帧（直接到 dataset/frames）
    # 默认 PNG 无损，--jpeg-frames 时使用 JPEG
    # 默认只提取 I-frame 关键帧，--all-frames 时提取所有帧
    lossless = not args.jpeg_frames
    keyframes_only = not args.all_frames
    frames_dir, _ = extract_frames(str(video_path), output_name, dataset_dir, args.fps, lossless=lossless, keyframes_only=keyframes_only)
    if not frames_dir:
        return
    
    # Step 2: 标注（直接到 dataset/annotations）
    # 确定分类器类型
    sign_classifier_type = args.sign_classifier
    if sign_classifier_type is None and args.clip_rag:
        sign_classifier_type = "clip"  # 兼容旧参数
    
    annotations_dir = await run_labeling_async(
        frames_dir, dataset_dir, output_name, args.workers, api_key, 
        use_rag=args.rag, model_type=model_type, use_clip_rag=args.clip_rag,
        sign_classifier_type=sign_classifier_type
    )
    if not annotations_dir:
        return
    
    # Step 3: 可视化（直接到 dataset/visualized）
    if args.skip_visualize:
        print(f"\n⏭️ 跳过可视化")
    else:
        generate_visualizations(frames_dir, annotations_dir, dataset_dir)
    
    # Step 4: 生成报告
    total_time = time.time() - start_time
    finalize_dataset(output_name, str(video_path), dataset_dir, fps=args.fps, elapsed_time=total_time)
    
    print("\n" + "=" * 70)
    print(f"🎉 完成！总耗时: {total_time/60:.1f} 分钟 ({total_time:.1f}秒)")
    print(f"📁 Dataset: {dataset_dir}/")
    print("=" * 70)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

