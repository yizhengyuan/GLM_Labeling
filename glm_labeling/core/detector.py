"""
目标检测器模块

封装 GLM-4.6V 目标检测逻辑。
"""

import os
import re
from typing import List, Dict, Any, Optional
from zai import ZaiClient

from ..config import get_config
from ..utils import (
    image_to_base64_url,
    get_image_size,
    convert_normalized_coords,
    parse_llm_json,
    get_category,
    normalize_vehicle_label,
    get_logger,
    retry_api_call
)


# 检测 Prompt 模板
DETECTION_PROMPT = """请检测图片中的以下4类物体，返回JSON格式。

## 检测类别与细粒度要求：

### 1. 行人类 (pedestrian) - 2种标签
- pedestrian: 单个或少量行人
- crowd: 人群（多人聚集）

### 2. 车辆类 (vehicle) - 5种标签
统一使用 vehicle，只区分行驶状态：

**状态判断规则**（按优先级）：
1. **刹车状态**: 尾灯明显变亮、红色刹车灯亮起 → `vehicle_braking`
2. **双闪状态**: 左右两侧转向灯同时亮起/闪烁 → `vehicle_double_flash`
3. **右转状态**: 车身朝右/车头转向右侧/右转车道转弯/仅右侧转向灯亮 → `vehicle_turning_right`
4. **左转状态**: 车身朝左/车头转向左侧/左转车道转弯/仅左侧转向灯亮 → `vehicle_turning_left`
5. **正常状态**: 直行或无法判断 → `vehicle`

注意：
- 不要标注第一人称视角的车辆（拍摄车辆本身）
- 所有机动车（轿车、卡车、公交、摩托车等）和自行车都统一标注为 vehicle

### 3. 交通标志类 (traffic_sign)
traffic_sign

### 4. 施工标志类 (construction)
traffic_cone, construction_barrier

## 返回格式示例：
[
  {"label": "vehicle_braking", "bbox_2d": [100, 200, 300, 400]},
  {"label": "vehicle", "bbox_2d": [900, 300, 1100, 500]},
  {"label": "traffic_sign", "bbox_2d": [50, 50, 80, 80]}
]

如果没有目标，返回 []
只返回JSON数组！"""


class ObjectDetector:
    """
    目标检测器
    
    封装 GLM-4.6V 的目标检测功能，支持重试和后处理。
    
    Usage:
        detector = ObjectDetector(api_key="xxx")
        results = detector.detect("image.jpg")
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: Optional[int] = None
    ):
        """
        初始化检测器
        
        Args:
            api_key: API Key（默认从配置读取）
            model_name: 模型名称（默认 glm-4.6v）
            max_retries: 最大重试次数
        """
        self.config = get_config()
        self.logger = get_logger()
        
        self.api_key = api_key or self.config.api_key
        self.model_name = model_name or self.config.model_name
        self.max_retries = max_retries or self.config.max_retries
        
        if not self.api_key:
            raise ValueError("API Key 未设置，请设置 ZAI_API_KEY 环境变量")
        
        self.client = ZaiClient(api_key=self.api_key)
    
    def detect(self, image_path: str, prompt: str = None) -> List[Dict[str, Any]]:
        """
        检测图片中的目标

        Args:
            image_path: 图片路径
            prompt: 自定义 prompt（可选）

        Returns:
            检测结果列表，每个元素包含 label, category, bbox

        Raises:
            FileNotFoundError: 图片文件不存在
            ValueError: 图片文件过大
        """
        # 输入验证
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        file_size = os.path.getsize(image_path)
        max_size = getattr(self.config, 'max_image_size', 20 * 1024 * 1024)  # 默认 20MB
        if file_size > max_size:
            raise ValueError(f"图片文件过大: {file_size / 1024 / 1024:.1f}MB (最大 {max_size / 1024 / 1024:.0f}MB)")

        base64_url = image_to_base64_url(image_path)
        width, height = get_image_size(image_path)
        
        detection_prompt = prompt or DETECTION_PROMPT

        # API 超时设置
        timeout = getattr(self.config, 'api_timeout', 60)

        def call_api():
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": base64_url}},
                        {"type": "text", "text": detection_prompt}
                    ]
                }],
                timeout=timeout
            )
        
        try:
            response = retry_api_call(
                call_api,
                max_retries=self.max_retries,
                delay=self.config.retry_delay,
                on_retry=lambda a, e: self.logger.warning(f"Retry {a}: {e}")
            )
            
            result_text = response.choices[0].message.content.strip()
            detections = parse_llm_json(result_text)
            
            if not detections:
                return []
            
            return self._post_process(detections, width, height)
            
        except Exception as e:
            self.logger.error(f"Detection failed for {image_path}: {e}")
            return []
    
    def _post_process(
        self, 
        detections: List[Dict], 
        width: int, 
        height: int
    ) -> List[Dict[str, Any]]:
        """后处理检测结果"""
        processed = []
        
        for det in detections:
            if "label" not in det or "bbox_2d" not in det:
                continue
            
            # 坐标转换
            bbox = convert_normalized_coords(
                det["bbox_2d"], 
                width, 
                height,
                base=self.config.coord_normalize_base
            )
            
            # 标签规范化
            label = det["label"].lower().replace(" ", "_").replace("-", "_")
            category = get_category(label)
            
            # 车辆标签统一
            if category == "vehicle":
                label = normalize_vehicle_label(label)
            
            processed.append({
                "label": label,
                "category": category,
                "bbox": bbox
            })
        
        return processed


def detect_objects(
    image_path: str,
    api_key: Optional[str] = None,
    use_rag: bool = False
) -> List[Dict[str, Any]]:
    """
    便捷函数：检测图片中的目标
    
    Args:
        image_path: 图片路径
        api_key: API Key（可选）
        use_rag: 是否启用 RAG 细粒度分类
        
    Returns:
        检测结果列表
    """
    detector = ObjectDetector(api_key=api_key)
    results = detector.detect(image_path)
    
    # TODO: 如果 use_rag，对 traffic_sign 进行二次分类
    
    return results
