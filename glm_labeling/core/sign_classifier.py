"""
交通标志分类器模块

实现两阶段 RAG 分类逻辑。
"""

import re
import os
import uuid
from pathlib import Path
from typing import Optional, List
from zai import ZaiClient

from ..config import get_config
from ..utils import (
    crop_region,
    image_to_base64,
    get_logger,
    retry_api_call
)


class SignClassifier:
    """
    交通标志两阶段分类器
    
    阶段1: 判断标志类型（限速/禁止/警告/指示）
    阶段2: 识别具体细节（如限速数字）
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.config = get_config()
        self.logger = get_logger()
        
        self.api_key = api_key or self.config.api_key
        self.model_name = model_name or self.config.model_name
        
        if not self.api_key:
            raise ValueError("API Key 未设置")
        
        self.client = ZaiClient(api_key=self.api_key)
    
    def classify(self, image_path: str, bbox: List[int]) -> str:
        """
        对交通标志进行细粒度分类
        
        Args:
            image_path: 原图路径
            bbox: 标志边界框 [x1, y1, x2, y2]
            
        Returns:
            细粒度标签
        """
        temp_path = None
        
        try:
            # 裁剪标志区域
            _, temp_path = crop_region(
                image_path, 
                bbox, 
                padding=self.config.sign_crop_padding
            )
            
            img_data = image_to_base64(temp_path)
            
            # 阶段1：判断类型
            sign_type = self._classify_type(img_data)
            
            # 阶段2：识别细节
            return self._classify_detail(img_data, sign_type)
            
        except Exception as e:
            self.logger.warning(f"Sign classification failed: {e}")
            return "traffic_sign"
        
        finally:
            # 清理临时文件
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    def _classify_type(self, img_data: str) -> str:
        """阶段1：判断标志类型"""
        prompt = """请判断这是什么类型的交通标志：
1. 限速标志（红圈白底，中间有数字）
2. 禁止标志（红圈）
3. 警告标志（三角形）
4. 指示/方向标志（蓝色或绿色）
5. 其他

只返回数字（1-5）。"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                temperature=0.1
            )
            
            text = response.choices[0].message.content.strip()
            match = re.search(r'[1-5]', text)
            return match.group() if match else "5"
            
        except Exception:
            return "5"
    
    def _classify_detail(self, img_data: str, sign_type: str) -> str:
        """阶段2：识别具体细节"""
        
        if sign_type == "1":  # 限速标志
            return self._recognize_speed_limit(img_data)
        
        elif sign_type == "4":  # 方向/指示标志
            return self._recognize_direction_sign(img_data)
        
        # 其他类型返回通用标签
        type_labels = {
            "2": "Prohibition_sign",
            "3": "Warning_sign",
            "5": "traffic_sign"
        }
        return type_labels.get(sign_type, "traffic_sign")
    
    def _recognize_speed_limit(self, img_data: str) -> str:
        """识别限速数字"""
        prompt = "请识别这个限速标志上的数字。只返回数字。"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                temperature=0.1
            )
            
            numbers = re.findall(r'\d+', response.choices[0].message.content)
            if numbers:
                return f"Speed_limit_{numbers[0]}_km_h"
            return "Speed_limit"
            
        except Exception:
            return "Speed_limit"
    
    def _recognize_direction_sign(self, img_data: str) -> str:
        """识别方向/指示标志"""
        prompt = """这是一个指示或方向标志。请判断：
1. 方向指示牌
2. 高速公路标志
3. 倒计时距离牌（100m/200m/300m斜条）
4. 其他

只返回数字（1-4）。"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                temperature=0.1
            )
            
            detail = re.search(r'[1-4]', response.choices[0].message.content)
            if detail:
                label_map = {
                    "1": "Direction_sign",
                    "2": "Expressway_sign",
                    "3": "100m_Countdown_markers",
                    "4": "Direction_other"
                }
                return label_map.get(detail.group(), "Direction_sign")
            return "Direction_sign"
            
        except Exception:
            return "Direction_sign"


def classify_sign(
    image_path: str,
    bbox: List[int],
    api_key: Optional[str] = None
) -> str:
    """
    便捷函数：对交通标志进行分类
    
    Args:
        image_path: 原图路径
        bbox: 标志边界框
        api_key: API Key（可选）
        
    Returns:
        细粒度标签
    """
    classifier = SignClassifier(api_key=api_key)
    return classifier.classify(image_path, bbox)
