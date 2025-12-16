"""
图像处理工具函数

提取自多个脚本中的重复代码，统一管理。
"""

import base64
import uuid
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image


def image_to_base64_url(image_path: str) -> str:
    """
    将图片文件转换为 Base64 Data URL
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        Base64 编码的 Data URL (data:image/xxx;base64,...)
    """
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(ext, 'image/jpeg')
    
    return f"data:{mime_type};base64,{image_data}"


def get_image_size(image_path: str) -> Tuple[int, int]:
    """
    获取图片尺寸
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        (width, height) 元组
    """
    with Image.open(image_path) as img:
        return img.width, img.height


def crop_region(
    image_path: str,
    bbox: list,
    padding: int = 10,
    save_path: Optional[str] = None
) -> Tuple[Image.Image, str]:
    """
    裁剪图片指定区域

    Args:
        image_path: 原图路径
        bbox: 边界框 [x1, y1, x2, y2]
        padding: 边界扩展像素
        save_path: 保存路径（可选，None 则自动生成临时路径）

    Returns:
        (裁剪后的 PIL Image, 保存路径)

    Raises:
        ValueError: bbox 尺寸无效
    """
    x1, y1, x2, y2 = bbox

    # 使用 context manager 防止内存泄漏
    with Image.open(image_path) as img:
        # 添加 padding 并确保不越界
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.width, x2 + padding)
        y2 = min(img.height, y2 + padding)

        # 验证裁剪区域有效性
        if (x2 - x1) < 1 or (y2 - y1) < 1:
            raise ValueError(f"Invalid crop dimensions: width={x2-x1}, height={y2-y1}")

        cropped = img.crop((x1, y1, x2, y2))

        # 生成保存路径
        if save_path is None:
            unique_id = uuid.uuid4()
            save_path = f"/tmp/glm_labeling/crop_{unique_id}.jpg"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        cropped.save(save_path, "JPEG")

    return cropped, save_path


def image_to_base64(image_path: str) -> str:
    """
    将图片转换为纯 Base64 字符串（不含 Data URL 前缀）
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        Base64 编码字符串
    """
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def convert_normalized_coords(
    bbox: list,
    width: int,
    height: int,
    base: int = 1000
) -> list:
    """
    将归一化坐标转换为绝对像素坐标

    GLM 模型输出的坐标是 0-1000 的归一化值

    Args:
        bbox: 归一化边界框 [x1, y1, x2, y2]
        width: 图片宽度
        height: 图片高度
        base: 归一化基数（默认 1000）

    Returns:
        绝对像素坐标 [x1, y1, x2, y2]，已校验边界
    """
    x1 = int(round(bbox[0] / base * width))
    y1 = int(round(bbox[1] / base * height))
    x2 = int(round(bbox[2] / base * width))
    y2 = int(round(bbox[3] / base * height))

    # 边界校验：确保坐标在有效范围内
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    # 确保 x1 < x2, y1 < y2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]
