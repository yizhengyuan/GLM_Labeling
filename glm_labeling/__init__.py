"""
GLM-4.6V 交通场景自动标注系统

核心模块：
- config: 统一配置管理
- utils: 通用工具函数
- core: 检测器和分类器
- cli: 命令行接口
"""

__version__ = "0.2.0"
__author__ = "GLM_Labeling Team"

from .config import Config, get_config, init_config

# 核心功能快捷导入
from .core import (
    ObjectDetector,
    detect_objects,
    SignClassifier,
    classify_sign,
    ParallelProcessor,
    process_images_parallel
)

