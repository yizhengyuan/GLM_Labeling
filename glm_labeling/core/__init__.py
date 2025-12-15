"""
核心检测模块

提供目标检测和交通标志分类的核心功能。
"""

from .detector import ObjectDetector, detect_objects
from .sign_classifier import SignClassifier, classify_sign
from .parallel import ParallelProcessor, process_images_parallel

