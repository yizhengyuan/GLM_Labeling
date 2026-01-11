"""
GLM Labeling Dashboard 模块

实时标注监控系统，提供：
- Web 界面实时查看标注进度
- WebSocket 推送标注结果
- AI 自动审查功能
"""

from .server import create_app
from .config import DashboardConfig

__all__ = ["create_app", "DashboardConfig"]






