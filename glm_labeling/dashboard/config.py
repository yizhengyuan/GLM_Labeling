"""
Dashboard 配置
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class DashboardConfig:
    """Dashboard 配置类"""
    
    # 服务配置
    host: str = "127.0.0.1"
    port: int = 8000
    
    # 标注配置
    workers: int = 5
    images_dir: str = "test_images/extracted_frames"
    output_dir: str = "output"
    
    # 审查配置
    enable_review: bool = True
    review_rate: float = 0.05  # 5% 随机深度审查
    
    # API 配置
    api_key: Optional[str] = None
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保路径存在
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 从环境变量获取 API Key
        if self.api_key is None:
            import os
            self.api_key = os.getenv("ZAI_API_KEY")






