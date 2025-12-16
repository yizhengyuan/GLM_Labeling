"""
统一配置管理模块

支持：
1. 环境变量读取
2. 默认值设置
3. 路径配置
4. API 配置
"""

import os
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """全局配置类"""
    
    # ==================== API 配置 ====================
    api_key: str = field(default_factory=lambda: os.getenv("ZAI_API_KEY", ""))
    model_name: str = "glm-4.6v"
    api_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 2.0  # 重试间隔（秒）
    
    # ==================== 路径配置 ====================
    # 项目根目录（自动检测）
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    # 输入路径
    images_dir: Path = field(default_factory=lambda: Path("test_images/extracted_frames"))
    signs_dir: Path = field(default_factory=lambda: Path("traffic_sign_data/images/signs/highres/png2560px"))
    
    # 输出路径
    output_dir: Path = field(default_factory=lambda: Path("output"))
    dataset_output_dir: Path = field(default_factory=lambda: Path("dataset_output"))
    temp_dir: Path = field(default_factory=lambda: Path("/tmp/glm_labeling"))
    
    # ==================== 检测配置 ====================
    # 坐标归一化基数（GLM 输出 0-1000）
    coord_normalize_base: int = 1000
    
    # RAG 配置
    rag_enabled: bool = False
    sign_crop_padding: int = 10
    
    # 并行配置
    default_workers: int = 5
    
    # ==================== 日志配置 ====================
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保临时目录存在
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 验证 API Key
        if not self.api_key:
            import warnings
            warnings.warn("ZAI_API_KEY 环境变量未设置，API 调用将失败")
    
    def get_output_path(self, prefix: str, suffix: str = "") -> Path:
        """获取输出目录路径"""
        dir_name = f"{prefix.lower()}_annotations{suffix}"
        path = self.output_dir / dir_name
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_dataset_path(self, name: str) -> Path:
        """获取数据集输出路径"""
        path = self.dataset_output_dir / f"{name}_dataset"
        path.mkdir(parents=True, exist_ok=True)
        return path


# 全局配置实例（单例模式，线程安全）
_config_instance: Optional[Config] = None
_config_lock = threading.Lock()


def get_config() -> Config:
    """获取全局配置实例（线程安全）"""
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            # Double-check locking
            if _config_instance is None:
                _config_instance = Config()
    return _config_instance


def init_config(**kwargs) -> Config:
    """初始化配置（可覆盖默认值）"""
    global _config_instance
    with _config_lock:
        _config_instance = Config(**kwargs)
    return _config_instance
