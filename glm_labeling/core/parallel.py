"""
并行处理器模块

提供批量图片的并行处理能力。
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import time
import threading

from ..config import get_config
from ..utils import (
    get_image_size,
    save_annotation,
    to_xanylabeling_format,
    get_logger,
    TaskProgress
)
from .detector import ObjectDetector
from .sign_classifier import SignClassifier


class ParallelProcessor:
    """
    并行批量处理器
    
    支持多线程并行处理图片，提供进度跟踪和错误处理。
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        workers: int = 5,
        use_rag: bool = False
    ):
        """
        初始化处理器
        
        Args:
            api_key: API Key
            workers: 并行线程数
            use_rag: 是否启用 RAG 细粒度分类
        """
        self.config = get_config()
        self.logger = get_logger()
        
        self.api_key = api_key or self.config.api_key
        self.workers = workers
        self.use_rag = use_rag

        # 统计数据（线程安全）
        self._stats_lock = threading.Lock()
        self.stats = {
            "pedestrian": 0,
            "vehicle": 0,
            "traffic_sign": 0,
            "construction": 0
        }
    
    def process_batch(
        self,
        image_paths: List[str],
        output_dir: Path,
        on_complete: Optional[Callable] = None,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        批量处理图片
        
        Args:
            image_paths: 图片路径列表
            output_dir: 输出目录
            on_complete: 单张完成回调 (path, detections, error) -> None
            resume: 是否启用断点续传（跳过已处理的图片）
            
        Returns:
            处理结果统计
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 断点续传：过滤已处理的图片
        if resume:
            original_count = len(image_paths)
            image_paths = self._filter_processed(image_paths, output_dir)
            skipped = original_count - len(image_paths)
            
            if skipped > 0:
                self.logger.info(f"📌 断点续传：跳过 {skipped} 张已处理图片")
            
            if not image_paths:
                self.logger.info("✅ 所有图片已处理完成，无需重新运行")
                return {
                    "success": 0, 
                    "failed": 0, 
                    "skipped": skipped,
                    "total_objects": 0,
                    "stats": self.stats
                }
        
        progress = TaskProgress(len(image_paths), "Parallel Detection")
        progress.start()
        
        start_time = time.time()
        results = {"success": 0, "failed": 0, "total_objects": 0}
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self._process_single, path): path
                for path in image_paths
            }
            
            # 收集结果
            for future in as_completed(futures):
                image_path = futures[future]
                image_name = Path(image_path).name
                
                try:
                    detections, error = future.result()
                    
                    if error:
                        progress.update(image_name, success=False, message=error)
                        results["failed"] += 1
                    else:
                        # 更新统计（线程安全）
                        with self._stats_lock:
                            for det in detections:
                                category = det.get("category", "unknown")
                                self.stats[category] = self.stats.get(category, 0) + 1
                                results["total_objects"] += 1
                        
                        # 保存标注
                        self._save_result(detections, image_path, output_dir)
                        
                        progress.update(
                            image_name, 
                            success=True, 
                            message=f"{len(detections)} objects"
                        )
                        results["success"] += 1
                    
                    # 回调
                    if on_complete:
                        on_complete(image_path, detections, error)
                        
                except Exception as e:
                    progress.update(image_name, success=False, message=str(e))
                    results["failed"] += 1
        
        elapsed = time.time() - start_time
        results["elapsed_seconds"] = elapsed
        results["per_image_seconds"] = elapsed / len(image_paths) if image_paths else 0
        results["stats"] = self.stats
        
        progress.finish(extra_stats=self.stats)
        
        return results
    
    def _process_single(self, image_path: str) -> tuple:
        """
        处理单张图片（在工作线程中执行）
        
        Returns:
            (detections, error)
        """
        try:
            # 每个线程创建独立的检测器
            detector = ObjectDetector(api_key=self.api_key)
            detections = detector.detect(image_path)
            
            # RAG 细粒度分类
            if self.use_rag and detections:
                classifier = SignClassifier(api_key=self.api_key)
                
                for det in detections:
                    if det["category"] == "traffic_sign" and det["label"] in ["traffic_sign", "sign"]:
                        det["label"] = classifier.classify(image_path, det["bbox"])
            
            return detections, None
            
        except Exception as e:
            return [], str(e)
    
    def _save_result(
        self, 
        detections: List[Dict], 
        image_path: str, 
        output_dir: Path
    ):
        """保存标注结果"""
        width, height = get_image_size(image_path)
        annotation = to_xanylabeling_format(detections, image_path, width, height)
        
        output_path = output_dir / f"{Path(image_path).stem}.json"
        save_annotation(annotation, output_path)
    
    def _filter_processed(
        self,
        image_paths: List[str],
        output_dir: Path
    ) -> List[str]:
        """
        过滤已处理的图片（断点续传）
        
        检查输出目录中是否已存在对应的 JSON 文件，
        如果存在则认为该图片已处理，跳过。
        
        Args:
            image_paths: 原始图片路径列表
            output_dir: 输出目录
            
        Returns:
            未处理的图片路径列表
        """
        unprocessed = []
        
        for path in image_paths:
            json_name = f"{Path(path).stem}.json"
            json_path = output_dir / json_name
            
            if not json_path.exists():
                unprocessed.append(path)
        
        return unprocessed


def process_images_parallel(
    image_paths: List[str],
    output_dir: str,
    workers: int = 5,
    use_rag: bool = False,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    便捷函数：并行处理图片批次
    
    Args:
        image_paths: 图片路径列表
        output_dir: 输出目录
        workers: 并行线程数
        use_rag: 是否启用 RAG
        api_key: API Key
        
    Returns:
        处理结果统计
    """
    processor = ParallelProcessor(
        api_key=api_key,
        workers=workers,
        use_rag=use_rag
    )
    return processor.process_batch(image_paths, Path(output_dir))
