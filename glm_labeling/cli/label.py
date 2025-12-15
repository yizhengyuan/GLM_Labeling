#!/usr/bin/env python3
"""
GLM-4.6V 并行自动标注 CLI

使用重构后的模块化代码，更简洁、更易维护。

用法:
    python3 -m glm_labeling.cli.label --prefix D2 --limit 50 --workers 5 --rag
"""

import argparse
from pathlib import Path

from ..config import get_config
from ..utils import get_logger
from ..core import ParallelProcessor


def main():
    parser = argparse.ArgumentParser(
        description="GLM-4.6V 并行自动标注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 -m glm_labeling.cli.label --prefix D2 --limit 50
  python3 -m glm_labeling.cli.label --prefix D2 --workers 10 --rag
        """
    )
    
    parser.add_argument(
        "--prefix", 
        type=str, 
        required=True, 
        help="图片前缀 (如 D1, D2)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="限制处理数量"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=5, 
        help="并行线程数 (默认 5)"
    )
    parser.add_argument(
        "--rag", 
        action="store_true", 
        help="启用 RAG 细粒度分类"
    )
    parser.add_argument(
        "--images-dir", 
        type=str, 
        default="test_images/extracted_frames",
        help="图片目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录 (默认: output/<prefix>_annotations)"
    )
    
    args = parser.parse_args()
    
    # 初始化
    config = get_config()
    logger = get_logger()
    
    if not config.api_key:
        logger.error("请设置 ZAI_API_KEY 环境变量")
        return 1
    
    # 获取图片列表
    images_dir = Path(args.images_dir)
    image_files = sorted(images_dir.glob(f"{args.prefix}_*.jpg"))
    
    if args.limit:
        image_files = image_files[:args.limit]
    
    if not image_files:
        logger.error(f"没有找到 {args.prefix} 开头的图片在 {images_dir}")
        return 1
    
    # 输出目录
    rag_suffix = "_rag" if args.rag else ""
    output_dir = args.output_dir or f"output/{args.prefix.lower()}_annotations{rag_suffix}"
    
    logger.info("=" * 60)
    logger.info(f"🚀 GLM-4.6V 并行自动标注")
    logger.info(f"   📁 图片数量: {len(image_files)}")
    logger.info(f"   🔧 并行线程: {args.workers}")
    logger.info(f"   🔍 RAG 模式: {'✅ 启用' if args.rag else '❌ 禁用'}")
    logger.info(f"   📂 输出目录: {output_dir}")
    logger.info("=" * 60)
    
    # 执行处理
    processor = ParallelProcessor(
        api_key=config.api_key,
        workers=args.workers,
        use_rag=args.rag
    )
    
    results = processor.process_batch(
        [str(p) for p in image_files],
        Path(output_dir)
    )
    
    # 输出结果
    logger.info(f"\n⏱️ 耗时: {results['elapsed_seconds']:.1f}s")
    logger.info(f"📊 平均: {results['per_image_seconds']:.2f}s/张")
    logger.info(f"✅ 成功: {results['success']} | ❌ 失败: {results['failed']}")
    
    return 0


if __name__ == "__main__":
    exit(main())
