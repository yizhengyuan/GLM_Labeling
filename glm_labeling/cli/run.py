"""
glm-label run 子命令

批量标注图片，复用现有的 ParallelProcessor。
"""

import click
from pathlib import Path


@click.command()
@click.option(
    "--prefix", "-p",
    required=True,
    help="图片前缀 (如 D1, D2, D3.100f)"
)
@click.option(
    "--limit", "-l",
    type=int,
    default=None,
    help="限制处理数量"
)
@click.option(
    "--workers", "-w",
    type=int,
    default=5,
    help="并行线程数 (默认 5)"
)
@click.option(
    "--rag",
    is_flag=True,
    help="启用 RAG 细粒度分类"
)
@click.option(
    "--images-dir", "-i",
    type=click.Path(exists=True),
    default="test_images/extracted_frames",
    help="图片目录"
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default=None,
    help="输出目录 (默认: output/<prefix>_annotations)"
)
@click.option(
    "--no-resume",
    is_flag=True,
    help="禁用断点续传，重新处理所有图片"
)
def run(prefix, limit, workers, rag, images_dir, output_dir, no_resume):
    """批量标注图片
    
    \b
    示例:
      glm-label run --prefix D1 --workers 5
      glm-label run --prefix D2 --limit 50 --rag
      glm-label run -p D3.100f -w 10 -o output/custom_dir
    """
    from ..config import get_config
    from ..utils import get_logger
    from ..core import ParallelProcessor
    
    config = get_config()
    logger = get_logger()
    
    if not config.api_key:
        click.echo("❌ 请设置 ZAI_API_KEY 环境变量", err=True)
        raise SystemExit(1)
    
    # 获取图片列表
    images_dir = Path(images_dir)
    
    # 支持多种图片格式和命名模式
    patterns = [f"{prefix}_*.jpg", f"{prefix}_*.png", f"{prefix}*.jpg"]
    image_files = []
    for pattern in patterns:
        image_files.extend(images_dir.glob(pattern))
    image_files = sorted(set(image_files))
    
    if limit:
        image_files = image_files[:limit]
    
    if not image_files:
        click.echo(f"❌ 没有找到 {prefix} 开头的图片在 {images_dir}", err=True)
        raise SystemExit(1)
    
    # 输出目录
    rag_suffix = "_rag" if rag else ""
    output_dir = output_dir or f"output/{prefix.lower()}_annotations{rag_suffix}"
    
    click.echo("=" * 60)
    click.echo(f"🚀 GLM-4.6V 并行自动标注")
    click.echo(f"   📁 图片数量: {len(image_files)}")
    click.echo(f"   🔧 并行线程: {workers}")
    click.echo(f"   🔍 RAG 模式: {'✅ 启用' if rag else '❌ 禁用'}")
    click.echo(f"   📂 输出目录: {output_dir}")
    click.echo(f"   🔄 断点续传: {'❌ 禁用' if no_resume else '✅ 启用'}")
    click.echo("=" * 60)
    
    # 执行处理
    processor = ParallelProcessor(
        api_key=config.api_key,
        workers=workers,
        use_rag=rag
    )
    
    results = processor.process_batch(
        [str(p) for p in image_files],
        Path(output_dir),
        resume=not no_resume
    )
    
    # 输出结果
    click.echo(f"\n⏱️  耗时: {results['elapsed_seconds']:.1f}s")
    click.echo(f"📊 平均: {results['per_image_seconds']:.2f}s/张")
    click.echo(f"✅ 成功: {results['success']} | ❌ 失败: {results['failed']}")
    
    if results.get('skipped', 0) > 0:
        click.echo(f"⏭️  跳过: {results['skipped']} (已处理)")






