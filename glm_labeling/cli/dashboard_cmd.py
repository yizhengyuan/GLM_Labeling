"""
glm-label dashboard 子命令

启动实时标注监控 Dashboard。
"""

import click


@click.command()
@click.option(
    "--port", "-p",
    type=int,
    default=8000,
    help="服务端口 (默认 8000)"
)
@click.option(
    "--host", "-h",
    default="127.0.0.1",
    help="绑定地址 (默认 127.0.0.1)"
)
@click.option(
    "--workers", "-w",
    type=int,
    default=5,
    help="标注并行线程数 (默认 5)"
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
    default="output",
    help="输出根目录"
)
@click.option(
    "--review-rate",
    type=float,
    default=0.05,
    help="AI 深度审查抽样率 (默认 0.05 即 5%)"
)
@click.option(
    "--no-review",
    is_flag=True,
    help="禁用 AI 审查功能"
)
def dashboard(port, host, workers, images_dir, output_dir, review_rate, no_review):
    """启动实时标注监控 Dashboard
    
    \b
    启动后访问 http://localhost:8000 查看界面
    
    \b
    示例:
      glm-label dashboard
      glm-label dashboard --port 8080 --workers 10
      glm-label dashboard --no-review
    """
    try:
        import uvicorn
    except ImportError:
        click.echo("❌ Dashboard 依赖未安装，请运行:", err=True)
        click.echo("   pip install glm-labeling[dashboard]", err=True)
        raise SystemExit(1)
    
    from ..dashboard import create_app
    from ..dashboard.config import DashboardConfig
    
    config = DashboardConfig(
        workers=workers,
        images_dir=images_dir,
        output_dir=output_dir,
        review_rate=review_rate if not no_review else 0,
        enable_review=not no_review,
    )
    
    app = create_app(config)
    
    click.echo("=" * 60)
    click.echo("🚀 GLM Labeling Dashboard")
    click.echo(f"   🌐 地址: http://{host}:{port}")
    click.echo(f"   🔧 标注线程: {workers}")
    click.echo(f"   🔍 AI 审查: {'❌ 禁用' if no_review else f'✅ 启用 ({review_rate*100:.0f}% 抽样)'}")
    click.echo(f"   📁 图片目录: {images_dir}")
    click.echo(f"   📂 输出目录: {output_dir}")
    click.echo("=" * 60)
    click.echo("\n按 Ctrl+C 停止服务\n")
    
    uvicorn.run(app, host=host, port=port, log_level="info")






