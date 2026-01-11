"""
glm-label visualize 子命令

生成可视化标注图片。
"""

import click
import json
from pathlib import Path
from PIL import Image, ImageDraw


# 类别颜色定义 (RGB)
CATEGORY_COLORS = {
    "pedestrian": (255, 0, 0),      # 红色 - 行人
    "vehicle": (0, 255, 0),         # 绿色 - 车辆
    "traffic_sign": (0, 100, 255),  # 蓝色 - 交通标志
    "construction": (255, 165, 0),  # 橙色 - 施工标志
    "unknown": (128, 128, 128),     # 灰色 - 未知
}


def visualize_single(image_path: str, json_path: str, output_path: str):
    """在图片上绘制标注框"""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    with open(json_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)
    
    for shape in annotation.get("shapes", []):
        label = shape["label"]
        points = shape["points"]
        category = shape.get("flags", {}).get("category", "unknown")
        
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        color = CATEGORY_COLORS.get(category, (128, 128, 128))
        
        # 绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 绘制标签背景
        text_bbox = draw.textbbox((x1, y1 - 18), label)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
            fill=color
        )
        draw.text((x1, y1 - 18), label, fill=(255, 255, 255))
    
    img.save(output_path)
    return len(annotation.get("shapes", []))


@click.command()
@click.option(
    "--prefix", "-p",
    required=True,
    help="图片前缀 (如 D1, D2)"
)
@click.option(
    "--images-dir", "-i",
    type=click.Path(exists=True),
    default="test_images/extracted_frames",
    help="原始图片目录"
)
@click.option(
    "--annotations-dir", "-a",
    type=click.Path(exists=True),
    default=None,
    help="标注文件目录 (默认: output/<prefix>_annotations)"
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default=None,
    help="输出目录 (默认: output/<prefix>_visualized)"
)
def visualize(prefix, images_dir, annotations_dir, output_dir):
    """生成可视化标注图片
    
    \b
    示例:
      glm-label visualize --prefix D1
      glm-label visualize -p D2 -o output/custom_vis
    """
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir or f"output/{prefix.lower()}_annotations")
    output_dir = Path(output_dir or f"output/{prefix.lower()}_visualized")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo("=" * 60)
    click.echo(f"🎨 可视化标注结果 - {prefix}")
    click.echo(f"   📁 标注目录: {annotations_dir}")
    click.echo(f"   📂 输出目录: {output_dir}")
    click.echo("=" * 60)
    
    json_files = list(annotations_dir.glob("*.json"))
    
    if not json_files:
        click.echo(f"❌ 没有找到标注文件在 {annotations_dir}", err=True)
        raise SystemExit(1)
    
    success_count = 0
    total_objects = 0
    
    with click.progressbar(sorted(json_files), label="处理中") as bar:
        for json_path in bar:
            # 尝试多种图片命名格式
            image_name = json_path.stem + ".jpg"
            image_path = images_dir / image_name
            
            if not image_path.exists():
                # 尝试其他格式
                image_path = images_dir / (json_path.stem + ".png")
            
            if image_path.exists():
                output_path = output_dir / f"{json_path.stem}_vis.jpg"
                try:
                    obj_count = visualize_single(
                        str(image_path), 
                        str(json_path), 
                        str(output_path)
                    )
                    success_count += 1
                    total_objects += obj_count
                except Exception as e:
                    click.echo(f"\n⚠️  {json_path.name}: {e}", err=True)
    
    click.echo("\n" + "=" * 60)
    click.echo(f"✅ 完成! {success_count} 张图片已可视化")
    click.echo(f"📊 共 {total_objects} 个标注框")
    click.echo(f"📁 结果保存到: {output_dir}")
    click.echo("=" * 60)






