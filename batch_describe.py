#!/usr/bin/env python3
"""
批量描述 test_images 文件夹中的前10张图片
"""

import os
import base64
from pathlib import Path
from zai import ZaiClient

# API Key
API_KEY = os.getenv("ZAI_API_KEY", "")

def image_to_base64_url(image_path: str) -> str:
    """将本地图片转换为 base64 data URL"""
    path = Path(image_path)
    ext = path.suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    mime_type = mime_types.get(ext, 'image/jpeg')
    
    with open(path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    return f"data:{mime_type};base64,{image_data}"


def describe_image(client: ZaiClient, image_path: str) -> str:
    """使用 GLM-4.6V 描述一张图片"""
    base64_url = image_to_base64_url(image_path)
    
    response = client.chat.completions.create(
        model="glm-4.6v",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_url
                        }
                    },
                    {
                        "type": "text",
                        "text": "请简洁描述这张图片的内容，包括：1) 场景类型 2) 主要物体 3) 环境特征。用2-3句话概括。"
                    }
                ]
            }
        ]
    )
    
    return response.choices[0].message.content


def main():
    if not API_KEY:
        print("❌ 请设置 ZAI_API_KEY 环境变量")
        return
    
    # 获取 test_images/extracted_frames 中的图片
    images_dir = Path("test_images/extracted_frames")
    
    if not images_dir.exists():
        print(f"❌ 目录不存在: {images_dir}")
        return
    
    # 获取所有 jpg 图片并排序
    images = sorted([f for f in images_dir.glob("*.jpg")])[:10]
    
    if not images:
        print("❌ 没有找到图片")
        return
    
    print("=" * 70)
    print(f"🖼️  GLM-4.6V 图片描述 - 前 {len(images)} 张图片")
    print("=" * 70)
    
    client = ZaiClient(api_key=API_KEY)
    
    results = []
    
    for i, img_path in enumerate(images, 1):
        print(f"\n📷 [{i}/{len(images)}] {img_path.name}")
        print("-" * 50)
        
        try:
            description = describe_image(client, str(img_path))
            print(f"📝 {description}")
            results.append({
                "image": img_path.name,
                "description": description
            })
        except Exception as e:
            print(f"❌ 错误: {e}")
            results.append({
                "image": img_path.name,
                "error": str(e)
            })
    
    print("\n" + "=" * 70)
    print("✅ 描述完成!")
    print("=" * 70)
    
    # 保存结果到 JSON
    import json
    output_file = "output/image_descriptions.json"
    os.makedirs("output", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
