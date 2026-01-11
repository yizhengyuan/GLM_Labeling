#!/usr/bin/env python3
"""
测试 HTSRB (Hong Kong Traffic Sign Recognition Benchmark) 模型

模型信息：
- 架构: ViT-B/16 (Vision Transformer Base, patch size 16)
- 类别数: 66 (0-64: 65种交通标志, 65: other)
- 测试准确率: 92.8%
- 输入尺寸: 224x224
"""

import json
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
from pathlib import Path
import sys

# 路径配置
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "htsrb_repo" / "vit_b_16_best_layer8.pth"
LABEL_MAPPING_PATH = BASE_DIR / "htsrb_repo" / "label_mapping.json"


def load_label_mapping():
    """加载标签映射"""
    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    # 转换为 list，按 index 排序
    idx_to_label = mapping["idx_to_label"]
    num_classes = mapping["num_classes"]
    labels = [idx_to_label[str(i)] for i in range(num_classes)]
    return labels


# 加载类别标签
CLASS_LABELS = load_label_mapping()


def load_model(model_path: Path, num_classes: int = 66):
    """加载 ViT-B/16 模型"""
    # 创建 ViT-B/16 模型
    model = vit_b_16(weights=None)

    # 替换分类头以匹配 66 类
    model.heads.head = nn.Linear(768, num_classes)

    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    print(f"模型加载成功！")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Test Acc: {checkpoint['test_acc']:.2f}%")

    return model


def get_transform():
    """获取图像预处理 transform"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict_image(model, image_path: Path, transform, top_k: int = 5):
    """预测单张图片"""
    # 加载图片
    image = Image.open(image_path).convert('RGB')

    # 预处理
    input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    # 推理
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    # 获取 top-k 预测
    top_probs, top_indices = torch.topk(probabilities, top_k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            'class_id': idx.item(),
            'class_name': CLASS_LABELS[idx.item()],
            'probability': prob.item()
        })

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="测试 HTSRB 模型")
    parser.add_argument("images", nargs="+", help="要测试的图片路径")
    parser.add_argument("--top-k", type=int, default=5, help="显示前 k 个预测结果")
    args = parser.parse_args()

    # 加载模型
    print("=" * 50)
    print("加载 HTSRB 模型...")
    model = load_model(MODEL_PATH)
    transform = get_transform()
    print("=" * 50)

    # 预测每张图片
    for image_path in args.images:
        path = Path(image_path)
        if not path.exists():
            print(f"\n❌ 文件不存在: {path}")
            continue

        print(f"\n📷 {path.name}")
        print("-" * 40)

        results = predict_image(model, path, transform, args.top_k)

        for i, r in enumerate(results):
            prob_bar = "█" * int(r['probability'] * 20)
            print(f"  {i+1}. [{r['class_id']:2d}] {r['class_name']:<30} {r['probability']*100:5.1f}% {prob_bar}")


if __name__ == "__main__":
    main()
