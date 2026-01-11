#!/usr/bin/env python3
"""
🧪 交通标志分类测试 UI v2

完整的测试闭环：浏览 → 分类 → 标注 → 修改Prompt → 重测

核心分类逻辑来自: glm_labeling.core.sign_classifier_v2
测好后，正式标注流程直接调用同一个模块。

用法:
    python scripts/test_sign_ui.py

需要安装:
    pip install gradio
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw

# ============================================================================
# 导入核心分类模块
# ============================================================================

from glm_labeling.core.sign_classifier_v2 import (
    SignClassifierV2,
    MOTORCYCLE_SAFETY_SIGNS,
    DEFAULT_CLASSIFY_PROMPT
)


# ============================================================================
# 可视化函数
# ============================================================================

def crop_and_visualize(image_path: str, bbox: List[int], label: str) -> Image.Image:
    """裁剪标志区域并添加标注"""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # 画框
    draw.rectangle(bbox, outline="red", width=3)
    
    # 添加标签
    short_label = label[:30] + "..." if len(label) > 30 else label
    draw.text((bbox[0], bbox[1] - 20), short_label, fill="red")
    
    return img


def get_sign_crop(image_path: str, bbox: List[int]) -> Image.Image:
    """获取放大的标志裁剪"""
    img = Image.open(image_path)
    padding = 20
    x1 = max(0, bbox[0] - padding)
    y1 = max(0, bbox[1] - padding)
    x2 = min(img.width, bbox[2] + padding)
    y2 = min(img.height, bbox[3] + padding)
    
    crop = img.crop((x1, y1, x2, y2))
    # 放大以便查看
    new_size = (crop.width * 4, crop.height * 4)
    return crop.resize(new_size, Image.Resampling.LANCZOS)


# ============================================================================
# 测试数据加载
# ============================================================================

def find_test_samples(archive_dir: str, label_filter: str = None, max_samples: int = 20) -> List[Dict]:
    """从归档数据中查找测试样本"""
    samples = []
    archive_path = Path(archive_dir)
    
    for dataset_dir in archive_path.glob("*_dataset"):
        for sub_dir in dataset_dir.glob("*_dataset"):
            ann_dir = sub_dir / "annotations"
            frames_dir = sub_dir / "frames"
            
            if not ann_dir.exists() or not frames_dir.exists():
                continue
            
            for json_path in ann_dir.glob("*.json"):
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
                
                for shape in data.get("shapes", []):
                    if shape.get("flags", {}).get("category") != "traffic_sign":
                        continue
                    
                    label = shape.get("label", "")
                    if label_filter and label_filter not in label:
                        continue
                    
                    frame_path = frames_dir / data.get("imagePath", "")
                    if not frame_path.exists():
                        continue
                    
                    pts = shape["points"]
                    samples.append({
                        "image": str(frame_path),
                        "bbox": [int(pts[0][0]), int(pts[0][1]), int(pts[1][0]), int(pts[1][1])],
                        "old_label": label,
                        "source": sub_dir.name,
                        "judgment": None,  # 人眼判断: correct/wrong/uncertain
                        "new_prediction": None,
                    })
                    
                    if len(samples) >= max_samples:
                        return samples
    
    return samples


# ============================================================================
# Gradio UI v2
# ============================================================================

def create_ui():
    """创建增强版 Gradio 界面"""
    try:
        import gradio as gr
    except ImportError:
        print("请安装 gradio: pip install gradio")
        return
    
    api_key = os.getenv("ZAI_API_KEY")
    if not api_key:
        print("⚠️ 请设置 ZAI_API_KEY 环境变量")
    
    # 状态变量
    archive_dir = "dataset_output/archive/v1_188signs_20241217"
    test_samples = []
    current_idx = [0]
    
    # 核心分类器实例（复用，避免重复创建）
    classifier = SignClassifierV2(api_key=api_key)
    
    # ========== 核心函数 ==========
    
    def load_samples(label_filter: str, max_count: int):
        nonlocal test_samples
        filter_val = label_filter if label_filter != "全部" else None
        test_samples = find_test_samples(archive_dir, filter_val, int(max_count))
        current_idx[0] = 0
        if test_samples:
            return (
                f"✅ 加载 {len(test_samples)} 个样本",
                *show_current(),
                get_stats_text()
            )
        return ("❌ 没有找到样本", None, None, "", "", "", get_stats_text())
    
    def show_current():
        if not test_samples:
            return None, None, "", "", ""
        
        sample = test_samples[current_idx[0]]
        full_img = crop_and_visualize(sample["image"], sample["bbox"], sample["old_label"])
        crop_img = get_sign_crop(sample["image"], sample["bbox"])
        
        # 判断状态显示
        judgment = sample.get("judgment")
        judgment_icon = {"correct": "✅", "wrong": "❌", "uncertain": "❓"}.get(judgment, "⚪")
        
        # 提取文件名
        img_path = Path(sample["image"])
        file_name = img_path.name
        
        info = f"📍 样本 {current_idx[0] + 1}/{len(test_samples)} {judgment_icon}\n"
        info += f"📄 文件: {file_name}\n"
        info += f"📁 路径: {sample['image']}\n"
        info += f"📦 来源: {sample['source']}\n"
        info += f"🏷️ 旧标签: {sample['old_label'][:40]}..."
        
        # 如果有新预测，显示对比
        result = ""
        if sample.get("new_prediction"):
            result = f"🔮 新预测: {sample['new_prediction']}\n"
            result += f"🏷️ 旧标签: {sample['old_label'][:50]}..."
        
        return full_img, crop_img, info, sample["old_label"], result
    
    def next_sample():
        if test_samples:
            current_idx[0] = (current_idx[0] + 1) % len(test_samples)
        return (*show_current(), get_stats_text())
    
    def prev_sample():
        if test_samples:
            current_idx[0] = (current_idx[0] - 1) % len(test_samples)
        return (*show_current(), get_stats_text())
    
    def run_classification(use_shuffle: bool, custom_prompt: str):
        """使用核心模块进行分类"""
        if not test_samples or not api_key:
            return "❌ 没有样本或 API Key 未设置", ""
        
        sample = test_samples[current_idx[0]]
        
        # 更新分类器配置
        classifier.use_shuffle = use_shuffle
        if custom_prompt.strip():
            classifier.set_prompt(custom_prompt)
        else:
            classifier.set_prompt(DEFAULT_CLASSIFY_PROMPT)
        
        # 运行异步分类（使用核心模块）
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        predicted, description, debug = loop.run_until_complete(
            classifier.classify(sample["image"], sample["bbox"])
        )
        loop.close()
        
        # 保存预测结果
        sample["new_prediction"] = predicted
        
        result = f"🔮 新预测: {predicted}\n\n"
        result += f"📝 模型描述:\n{description}\n\n"
        result += f"🏷️ 旧标签: {sample['old_label'][:60]}...\n\n"
        
        if predicted == "other":
            result += "ℹ️ 分类为 other（导航/方向/倒计时等）"
        elif predicted == sample['old_label']:
            result += "✅ 与旧标签一致"
        else:
            result += "⚠️ 与旧标签不同"
        
        return result, debug
    
    # ========== 人眼标注 ==========
    
    def mark_correct():
        if test_samples:
            test_samples[current_idx[0]]["judgment"] = "correct"
        return (*next_sample(),)
    
    def mark_wrong():
        if test_samples:
            test_samples[current_idx[0]]["judgment"] = "wrong"
        return (*next_sample(),)
    
    def mark_uncertain():
        if test_samples:
            test_samples[current_idx[0]]["judgment"] = "uncertain"
        return (*next_sample(),)
    
    # ========== 统计 ==========
    
    def get_stats_text():
        if not test_samples:
            return "📊 统计: 暂无数据"
        
        total = len(test_samples)
        tested = sum(1 for s in test_samples if s.get("new_prediction"))
        correct = sum(1 for s in test_samples if s.get("judgment") == "correct")
        wrong = sum(1 for s in test_samples if s.get("judgment") == "wrong")
        uncertain = sum(1 for s in test_samples if s.get("judgment") == "uncertain")
        
        accuracy = (correct / (correct + wrong) * 100) if (correct + wrong) > 0 else 0
        
        return (f"📊 已标注: {correct + wrong + uncertain}/{total} | "
                f"✅ 正确: {correct} | ❌ 错误: {wrong} | ❓ 不确定: {uncertain} | "
                f"准确率: {accuracy:.1f}%")
    
    # ========== 导出 ==========
    
    def export_results():
        if not test_samples:
            return "❌ 没有数据可导出"
        
        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_samples": len(test_samples),
            "prompt_used": classifier.prompt_template,
            "core_module": "glm_labeling.core.sign_classifier_v2",
            "results": []
        }
        
        for s in test_samples:
            export_data["results"].append({
                "image": s["image"],
                "bbox": s["bbox"],
                "old_label": s["old_label"],
                "new_prediction": s.get("new_prediction"),
                "judgment": s.get("judgment"),
                "source": s["source"]
            })
        
        # 统计
        correct = sum(1 for s in test_samples if s.get("judgment") == "correct")
        wrong = sum(1 for s in test_samples if s.get("judgment") == "wrong")
        export_data["stats"] = {
            "correct": correct,
            "wrong": wrong,
            "accuracy": (correct / (correct + wrong) * 100) if (correct + wrong) > 0 else 0
        }
        
        # 保存
        output_path = f"tests/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("tests", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return f"✅ 已导出到 {output_path}"
    
    # ========== 只看错误 ==========
    
    def show_only_wrong():
        if not test_samples:
            return "❌ 没有数据", None, None, "", "", "", get_stats_text()
        
        # 找第一个标记为错误的
        for i, s in enumerate(test_samples):
            if s.get("judgment") == "wrong":
                current_idx[0] = i
                return (f"🔍 显示第 {i+1} 个错误样本", *show_current(), get_stats_text())
        
        return ("ℹ️ 没有标记为错误的样本", *show_current(), get_stats_text())
    
    def next_wrong():
        if not test_samples:
            return (*show_current(), get_stats_text())
        
        start = current_idx[0]
        for i in range(1, len(test_samples)):
            idx = (start + i) % len(test_samples)
            if test_samples[idx].get("judgment") == "wrong":
                current_idx[0] = idx
                break
        
        return (*show_current(), get_stats_text())
    
    # ========== 构建界面 ==========
    
    with gr.Blocks(title="🧪 交通标志分类测试 v2") as demo:
        gr.Markdown("# 🧪 交通标志分类测试 UI v2")
        gr.Markdown("""
完整闭环：**浏览** → **分类** → **人眼判断** → **修改Prompt** → **重测**

📦 核心模块：`glm_labeling.core.sign_classifier_v2` (测好后正式流程直接调用)
""")
        
        # ===== 加载区域 =====
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    label_filter = gr.Dropdown(
                        choices=["全部", "100m_Countdown", "Speed_limit", "No_stopping", "Direction"],
                        value="100m_Countdown",
                        label="标签过滤"
                    )
                    max_samples = gr.Slider(5, 100, value=20, step=5, label="最大样本数")
                    load_btn = gr.Button("🔄 加载样本", variant="primary")
            
            with gr.Column():
                status_text = gr.Textbox(label="状态", interactive=False, lines=1)
        
        # ===== 图片区域 =====
        with gr.Row():
            with gr.Column():
                full_image = gr.Image(label="🖼️ 完整图片（红框标注位置）", type="pil", height=400)
            
            with gr.Column():
                crop_image = gr.Image(label="🔍 标志放大 (4x)", type="pil", height=200)
                info_text = gr.Textbox(label="样本信息", interactive=False, lines=5)
        
        # ===== 导航按钮 =====
        with gr.Row():
            prev_btn = gr.Button("⬅️ 上一张")
            next_btn = gr.Button("➡️ 下一张")
            wrong_btn = gr.Button("🔍 只看错误", variant="secondary")
            next_wrong_btn = gr.Button("➡️ 下一个错误")
        
        # ===== 分类区域 =====
        gr.Markdown("---")
        gr.Markdown("### 🧪 运行分类测试（使用核心模块）")
        
        with gr.Row():
            with gr.Column():
                shuffle_checkbox = gr.Checkbox(value=True, label="🔀 随机打乱候选顺序（消除位置偏差）")
                classify_btn = gr.Button("🚀 运行分类", variant="primary", size="lg")
            
            with gr.Column():
                result_text = gr.Textbox(label="分类结果", interactive=False, lines=6)
        
        # ===== 人眼标注区域 =====
        gr.Markdown("---")
        gr.Markdown("### 👁️ 人眼判断（标注后自动跳转下一张）")
        
        with gr.Row():
            correct_btn = gr.Button("✅ 正确", variant="primary", size="lg")
            wrong_btn2 = gr.Button("❌ 错误", variant="stop", size="lg")
            uncertain_btn = gr.Button("❓ 不确定", variant="secondary", size="lg")
        
        # ===== 统计面板 =====
        stats_text = gr.Textbox(label="📊 实时统计", interactive=False, lines=1)
        
        # ===== Prompt 编辑器 =====
        gr.Markdown("---")
        with gr.Accordion("📝 Prompt 编辑器（点击展开）", open=False):
            prompt_editor = gr.Textbox(
                label="分类 Prompt 模板",
                value=DEFAULT_CLASSIFY_PROMPT,
                lines=15,
                info="使用 {candidate_list} 作为候选列表占位符"
            )
            with gr.Row():
                reset_prompt_btn = gr.Button("🔄 重置为默认")
                gr.Markdown("修改后点击「运行分类」即可使用新 Prompt")
        
        # ===== 导出区域 =====
        gr.Markdown("---")
        with gr.Row():
            export_btn = gr.Button("📥 导出测试结果")
            export_status = gr.Textbox(label="导出状态", interactive=False)
        
        # ===== 调试信息 =====
        with gr.Accordion("🔧 调试信息（点击展开）", open=False):
            debug_text = gr.Textbox(label="模型原始输出", interactive=False, lines=8)
            old_label_hidden = gr.Textbox(visible=False)
        
        # ========== 事件绑定 ==========
        
        # 加载样本
        load_btn.click(
            load_samples,
            inputs=[label_filter, max_samples],
            outputs=[status_text, full_image, crop_image, info_text, old_label_hidden, result_text, stats_text]
        )
        
        # 导航
        next_btn.click(
            next_sample,
            outputs=[full_image, crop_image, info_text, old_label_hidden, result_text, stats_text]
        )
        prev_btn.click(
            prev_sample,
            outputs=[full_image, crop_image, info_text, old_label_hidden, result_text, stats_text]
        )
        
        # 分类
        classify_btn.click(
            run_classification,
            inputs=[shuffle_checkbox, prompt_editor],
            outputs=[result_text, debug_text]
        )
        
        # 人眼标注
        correct_btn.click(
            mark_correct,
            outputs=[full_image, crop_image, info_text, old_label_hidden, result_text, stats_text]
        )
        wrong_btn2.click(
            mark_wrong,
            outputs=[full_image, crop_image, info_text, old_label_hidden, result_text, stats_text]
        )
        uncertain_btn.click(
            mark_uncertain,
            outputs=[full_image, crop_image, info_text, old_label_hidden, result_text, stats_text]
        )
        
        # 只看错误
        wrong_btn.click(
            show_only_wrong,
            outputs=[status_text, full_image, crop_image, info_text, old_label_hidden, result_text, stats_text]
        )
        next_wrong_btn.click(
            next_wrong,
            outputs=[full_image, crop_image, info_text, old_label_hidden, result_text, stats_text]
        )
        
        # Prompt 重置
        reset_prompt_btn.click(
            lambda: DEFAULT_CLASSIFY_PROMPT,
            outputs=[prompt_editor]
        )
        
        # 导出
        export_btn.click(
            export_results,
            outputs=[export_status]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    if demo:
        print("\n" + "="*60)
        print("🧪 交通标志分类测试 UI v2")
        print("="*60)
        print("📦 核心模块: glm_labeling.core.sign_classifier_v2")
        print("📍 浏览器访问: http://127.0.0.1:7861")
        print("="*60)
        demo.launch(share=False, server_port=7861)
