#!/usr/bin/env python3
"""
AI Model Evaluation UI
用于手动评估AI模型分类结果的UI工具

参考标志图片来自 raw_data/signs 目录
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import flask
from flask import Flask, render_template_string, request, jsonify, send_file

app = Flask(__name__)

# 参考标志目录（188种标准交通标志）
REFERENCE_SIGNS_DIR = Path(__file__).parent.parent / "raw_data" / "188_signs_revised"


class EvaluationData:
    """评估数据管理类"""

    def __init__(self, base_dir: str, reference_dir: Path = None, frequency_file: Path = None):
        self.base_dir = Path(base_dir)
        self.predictions_file = self.base_dir / "predictions.json"
        # 使用目录名作为标签文件名前缀
        self.annotations_file = self.base_dir / f"{self.base_dir.name}_labels.json"
        self.export_file = self.base_dir / "evaluation_report.json"
        self.reference_dir = reference_dir or REFERENCE_SIGNS_DIR
        self.frequency_file = frequency_file

        self.predictions: Dict = {}
        self.annotations: Dict = {}
        self.labels: set = set()
        self.reference_signs: List[Dict] = []
        self.sign_frequency: Dict[str, int] = {}

        self._load_frequency()
        self.load_data()
        self._load_reference_signs()

    def load_data(self):
        """加载预测数据和已有的标注"""
        # 加载预测结果
        if self.predictions_file.exists():
            with open(self.predictions_file, 'r') as f:
                data = json.load(f)
                self.predictions = data.get('predictions', {})
                self.labels = set(p.get('label', '') for p in self.predictions.values())
        else:
            print(f"警告: 未找到 {self.predictions_file}")

        # 加载已有的标注
        if self.annotations_file.exists():
            with open(self.annotations_file, 'r') as f:
                self.annotations = json.load(f)

    def save_annotation(self, image_name: str, is_correct: bool, correct_label: Optional[str] = None):
        """保存标注结果"""
        self.annotations[image_name] = {
            'is_correct': is_correct,
            'correct_label': correct_label if not is_correct else None,
            'timestamp': datetime.now().isoformat()
        }
        self._save_annotations()

    def _load_frequency(self):
        """加载历史标注频率"""
        self.sign_frequency = {}
        if self.frequency_file and self.frequency_file.exists():
            try:
                with open(self.frequency_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.sign_frequency = data.get("label_statistics", {})
                    print(f"加载历史频率: {len(self.sign_frequency)} 种标志")
                    if self.sign_frequency:
                        top3 = sorted(self.sign_frequency.items(), key=lambda x: -x[1])[:3]
                        top3_str = ", ".join([f"{k}({v})" for k, v in top3])
                        print(f"  TOP 3: {top3_str}")
            except Exception as e:
                print(f"警告: 无法加载频率文件: {e}")

    def _save_annotations(self):
        """保存标注到文件，包含统计信息"""
        # 计算统计
        total = len([k for k in self.annotations if not k.startswith('_')])
        correct = sum(1 for k, v in self.annotations.items() if not k.startswith('_') and v.get('is_correct') == True)
        incorrect = sum(1 for k, v in self.annotations.items() if not k.startswith('_') and v.get('is_correct') == False)
        accuracy = round(correct / (correct + incorrect) * 100, 1) if (correct + incorrect) > 0 else 0

        # 添加统计信息
        self.annotations['_summary'] = {
            'total': total,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': accuracy
        }

        with open(self.annotations_file, 'w') as f:
            json.dump(self.annotations, f, indent=2, ensure_ascii=False)

    def _load_reference_signs(self):
        """加载参考标志图片列表，按历史频率排序"""
        self.reference_signs = []
        if not self.reference_dir.exists():
            print(f"警告: 参考标志目录不存在 {self.reference_dir}")
            return

        sign_files = sorted(self.reference_dir.glob("*.png"))

        signs_data = []
        for f in sign_files:
            name = f.stem
            display_name = name.replace("_", " ")
            freq = self.sign_frequency.get(name, 0)
            signs_data.append({
                "name": name,
                "display_name": display_name,
                "path": str(f),
                "relative_path": str(f.relative_to(self.reference_dir.parent.parent)),
                "frequency": freq
            })

        # 按频率降序排序，频率为 0 的按字母排序
        sorted_signs = sorted(
            signs_data,
            key=lambda x: (-x["frequency"], x["name"])
        )

        self.reference_signs = sorted_signs

        with_freq = sum(1 for s in sorted_signs if s["frequency"] > 0)
        print(f"加载了 {len(self.reference_signs)} 个参考标志 ({with_freq} 种有历史记录)")

    def get_predictions_list(self) -> List[Dict]:
        """获取预测结果列表（带标注状态）"""
        result = []
        for img_name, pred in self.predictions.items():
            annotated = img_name in self.annotations
            annotation = self.annotations.get(img_name, {})
            result.append({
                'image_name': img_name,
                'predicted_label': pred.get('label', ''),
                'confidence': pred.get('confidence', 0),
                'annotated': annotated,
                'is_correct': annotation.get('is_correct'),
                'correct_label': annotation.get('correct_label')
            })
        return result

    def get_statistics(self) -> Dict:
        """计算统计信息"""
        total = len(self.predictions)
        annotated = len(self.annotations)

        if annotated == 0:
            return {
                'total': total,
                'annotated': 0,
                'unannotated': total,
                'correct': 0,
                'incorrect': 0,
                'accuracy': 0,
                'label_accuracy': {},
                'confusion_matrix': {}
            }

        correct = sum(1 for a in self.annotations.values() if a.get('is_correct'))
        incorrect = annotated - correct
        accuracy = correct / annotated if annotated > 0 else 0

        # 按标签统计准确率
        label_stats: Dict[str, Dict] = {}
        for img_name, pred in self.predictions.items():
            if img_name not in self.annotations:
                continue
            label = pred.get('label', 'unknown')
            if label not in label_stats:
                label_stats[label] = {'correct': 0, 'total': 0}
            label_stats[label]['total'] += 1
            if self.annotations[img_name].get('is_correct'):
                label_stats[label]['correct'] += 1

        label_accuracy = {
            label: {
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                'total': stats['total'],
                'correct': stats['correct']
            }
            for label, stats in label_stats.items()
        }

        # 混淆矩阵（预测标签 vs 实际标签）
        confusion = {}
        for img_name, pred in self.predictions.items():
            if img_name not in self.annotations:
                continue
            predicted = pred.get('label', 'unknown')
            annotation = self.annotations[img_name]
            if not annotation.get('is_correct'):
                actual = annotation.get('correct_label', 'unknown')
                key = f"{predicted} -> {actual}"
                confusion[key] = confusion.get(key, 0) + 1

        return {
            'total': total,
            'annotated': annotated,
            'unannotated': total - annotated,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': accuracy,
            'label_accuracy': label_accuracy,
            'confusion_matrix': confusion
        }

    def export_report(self) -> Dict:
        """导出完整评估报告"""
        stats = self.get_statistics()

        report = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_images': stats['total'],
                'annotated_images': stats['annotated']
            },
            'statistics': stats,
            'detailed_annotations': []
        }

        for img_name, pred in self.predictions.items():
            annotation = self.annotations.get(img_name)
            report['detailed_annotations'].append({
                'image': img_name,
                'predicted_label': pred.get('label', ''),
                'confidence': pred.get('confidence', 0),
                'is_correct': annotation.get('is_correct') if annotation else None,
                'correct_label': annotation.get('correct_label') if annotation else None,
                'annotated_at': annotation.get('timestamp') if annotation else None
            })

        # 保存报告
        with open(self.export_file, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report


# 全局数据管理器
eval_data: Optional[EvaluationData] = None

# 默认历史频率文件（用于排序参考标志）
DEFAULT_FREQUENCY_FILE = Path(__file__).parent.parent / "extracted_signs" / "DJI_batch1" / "label_results_batch1.json"


def get_eval_data() -> EvaluationData:
    """获取当前评估数据实例"""
    global eval_data
    if eval_data is None:
        # 默认路径
        base_dir = Path(__file__).parent.parent / "extracted_signs" / "auto_label_tests" / "DJI_0020"
        eval_data = EvaluationData(str(base_dir), frequency_file=DEFAULT_FREQUENCY_FILE)
    return eval_data


@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/references')
def get_references():
    """获取参考标志列表"""
    data = get_eval_data()
    return jsonify({"references": data.reference_signs})


@app.route('/api/reference-image/<path:image_path>')
def get_reference_image(image_path):
    """获取参考标志图片"""
    from urllib.parse import unquote
    # 解码 URL 编码的路径
    decoded_path = unquote(image_path)
    full_path = REFERENCE_SIGNS_DIR.parent.parent / decoded_path
    if full_path.exists():
        return send_file(str(full_path))
    return "Not found", 404


@app.route('/api/set_folder', methods=['POST'])
def set_folder():
    """设置要评估的文件夹"""
    data = request.json
    folder_name = data.get('folder')

    base_dir = Path(__file__).parent.parent / "extracted_signs" / "auto_label_tests"
    target_dir = base_dir / folder_name

    if not target_dir.exists():
        return jsonify({'error': f'文件夹不存在: {folder_name}'}), 404

    predictions_file = target_dir / "predictions.json"
    if not predictions_file.exists():
        return jsonify({'error': f'未找到 predictions.json 文件'}), 404

    global eval_data
    eval_data = EvaluationData(str(target_dir), frequency_file=DEFAULT_FREQUENCY_FILE)

    return jsonify({
        'message': f'已加载 {folder_name}',
        'total': len(eval_data.predictions)
    })


@app.route('/api/folders')
def list_folders():
    """列出可用的评估文件夹"""
    # 扫描多个目录
    search_dirs = [
        Path(__file__).parent.parent / "extracted_signs" / "auto_label_tests",
        Path(__file__).parent.parent / "extracted_signs"  # 也扫描 extracted_signs 根目录
    ]
    folders = []
    seen_names = set()

    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        for item in base_dir.iterdir():
            if item.is_dir() and (item / "predictions.json").exists():
                # 避免重复
                folder_key = str(item)
                if folder_key in seen_names:
                    continue
                seen_names.add(folder_key)

                # 统计信息
                pred_file = item / "predictions.json"
                try:
                    with open(pred_file, 'r') as f:
                        pred_data = json.load(f)
                        total = len(pred_data.get('predictions', {}))
                except:
                    total = 0

                # 检查是否有标注
                ann_file = item / "annotations.json"
                annotated = 0
                if ann_file.exists():
                    try:
                        with open(ann_file, 'r') as f:
                            annotations = json.load(f)
                            annotated = len(annotations)
                    except:
                        pass

                folders.append({
                    'name': item.name,
                    'path': str(item),  # 添加完整路径
                    'total': total,
                    'annotated': annotated
                })

    # 按名称排序
    folders.sort(key=lambda x: x['name'])
    return jsonify({'folders': folders})


@app.route('/api/image/<path:filename>')
def get_image(filename):
    """获取图片"""
    from urllib.parse import unquote
    data = get_eval_data()
    # 解码 URL 编码的文件名
    decoded_filename = unquote(filename)

    # 尝试多个可能的路径
    possible_paths = [
        data.base_dir / decoded_filename,
        data.base_dir / "pictures" / decoded_filename,
    ]

    for image_path in possible_paths:
        if image_path.exists():
            return send_file(str(image_path))

    return "Not found", 404


@app.route('/api/predictions')
def get_predictions():
    """获取所有预测结果"""
    data = get_eval_data()
    filter_type = request.args.get('filter', 'all')  # all, unannotated, annotated, correct, incorrect
    label_filter = request.args.get('label', '')

    predictions = data.get_predictions_list()

    # 过滤
    if filter_type == 'unannotated':
        predictions = [p for p in predictions if not p['annotated']]
    elif filter_type == 'annotated':
        predictions = [p for p in predictions if p['annotated']]
    elif filter_type == 'correct':
        predictions = [p for p in predictions if p.get('is_correct') is True]
    elif filter_type == 'incorrect':
        predictions = [p for p in predictions if p.get('is_correct') is False]

    if label_filter:
        predictions = [p for p in predictions if p['predicted_label'] == label_filter]

    return jsonify({
        'predictions': predictions,
        'total': len(predictions),
        'labels': sorted(list(data.labels))
    })


@app.route('/api/prediction/<filename>')
def get_prediction(filename):
    """获取单个预测结果"""
    data = get_eval_data()

    if filename not in data.predictions:
        return jsonify({'error': '未找到该图片的预测'}), 404

    pred = data.predictions[filename]
    annotation = data.annotations.get(filename, {})

    # 获取相邻图片（用于导航）
    filenames = list(data.predictions.keys())
    try:
        current_idx = filenames.index(filename)
    except ValueError:
        current_idx = 0

    return jsonify({
        'image_name': filename,
        'predicted_label': pred.get('label', ''),
        'confidence': pred.get('confidence', 0),
        'annotated': filename in data.annotations,
        'is_correct': annotation.get('is_correct'),
        'correct_label': annotation.get('correct_label'),
        'current_index': current_idx,
        'total_count': len(filenames),
        'prev_image': filenames[current_idx - 1] if current_idx > 0 else None,
        'next_image': filenames[current_idx + 1] if current_idx < len(filenames) - 1 else None
    })


@app.route('/api/annotate', methods=['POST'])
def annotate():
    """提交标注"""
    req_data = request.json
    filename = req_data.get('filename')
    is_correct = req_data.get('is_correct')
    correct_label = req_data.get('correct_label')

    if not filename or is_correct is None:
        return jsonify({'error': '缺少必要参数'}), 400

    data = get_eval_data()
    data.save_annotation(filename, is_correct, correct_label)

    return jsonify({'success': True})


@app.route('/api/statistics')
def statistics():
    """获取统计信息"""
    data = get_eval_data()
    stats = data.get_statistics()
    return jsonify(stats)


@app.route('/api/export')
def export_report():
    """导出评估报告"""
    data = get_eval_data()
    report = data.export_report()
    return jsonify(report)


@app.route('/api/batch_annotate', methods=['POST'])
def batch_annotate():
    """批量标注（跳过）"""
    req_data = request.json
    filenames = req_data.get('filenames', [])
    is_correct = req_data.get('is_correct', True)

    data = get_eval_data()
    for filename in filenames:
        if filename in data.predictions and filename not in data.annotations:
            data.save_annotation(filename, is_correct)

    return jsonify({'success': True, 'count': len(filenames)})


def main():
    import argparse

    parser = argparse.ArgumentParser(description='AI模型评估UI')
    parser.add_argument('--data-dir', type=str,
                        help='评估数据目录路径（包含predictions.json）')
    parser.add_argument('--frequency-file', type=str,
                        help=f'历史频率文件路径（默认: {DEFAULT_FREQUENCY_FILE}）')
    parser.add_argument('--port', type=int, default=5005,
                        help='服务端口')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='服务地址')

    args = parser.parse_args()

    global eval_data
    if args.data_dir:
        freq_file = Path(args.frequency_file) if args.frequency_file else DEFAULT_FREQUENCY_FILE
        eval_data = EvaluationData(args.data_dir, frequency_file=freq_file)

    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║           AI 模型评估 UI                                   ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  访问地址: http://{args.host}:{args.port}                      ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    app.run(host=args.host, port=args.port, debug=True)


# ============ HTML Template ============
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 模型分类评估</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #ffffff;
            color: #1a1a1a;
            height: 100vh;
            overflow: hidden;
        }

        /* 顶部导航栏 */
        .header {
            padding: 12px 24px;
            background: #f8f9fa;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #e0e0e0;
        }
        .header h1 {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1a1a1a;
        }
        .header h1 span { color: #0066cc; }

        .header-controls {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .folder-select {
            background: #ffffff;
            color: #1a1a1a;
            border: 1px solid #d0d0d0;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 13px;
            cursor: pointer;
            min-width: 160px;
        }

        .stats-bar {
            display: flex;
            gap: 20px;
            font-size: 13px;
            color: #666;
        }
        .stat-value {
            font-weight: 600;
            color: #0066cc;
        }
        .stat-value.correct { color: #28a745; }
        .stat-value.incorrect { color: #dc3545; }
        .stat-value.accuracy { color: #fd7e14; }

        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .btn:hover { transform: translateY(-1px); }
        .btn-primary {
            background: #0066cc;
            color: white;
        }
        .btn-primary:hover { background: #0052a3; }

        /* 进度条 */
        .progress-bar {
            height: 3px;
            background: #e0e0e0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #0066cc);
            transition: width 0.3s;
        }

        /* 主内容区 */
        .main-container {
            display: flex;
            height: calc(100vh - 100px);
        }

        /* 左侧：待标注图片 */
        .left-panel {
            width: 40%;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            border-right: 1px solid #e0e0e0;
            background: #fafafa;
        }

        .image-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f0f0f0;
            border-radius: 12px;
            width: 100%;
            margin-bottom: 16px;
            border: 1px solid #e0e0e0;
        }
        .image-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 8px;
        }

        .image-info {
            color: #666;
            font-size: 13px;
            text-align: center;
            margin-bottom: 8px;
        }

        .prediction-info {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
            padding: 10px 16px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }
        .prediction-label {
            font-weight: 500;
            color: #1a1a1a;
        }
        .confidence-badge {
            padding: 4px 8px;
            background: #e9ecef;
            border-radius: 4px;
            font-size: 11px;
        }
        .confidence-badge.high { background: #d4edda; color: #155724; }
        .confidence-badge.medium { background: #fff3cd; color: #856404; }
        .confidence-badge.low { background: #f8d7da; color: #721c24; }

        .annotation-status {
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            margin-bottom: 12px;
        }
        .annotation-status.unannotated {
            background: #e9ecef;
            color: #666;
        }
        .annotation-status.correct {
            background: #d4edda;
            color: #155724;
        }
        .annotation-status.incorrect {
            background: #f8d7da;
            color: #721c24;
        }

        .nav-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
        }
        .btn-nav {
            background: #f0f0f0;
            border: 1px solid #d0d0d0;
            color: #333;
        }
        .btn-nav:hover {
            background: #e8e8e8;
            border-color: #0066cc;
            color: #0066cc;
        }
        .btn-correct {
            background: #28a745;
            color: white;
        }
        .btn-correct:hover { background: #218838; }
        .btn-incorrect {
            background: #dc3545;
            color: white;
        }
        .btn-incorrect:hover { background: #c82333; }
        .btn-other {
            background: #fd7e14;
            color: white;
        }
        .btn-other:hover { background: #e06c0a; }

        /* 右侧：参考标志 */
        .right-panel {
            width: 60%;
            display: flex;
            flex-direction: column;
            background: #ffffff;
        }

        .search-bar {
            padding: 12px 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
        }
        .search-input {
            width: 100%;
            padding: 10px 14px;
            background: #ffffff;
            border: 1px solid #d0d0d0;
            border-radius: 8px;
            color: #1a1a1a;
            font-size: 14px;
        }
        .search-input:focus {
            outline: none;
            border-color: #0066cc;
        }

        .reference-grid {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 10px;
            align-content: start;
        }
        .reference-item {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 6px;
            cursor: pointer;
            border: 2px solid #e0e0e0;
            transition: all 0.2s;
        }
        .reference-item:hover {
            border-color: #0066cc;
            transform: translateY(-2px);
        }
        .reference-item.selected {
            border-color: #28a745;
            background: #e8f5e9;
        }
        .reference-item img {
            width: 100%;
            aspect-ratio: 1;
            object-fit: contain;
            background: #ffffff;
            border-radius: 4px;
        }
        .reference-item .name {
            margin-top: 4px;
            font-size: 10px;
            color: #666;
            text-align: center;
            line-height: 1.3;
            max-height: 2.6em;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        .shortcuts-hint {
            padding: 8px 20px;
            background: #f8f9fa;
            color: #666;
            font-size: 12px;
            border-top: 1px solid #e0e0e0;
        }
        .shortcuts-hint kbd {
            background: #e9ecef;
            padding: 2px 6px;
            border-radius: 4px;
            margin: 0 2px;
            font-family: monospace;
        }

        /* 报告模态框 */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        .modal.show { display: flex; }
        .modal-content {
            background: white;
            border-radius: 12px;
            padding: 24px;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .modal-title {
            font-size: 18px;
            font-weight: 500;
        }
        .modal-close {
            background: none;
            border: none;
            color: #999;
            font-size: 24px;
            cursor: pointer;
        }

        .report-section {
            margin-bottom: 20px;
        }
        .report-section-title {
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 10px;
            color: #0066cc;
        }
        .report-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }
        .report-item {
            background: #f8f9fa;
            padding: 12px;
            border-radius: 8px;
        }
        .report-item-label {
            font-size: 11px;
            color: #666;
        }
        .report-item-value {
            font-size: 20px;
            font-weight: 500;
            margin-top: 4px;
        }
        .label-accuracy-item {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid #e9ecef;
            font-size: 12px;
        }
        .confusion-item {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            color: #666;
            font-size: 11px;
        }

        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #999;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1><span>AI 模型</span> 分类评估</h1>
        <div class="header-controls">
            <select class="folder-select" id="folderSelect">
                <option value="">选择文件夹...</option>
            </select>
            <div class="stats-bar">
                <span>总计: <span class="stat-value" id="totalStat">0</span></span>
                <span>已标注: <span class="stat-value" id="annotatedStat">0</span></span>
                <span>正确: <span class="stat-value correct" id="correctStat">0</span></span>
                <span>错误: <span class="stat-value incorrect" id="incorrectStat">0</span></span>
                <span>准确率: <span class="stat-value accuracy" id="accuracyStat">0%</span></span>
            </div>
            <button class="btn btn-primary" onclick="showReport()">📊 报告</button>
        </div>
    </div>
    <div class="progress-bar">
        <div class="progress-fill" id="progressFill" style="width: 0%"></div>
    </div>

    <div class="main-container">
        <!-- 左侧：待标注图片 -->
        <div class="left-panel">
            <div class="image-container" id="imageContainer">
                <div class="empty-state">
                    <div style="font-size: 48px;">📷</div>
                    <div>请先选择文件夹</div>
                </div>
            </div>
            <div class="image-info">
                <span id="imageName">-</span>
                <span id="imageIndex"></span>
            </div>
            <div class="prediction-info" id="predictionInfo" style="display: none;">
                <span style="color: #666;">AI 预测:</span>
                <span class="prediction-label" id="predictedLabel">-</span>
                <span class="confidence-badge" id="confidenceBadge">-</span>
            </div>
            <div class="annotation-status unannotated" id="annotationStatus" style="display: none;">未标注</div>
            <div class="nav-buttons">
                <button class="btn btn-nav" onclick="navigate(-1)">◀ 上一张</button>
                <button class="btn btn-correct" onclick="annotate(true)">✓ 正确</button>
                <button class="btn btn-nav" onclick="navigate(1)">下一张 ▶</button>
            </div>
            <div class="special-buttons" style="margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; justify-content: center;">
                <button class="btn btn-special" onclick="markAs('other')" style="background: #6c757d; color: white;">❓ Other</button>
                <button class="btn btn-special" onclick="markAs('blur')" style="background: #4a3728; color: white;">💨 Blur</button>
                <button class="btn btn-special" onclick="markAs('glare')" style="background: #6b4423; color: white;">☀️ Glare</button>
                <button class="btn btn-special" onclick="markAs('lowlight')" style="background: #1e3a5f; color: white;">🌙 Lowlight</button>
            </div>
        </div>

        <!-- 右侧：参考标志 -->
        <div class="right-panel">
            <div class="search-bar">
                <input type="text" class="search-input" id="searchInput"
                       placeholder="搜索标志名称... (按 / 聚焦, Esc 清空)">
            </div>
            <div class="reference-grid" id="referenceGrid">
                <!-- 参考标志将在这里动态生成 -->
            </div>
            <div class="shortcuts-hint">
                快捷键: <kbd>/</kbd> 搜索 | <kbd>Esc</kbd> 清空 | <kbd>←</kbd><kbd>→</kbd> 导航 | <kbd>1</kbd> 正确 | <kbd>2</kbd> Other | <kbd>3</kbd> Blur | <kbd>4</kbd> Glare | <kbd>5</kbd> Lowlight
            </div>
        </div>
    </div>

    <!-- 报告模态框 -->
    <div class="modal" id="reportModal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 class="modal-title">评估报告</h3>
                <button class="modal-close" onclick="closeReport()">&times;</button>
            </div>
            <div id="reportContent"></div>
            <div style="margin-top: 20px; text-align: right;">
                <button class="btn btn-primary" onclick="downloadReport()">下载报告</button>
                <button class="btn" onclick="closeReport()">关闭</button>
            </div>
        </div>
    </div>

    <script>
        let currentData = null;
        let referenceSignsData = [];
        let currentImageName = null;
        let allImageNames = [];

        // 加载文件夹列表
        async function loadFolders() {
            const res = await fetch('/api/folders');
            const data = await res.json();

            const select = document.getElementById('folderSelect');
            select.innerHTML = '<option value="">选择文件夹...</option>';
            data.folders.forEach(folder => {
                const option = document.createElement('option');
                option.value = folder.name;
                option.textContent = `${folder.name} (${folder.annotated}/${folder.total} 已标注)`;
                select.appendChild(option);
            });
        }

        // 加载参考标志
        async function loadReferenceSigns() {
            const res = await fetch('/api/references');
            const data = await res.json();
            referenceSignsData = data.references;
            renderReferences(referenceSignsData);
        }

        function renderReferences(signs) {
            const grid = document.getElementById('referenceGrid');
            grid.innerHTML = signs.map(sign => `
                <div class="reference-item" data-name="${sign.name}" onclick="selectCorrectLabel('${sign.name}')">
                    <img src="/api/reference-image/${encodeURIComponent(sign.relative_path)}" alt="${sign.display_name}">
                    <div class="name">${sign.display_name}</div>
                </div>
            `).join('');
        }

        // 选择文件夹
        async function selectFolder(folderName) {
            if (!folderName) return;

            const res = await fetch('/api/set_folder', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({folder: folderName})
            });
            const data = await res.json();

            // 重新加载参考标志（因为可能更新了）
            await loadReferenceSigns();
            await loadStatistics();

            // 加载第一张未标注的图片
            await loadFirstUnannotated();
        }

        async function loadFirstUnannotated() {
            const res = await fetch('/api/predictions?filter=unannotated');
            const data = await res.json();

            if (data.predictions.length > 0) {
                allImageNames = data.predictions.map(p => p.image_name);
                await loadImage(data.predictions[0].image_name);
            } else {
                // 没有未标注的，加载第一张
                const res2 = await fetch('/api/predictions');
                const data2 = await res2.json();
                if (data2.predictions.length > 0) {
                    allImageNames = data2.predictions.map(p => p.image_name);
                    await loadImage(data2.predictions[0].image_name);
                }
            }
        }

        async function loadImage(filename) {
            currentImageName = filename;

            const res = await fetch(`/api/prediction/${encodeURIComponent(filename)}`);
            const data = await res.json();

            // 更新图片
            document.getElementById('imageContainer').innerHTML =
                `<img src="/api/image/${encodeURIComponent(filename)}" alt="${filename}">`;

            // 更新信息
            document.getElementById('imageName').textContent = data.image_name;
            document.getElementById('imageIndex').textContent = `${data.current_index + 1} / ${data.total_count}`;

            // 更新预测信息
            document.getElementById('predictedLabel').textContent = data.predicted_label.replace(/_/g, ' ');
            const confidence = data.confidence * 100;
            document.getElementById('confidenceBadge').textContent = confidence.toFixed(1) + '%';
            const confidenceLevel = data.confidence >= 0.9 ? 'high' : data.confidence >= 0.7 ? 'medium' : 'low';
            document.getElementById('confidenceBadge').className = `confidence-badge ${confidenceLevel}`;
            document.getElementById('predictionInfo').style.display = 'flex';

            // 更新标注状态
            const statusEl = document.getElementById('annotationStatus');
            statusEl.style.display = 'block';

            if (data.annotated) {
                if (data.is_correct) {
                    statusEl.textContent = '✓ 已标注: 正确';
                    statusEl.className = 'annotation-status correct';
                } else {
                    statusEl.textContent = `✗ 已标注: 应为 ${data.correct_label?.replace(/_/g, ' ') || '(未知)'}`;
                    statusEl.className = 'annotation-status incorrect';
                }
            } else {
                statusEl.textContent = '未标注';
                statusEl.className = 'annotation-status unannotated';
            }

            // 高亮对应的参考图
            document.querySelectorAll('.reference-item').forEach(item => {
                item.classList.toggle('selected', item.dataset.name === data.predicted_label);
            });

            // 保存导航信息
            window.prevImage = data.prev_image;
            window.nextImage = data.next_image;
        }

        async function navigate(direction) {
            const target = direction === -1 ? window.prevImage : window.nextImage;
            if (target) {
                await loadImage(target);
            }
        }

        async function annotate(isCorrect) {
            if (!currentImageName) return;

            if (isCorrect) {
                await submitAnnotation(currentImageName, true, null);
            }
        }

        async function markAs(label) {
            if (!currentImageName) return;
            // 标记为错误，正确标签为指定的 label
            await submitAnnotation(currentImageName, false, label);
        }

        async function selectCorrectLabel(signName) {
            // 点击参考图标记为错误并指定正确标签
            if (!currentImageName) return;
            // 将当前图片标记为错误，使用点击的标签作为正确标签
            await submitAnnotation(currentImageName, false, signName);
        }

        async function submitAnnotation(filename, isCorrect, correctLabel) {
            const res = await fetch('/api/annotate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    filename,
                    is_correct: isCorrect,
                    correct_label: correctLabel
                })
            });

            if (res.ok) {
                await loadStatistics();
                // 自动跳转到下一张未标注的图片
                await loadFirstUnannotated();
            }
        }

        async function loadStatistics() {
            const res = await fetch('/api/statistics');
            const stats = await res.json();

            document.getElementById('totalStat').textContent = stats.total;
            document.getElementById('annotatedStat').textContent = stats.annotated;
            document.getElementById('correctStat').textContent = stats.correct;
            document.getElementById('incorrectStat').textContent = stats.incorrect;
            document.getElementById('accuracyStat').textContent = (stats.accuracy * 100).toFixed(1) + '%';

            const progress = stats.total > 0 ? (stats.annotated / stats.total) * 100 : 0;
            document.getElementById('progressFill').style.width = progress + '%';
        }

        async function showReport() {
            const res = await fetch('/api/export');
            const report = await res.json();

            const content = document.getElementById('reportContent');
            content.innerHTML = `
                <div class="report-section">
                    <div class="report-section-title">总体统计</div>
                    <div class="report-grid">
                        <div class="report-item">
                            <div class="report-item-label">总图片数</div>
                            <div class="report-item-value">${report.statistics.total}</div>
                        </div>
                        <div class="report-item">
                            <div class="report-item-label">已标注</div>
                            <div class="report-item-value">${report.statistics.annotated}</div>
                        </div>
                        <div class="report-item">
                            <div class="report-item-label">正确数量</div>
                            <div class="report-item-value" style="color: #28a745;">${report.statistics.correct}</div>
                        </div>
                        <div class="report-item">
                            <div class="report-item-label">错误数量</div>
                            <div class="report-item-value" style="color: #dc3545;">${report.statistics.incorrect}</div>
                        </div>
                        <div class="report-item" style="grid-column: span 2;">
                            <div class="report-item-label">准确率</div>
                            <div class="report-item-value" style="color: #fd7e14;">
                                ${(report.statistics.accuracy * 100).toFixed(2)}%
                            </div>
                        </div>
                    </div>
                </div>
                <div class="report-section">
                    <div class="report-section-title">各标签准确率</div>
                    <div style="max-height: 200px; overflow-y: auto;">
                        ${Object.entries(report.statistics.label_accuracy || {})
                            .sort((a, b) => a[1].total - b[1].total)
                            .map(([label, data]) => `
                                <div class="label-accuracy-item">
                                    <span>${label.replace(/_/g, ' ')}</span>
                                    <span>${(data.accuracy * 100).toFixed(1)}% (${data.correct}/${data.total})</span>
                                </div>
                            `).join('')}
                    </div>
                </div>
                <div class="report-section">
                    <div class="report-section-title">常见错误</div>
                    <div style="max-height: 150px; overflow-y: auto;">
                        ${Object.entries(report.statistics.confusion_matrix || {})
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 15)
                            .map(([key, count]) => `
                                <div class="confusion-item">
                                    <span>${key.replace(/_/g, ' -> ')}</span>
                                    <span>${count} 次</span>
                                </div>
                            `).join('')}
                    </div>
                </div>
            `;

            document.getElementById('reportModal').classList.add('show');
        }

        function closeReport() {
            document.getElementById('reportModal').classList.remove('show');
        }

        function downloadReport() {
            fetch('/api/export')
                .then(res => res.json())
                .then(report => {
                    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `evaluation_report_${new Date().toISOString().slice(0, 10)}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                });
        }

        // 搜索功能
        const searchInput = document.getElementById('searchInput');
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            if (!query) {
                renderReferences(referenceSignsData);
                return;
            }
            const filtered = referenceSignsData.filter(sign =>
                sign.display_name.toLowerCase().includes(query) ||
                sign.name.toLowerCase().includes(query)
            );
            renderReferences(filtered);
        });

        // 键盘快捷键
        document.addEventListener('keydown', (e) => {
            if (e.target === searchInput) {
                if (e.key === 'Escape') {
                    searchInput.value = '';
                    searchInput.blur();
                    renderReferences(referenceSignsData);
                }
                return;
            }

            switch(e.key) {
                case '/':
                    e.preventDefault();
                    searchInput.focus();
                    break;
                case '1':
                    annotate(true);
                    break;
                case '2':
                    markAs('other');
                    break;
                case '3':
                    markAs('blur');
                    break;
                case '4':
                    markAs('glare');
                    break;
                case '5':
                    markAs('lowlight');
                    break;
                case 'ArrowLeft':
                    navigate(-1);
                    break;
                case 'ArrowRight':
                    navigate(1);
                    break;
                case 'Escape':
                    searchInput.value = '';
                    renderReferences(referenceSignsData);
                    break;
            }
        });

        // 文件夹选择
        document.getElementById('folderSelect').addEventListener('change', (e) => {
            selectFolder(e.target.value);
        });

        // 初始化
        loadFolders();
        loadReferenceSigns();
    </script>
</body>
</html>
"""


if __name__ == '__main__':
    main()
