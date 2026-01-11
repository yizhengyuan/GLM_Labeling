#!/usr/bin/env python3
"""
Unclear Image Reclassification UI
将之前标注为 unclear 的图片重新分类为 lowlight/blur/glare

参考 Diary-20260101.md:
- lowlight: 低光照 (夜间、隧道、阴影)
- blur: 模糊 (运动模糊、对焦模糊)
- glare: 眩光 (阳光直射、逆光、反光)
"""

import json
import os
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request, send_from_directory

app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).parent.parent
IMAGE_DIR = BASE_DIR / "unclear_reclassification" / "images"
OUTPUT_DIR = BASE_DIR / "unclear_reclassification"
PROGRESS_FILE = OUTPUT_DIR / "reclassification_progress.json"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

# Three interference types
LABELS = {
    "lowlight": {
        "name": "低光照 Lowlight",
        "emoji": "",
        "description": "夜间、隧道、阴影等光照不足场景",
        "color": "#1e3a5f"
    },
    "blur": {
        "name": "模糊 Blur",
        "emoji": "",
        "description": "运动模糊、对焦模糊等",
        "color": "#4a3728"
    },
    "glare": {
        "name": "眩光 Glare",
        "emoji": "",
        "description": "阳光直射、逆光、反光等",
        "color": "#6b4423"
    }
}


def get_image_list():
    """Get all PNG images from the directory."""
    images = sorted([f for f in IMAGE_DIR.glob("*.png")])
    return images


def load_progress():
    """Load classification progress."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"classified": {}, "total": 0}


def save_progress(progress):
    """Save classification progress."""
    progress["updated_at"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def get_unclassified_images(progress):
    """Get list of images that haven't been classified yet."""
    classified = set(progress.get("classified", {}).keys())
    images = get_image_list()
    return [img for img in images if img.name not in classified]


def get_stats(progress):
    """Get classification statistics."""
    classified = progress.get("classified", {})
    stats = {
        "total": len(get_image_list()),
        "classified": len(classified),
        "remaining": len(get_unclassified_images(progress)),
        "lowlight": sum(1 for v in classified.values() if v == "lowlight"),
        "blur": sum(1 for v in classified.values() if v == "blur"),
        "glare": sum(1 for v in classified.values() if v == "glare"),
    }
    return stats


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unclear 图片重新分类</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }

        .header {
            background: #16213e;
            padding: 15px 20px;
            border-bottom: 1px solid #333;
        }

        .header h1 {
            font-size: 20px;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 13px;
            color: #888;
        }

        .stats-bar {
            display: flex;
            gap: 20px;
            padding: 10px 20px;
            background: #0f3460;
            border-bottom: 1px solid #333;
        }

        .stat-item {
            font-size: 13px;
        }

        .stat-value {
            font-weight: bold;
            color: #4cc9f0;
        }

        .progress-bar {
            height: 4px;
            background: #333;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4cc9f0, #4361ee);
            transition: width 0.3s;
        }

        .main-container {
            display: flex;
            height: calc(100vh - 110px);
        }

        .image-list {
            width: 280px;
            background: #16213e;
            border-right: 1px solid #333;
            overflow-y: auto;
            padding: 10px;
        }

        .list-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px;
            border-radius: 6px;
            cursor: pointer;
            margin-bottom: 4px;
            transition: background 0.2s;
        }

        .list-item:hover {
            background: #1f4068;
        }

        .list-item.active {
            background: #1f4068;
            border-left: 3px solid #4cc9f0;
        }

        .list-item.classified {
            opacity: 0.6;
        }

        .list-item .label-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .list-item .label-lowlight .label-indicator { background: #1e3a5f; }
        .list-item .label-blur .label-indicator { background: #4a3728; }
        .list-item .label-glare .label-indicator { background: #6b4423; }

        .thumbnail {
            width: 50px;
            height: 50px;
            object-fit: contain;
            background: #000;
            border-radius: 4px;
        }

        .item-name {
            font-size: 11px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            flex: 1;
        }

        .image-viewer {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .image-container {
            max-width: 100%;
            max-height: calc(100vh - 320px);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .main-image {
            max-width: 100%;
            max-height: calc(100vh - 320px);
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }

        .button-row {
            display: flex;
            gap: 20px;
            margin-top: 20px;
        }

        .label-btn {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            padding: 20px 30px;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
            min-width: 140px;
        }

        .label-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.4);
        }

        .label-btn:active {
            transform: translateY(0);
        }

        .label-btn .emoji {
            font-size: 32px;
        }

        .label-btn .name {
            font-size: 13px;
        }

        .label-btn .desc {
            font-size: 10px;
            opacity: 0.7;
        }

        .btn-lowlight {
            background: #1e3a5f;
        }
        .btn-lowlight:hover {
            background: #2a4a7f;
        }

        .btn-blur {
            background: #4a3728;
        }
        .btn-blur:hover {
            background: #6a4738;
        }

        .btn-glare {
            background: #6b4423;
        }
        .btn-glare:hover {
            background: #8b5433;
        }

        .nav-controls {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }

        .nav-btn {
            padding: 10px 20px;
            background: #333;
            border: none;
            border-radius: 6px;
            color: #fff;
            cursor: pointer;
            font-size: 13px;
        }

        .nav-btn:hover {
            background: #444;
        }

        .empty-state {
            text-align: center;
            padding: 40px;
        }

        .empty-state h2 {
            color: #4cc9f0;
            margin-bottom: 10px;
        }

        .empty-state p {
            color: #888;
        }

        .keyboard-hint {
            margin-top: 15px;
            font-size: 11px;
            color: #666;
            text-align: center;
        }

        .export-btn {
            padding: 8px 16px;
            background: #4361ee;
            border: none;
            border-radius: 6px;
            color: #fff;
            cursor: pointer;
            font-size: 12px;
            margin-left: auto;
        }

        .export-btn:hover {
            background: #5371fe;
        }

        .filter-tabs {
            display: flex;
            gap: 5px;
            padding: 10px;
            background: #0f3460;
        }

        .filter-tab {
            padding: 6px 12px;
            background: transparent;
            border: none;
            color: #888;
            cursor: pointer;
            font-size: 12px;
            border-radius: 4px;
        }

        .filter-tab:hover {
            background: #1a4068;
        }

        .filter-tab.active {
            background: #1f4068;
            color: #fff;
        }
    </style>
</head>
<body>
    <div class="header">
        <div style="display: flex; align-items: center;">
            <h1> Unclear 图片重新分类</h1>
            <button class="export-btn" onclick="exportResults()">导出结果</button>
        </div>
        <p>将 unclear 图片分类为：低光照 / 模糊 / 眩光</p>
    </div>

    <div class="stats-bar">
        <div class="stat-item">总计: <span class="stat-value" id="stat-total">-</span></div>
        <div class="stat-item">已分类: <span class="stat-value" id="stat-classified">-</span></div>
        <div class="stat-item">待处理: <span class="stat-value" id="stat-remaining">-</span></div>
        <div class="stat-item" style="color: #1e3a5f;">低光照: <span class="stat-value" id="stat-lowlight">-</span></div>
        <div class="stat-item" style="color: #a08060;">模糊: <span class="stat-value" id="stat-blur">-</span></div>
        <div class="stat-item" style="color: #c09050;">眩光: <span class="stat-value" id="stat-glare">-</span></div>
    </div>

    <div class="progress-bar">
        <div class="progress-fill" id="progress-fill"></div>
    </div>

    <div class="filter-tabs">
        <button class="filter-tab active" onclick="setFilter('all')">全部</button>
        <button class="filter-tab" onclick="setFilter('unclassified')">未分类</button>
        <button class="filter-tab" onclick="setFilter('lowlight')">低光照</button>
        <button class="filter-tab" onclick="setFilter('blur')">模糊</button>
        <button class="filter-tab" onclick="setFilter('glare')">眩光</button>
    </div>

    <div class="main-container">
        <div class="image-list" id="image-list"></div>

        <div class="image-viewer" id="image-viewer">
            <!-- Content loaded dynamically -->
        </div>
    </div>

    <script>
        let images = [];
        let currentIndex = 0;
        let classified = {};
        let currentFilter = 'all';

        // Load images and progress
        async function loadData() {
            const response = await fetch('/api/data');
            const data = await response.json();
            images = data.images;
            classified = data.classified;
            renderImageList();
            loadStats();
            if (images.length > 0) {
                loadImage(0);
            }
        }

        // Render the sidebar list
        function renderImageList() {
            const list = document.getElementById('image-list');
            list.innerHTML = '';

            const filteredImages = images.filter((img, idx) => {
                const label = classified[img];
                if (currentFilter === 'all') return true;
                if (currentFilter === 'unclassified') return !label;
                return label === currentFilter;
            });

            filteredImages.forEach(img => {
                const label = classified[img];
                const labelClass = label ? `label-${label}` : '';
                const div = document.createElement('div');
                div.className = `list-item ${labelClass} ${label ? 'classified' : ''}`;
                div.onclick = () => {
                    const idx = images.indexOf(img);
                    loadImage(idx);
                };

                div.innerHTML = `
                    <div class="${labelClass}">
                        <div class="label-indicator"></div>
                    </div>
                    <img class="thumbnail" src="/api/image/${encodeURIComponent(img)}" loading="lazy">
                    <div class="item-name">${img}</div>
                `;
                list.appendChild(div);
            });
        }

        // Load a specific image
        function loadImage(index) {
            currentIndex = index;
            const img = images[index];

            // Find next unclassified index
            let nextIndex = index + 1;
            while (nextIndex < images.length && classified[images[nextIndex]]) {
                nextIndex++;
            }

            // Find prev unclassified index
            let prevIndex = index - 1;
            while (prevIndex >= 0 && classified[images[prevIndex]]) {
                prevIndex--;
            }

            const viewer = document.getElementById('image-viewer');
            const isClassified = !!classified[img];

            if (isClassified) {
                viewer.innerHTML = `
                    <div class="empty-state">
                        <h2>已分类</h2>
                        <p>此图片已标记为: <strong>${LABELS[classified[img]].name}</strong></p>
                        <div class="image-container">
                            <img class="main-image" src="/api/image/${encodeURIComponent(img)}" style="max-height: 400px;">
                        </div>
                        <button class="nav-btn" onclick="loadImage(${index + 1 < images.length ? index + 1 : 0})">下一张</button>
                    </div>
                `;
            } else {
                viewer.innerHTML = `
                    <div class="image-container">
                        <img class="main-image" src="/api/image/${encodeURIComponent(img)}" id="current-image">
                    </div>

                    <div class="button-row">
                        <button class="label-btn btn-lowlight" onclick="classify('lowlight')">
                            <span class="emoji"></span>
                            <span class="name">低光照</span>
                            <span class="desc">夜间、隧道、阴影</span>
                        </button>
                        <button class="label-btn btn-blur" onclick="classify('blur')">
                            <span class="emoji"></span>
                            <span class="name">模糊</span>
                            <span class="desc">运动模糊、对焦模糊</span>
                        </button>
                        <button class="label-btn btn-glare" onclick="classify('glare')">
                            <span class="emoji"></span>
                            <span class="name">眩光</span>
                            <span class="desc">阳光直射、逆光</span>
                        </button>
                    </div>

                    <div class="nav-controls">
                        <button class="nav-btn" onclick="loadImage(${prevIndex >= 0 ? prevIndex : index})">
                            ← 上一张未分类
                        </button>
                        <button class="nav-btn" onclick="loadImage(${nextIndex < images.length ? nextIndex : index})">
                            下一张未分类 →
                        </button>
                    </div>

                    <div class="keyboard-hint">
                        快捷键: 1=低光照, 2=模糊, 3=眩光 | ←→ = 切换图片
                    </div>
                `;
            }

            // Update active state in list
            document.querySelectorAll('.list-item').forEach(item => item.classList.remove('active'));
            const activeItem = document.querySelector(`.list-item:nth-child(${index + 1})`);
            if (activeItem) activeItem.classList.add('active');
        }

        // Classify current image
        async function classify(label) {
            const img = images[currentIndex];

            const response = await fetch('/api/classify', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image: img, label: label})
            });

            if (response.ok) {
                classified[img] = label;
                renderImageList();
                loadStats();

                // Move to next unclassified image
                let nextIndex = currentIndex + 1;
                while (nextIndex < images.length && classified[images[nextIndex]]) {
                    nextIndex++;
                }

                if (nextIndex < images.length) {
                    loadImage(nextIndex);
                } else {
                    // Check if we're done
                    const remaining = images.filter(img => !classified[img]);
                    if (remaining.length === 0) {
                        document.getElementById('image-viewer').innerHTML = `
                            <div class="empty-state">
                                <h2>完成！</h2>
                                <p>所有图片都已分类完毕</p>
                                <button class="nav-btn" onclick="exportResults()">导出结果</button>
                            </div>
                        `;
                    } else {
                        loadImage(0);
                    }
                }
            }
        }

        // Load statistics
        async function loadStats() {
            const response = await fetch('/api/stats');
            const stats = await response.json();

            document.getElementById('stat-total').textContent = stats.total;
            document.getElementById('stat-classified').textContent = stats.classified;
            document.getElementById('stat-remaining').textContent = stats.remaining;
            document.getElementById('stat-lowlight').textContent = stats.lowlight;
            document.getElementById('stat-blur').textContent = stats.blur;
            document.getElementById('stat-glare').textContent = stats.glare;

            const progress = stats.total > 0 ? (stats.classified / stats.total * 100) : 0;
            document.getElementById('progress-fill').style.width = progress + '%';
        }

        // Set filter
        function setFilter(filter) {
            currentFilter = filter;
            document.querySelectorAll('.filter-tab').forEach(tab => {
                tab.classList.toggle('active', tab.textContent.includes(
                    filter === 'all' ? '全部' :
                    filter === 'unclassified' ? '未分类' :
                    filter === 'lowlight' ? '低光照' :
                    filter === 'blur' ? '模糊' : '眩光'
                ));
            });
            renderImageList();
        }

        // Export results
        async function exportResults() {
            const response = await fetch('/api/export');
            const data = await response.json();

            // Download JSON
            const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `unclear_reclassified_${new Date().toISOString().slice(0,10)}.json`;
            a.click();
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === '1') classify('lowlight');
            if (e.key === '2') classify('blur');
            if (e.key === '3') classify('glare');
            if (e.key === 'ArrowRight') {
                const nextIndex = currentIndex + 1 < images.length ? currentIndex + 1 : 0;
                loadImage(nextIndex);
            }
            if (e.key === 'ArrowLeft') {
                const prevIndex = currentIndex - 1 >= 0 ? currentIndex - 1 : images.length - 1;
                loadImage(prevIndex);
            }
        });

        // Label info for JS
        const LABELS = {
            lowlight: {name: '低光照 Lowlight'},
            blur: {name: '模糊 Blur'},
            glare: {name: '眩光 Glare'}
        };

        // Initialize
        loadData();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/image/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)


@app.route('/api/data')
def get_data():
    """Get all images and current classification progress."""
    images = [f.name for f in get_image_list()]
    progress = load_progress()
    return jsonify({
        "images": images,
        "classified": progress.get("classified", {})
    })


@app.route('/api/stats')
def get_stats_api():
    """Get classification statistics."""
    progress = load_progress()
    return jsonify(get_stats(progress))


@app.route('/api/classify', methods=['POST'])
def classify_image():
    """Classify an image."""
    data = request.json
    image = data.get('image')
    label = data.get('label')

    if not image or label not in LABELS:
        return jsonify({"error": "Invalid data"}), 400

    progress = load_progress()
    if "classified" not in progress:
        progress["classified"] = {}

    progress["classified"][image] = label
    save_progress(progress)

    return jsonify({"success": True})


@app.route('/api/export')
def export_data():
    """Export all classification results."""
    progress = load_progress()

    # Group by label
    results = {
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "total_images": len(get_image_list()),
            "total_classified": len(progress.get("classified", {}))
        },
        "labels": {
            "lowlight": {"name": "低光照 Lowlight", "images": []},
            "blur": {"name": "模糊 Blur", "images": []},
            "glare": {"name": "眩光 Glare", "images": []}
        }
    }

    for img, label in progress.get("classified", {}).items():
        if label in results["labels"]:
            results["labels"][label]["images"].append(img)

    return jsonify(results)


if __name__ == '__main__':
    print("=" * 50)
    print("Unclear 图片重新分类 UI")
    print("=" * 50)
    print(f"图片目录: {IMAGE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"图片总数: {len(get_image_list())}")
    print("=" * 50)
    print("打开浏览器访问: http://localhost:5001")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5001, debug=True)
