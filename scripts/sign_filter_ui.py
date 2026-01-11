#!/usr/bin/env python3
"""
交通标志筛选 UI

用于快速筛选 4K 抠图的交通标志：
- valid: 清晰可辨的标准交通标志
- unclear: 看不清，需要参考原图
- not_sign: 不是 188 种标准标志（施工牌、广告等）

快捷键：
- 1 或 V: valid (有效标志)
- 2 或 U: unclear (不清晰)
- 3 或 N: not_sign (非标准标志)
- Z: 撤销上一个
- Q: 保存并退出
- 左/右箭头: 上一张/下一张（不标记）
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# 检查依赖
try:
    from flask import Flask, render_template_string, jsonify, request, send_file
except ImportError:
    print("请安装 Flask: pip install flask")
    sys.exit(1)


# ============ 配置 ============

EXTRACTED_SIGNS_DIR = Path(__file__).parent.parent / "extracted_signs"

# Batch 配置
BATCH_CONFIG = {
    "batch1": {
        "signs_dir": EXTRACTED_SIGNS_DIR / "DJI_batch1",
        "output_file": EXTRACTED_SIGNS_DIR / "DJI_batch1" / "filter_results_batch1.json",
    },
    "batch2": {
        "signs_dir": EXTRACTED_SIGNS_DIR / "DJI_batch2",
        "output_file": EXTRACTED_SIGNS_DIR / "DJI_batch2" / "filter_results_batch2.json",
    },
}

# 默认使用 batch2
DEFAULT_BATCH = "batch2"


# ============ 数据管理 ============

class FilterManager:
    def __init__(self, signs_dir: Path, output_file: Path):
        self.signs_dir = signs_dir
        self.output_file = output_file
        self.images: list[Path] = []
        self.results: dict[str, str] = {}  # filename -> category
        self.current_index = 0
        self.history: list[str] = []  # 用于撤销

        self._load_images()
        self._load_results()

    def _load_images(self):
        """加载所有图片"""
        self.images = sorted(
            self.signs_dir.rglob("*.png"),
            key=lambda p: p.name
        )
        print(f"找到 {len(self.images)} 张图片")

    def _load_results(self):
        """加载已有的筛选结果"""
        if self.output_file.exists():
            with open(self.output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.results = data.get("results", {})
                print(f"已加载 {len(self.results)} 条筛选记录")

        # 跳到第一个未标记的图片
        for i, img in enumerate(self.images):
            if img.name not in self.results:
                self.current_index = i
                break

    def save_results(self):
        """保存筛选结果"""
        data = {
            "updated_at": datetime.now().isoformat(),
            "total_images": len(self.images),
            "labeled_count": len(self.results),
            "stats": {
                "valid": sum(1 for v in self.results.values() if v == "valid"),
                "unclear": sum(1 for v in self.results.values() if v == "unclear"),
                "not_sign": sum(1 for v in self.results.values() if v == "not_sign"),
            },
            "results": self.results
        }
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(self.results)} 条记录到 {self.output_file}")

    def get_current_image(self) -> Optional[Path]:
        """获取当前图片"""
        if 0 <= self.current_index < len(self.images):
            return self.images[self.current_index]
        return None

    def label_current(self, category: str) -> bool:
        """标记当前图片并前进到下一张"""
        img = self.get_current_image()
        if img is None:
            return False

        self.results[img.name] = category
        self.history.append(img.name)

        # 前进到下一张未标记的
        self._move_to_next_unlabeled()
        return True

    def _move_to_next_unlabeled(self):
        """移动到下一个未标记的图片"""
        start = self.current_index + 1
        for i in range(start, len(self.images)):
            if self.images[i].name not in self.results:
                self.current_index = i
                return
        # 如果后面没有了，从头找
        for i in range(0, start):
            if self.images[i].name not in self.results:
                self.current_index = i
                return
        # 全部标记完了
        self.current_index = len(self.images)

    def undo(self) -> bool:
        """撤销上一个标记"""
        if not self.history:
            return False

        last_name = self.history.pop()
        if last_name in self.results:
            del self.results[last_name]

        # 找到这张图的索引
        for i, img in enumerate(self.images):
            if img.name == last_name:
                self.current_index = i
                break

        return True

    def go_prev(self):
        """上一张"""
        if self.current_index > 0:
            self.current_index -= 1

    def go_next(self):
        """下一张"""
        if self.current_index < len(self.images) - 1:
            self.current_index += 1

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total": len(self.images),
            "labeled": len(self.results),
            "remaining": len(self.images) - len(self.results),
            "valid": sum(1 for v in self.results.values() if v == "valid"),
            "unclear": sum(1 for v in self.results.values() if v == "unclear"),
            "not_sign": sum(1 for v in self.results.values() if v == "not_sign"),
            "current_index": self.current_index,
        }


# ============ Flask App ============

app = Flask(__name__)
manager: Optional[FilterManager] = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>交通标志筛选</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #fff;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            padding: 16px 24px;
            background: rgba(0,0,0,0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .header h1 { font-size: 1.4rem; font-weight: 600; }
        .header h1 span { color: #00d9ff; }
        .stats {
            display: flex;
            gap: 20px;
            font-size: 0.9rem;
        }
        .stat { display: flex; align-items: center; gap: 6px; }
        .stat-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }
        .stat-dot.valid { background: #51cf66; }
        .stat-dot.unclear { background: #ffa94d; }
        .stat-dot.not_sign { background: #ff6b6b; }
        .stat-dot.remaining { background: #666; }

        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 24px;
        }
        .image-container {
            background: rgba(0,0,0,0.4);
            border-radius: 16px;
            padding: 24px;
            max-width: 90vw;
            max-height: 60vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .image-container img {
            max-width: 100%;
            max-height: 50vh;
            object-fit: contain;
            border-radius: 8px;
        }
        .image-info {
            margin-top: 16px;
            color: #888;
            font-size: 0.9rem;
            text-align: center;
        }
        .current-label {
            margin-top: 8px;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85rem;
            display: inline-block;
        }
        .current-label.valid { background: #51cf66; color: #000; }
        .current-label.unclear { background: #ffa94d; color: #000; }
        .current-label.not_sign { background: #ff6b6b; color: #fff; }

        .controls {
            padding: 24px;
            background: rgba(0,0,0,0.3);
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        .btn-group {
            display: flex;
            justify-content: center;
            gap: 16px;
            margin-bottom: 16px;
        }
        .btn {
            padding: 16px 32px;
            border: none;
            border-radius: 12px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.1s, box-shadow 0.2s;
            min-width: 160px;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn:active { transform: translateY(0); }
        .btn-valid { background: #51cf66; color: #000; }
        .btn-valid:hover { box-shadow: 0 4px 20px rgba(81,207,102,0.4); }
        .btn-unclear { background: #ffa94d; color: #000; }
        .btn-unclear:hover { box-shadow: 0 4px 20px rgba(255,169,77,0.4); }
        .btn-not_sign { background: #ff6b6b; color: #fff; }
        .btn-not_sign:hover { box-shadow: 0 4px 20px rgba(255,107,107,0.4); }

        .nav-group {
            display: flex;
            justify-content: center;
            gap: 12px;
        }
        .btn-nav {
            padding: 10px 20px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            color: #fff;
            font-size: 0.9rem;
            cursor: pointer;
        }
        .btn-nav:hover { background: rgba(255,255,255,0.2); }

        .shortcuts {
            text-align: center;
            margin-top: 16px;
            color: #666;
            font-size: 0.85rem;
        }
        .shortcuts kbd {
            background: rgba(255,255,255,0.1);
            padding: 2px 8px;
            border-radius: 4px;
            margin: 0 2px;
        }

        .progress-bar {
            height: 4px;
            background: rgba(255,255,255,0.1);
            border-radius: 2px;
            margin-top: 16px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #51cf66, #00d9ff);
            transition: width 0.3s;
        }

        .done-message {
            text-align: center;
            padding: 60px;
        }
        .done-message h2 { font-size: 2rem; color: #51cf66; margin-bottom: 16px; }
        .done-message p { color: #888; }
    </style>
</head>
<body>
    <div class="header">
        <h1><span>交通标志</span> 筛选工具</h1>
        <div class="stats">
            <div class="stat"><span class="stat-dot valid"></span> Valid: <span id="stat-valid">0</span></div>
            <div class="stat"><span class="stat-dot unclear"></span> Unclear: <span id="stat-unclear">0</span></div>
            <div class="stat"><span class="stat-dot not_sign"></span> Not Sign: <span id="stat-not_sign">0</span></div>
            <div class="stat"><span class="stat-dot remaining"></span> 剩余: <span id="stat-remaining">0</span></div>
        </div>
    </div>

    <div class="main" id="main-content">
        <div class="image-container">
            <img id="current-image" src="" alt="当前图片">
        </div>
        <div class="image-info">
            <span id="image-name">加载中...</span>
            <span id="image-index"></span>
        </div>
        <div id="current-label-container"></div>
    </div>

    <div class="controls">
        <div class="btn-group">
            <button class="btn btn-valid" onclick="label('valid')">1. Valid ✓</button>
            <button class="btn btn-unclear" onclick="label('unclear')">2. Unclear ?</button>
            <button class="btn btn-not_sign" onclick="label('not_sign')">3. Not Sign ✗</button>
        </div>
        <div class="nav-group">
            <button class="btn-nav" onclick="navigate('prev')">← 上一张</button>
            <button class="btn-nav" onclick="undo()">↩ 撤销 (Z)</button>
            <button class="btn-nav" onclick="navigate('next')">下一张 →</button>
            <button class="btn-nav" onclick="saveAndQuit()" style="background: #00d9ff; color: #000;">保存退出 (Q)</button>
        </div>
        <div class="shortcuts">
            快捷键: <kbd>1</kbd>/<kbd>V</kbd> Valid | <kbd>2</kbd>/<kbd>U</kbd> Unclear | <kbd>3</kbd>/<kbd>N</kbd> Not Sign | <kbd>Z</kbd> 撤销 | <kbd>←</kbd><kbd>→</kbd> 导航 | <kbd>Q</kbd> 保存退出
        </div>
        <div class="progress-bar">
            <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
        </div>
    </div>

    <script>
        let currentData = null;

        async function loadCurrent() {
            const res = await fetch('/api/current');
            const data = await res.json();
            currentData = data;

            if (data.done) {
                document.getElementById('main-content').innerHTML = `
                    <div class="done-message">
                        <h2>🎉 全部完成！</h2>
                        <p>共筛选 ${data.stats.labeled} 张图片</p>
                        <p style="margin-top: 12px">
                            Valid: ${data.stats.valid} |
                            Unclear: ${data.stats.unclear} |
                            Not Sign: ${data.stats.not_sign}
                        </p>
                    </div>
                `;
                return;
            }

            document.getElementById('current-image').src = '/api/image/' + data.image_path;
            document.getElementById('image-name').textContent = data.image_name;
            document.getElementById('image-index').textContent = ` (${data.stats.current_index + 1}/${data.stats.total})`;

            // 显示当前标签（如果已标记）
            const labelContainer = document.getElementById('current-label-container');
            if (data.current_label) {
                labelContainer.innerHTML = `<span class="current-label ${data.current_label}">${data.current_label}</span>`;
            } else {
                labelContainer.innerHTML = '';
            }

            updateStats(data.stats);
        }

        function updateStats(stats) {
            document.getElementById('stat-valid').textContent = stats.valid;
            document.getElementById('stat-unclear').textContent = stats.unclear;
            document.getElementById('stat-not_sign').textContent = stats.not_sign;
            document.getElementById('stat-remaining').textContent = stats.remaining;

            const progress = (stats.labeled / stats.total) * 100;
            document.getElementById('progress-fill').style.width = progress + '%';
        }

        async function label(category) {
            await fetch('/api/label', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({category})
            });
            loadCurrent();
        }

        async function navigate(direction) {
            await fetch('/api/navigate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({direction})
            });
            loadCurrent();
        }

        async function undo() {
            await fetch('/api/undo', {method: 'POST'});
            loadCurrent();
        }

        async function saveAndQuit() {
            await fetch('/api/save', {method: 'POST'});
            alert('已保存！可以关闭页面了。');
        }

        // 键盘快捷键
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;

            switch(e.key) {
                case '1':
                case 'v':
                case 'V':
                    label('valid');
                    break;
                case '2':
                case 'u':
                case 'U':
                    label('unclear');
                    break;
                case '3':
                case 'n':
                case 'N':
                    label('not_sign');
                    break;
                case 'z':
                case 'Z':
                    undo();
                    break;
                case 'q':
                case 'Q':
                    saveAndQuit();
                    break;
                case 'ArrowLeft':
                    navigate('prev');
                    break;
                case 'ArrowRight':
                    navigate('next');
                    break;
            }
        });

        // 初始加载
        loadCurrent();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/current")
def get_current():
    img = manager.get_current_image()
    stats = manager.get_stats()

    if img is None or manager.current_index >= len(manager.images):
        return jsonify({
            "done": True,
            "stats": stats
        })

    # 获取相对路径
    rel_path = img.relative_to(manager.signs_dir)

    return jsonify({
        "done": False,
        "image_name": img.name,
        "image_path": str(rel_path),
        "current_label": manager.results.get(img.name),
        "stats": stats
    })


@app.route("/api/image/<path:image_path>")
def get_image(image_path):
    full_path = manager.signs_dir / image_path
    if full_path.exists():
        return send_file(full_path)
    return "Not found", 404


@app.route("/api/label", methods=["POST"])
def label_image():
    data = request.json
    category = data.get("category")
    if category not in ["valid", "unclear", "not_sign"]:
        return jsonify({"error": "Invalid category"}), 400

    manager.label_current(category)
    return jsonify({"success": True})


@app.route("/api/navigate", methods=["POST"])
def navigate():
    data = request.json
    direction = data.get("direction")
    if direction == "prev":
        manager.go_prev()
    elif direction == "next":
        manager.go_next()
    return jsonify({"success": True})


@app.route("/api/undo", methods=["POST"])
def undo():
    manager.undo()
    return jsonify({"success": True})


@app.route("/api/save", methods=["POST"])
def save():
    manager.save_results()
    return jsonify({"success": True})


# ============ 主函数 ============

def main():
    global manager

    import argparse
    parser = argparse.ArgumentParser(description="交通标志筛选 UI")
    parser.add_argument("--port", type=int, default=8080, help="端口号")
    parser.add_argument("--batch", type=str, default=DEFAULT_BATCH,
                        choices=list(BATCH_CONFIG.keys()),
                        help=f"选择批次 (默认: {DEFAULT_BATCH})")
    parser.add_argument("--signs-dir", type=str, default=None, help="标志图片目录 (覆盖批次配置)")
    parser.add_argument("--output", type=str, default=None, help="输出文件 (覆盖批次配置)")
    args = parser.parse_args()

    # 使用批次配置或自定义路径
    batch_config = BATCH_CONFIG[args.batch]
    signs_dir = Path(args.signs_dir) if args.signs_dir else batch_config["signs_dir"]
    output_file = Path(args.output) if args.output else batch_config["output_file"]

    if not signs_dir.exists():
        print(f"错误: 目录不存在 {signs_dir}")
        sys.exit(1)

    manager = FilterManager(signs_dir, output_file)

    print(f"\n{'='*50}")
    print(f"交通标志筛选 UI - {args.batch.upper()}")
    print(f"{'='*50}")
    print(f"图片目录: {signs_dir}")
    print(f"输出文件: {output_file}")
    print(f"总图片数: {len(manager.images)}")
    print(f"已标记数: {len(manager.results)}")
    print(f"{'='*50}")
    print(f"\n打开浏览器访问: http://localhost:{args.port}\n")

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
