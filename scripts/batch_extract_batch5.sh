#!/bin/bash
# Batch 5: 2个视频 (0028, 0029)，59分钟
# 输出到 extracted_signs/DJI_batch5/
set -e
cd /Users/justin/Desktop/GLM_Labeling
source venv/bin/activate
[ -z "$ZAI_API_KEY" ] && echo "请设置 ZAI_API_KEY" && exit 1

BATCH_OUTPUT="extracted_signs/DJI_batch5"
mkdir -p "$BATCH_OUTPUT"

SOURCE_DIR="/Volumes/LQ1000/DJI_4K"

# Batch 5: 0028(29分钟) + 0029(30分钟) = 59分钟
VIDEOS=(
    "DJI_20250705091733_0028_D.MP4"
    "DJI_20250705101207_0029_D.MP4"
)

echo "========================================"
echo "Batch 5: 提取交通标志 (2个视频, 59分钟)"
echo "视频: 0028, 0029"
echo "输出: $BATCH_OUTPUT"
echo "开始: $(date)"
echo "========================================"

for VIDEO in "${VIDEOS[@]}"; do
    SHORT_NAME=$(echo "$VIDEO" | grep -oE '_[0-9]{4}_D' | tr -d '_D' | sed 's/^/DJI_/')
    CLIPS_DIR=$(echo "$VIDEO" | sed 's/\.MP4$//')

    VIDEO_PATH="$SOURCE_DIR/$VIDEO"
    CLIPS_PATH="raw_data/videos/clips/$CLIPS_DIR"
    SIGN_OUTPUT="$BATCH_OUTPUT/$SHORT_NAME"

    [ -d "$SIGN_OUTPUT" ] && [ "$(ls -A $SIGN_OUTPUT 2>/dev/null)" ] && echo "跳过: $SHORT_NAME 已完成" && continue

    echo ""
    echo ">>> $SHORT_NAME | $(date)"

    if [ ! -d "$CLIPS_PATH" ]; then
        echo "[1/4] 分割视频 (120秒/片段)..."
        python scripts/split_video.py "$VIDEO_PATH" --prefix "$SHORT_NAME" --segment-time 120
        CLIPS_PATH=$(ls -d raw_data/videos/clips/${CLIPS_DIR}* 2>/dev/null | head -1)
    else
        echo "[1/4] 跳过分割 (已存在)"
    fi

    echo "[2/4] VLM 检测 (JPEG 抽帧)..."
    find "$CLIPS_PATH" -name "*.mp4" | sort | \
        xargs -P 2 -I {} python scripts/video_to_dataset_async.py --video "{}" --workers 4 --no-rag --jpeg-frames

    echo "[3/4] 裁剪标志图片..."
    mkdir -p "$SIGN_OUTPUT"
    # 从 annotations 中裁剪 traffic_sign
    for dataset_dir in dataset_output/${SHORT_NAME}*_dataset; do
        if [ -d "$dataset_dir/annotations" ] && [ -d "$dataset_dir/frames" ]; then
            python -c "
import json
from pathlib import Path
from PIL import Image

dataset_dir = Path('$dataset_dir')
output_dir = Path('$SIGN_OUTPUT')
output_dir.mkdir(parents=True, exist_ok=True)

for json_file in sorted((dataset_dir / 'annotations').glob('*.json')):
    try:
        with open(json_file) as f:
            data = json.load(f)
        frame_name = data.get('imagePath', json_file.stem + '.jpg')
        frame_path = dataset_dir / 'frames' / frame_name
        if not frame_path.exists():
            continue
        image = Image.open(frame_path).convert('RGB')
        sign_idx = 0
        for shape in data.get('shapes', []):
            if shape.get('flags', {}).get('category') != 'traffic_sign':
                continue
            points = shape.get('points', [])
            if len(points) != 2:
                continue
            x1, y1 = int(points[0][0]), int(points[0][1])
            x2, y2 = int(points[1][0]), int(points[1][1])
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            sign_idx += 1
            # 添加 10px padding
            x1, y1 = max(0, x1-10), max(0, y1-10)
            x2, y2 = min(image.width, x2+10), min(image.height, y2+10)
            cropped = image.crop((x1, y1, x2, y2))
            out_name = f'{json_file.stem}_sign{sign_idx}_{x1}_{y1}_{x2}_{y2}.png'
            cropped.save(output_dir / out_name, 'PNG')
    except Exception as e:
        print(f'  错误: {json_file.name}: {e}')
"
        fi
    done

    echo "[4/4] 统计..."

    COUNT=$(ls "$SIGN_OUTPUT"/*.png 2>/dev/null | wc -l | tr -d ' ')
    echo "提取标志: $COUNT 张"

    # 清理中间文件（clips 和 dataset_output）
    rm -rf "$CLIPS_PATH"
    rm -rf dataset_output/${SHORT_NAME}*_dataset
    echo "<<< $SHORT_NAME 完成 | 剩余空间: $(df -h / | tail -1 | awk '{print $4}')"
done

TOTAL=$(find "$BATCH_OUTPUT" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "========================================"
echo "Batch 5 完成! $(date)"
echo "总计提取: $TOTAL 张标志图片"
echo "输出目录: $BATCH_OUTPUT"
echo ""
echo "下一步: 运行标注工具"
echo "  python scripts/sign_labeling_ui.py --batch 5"
echo "========================================"
