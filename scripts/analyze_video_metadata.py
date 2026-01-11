#!/usr/bin/env python3
"""用 Gemini 分析视频的道路类型和地区特征"""

import os
import json
import time
from pathlib import Path
from google import genai

PROMPT = """请分析这段骑行视频，判断以下信息：

1. **道路类型** (可多选):
   - highway: 高速公路/快速路（有隔离带、多车道、高速行驶）
   - main_road: 主干道（城市主要道路、红绿灯、双向多车道）
   - urban_street: 城市街道（住宅区、商业区、窄路）

2. **地区特征** (根据路牌、建筑、地标判断):
   - 如果看到路牌或地标，请标注具体地名
   - 香港常见地区：北区、大埔、沙田、西贡、将军澳等

3. **场景描述**: 简短描述视频中的主要道路环境

请严格按以下 JSON 格式输出：
{
  "road_types": ["main_road", "urban_street"],
  "road_type_percentages": {"main_road": 60, "urban_street": 40},
  "district_signs": ["大埔", "吐露港"],
  "scene_description": "视频主要在大埔区的主干道行驶，途经吐露港公路"
}
"""

def analyze_video(video_path: str, client, model: str = "gemini-2.0-flash") -> dict:
    print(f"   📤 上传: {Path(video_path).name}")
    video_file = client.files.upload(file=video_path)
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
    
    print(f"   🔍 分析中...")
    response = client.models.generate_content(model=model, contents=[video_file, PROMPT])
    client.files.delete(name=video_file.name)
    
    # 解析 JSON
    text = response.text
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    return {"error": "解析失败", "raw": text[:500]}

def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("请设置 GOOGLE_API_KEY")
        return
    
    client = genai.Client(api_key=api_key)
    
    # 加载视频清单
    with open("video_inventory.json") as f:
        data = json.load(f)
    
    print(f"共 {len(data['videos'])} 个视频待分析\n")
    
    for i, v in enumerate(data["videos"], 1):
        path = v["path"]
        print(f"[{i}/{len(data['videos'])}] {v['file']}")
        
        if not Path(path).exists():
            print(f"   ⚠️ 文件不存在，跳过")
            continue
        
        try:
            result = analyze_video(path, client)
            v["road_analysis"] = result
            print(f"   ✅ {result.get('road_types', [])} | {result.get('district_signs', [])}")
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            v["road_analysis"] = {"error": str(e)}
        
        # 保存进度
        with open("video_inventory.json", "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        time.sleep(1)  # 避免 rate limit
    
    print("\n✅ 分析完成，结果已保存到 video_inventory.json")

if __name__ == "__main__":
    main()
