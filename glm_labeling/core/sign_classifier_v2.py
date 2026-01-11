#!/usr/bin/env python3
"""
🎯 交通标志分类器 v2 - 核心模块

独立的分类逻辑，可用于：
1. 快速测试与调优
2. 正式标注流程调用

特点：
- 69 个摩托车安全相关标志 + other
- 随机打乱候选顺序（消除位置偏差）
- 先描述后匹配（提高准确率）
- 支持自定义 Prompt
- 支持 GLM 和 Gemini 双后端

用法:
    from glm_labeling.core.sign_classifier_v2 import SignClassifierV2

    # GLM 后端
    classifier = SignClassifierV2(api_key="xxx", model_choice="glm")
    
    # Gemini 后端
    classifier = SignClassifierV2(model_choice="gemini-2.5-flash")
    
    label, desc, raw = await classifier.classify(image_path, bbox)
"""

import os
import re
import uuid
import base64
import random
from pathlib import Path
from typing import List, Tuple, Optional

import httpx
from PIL import Image

# Gemini SDK (可选导入)
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ============================================================================
# 69 个摩托车安全相关标志
# ============================================================================

MOTORCYCLE_SAFETY_SIGNS = [
    # 直接与摩托车相关 (3个)
    "No_motor_cycles_or_motor_tricycles",
    "No_motor_vehicles_except_motor_cyclists_and_motor_tricycles",
    "Parking_place_for_motor_cycles_only",
    
    # 速度控制类 (6个)
    "Speed_limit_(in_km_h)",
    "Variable_speed_limit_(in_km_h)",
    "Reduce_speed_now",
    "Keep_in_low_gear",
    "Use_low_gear",
    "Use_low_gear_for_distance_shown",
    
    # 路面状况警告类 (8个)
    "Slippery_road_ahead",
    "Loose_chippings_ahead",
    "Uneven_road_surface_ahead",
    "Road_hump_ahead",
    "Ramp_or_sudden_change_of_road_level_ahead",
    "Ramp_or_sudden_change_of_road_level",
    "Risk_of_falling_or_fallen_rocks_ahead",
    "Road_works_ahead",
    
    # 弯道与坡度类 (6个)
    "Bend_to_left_ahead",
    "Double_bend_ahead_first_to_right",
    "Sharp_deviation_of_route_to_left",
    "Steep_hill_downwards_ahead",
    "Steep_hill_upwards_ahead",
    "Road_narrows_on_both_sides_ahead",
    
    # 禁止与限制类 (10个)
    "No_overtaking",
    "No_entry_for_all_vehicles",
    "No_entry_for_vehicles",
    "No_motor_vehicles",
    "No_stopping_at_any_time",
    "No_stopping",
    "One_way_traffic",
    "One_way_road_ahead",
    "Ahead_only",
    "Keep_right_(keep_left_if_symbol_reversed)",
    
    # 让行与停车类 (5个)
    "Stop_and_give_way",
    "Give_way_to_traffic_on_major_road",
    "Distance_to__Stop__line",
    "Distance_to__Give_way__line",
    "Stop_or_give_way_ahead_(with_distance_to_line_ahead_given_below)",
    
    # 路口与合流类 (8个)
    "Cross_roads_ahead",
    "T-junction_ahead",
    "Side_road_to_left_ahead",
    "Staggered_junction_ahead",
    "Traffic_merging_from_left",
    "Merging_into_main_traffic_on_left",
    "Two-way_traffic_ahead",
    "Two-way_traffic_across_a_one-way_road_ahead",
    
    # 信号灯与执法类 (6个)
    "Traffic_lights_ahead",
    "Traffic_signals_ahead",
    "Red_light_camera_control_zone",
    "Red_light_speed_camera_ahead",
    "Prepare_to_stop_if_signalled_to_do_so",
    "Vehicles_must_stop_at_the_sign_(sign_used_by_police)",
    
    # 弱势道路使用者警告类 (9个)
    "Pedestrian_crossing_ahead",
    "Pedestrians_Ahead",
    "Pedestrian_on_or_crossing_road_ahead",
    "Children_ahead",
    "School_ahead",
    "Playground_ahead",
    "Cyclists_ahead",
    "Disabled_persons_ahead",
    "Visually_impaired_persons_ahead",
    
    # 事故黑点与天气类 (3个)
    "Traffic_Accident_blackspot_ahead",
    "Pedestrian_Accident_blackspot_ahead",
    "Fog_or_mist_ahead",
    
    # 高度/重量限制类 (5个)
    "Restricted_headroom_ahead",
    "No_vehicles_over_height_shown_(including_load)",
    "No_vehicles_over_width_shown_(including_load)",
    "No_vehicles_over_gross_vehicle_weight_shown_(including_load)",
    "No_vehicles_over_axle_weight_shown_(including_load)",
]


# ============================================================================
# 默认 Prompt 模板
# ============================================================================

DEFAULT_CLASSIFY_PROMPT = """请仔细观察这个交通标志图片。

## 第一步：描述你看到的特征
请描述：
- 颜色：主体颜色是什么？（红/蓝/绿/黄/白/黑）
- 形状：什么形状？（圆形/三角形/方形/菱形）
- 内容：有什么文字、数字或图案？

## 第二步：判断是否需要返回 0（优先检查！）

以下类型必须返回 0，不要强行匹配：

1.【方向/导航/路名牌】
  - 蓝底或绿底，有地名、路名、箭头、公里数
  - 例如：往某地 X km、高速出口指示、街道名称牌

2.【距离倒计时牌】
  - 绿底配白色斜条（1条/2条/3条）
  - 表示距离出口 100m/200m/300m

3.【公交/轻轨专用道标志】
  - 有"Bus"、"巴士"、公交车图案的蓝底牌

4.【看不清/无法确定】
  - 图片模糊、过曝、遮挡、太小

如果是以上任何一种 → 直接返回 0

## 第三步：从以下 69 个选项中选择最匹配的
{candidate_list}

## 返回格式：
描述：[你看到的特征]
判断：[是否属于必须返回0的类型]
选择：[编号，如不确定则填0]"""


# 数字细化 Prompt
DETAIL_PROMPTS = {
    "Speed_limit_(in_km_h)": {
        "question": "请识别这个限速标志上显示的具体数字（如 20, 30, 50, 70, 100）。只返回数字。",
        "format": "Speed_limit_{}_km_h"
    },
    "Variable_speed_limit_(in_km_h)": {
        "question": "请识别这个可变限速标志上显示的数字。只返回数字。",
        "format": "Variable_speed_limit_{}_km_h"
    },
}


# ============================================================================
# 核心分类器
# ============================================================================

class SignClassifierV2:
    """
    交通标志分类器 v2
    
    特点：
    - 69 个核心标志 + other
    - 随机打乱消除位置偏差
    - 先描述后匹配
    - 支持 GLM 和 Gemini 双后端
    """
    
    # Gemini 模型映射
    GEMINI_MODELS = {
        "gemini": "gemini-2.5-flash",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.0-flash": "gemini-2.0-flash",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-3-pro": "gemini-3-pro-preview",
    }
    
    def __init__(
        self,
        api_key: str = None,
        api_base: str = "https://api.z.ai/api/paas/v4",
        model: str = "glm-4.6v",
        model_choice: str = "glm",
        timeout: float = 45.0,
        use_shuffle: bool = True,
        prompt_template: str = None
    ):
        """
        初始化分类器
        
        Args:
            api_key: API Key（GLM 用 ZAI_API_KEY，Gemini 用 GOOGLE_API_KEY）
            api_base: API 基础 URL (仅 GLM)
            model: GLM 模型名称
            model_choice: 后端选择 ("glm" / "gemini" / "gemini-2.5-flash" 等)
            timeout: 请求超时时间
            use_shuffle: 是否随机打乱候选顺序
            prompt_template: 自定义 Prompt 模板
        """
        self.model_choice = model_choice
        self.timeout = timeout
        self.use_shuffle = use_shuffle
        self.prompt_template = prompt_template or DEFAULT_CLASSIFY_PROMPT
        
        self._client = None
        self._gemini_client = None
        
        # 根据模型选择初始化
        if model_choice.startswith("gemini"):
            if not GEMINI_AVAILABLE:
                raise ImportError("Gemini SDK 未安装，请运行: pip install google-genai")
            
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("请设置 GOOGLE_API_KEY 环境变量")
            
            self.gemini_model_name = self.GEMINI_MODELS.get(model_choice, "gemini-2.5-flash")
            self._gemini_client = genai.Client(api_key=google_api_key)
        else:
            # GLM 后端
            self.api_key = api_key or os.getenv("ZAI_API_KEY")
            self.api_base = api_base
            self.model = model
    
    @property
    def client(self) -> httpx.AsyncClient:
        """懒加载 HTTP 客户端 (GLM)"""
        if self._client is None and not self.model_choice.startswith("gemini"):
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=self.timeout
            )
        return self._client
    
    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def set_prompt(self, prompt: str):
        """动态更新 Prompt"""
        self.prompt_template = prompt
    
    async def classify(
        self,
        image_path: str,
        bbox: List[int],
        refine_numbers: bool = True
    ) -> Tuple[str, str, str]:
        """
        分类交通标志
        
        Args:
            image_path: 图片路径
            bbox: 边界框 [x1, y1, x2, y2]
            refine_numbers: 是否进一步识别数字（如限速值）
        
        Returns:
            (label, description, raw_output)
            - label: 分类标签（69种之一或 "other"）
            - description: 模型描述
            - raw_output: 原始输出（调试用）
        """
        temp_path = None
        
        try:
            # 1. 裁剪标志区域
            img = Image.open(image_path)
            padding = 10
            x1 = max(0, bbox[0] - padding)
            y1 = max(0, bbox[1] - padding)
            x2 = min(img.width, bbox[2] + padding)
            y2 = min(img.height, bbox[3] + padding)
            
            sign_crop = img.crop((x1, y1, x2, y2))
            temp_path = f"/tmp/sign_v2_{uuid.uuid4()}.jpg"
            sign_crop.save(temp_path, "JPEG")
            
            # 2. 转 base64
            with open(temp_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            base64_url = f"data:image/jpeg;base64,{img_data}"
            
            # 3. 准备候选列表（可选随机打乱）
            candidates = MOTORCYCLE_SAFETY_SIGNS.copy()
            if self.use_shuffle:
                random.shuffle(candidates)
            
            candidate_list = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
            
            # 4. 构建 Prompt
            prompt = self.prompt_template.format(candidate_list=candidate_list)
            
            # 5. 调用 API (区分 GLM / Gemini)
            if self.model_choice.startswith("gemini"):
                # Gemini API
                with open(temp_path, "rb") as f:
                    img_bytes = f.read()
                
                response = self._gemini_client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        prompt
                    ],
                    config=types.GenerateContentConfig(temperature=0.1)
                )
                raw_output = response.text.strip()
            else:
                # GLM API
                response = await self.client.post(
                    "/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": base64_url}},
                                {"type": "text", "text": prompt}
                            ]
                        }],
                        "temperature": 0.1
                    }
                )
                response.raise_for_status()
                raw_output = response.json()["choices"][0]["message"]["content"].strip()
            
            # 6. 解析结果
            description = ""
            if "描述：" in raw_output or "描述:" in raw_output:
                desc_match = re.search(r'描述[：:]\s*(.+?)(?=选择|$)', raw_output, re.DOTALL)
                if desc_match:
                    description = desc_match.group(1).strip()
            
            # 解析编号
            label = "other"
            numbers = re.findall(r'选择[：:]\s*(\d+)', raw_output)
            if not numbers:
                # 兜底：最后一行的数字
                numbers = re.findall(r'\d+', raw_output.split('\n')[-1])
            
            if numbers:
                idx = int(numbers[0]) - 1
                if 0 <= idx < len(candidates):
                    label = candidates[idx]
                # idx < 0（返回 0）→ 保持 other
            
            # 7. 数字细化（如限速具体值）
            if refine_numbers and label in DETAIL_PROMPTS:
                detail_info = DETAIL_PROMPTS[label]
                
                if self.model_choice.startswith("gemini"):
                    # Gemini API
                    response2 = self._gemini_client.models.generate_content(
                        model=self.gemini_model_name,
                        contents=[
                            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                            detail_info["question"]
                        ],
                        config=types.GenerateContentConfig(temperature=0.1)
                    )
                    detail_text = response2.text.strip()
                else:
                    # GLM API
                    response2 = await self.client.post(
                        "/chat/completions",
                        json={
                            "model": self.model,
                            "messages": [{
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": base64_url}},
                                    {"type": "text", "text": detail_info["question"]}
                                ]
                            }],
                            "temperature": 0.1
                        }
                    )
                    response2.raise_for_status()
                    detail_text = response2.json()["choices"][0]["message"]["content"].strip()
                
                detail_numbers = re.findall(r'\d+', detail_text)
                if detail_numbers:
                    label = detail_info["format"].format(detail_numbers[0])
            
            return label, description, raw_output
            
        except Exception as e:
            return "error", str(e), ""
        
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


# ============================================================================
# 便捷函数（用于简单场景）
# ============================================================================

async def classify_sign(
    image_path: str,
    bbox: List[int],
    api_key: str = None,
    use_shuffle: bool = True,
    prompt: str = None
) -> Tuple[str, str, str]:
    """
    快捷分类函数
    
    用法:
        label, desc, raw = await classify_sign("img.jpg", [100, 200, 150, 250])
    """
    classifier = SignClassifierV2(
        api_key=api_key,
        use_shuffle=use_shuffle,
        prompt_template=prompt
    )
    try:
        return await classifier.classify(image_path, bbox)
    finally:
        await classifier.close()


# ============================================================================
# 测试入口
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import sys
    
    async def test():
        if len(sys.argv) < 5:
            print("用法: python sign_classifier_v2.py <image_path> <x1> <y1> <x2> <y2>")
            print("示例: python sign_classifier_v2.py test.jpg 100 200 150 250")
            return
        
        image_path = sys.argv[1]
        bbox = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])]
        
        print(f"🔍 测试分类: {image_path}")
        print(f"📦 边界框: {bbox}")
        print()
        
        label, desc, raw = await classify_sign(image_path, bbox)
        
        print(f"🏷️ 分类结果: {label}")
        print(f"📝 描述: {desc}")
        print(f"\n🔧 原始输出:\n{raw}")
    
    asyncio.run(test())

