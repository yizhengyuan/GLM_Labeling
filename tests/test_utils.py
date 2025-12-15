"""
单元测试：工具函数
"""

import pytest
import json
import tempfile
from pathlib import Path

from glm_labeling.utils import (
    parse_llm_json,
    get_category,
    normalize_vehicle_label,
    normalize_label,
    convert_normalized_coords,
    get_category_emoji
)


class TestJsonUtils:
    """JSON 解析工具测试"""
    
    def test_parse_simple_json(self):
        """测试简单 JSON 解析"""
        text = '[{"label": "car", "bbox_2d": [100, 200, 300, 400]}]'
        result = parse_llm_json(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["label"] == "car"
    
    def test_parse_markdown_wrapped_json(self):
        """测试 Markdown 包裹的 JSON"""
        text = '''```json
[{"label": "vehicle", "bbox_2d": [50, 50, 100, 100]}]
```'''
        result = parse_llm_json(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["label"] == "vehicle"
    
    def test_parse_empty_response(self):
        """测试空响应"""
        assert parse_llm_json("") is None
        assert parse_llm_json("   ") is None
        assert parse_llm_json("[]") == []
    
    def test_parse_with_extra_text(self):
        """测试带额外文字的 JSON"""
        text = 'Here is the result: [{"label": "pedestrian", "bbox_2d": [0, 0, 50, 50]}] Done!'
        result = parse_llm_json(text)
        assert result is not None
        assert result[0]["label"] == "pedestrian"


class TestLabelUtils:
    """标签处理工具测试"""
    
    def test_get_category_pedestrian(self):
        """测试行人类别识别"""
        assert get_category("pedestrian") == "pedestrian"
        assert get_category("person") == "pedestrian"
        assert get_category("cyclist") == "pedestrian"
        assert get_category("crowd") == "pedestrian"
    
    def test_get_category_vehicle(self):
        """测试车辆类别识别"""
        assert get_category("car") == "vehicle"
        assert get_category("truck") == "vehicle"
        assert get_category("bus") == "vehicle"
        assert get_category("motorcycle") == "vehicle"
        assert get_category("vehicle_braking") == "vehicle"
    
    def test_get_category_traffic_sign(self):
        """测试交通标志类别识别"""
        assert get_category("traffic_sign") == "traffic_sign"
        assert get_category("speed_limit") == "traffic_sign"
        assert get_category("stop_sign") == "traffic_sign"
    
    def test_get_category_construction(self):
        """测试施工标志类别识别"""
        assert get_category("traffic_cone") == "construction"
        assert get_category("construction_barrier") == "construction"
    
    def test_normalize_vehicle_label(self):
        """测试车辆标签规范化"""
        # 基础车辆类型
        assert normalize_vehicle_label("car") == "vehicle"
        assert normalize_vehicle_label("truck") == "vehicle"
        assert normalize_vehicle_label("bus") == "vehicle"
        
        # 带状态的车辆
        assert normalize_vehicle_label("car_braking") == "vehicle_braking"
        assert normalize_vehicle_label("truck_turning_left") == "vehicle_turning_left"
        assert normalize_vehicle_label("bus_turning_right") == "vehicle_turning_right"
        assert normalize_vehicle_label("car_double_flash") == "vehicle_double_flash"
        
        # 已经是 vehicle 格式
        assert normalize_vehicle_label("vehicle") == "vehicle"
        assert normalize_vehicle_label("vehicle_braking") == "vehicle_braking"
    
    def test_normalize_label(self):
        """测试标签标准化"""
        assert normalize_label("Traffic Sign") == "traffic_sign"
        assert normalize_label("Car-Braking") == "car_braking"
        assert normalize_label("PEDESTRIAN") == "pedestrian"
    
    def test_get_category_emoji(self):
        """测试类别 emoji"""
        assert get_category_emoji("pedestrian") == "🔴"
        assert get_category_emoji("vehicle") == "🟢"
        assert get_category_emoji("traffic_sign") == "🔵"
        assert get_category_emoji("construction") == "🟠"
        assert get_category_emoji("unknown") == "⚪"


class TestImageUtils:
    """图像处理工具测试"""
    
    def test_convert_normalized_coords(self):
        """测试坐标转换"""
        # 1000x1000 归一化到 1920x1080
        bbox = [100, 200, 300, 400]
        result = convert_normalized_coords(bbox, 1920, 1080, base=1000)
        
        assert result[0] == 192   # 100/1000 * 1920
        assert result[1] == 216   # 200/1000 * 1080
        assert result[2] == 576   # 300/1000 * 1920
        assert result[3] == 432   # 400/1000 * 1080
    
    def test_convert_coords_edge_cases(self):
        """测试坐标转换边界情况"""
        # 全图
        bbox = [0, 0, 1000, 1000]
        result = convert_normalized_coords(bbox, 1920, 1080, base=1000)
        assert result == [0, 0, 1920, 1080]
        
        # 零坐标
        bbox = [0, 0, 0, 0]
        result = convert_normalized_coords(bbox, 1920, 1080, base=1000)
        assert result == [0, 0, 0, 0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
