"""
标签处理工具

标签规范化、类别判断等。
"""

from typing import Optional


# 类别关键词映射（按 ID 顺序）
# ID 0: traffic_sign, ID 1: pedestrian, ID 2: vehicle, ID 3: small_obstacle
CATEGORY_KEYWORDS = {
    "traffic_sign": ["sign", "speed", "limit", "no_", "traffic", "light", "stop", 
                     "give_way", "direction", "exit", "lane", "countdown"],
    "pedestrian": ["pedestrian", "person", "people", "child", "cyclist", "crowd"],
    "vehicle": ["car", "truck", "bus", "motorcycle", "bicycle", "van", "suv", "taxi", "vehicle"],
    "small_obstacle": ["cone", "construction", "barrier", "road_work", "detour"],
}

# 车辆类型列表
VEHICLE_TYPES = ["car", "truck", "bus", "van", "motorcycle", "bicycle", "taxi", "suv"]

# 车辆状态后缀
VEHICLE_STATES = ["_braking", "_double_flash", "_turning_left", "_turning_right"]


def get_category(label: str) -> str:
    """
    根据标签获取粗粒度类别
    
    Args:
        label: 原始标签
        
    Returns:
        类别名称：traffic_sign(ID0), pedestrian(ID1), vehicle(ID2), small_obstacle(ID3), unknown
    """
    label_lower = label.lower().replace(" ", "_").replace("-", "_")
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in label_lower for kw in keywords):
            return category
    
    return "unknown"


def normalize_label(label: str) -> str:
    """
    标准化标签格式
    
    - 转小写
    - 空格和连字符替换为下划线
    
    Args:
        label: 原始标签
        
    Returns:
        标准化后的标签
    """
    return label.lower().replace(" ", "_").replace("-", "_")


def normalize_vehicle_label(label: str) -> str:
    """
    将车辆类型标签规范化为 vehicle 格式
    
    例如：
    - car -> vehicle
    - car_braking -> vehicle_braking
    - truck_turning_left -> vehicle_turning_left
    - motorcycle -> vehicle
    
    Args:
        label: 原始车辆标签
        
    Returns:
        规范化后的 vehicle 格式标签
    """
    label_lower = normalize_label(label)
    
    # 检查是否以车辆类型开头
    for vtype in VEHICLE_TYPES:
        if label_lower.startswith(vtype):
            suffix = label_lower[len(vtype):]
            
            # 如果有标准状态后缀
            if suffix in VEHICLE_STATES:
                return "vehicle" + suffix
            
            # 检查是否包含状态关键词
            if "braking" in suffix or "brake" in suffix:
                return "vehicle_braking"
            elif "double_flash" in suffix or "hazard" in suffix:
                return "vehicle_double_flash"
            elif "turning_left" in suffix or "turn_left" in suffix or "left_turn" in suffix:
                return "vehicle_turning_left"
            elif "turning_right" in suffix or "turn_right" in suffix or "right_turn" in suffix:
                return "vehicle_turning_right"
            else:
                return "vehicle"
    
    # 如果已经是 vehicle 格式
    if label_lower.startswith("vehicle"):
        return label_lower
    
    return label


def get_category_color(category: str) -> str:
    """
    获取类别对应的显示颜色
    
    Args:
        category: 类别名称
        
    Returns:
        颜色名称
    """
    colors = {
        "traffic_sign": "blue",
        "pedestrian": "red",
        "vehicle": "green",
        "small_obstacle": "orange",
        "unknown": "gray"
    }
    return colors.get(category, "gray")


def get_category_emoji(category: str) -> str:
    """
    获取类别对应的 emoji
    
    Args:
        category: 类别名称
        
    Returns:
        emoji 字符
    """
    emojis = {
        "traffic_sign": "🔵",
        "pedestrian": "🔴",
        "vehicle": "🟢",
        "small_obstacle": "🟠",
        "unknown": "⚪"
    }
    return emojis.get(category, "⚪")
