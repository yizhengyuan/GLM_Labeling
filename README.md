# 🚗 GLM-4.6V 交通场景自动标注系统

> 基于大模型的自动化图像标注流水线，实现交通场景中行人、车辆、交通标志的智能检测与标注。

---

## 🎯 项目简介

本项目使用 **GLM-4.6V** 多模态大模型，实现对交通场景视频关键帧的自动标注。支持检测行人、车辆、交通标志、施工标志等多类目标，输出与主流标注软件（如 X-AnyLabeling）兼容的 JSON 格式。

---

## 🔄 核心流程

<p align="center">
  <img src="GLM_Labeling_Manner_1.png" alt="GLM-4.6V 自动标注流程" width="800">
</p>

### 1️⃣ 数据准备 (Data Preparation)
脚本根据命令行参数筛选指定前缀的目标图片，并将其转换为 Base64 编码格式，完成 API 调用前的预处理工作。

### 2️⃣ 模型推理 (Model Inference)
调用 **GLM-4.6V** 多模态大模型，通过预设的提示词（Prompt）引导模型对图像进行语义理解，识别出行人、车辆、交通标志等特定目标，并返回原始检测数据。

### 3️⃣ 数据规范 (Post-Processing)
对模型返回的原始数据进行二次加工：
- **坐标转换**：将模型输出的归一化相对坐标，换算为图像实际的绝对像素坐标
- **标签清洗**：通过映射字典执行标准化逻辑，将同义词、中文或不规范标签统一转换为标准的英文标签

### 4️⃣ 结果导出 (Result Export)
将清洗后的结构化数据封装为 JSON 格式并保存。该格式与主流标注软件（如 X-AnyLabeling）兼容，支持直接加载以进行人工校验或二次编辑。

---

## 🔍 RAG 细粒度分类

对于交通标志的细粒度分类，本项目采用 **RAG（检索增强生成）** 技术，实现对 188 种标准交通标志的精准识别。

<p align="center">
  <img src="GLM_Labeling_Manner_2_new.png" alt="RAG 细粒度分类流程" width="800">
</p>

### 核心流程
1. **一阶段粗分类**：通过 GLM-4.6V 识别交通标志的大类（限速、禁止、警告、指示等）
2. **二阶段精排**：根据大类从向量库中检索候选标志图片
3. **多模态匹配**：将裁剪的标志区域与候选图片进行视觉对比
4. **最终决策**：输出最匹配的标准标志名称

### 支持的交通标志类型
- **限速标志**：`Speed_limit_50_km_h`, `Speed_limit_70_km_h` 等
- **警告标志**：`Road_works_ahead`, `Slippery_road_ahead` 等
- **指示标志**：`Direction_sign`, `Expressway_sign` 等
- **禁止标志**：`No_entry`, `No_parking`, `No_stopping` 等

## 📊 检测类别

| 类别 | 标签示例 | 颜色 |
|------|---------|------|
| 🔴 行人 | `pedestrian`, `crowd` | 红色 |
| 🟢 车辆 | `vehicle`, `vehicle_braking`, `vehicle_turning_left`, `vehicle_turning_right`, `vehicle_double_flash` | 绿色 |
| 🔵 交通标志 | `Speed_limit_50_km_h`, `No_stopping`, `Direction_sign`, `Warning_sign`, `Give_way`, `Road_works_ahead` 等 **188 种** | 蓝色 |
| 🟠 施工标志 | `traffic_cone`, `construction_barrier` | 橙色 |

### 🚗 车辆行为标签说明

当前系统对车辆采用**统一类型 + 行为状态**的标签设计：
- `vehicle` - 正常行驶
- `vehicle_braking` - 刹车（刹车灯亮起）
- `vehicle_turning_left` - 左转（转向灯/车身姿态）
- `vehicle_turning_right` - 右转（转向灯/车身姿态）
- `vehicle_double_flash` - 双闪（危险警告灯）

> 💡 **扩展能力**：如需区分车辆基础类型，可通过修改 prompt 增加细分标签（如 `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `taxi`, `suv` 等），实现更丰富的车辆分类。

### 🚦 交通标志细粒度识别

系统支持对 **188 种交通标志**的细粒度识别，标志库来源于[香港运输署官网](https://www.td.gov.hk/tc/road_safety/road_users_code/index/chapter_7_702_702.html)。

主要类别包括：
- **限速标志**: `Speed_limit_50_km_h`, `Speed_limit_70_km_h`, `Variable_speed_limit` 等
- **禁止标志**: `No_entry`, `No_parking`, `No_stopping`, `No_overtaking` 等
- **警告标志**: `Road_works_ahead`, `Slippery_road_ahead`, `Children_ahead` 等
- **指示标志**: `Direction_sign`, `Expressway_sign`, `One_way_traffic` 等
- **倒计时牌**: `100m_Countdown_markers`, `200m_Countdown_markers` 等

---

## 🗂️ 数据集结构

以 D2 数据集为例（`D2` 为源视频文件名）：

```
D2_dataset/
├── 📄 SUMMARY.md                   # 数据标注总结报告
├── 📊 stats.json                   # 统计数据 (JSON格式)
├── 🎬 video/
│   └── D2.mp4                      # 源视频
├── 🖼️ frames/
│   └── D2_frame_*.jpg              # 原始关键帧 (283张)
├── 📋 annotations/
│   └── D2_frame_*.json             # JSON标注文件 (283个)
└── 🎨 visualized/
    └── D2_frame_*_vis.jpg          # 可视化标注图片 (283张)
```

---

## 🚀 快速开始

### 1. 创建虚拟环境（推荐）
```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置 API Key
```bash
export ZAI_API_KEY="your_api_key_here"
```

### 4. 运行标注
```bash
# 基础标注 - D1 系列图片
python3 auto_labeling_universal.py --prefix D1

# RAG 增强标注（交通标志细粒度分类）
python3 auto_labeling_rag.py --prefix D1 --rag

# 两阶段精细分类（推荐，最准确）
python3 two_stage_classifier.py --test test_images/extracted_frames/D1_frame_0006.jpg --bbox "733,270,776,300"
```

### 5. 生成可视化
```bash
python3 visualize_universal.py --prefix D1
```

### 6. 生成报告
```bash
python3 generate_report.py --prefix D1
```

### 7. 退出虚拟环境
```bash
deactivate
```

---

## 📁 项目文件说明

| 文件 | 说明 |
|------|------|
| `auto_labeling_universal.py` | 通用标注脚本，支持任意前缀的图片 |
| `auto_labeling_rag.py` | RAG 增强标注，交通标志细粒度分类（188 种标准标志） |
| `two_stage_classifier.py` | 两阶段分类器（推荐），先识别类型再识别细节 |
| `visualize_universal.py` | 通用可视化脚本 |
| `generate_report.py` | 标注报告生成脚本 |
| `rag_sign_classifier.py` | RAG 向量库管理工具 |
| `demo_rag_flow.py` | RAG 流程演示脚本 |

---

## 📄 输出格式

标注结果采用 **X-AnyLabeling** 兼容的 JSON 格式：

```json
{
  "version": "0.4.1",
  "shapes": [
    {
      "label": "car",
      "points": [[100, 200], [300, 400]],
      "shape_type": "rectangle",
      "flags": {"category": "vehicle"}
    }
  ],
  "imagePath": "D1_frame_0001.jpg",
  "imageHeight": 1080,
  "imageWidth": 1920
}
```

---

## 🔧 技术栈

- **多模态模型**: GLM-4.6V
- **Python SDK**: zai-sdk
- **图像处理**: Pillow
- **输出格式**: X-AnyLabeling JSON

---

## 📝 License

MIT License

---

<p align="center">
  <b>🚀 Powered by GLM-4.6V</b>
</p>