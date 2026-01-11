# 基于 GLARE 论文的数据标注框架

> 参考论文: *GLARE: A Dataset for Traffic Sign Detection in Sun Glare*
>
> IEEE: https://ieeexplore.ieee.org/document/10197287
>
> GitHub: https://github.com/NicholasCG/GLARE_Dataset

---

## 一、GLARE 数据集概览

### 1.1 基本信息

| 属性 | GLARE 数据集 |
|------|-------------|
| 图片数量 | 2,157 张 |
| 标志类别 | 41 类 (美国交通标志) |
| 来源视频 | 33 个行车记录仪视频 |
| 原始素材 | 38 小时，463 个视频 |
| 最终片段 | 189 个 tracks，共 18 分 11 秒 |
| 图像分辨率 | 720×480 (480p) ~ 1920×1080 (1080p) |
| 标注格式 | Bounding Box CSV (与 LISA 兼容) |

### 1.2 与其他数据集对比

| 数据集 | 图片数 | 类别 | 特点 | 国家 | 年份 |
|--------|--------|------|------|------|------|
| GTSRB | 51,839 | 43 | 一般遮挡 | 德国 | 2011 |
| LISA | 6,610 | 49 | 一般遮挡 | 美国 | 2012 |
| TT-100K | 100,000 | 221 | 一般遮挡 | 中国 | 2016 |
| MTSD | 100,000 | 313 | 一般遮挡 | 全球 | 2019 |
| **GLARE** | 2,157 | 41 | **强眩光** | 美国 | 2022 |

**GLARE 的独特价值**: 首个专注于强眩光条件的交通标志数据集

---

## 二、数据收集流程

### 2.1 视频采集

```
原始素材: 38 小时, 463 个视频
    ↓ 筛选含眩光视频
含眩光视频: 163 个
    ↓ 提取眩光+标志片段
有效片段: 189 个 tracks
    ↓ 帧采样
最终图片: 2,157 张
```

### 2.2 筛选标准

1. **必须同时包含**: 眩光 + 交通标志
2. **片段定义**: 连续存在眩光的视频段落
3. **缓冲处理**: 每个片段开头增加约 0.5 秒，便于定位眩光起点

### 2.3 帧采样策略

| 策略 | 说明 |
|------|------|
| 采样间隔 | 每 5 帧标注一次 |
| 单标志上限 | 每个标志最多 30 张图片 |
| 目的 | 避免同一标志过度曝光，保持类别平衡 |

---

## 三、标注流程

### 3.1 两阶段标注

```
阶段一: Bounding Box 定位
    - 2 人同时标注
    - 使用 OpenCV Video Label 工具
    - 自动跟踪算法: Re3

阶段二: 质量审核
    - 2 人审核
    - 剔除错误标注
    - 重新标注问题样本
```

### 3.2 标注工具

- **工具**: [OpenCV Video Label](https://github.com/natdebru/opencv-video-label)
- **跟踪算法**: Re3 (Real-time Recurrent Regression)
- **输出格式**: CSV (与 LISA 兼容)

### 3.3 标注属性

| 属性 | 说明 |
|------|------|
| Bounding Box | 左上角坐标 + 宽高 |
| 标志类别 | 41 类之一 |
| Occluded | 是否被遮挡 |
| On another road | 是否在其他道路上 |

---

## 四、眩光类型分类

### 4.1 四种眩光类型

| 类型 | 英文 | 描述 |
|------|------|------|
| 类型 1 | Clear sun without clouds | 无云遮挡的清晰太阳，画面中有明亮的圆形太阳 |
| 类型 2 | Clear sun with clouds | 有云但太阳可见，云层增加了整体亮度 |
| 类型 3 | Clouds with non-visible sun | 云层遮挡太阳不可见，但仍有视觉干扰 |
| 类型 4 | Sun with other interference | 相机设置或其他因素导致的类似眩光效果 |

### 4.2 眩光特征

- 眩光可能覆盖标志区域，也可能在标志附近
- 图像整体亮度提高
- 对比度降低，细节丢失
- 可能出现光晕、光斑

---

## 五、实验结果与启示

### 5.1 检测性能对比

| 训练数据 | mAP@0.5 | mAP@0.5:0.95 |
|---------|---------|--------------|
| 仅 LISA (无眩光) | 34.7 | 19.4 |
| 仅 GLARE (眩光) | 59.2 | 39.6 |
| **混合数据** | **67.9** | **42.3** |

### 5.2 关键发现

1. **眩光显著降低检测性能**: 仅用正常数据训练的模型，在眩光条件下表现差
2. **混合训练最优**: LISA + GLARE 混合训练效果最佳
3. **YOLO 系列受影响大**: 实时检测常用的 YOLO 系列在眩光条件下性能下降明显

### 5.3 最佳模型

| 场景 | 最佳模型 |
|------|---------|
| 仅 GLARE 训练 | Faster-RCNN + Swin Transformer |
| 仅 LISA 训练 | TOOD |
| 混合数据训练 | TOOD (mAP@0.5:0.95) / Swin (mAP@0.5) |

---

## 六、我们项目的标注框架

### 6.1 扩展干扰类型

根据李老师要求，我们需要扩展到三类视觉干扰：

| 干扰类型 | 英文 | 子类型 | 示例场景 |
|---------|------|--------|---------|
| 低光照 | Lowlight | 夜间、隧道、阴影 | 夜间视频、隧道内 |
| 模糊 | Blur | 运动模糊、对焦模糊、雨水模糊 | 快速转弯、雨天 |
| 反光/眩光 | Reflection/Glare | 阳光直射、车灯反光、镜面反射 | 日出日落、逆光行驶 |

### 6.2 标注属性设计

```yaml
# 基础属性 (参考 GLARE)
bounding_box:
  x: int          # 左上角 x
  y: int          # 左上角 y
  width: int      # 宽度
  height: int     # 高度

sign_class: str   # 标志类别

# 遮挡属性 (参考 GLARE)
occluded: bool           # 是否被遮挡
on_another_road: bool    # 是否在其他道路上

# 干扰属性 (扩展)
interference:
  type: enum      # none / lowlight / blur / glare
  severity: enum  # mild / moderate / severe

# 图像质量 (扩展)
image_quality:
  overall: enum   # good / acceptable / poor
  notes: str      # 备注
```

### 6.3 我们的数据来源

| 来源 | 视频数 | 时长 | 干扰类型覆盖 |
|------|--------|------|-------------|
| DJI_1080p 日间 | 12 | ~150 min | Glare (部分) |
| DJI_1080p 夜间 | 4 | ~53 min | Lowlight |
| DJI2 日间 | 6 | ~195 min | Glare (部分) |
| DJI2 夜间 | 3 | ~44 min | Lowlight |
| DJI2 傍晚 | 1 | ~17 min | Lowlight + Glare |
| **合计** | **26** | **~459 min** | - |

### 6.4 采样策略建议

参考 GLARE 的做法：

| 策略 | GLARE | 我们的建议 |
|------|-------|-----------|
| 采样间隔 | 每 5 帧 | 每 5-10 帧 (根据场景变化调整) |
| 单标志上限 | 30 张 | 20-30 张 |
| 干扰类型平衡 | 仅眩光 | Lowlight : Blur : Glare ≈ 1:1:1 |
| 质量审核 | 2 人 | 至少 1 人复核 |

### 6.5 标注文件格式

建议使用 CSV 格式 (与 LISA/GLARE 兼容):

```csv
filename,width,height,class,xmin,ymin,xmax,ymax,occluded,on_another_road,interference_type,interference_severity,image_quality
frame_001.jpg,1920,1080,stop_sign,100,200,150,280,0,0,glare,moderate,acceptable
frame_002.jpg,1920,1080,speed_limit_50,500,300,580,420,1,0,lowlight,severe,poor
```

---

## 七、下一步工作

1. **视频筛选**: 从 26 个视频中标记含干扰的片段
2. **标注工具选择**: 评估 OpenCV Video Label 或其他工具
3. **试标注**: 选择 2-3 个视频进行试标注，验证框架可行性
4. **迭代优化**: 根据试标注结果调整标注规范

---

## 参考资料

1. Gray, N., et al. (2022). GLARE: A Dataset for Traffic Sign Detection in Sun Glare. *IEEE*.
2. LISA Traffic Sign Dataset: http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html
3. OpenCV Video Label Tool: https://github.com/natdebru/opencv-video-label
