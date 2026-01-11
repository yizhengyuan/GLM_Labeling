"""
AI 审查模块

分层递进式标注质量审查：
- L0: 规则检查（100%，零成本）
- L1: 时序一致性检查（连续帧对比）
- L2: AI 快速扫描（规则触发时）
- L3: AI 深度复核（随机抽样）
"""

import random
from typing import List, Dict, Optional, Any
from pathlib import Path

from .models import Issue, IssueType, IssueSeverity


class AnnotationReviewer:
    """
    分层递进式审查器
    """
    
    def __init__(
        self,
        api_key: str = None,
        sample_rate: float = 0.05,  # 5% 随机深度审查
        enable_ai: bool = True
    ):
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.enable_ai = enable_ai
        
        # 上一帧的检测结果（用于时序检查）
        self._prev_frame_id: Optional[str] = None
        self._prev_detections: Optional[List[dict]] = None
    
    def review(
        self,
        frame_id: str,
        image_path: str,
        detections: List[dict]
    ) -> List[Issue]:
        """
        审查单帧标注
        
        Args:
            frame_id: 帧ID
            image_path: 图片路径
            detections: 检测结果
            
        Returns:
            问题列表
        """
        issues = []
        
        # ========== L0: 规则检查 (100%) ==========
        issues.extend(self._check_bbox_rules(frame_id, image_path, detections))
        issues.extend(self._check_label_rules(frame_id, detections))
        
        # ========== L1: 时序一致性检查 ==========
        if self._prev_detections is not None:
            issues.extend(self._check_temporal(
                frame_id,
                detections,
                self._prev_detections
            ))
        
        # 更新前一帧缓存
        self._prev_frame_id = frame_id
        self._prev_detections = detections.copy() if detections else []
        
        # ========== L2: AI 快速扫描 (规则触发时) ==========
        if self.enable_ai and len(issues) > 0:
            # 可以在这里调用 AI 确认问题
            # 目前先跳过，减少 API 调用
            pass
        
        # ========== L3: AI 深度复核 (随机抽样) ==========
        if self.enable_ai and self.api_key and random.random() < self.sample_rate:
            deep_issues = self._ai_deep_review(frame_id, image_path, detections)
            issues.extend(deep_issues)
        
        return issues
    
    def _check_bbox_rules(
        self,
        frame_id: str,
        image_path: str,
        detections: List[dict]
    ) -> List[Issue]:
        """L0: bbox 规则检查"""
        issues = []
        
        try:
            from ..utils import get_image_size
            width, height = get_image_size(image_path)
            image_area = width * height
        except Exception:
            return issues
        
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            label = det.get("label", "unknown")
            
            # bbox 太小
            if area < 100:
                issues.append(Issue(
                    frame_id=frame_id,
                    issue_type=IssueType.BBOX_TOO_SMALL,
                    severity=IssueSeverity.WARNING,
                    description=f"{label} 的 bbox 面积过小 ({area}px²)",
                    suggestion="可能是误检，建议删除或检查",
                    detected_by="rule",
                    bbox=bbox
                ))
            
            # bbox 太大
            if area > image_area * 0.7:
                issues.append(Issue(
                    frame_id=frame_id,
                    issue_type=IssueType.BBOX_TOO_LARGE,
                    severity=IssueSeverity.WARNING,
                    description=f"{label} 的 bbox 占图片 {area/image_area*100:.0f}%",
                    suggestion="检查是否错误框选了整个场景",
                    detected_by="rule",
                    bbox=bbox
                ))
            
            # bbox 坐标异常
            if x1 >= x2 or y1 >= y2:
                issues.append(Issue(
                    frame_id=frame_id,
                    issue_type=IssueType.BBOX_TOO_SMALL,
                    severity=IssueSeverity.ERROR,
                    description=f"{label} 的 bbox 坐标异常 ({bbox})",
                    suggestion="bbox 坐标顺序错误",
                    detected_by="rule",
                    bbox=bbox
                ))
        
        # 检查 bbox 重叠
        issues.extend(self._check_bbox_overlap(frame_id, detections))
        
        return issues
    
    def _check_bbox_overlap(
        self,
        frame_id: str,
        detections: List[dict]
    ) -> List[Issue]:
        """检查 bbox 重叠"""
        issues = []
        
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections):
                if i >= j:
                    continue
                
                bbox1 = det1.get("bbox", [])
                bbox2 = det2.get("bbox", [])
                
                if len(bbox1) != 4 or len(bbox2) != 4:
                    continue
                
                # 计算 IoU
                iou = self._calculate_iou(bbox1, bbox2)
                
                if iou > 0.9:  # 高度重叠
                    issues.append(Issue(
                        frame_id=frame_id,
                        issue_type=IssueType.BBOX_OVERLAP,
                        severity=IssueSeverity.WARNING,
                        description=f"{det1['label']} 和 {det2['label']} 高度重叠 (IoU={iou:.2f})",
                        suggestion="可能是重复检测，建议删除一个",
                        detected_by="rule",
                    ))
        
        return issues
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """计算两个 bbox 的 IoU"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _check_label_rules(
        self,
        frame_id: str,
        detections: List[dict]
    ) -> List[Issue]:
        """L0: 标签规则检查"""
        issues = []
        
        valid_categories = {"pedestrian", "vehicle", "traffic_sign", "construction"}
        
        for det in detections:
            category = det.get("category", "unknown")
            label = det.get("label", "unknown")
            
            if category == "unknown" or category not in valid_categories:
                issues.append(Issue(
                    frame_id=frame_id,
                    issue_type=IssueType.UNKNOWN_LABEL,
                    severity=IssueSeverity.INFO,
                    description=f"未识别的类别: {label} (category={category})",
                    suggestion="考虑归类到已知类别或添加新类别",
                    detected_by="rule",
                ))
        
        return issues
    
    def _check_temporal(
        self,
        frame_id: str,
        curr: List[dict],
        prev: List[dict]
    ) -> List[Issue]:
        """L1: 时序一致性检查"""
        issues = []
        
        curr_count = len(curr)
        prev_count = len(prev)
        
        # 物体全部消失
        if curr_count == 0 and prev_count > 3:
            issues.append(Issue(
                frame_id=frame_id,
                issue_type=IssueType.TEMPORAL_DISAPPEAR,
                severity=IssueSeverity.WARNING,
                description=f"物体突然全部消失（{prev_count} → 0）",
                suggestion="可能是检测失败，建议人工复核",
                detected_by="rule",
            ))
        
        # 物体数量剧增
        if prev_count > 0 and curr_count > prev_count * 3:
            issues.append(Issue(
                frame_id=frame_id,
                issue_type=IssueType.TEMPORAL_APPEAR,
                severity=IssueSeverity.INFO,
                description=f"物体数量剧增（{prev_count} → {curr_count}）",
                suggestion="可能是场景切换，建议确认",
                detected_by="rule",
            ))
        
        return issues
    
    def _ai_deep_review(
        self,
        frame_id: str,
        image_path: str,
        detections: List[dict]
    ) -> List[Issue]:
        """L3: AI 深度复核"""
        issues = []
        
        # 如果没有 API key，跳过
        if not self.api_key:
            return issues
        
        try:
            from ..utils import image_to_base64_url
            from zai import ZaiClient
            
            client = ZaiClient(api_key=self.api_key)
            base64_url = image_to_base64_url(image_path)
            
            # 构建检测结果描述
            det_desc = "\n".join([
                f"- {d['label']} at {d['bbox']}"
                for d in detections
            ]) if detections else "（无检测结果）"
            
            prompt = f"""请检查这张图片的标注是否完整准确。

当前标注结果：
{det_desc}

请检查：
1. 是否有明显漏检的行人、车辆或交通标志？
2. 现有标签是否正确？

如果发现问题，请简要描述。如果没有问题，回复"无问题"。"""

            response = client.chat.completions.create(
                model="glm-4.6v",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": base64_url}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                timeout=30
            )
            
            result = response.choices[0].message.content.strip()
            
            # 解析结果
            if "无问题" not in result and len(result) > 10:
                issues.append(Issue(
                    frame_id=frame_id,
                    issue_type=IssueType.MISSING_DETECTION,
                    severity=IssueSeverity.WARNING,
                    description=f"AI 复核发现问题: {result[:200]}",
                    suggestion="建议人工复核",
                    detected_by="ai_deep",
                ))
                
        except Exception as e:
            # AI 审查失败不影响主流程
            pass
        
        return issues
    
    def reset(self):
        """重置审查器状态"""
        self._prev_frame_id = None
        self._prev_detections = None






