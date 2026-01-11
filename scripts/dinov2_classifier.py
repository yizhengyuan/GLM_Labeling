#!/usr/bin/env python3
"""
🦖 DINOv2 + Chroma 交通标志分类器

使用 Meta 的 DINOv2 模型进行特征提取，比 CLIP 更强的视觉表示能力。
DINOv2 通过自监督学习获得了出色的细粒度视觉特征，特别适合交通标志匹配。

优势:
    - 更强的视觉特征: DINOv2 在多种下游任务上超越 CLIP
    - 更好的细粒度区分: 对形状、颜色、细节的感知更精确
    - 对背景干扰更鲁棒: 自监督训练使其对复杂背景更不敏感

用法:
    # 1. 首次运行：建立向量库
    python scripts/dinov2_classifier.py --build
    
    # 2. 测试分类
    python scripts/dinov2_classifier.py --test path/to/sign.jpg
    
    # 3. 作为模块导入
    from scripts.dinov2_classifier import DINOv2SignClassifier
    classifier = DINOv2SignClassifier()
    label, score = classifier.classify(image_path, bbox)
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import uuid

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# ============================================================================
# 配置
# ============================================================================

# 标志图片目录
SIGNS_DIR = Path("raw_data/signs")

# Chroma 数据库目录（DINOv2 专用）
CHROMA_DB_DIR = Path("raw_data/chroma_dinov2_db")

# DINOv2 模型配置
# 可选: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
# vitl14 是效果和速度的良好平衡，vitg14 最强但需要更多显存
DINOV2_MODEL_NAME = "dinov2_vitl14"

# 69 个摩托车安全相关标志（与其他分类器保持一致）
MOTORCYCLE_SAFETY_SIGNS = [
    "No_motor_cycles_or_motor_tricycles",
    "No_motor_vehicles_except_motor_cyclists_and_motor_tricycles",
    "Parking_place_for_motor_cycles_only",
    "Speed_limit_(in_km_h)",
    "Variable_speed_limit_(in_km_h)",
    "Reduce_speed_now",
    "Keep_in_low_gear",
    "Use_low_gear",
    "Use_low_gear_for_distance_shown",
    "Slippery_road_ahead",
    "Loose_chippings_ahead",
    "Uneven_road_surface_ahead",
    "Road_hump_ahead",
    "Ramp_or_sudden_change_of_road_level_ahead",
    "Ramp_or_sudden_change_of_road_level",
    "Risk_of_falling_or_fallen_rocks_ahead",
    "Road_works_ahead",
    "Bend_to_left_ahead",
    "Double_bend_ahead_first_to_right",
    "Sharp_deviation_of_route_to_left",
    "Steep_hill_downwards_ahead",
    "Steep_hill_upwards_ahead",
    "Road_narrows_on_both_sides_ahead",
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
    "Stop_and_give_way",
    "Give_way_to_traffic_on_major_road",
    "Distance_to__Stop__line",
    "Distance_to__Give_way__line",
    "Stop_or_give_way_ahead_(with_distance_to_line_ahead_given_below)",
    "Cross_roads_ahead",
    "T-junction_ahead",
    "Side_road_to_left_ahead",
    "Staggered_junction_ahead",
    "Traffic_merging_from_left",
    "Merging_into_main_traffic_on_left",
    "Two-way_traffic_ahead",
    "Two-way_traffic_across_a_one-way_road_ahead",
    "Traffic_lights_ahead",
    "Traffic_signals_ahead",
    "Red_light_camera_control_zone",
    "Red_light_speed_camera_ahead",
    "Prepare_to_stop_if_signalled_to_do_so",
    "Vehicles_must_stop_at_the_sign_(sign_used_by_police)",
    "Pedestrian_crossing_ahead",
    "Pedestrians_Ahead",
    "Pedestrian_on_or_crossing_road_ahead",
    "Children_ahead",
    "School_ahead",
    "Playground_ahead",
    "Cyclists_ahead",
    "Disabled_persons_ahead",
    "Visually_impaired_persons_ahead",
    "Traffic_Accident_blackspot_ahead",
    "Pedestrian_Accident_blackspot_ahead",
    "Fog_or_mist_ahead",
    "Restricted_headroom_ahead",
    "No_vehicles_over_height_shown_(including_load)",
    "No_vehicles_over_width_shown_(including_load)",
    "No_vehicles_over_gross_vehicle_weight_shown_(including_load)",
    "No_vehicles_over_axle_weight_shown_(including_load)",
]


# ============================================================================
# DINOv2 模型封装
# ============================================================================

class DINOv2Encoder:
    """DINOv2 图像编码器"""
    
    def __init__(self, model_name: str = DINOV2_MODEL_NAME):
        """
        初始化 DINOv2 模型
        
        Args:
            model_name: 模型名称
                - dinov2_vits14: Small (21M params, 最快)
                - dinov2_vitb14: Base (86M params)
                - dinov2_vitl14: Large (300M params, 推荐)
                - dinov2_vitg14: Giant (1.1B params, 最强)
        """
        print(f"🦖 加载 DINOv2 模型: {model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"   设备: {self.device}")
        
        # 从 torch hub 加载 DINOv2
        self.use_timm = False
        self.input_size = 224  # 默认 hub 模型输入大小
        
        try:
            # 先尝试从缓存加载
            self.model = torch.hub.load('facebookresearch/dinov2', model_name, trust_repo=True)
        except Exception as e:
            print(f"   ⚠️ 从 hub 加载失败: {e}")
            print(f"   尝试使用本地 timm 模型...")
            # 备选方案：使用 timm 加载类似模型
            try:
                import timm
                # 使用 timm 的 ViT 模型作为替代
                timm_model_map = {
                    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
                    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
                    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
                    "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m",
                }
                timm_name = timm_model_map.get(model_name, "vit_large_patch14_dinov2.lvd142m")
                print(f"   使用 timm 模型: {timm_name}")
                self.model = timm.create_model(timm_name, pretrained=True, num_classes=0)
                self.use_timm = True
                self.input_size = 518  # timm DINOv2 需要 518x518
            except Exception as e2:
                raise RuntimeError(f"无法加载 DINOv2 模型: {e}, {e2}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # DINOv2 预处理 - 根据模型类型调整输入大小
        resize_size = int(self.input_size * 256 / 224)  # 保持缩放比例
        self.transform = transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print(f"   ✅ DINOv2 模型加载完成 (输入: {self.input_size}x{self.input_size})")
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        编码单张图片
        
        Args:
            image: PIL Image 对象
        
        Returns:
            归一化后的特征向量 (numpy array)
        """
        with torch.no_grad():
            # 预处理
            image_input = self.transform(image).unsqueeze(0).to(self.device)
            
            # 提取特征 (CLS token)
            features = self.model(image_input)
            
            # L2 归一化
            features = F.normalize(features, p=2, dim=-1)
            
            return features.cpu().numpy().flatten()
    
    def encode_image_path(self, image_path: str) -> np.ndarray:
        """从路径加载并编码图片"""
        image = Image.open(image_path).convert("RGB")
        return self.encode_image(image)


# ============================================================================
# Chroma 向量数据库封装
# ============================================================================

class SignVectorDB:
    """交通标志向量数据库"""
    
    def __init__(self, db_dir: Path = CHROMA_DB_DIR, collection_name: str = "traffic_signs_dinov2"):
        """
        初始化 Chroma 数据库
        
        Args:
            db_dir: 数据库存储目录
            collection_name: 集合名称
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("请安装 chromadb: pip install chromadb")
        
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用持久化客户端
        self.client = chromadb.PersistentClient(
            path=str(self.db_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 获取或创建集合（使用余弦相似度）
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"📦 Chroma 数据库: {self.db_dir}")
        print(f"   集合: {collection_name} ({self.collection.count()} 条记录)")
    
    def add_signs(self, embeddings: List[np.ndarray], labels: List[str], image_paths: List[str]):
        """
        批量添加标志向量
        
        Args:
            embeddings: 特征向量列表
            labels: 标签列表
            image_paths: 图片路径列表
        """
        ids = [f"sign_{i}_{label[:50]}" for i, label in enumerate(labels)]
        
        self.collection.add(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],
            metadatas=[{"label": label, "image_path": path} for label, path in zip(labels, image_paths)],
            documents=labels
        )
        
        print(f"   ✅ 添加 {len(labels)} 个标志向量")
    
    def query(self, query_embedding: np.ndarray, top_k: int = 1) -> List[dict]:
        """
        查询最相似的标志
        
        Args:
            query_embedding: 查询向量
            top_k: 返回前 k 个结果
        
        Returns:
            结果列表，每个元素包含 label, score, image_path
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        
        output = []
        if results["metadatas"] and results["distances"]:
            for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
                # Chroma 使用距离，余弦相似度 = 1 - distance
                score = 1 - distance
                output.append({
                    "label": metadata["label"],
                    "score": score,
                    "image_path": metadata.get("image_path", "")
                })
        
        return output
    
    def clear(self):
        """清空集合"""
        collection_name = "traffic_signs_dinov2"
        self.client.delete_collection(collection_name)
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("   🗑️ 已清空向量库")


# ============================================================================
# 主分类器
# ============================================================================

class DINOv2SignClassifier:
    """
    DINOv2 + Chroma 交通标志分类器
    
    用法:
        classifier = DINOv2SignClassifier()
        
        # 建库（首次运行）
        classifier.build_database()
        
        # 分类
        label, score = classifier.classify("image.jpg", [100, 200, 150, 250])
    """
    
    def __init__(
        self, 
        signs_dir: Path = SIGNS_DIR,
        db_dir: Path = CHROMA_DB_DIR,
        model_name: str = DINOV2_MODEL_NAME,
        use_69_signs: bool = True,
        similarity_threshold: float = 0.5
    ):
        """
        初始化分类器
        
        Args:
            signs_dir: 标志图片目录
            db_dir: 向量数据库目录
            model_name: DINOv2 模型名称
            use_69_signs: 是否只使用 69 个摩托车安全相关标志
            similarity_threshold: 相似度阈值，低于此值返回 "other"
        """
        self.signs_dir = Path(signs_dir)
        self.db_dir = Path(db_dir)
        self.model_name = model_name
        self.use_69_signs = use_69_signs
        self.similarity_threshold = similarity_threshold
        
        # 懒加载
        self._encoder = None
        self._db = None
    
    @property
    def encoder(self) -> DINOv2Encoder:
        """懒加载 DINOv2 编码器"""
        if self._encoder is None:
            self._encoder = DINOv2Encoder(self.model_name)
        return self._encoder
    
    @property
    def db(self) -> SignVectorDB:
        """懒加载向量数据库"""
        if self._db is None:
            self._db = SignVectorDB(self.db_dir)
        return self._db
    
    def build_database(self, rebuild: bool = False):
        """
        建立向量数据库
        
        Args:
            rebuild: 是否重建（清空现有数据）
        """
        print("=" * 60)
        print("🏗️ 建立 DINOv2 交通标志向量数据库")
        print("=" * 60)
        
        if rebuild:
            self.db.clear()
        
        # 检查是否已有数据
        if self.db.collection.count() > 0:
            print(f"   ⚠️ 数据库已有 {self.db.collection.count()} 条记录，跳过建库")
            print(f"   若需重建，请使用 --rebuild 参数")
            return
        
        # 获取标志图片
        if self.use_69_signs:
            sign_names = MOTORCYCLE_SAFETY_SIGNS
            print(f"   📋 使用 69 个摩托车安全相关标志")
        else:
            sign_names = [f.stem for f in sorted(self.signs_dir.glob("*.png"))]
            print(f"   📋 使用全部 {len(sign_names)} 个标志")
        
        # 编码并添加
        embeddings = []
        labels = []
        paths = []
        
        print(f"\n🔄 编码标志图片...")
        
        # 先确保编码器加载成功
        try:
            _ = self.encoder  # 触发懒加载
        except Exception as e:
            print(f"   ❌ 无法加载 DINOv2 模型: {e}")
            return
        
        for i, name in enumerate(sign_names):
            # 查找对应的图片文件
            image_path = self.signs_dir / f"{name}.png"
            if not image_path.exists():
                # 尝试模糊匹配
                matches = list(self.signs_dir.glob(f"{name}*.png"))
                if matches:
                    image_path = matches[0]
                else:
                    print(f"   ⚠️ 未找到图片: {name}")
                    continue
            
            try:
                embedding = self.encoder.encode_image_path(str(image_path))
                embeddings.append(embedding)
                labels.append(name)
                paths.append(str(image_path))
                
                if (i + 1) % 10 == 0:
                    print(f"   已处理 {i + 1}/{len(sign_names)}")
                    
            except Exception as e:
                print(f"   ❌ 编码失败 {name}: {e}")
        
        # 添加到数据库
        print(f"\n💾 添加到向量数据库...")
        self.db.add_signs(embeddings, labels, paths)
        
        print(f"\n✅ 建库完成！共 {len(labels)} 个标志")
        print("=" * 60)
    
    def classify(
        self, 
        image_path: str, 
        bbox: List[int] = None,
        top_k: int = 1
    ) -> Tuple[str, float]:
        """
        分类交通标志
        
        Args:
            image_path: 原图路径
            bbox: 边界框 [x1, y1, x2, y2]，如果为 None 则使用整张图片
            top_k: 返回前 k 个结果
        
        Returns:
            (label, score) - 标签和相似度分数
        """
        # 加载并裁剪图片
        image = Image.open(image_path).convert("RGB")
        
        if bbox:
            padding = 5
            x1 = max(0, bbox[0] - padding)
            y1 = max(0, bbox[1] - padding)
            x2 = min(image.width, bbox[2] + padding)
            y2 = min(image.height, bbox[3] + padding)
            image = image.crop((x1, y1, x2, y2))
        
        # 编码
        query_embedding = self.encoder.encode_image(image)
        
        # 查询
        results = self.db.query(query_embedding, top_k=top_k)
        
        if not results:
            return "other", 0.0
        
        top_result = results[0]
        
        # 检查阈值
        if top_result["score"] < self.similarity_threshold:
            return "other", top_result["score"]
        
        return top_result["label"], top_result["score"]
    
    def classify_with_details(
        self, 
        image_path: str, 
        bbox: List[int] = None,
        top_k: int = 3
    ) -> List[dict]:
        """
        分类并返回详细结果
        
        Returns:
            Top-K 结果列表
        """
        image = Image.open(image_path).convert("RGB")
        
        if bbox:
            padding = 5
            x1 = max(0, bbox[0] - padding)
            y1 = max(0, bbox[1] - padding)
            x2 = min(image.width, bbox[2] + padding)
            y2 = min(image.height, bbox[3] + padding)
            image = image.crop((x1, y1, x2, y2))
        
        query_embedding = self.encoder.encode_image(image)
        return self.db.query(query_embedding, top_k=top_k)
    
    def classify_with_details_from_image(
        self, 
        image: Image.Image,
        top_k: int = 3
    ) -> List[dict]:
        """
        从 PIL Image 直接分类并返回详细结果
        
        Args:
            image: PIL Image 对象
            top_k: 返回前 k 个结果
        
        Returns:
            Top-K 结果列表
        """
        query_embedding = self.encoder.encode_image(image.convert("RGB"))
        return self.db.query(query_embedding, top_k=top_k)


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="DINOv2 + Chroma 交通标志分类器")
    parser.add_argument("--build", action="store_true", help="建立向量数据库")
    parser.add_argument("--rebuild", action="store_true", help="重建向量数据库（清空现有数据）")
    parser.add_argument("--test", type=str, help="测试分类，指定图片路径")
    parser.add_argument("--bbox", type=str, help="边界框，格式: x1,y1,x2,y2")
    parser.add_argument("--all-signs", action="store_true", help="使用全部 188 个标志（默认只用 69 个）")
    parser.add_argument("--top-k", type=int, default=5, help="返回前 K 个结果（默认 5）")
    parser.add_argument("--model", type=str, default=DINOV2_MODEL_NAME,
                       choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                       help=f"DINOv2 模型 (默认: {DINOV2_MODEL_NAME})")
    args = parser.parse_args()
    
    classifier = DINOv2SignClassifier(
        use_69_signs=not args.all_signs,
        model_name=args.model
    )
    
    if args.build or args.rebuild:
        classifier.build_database(rebuild=args.rebuild)
    
    if args.test:
        print("\n" + "=" * 60)
        print(f"🔍 测试分类: {args.test}")
        print("=" * 60)
        
        bbox = None
        if args.bbox:
            bbox = [int(x) for x in args.bbox.split(",")]
            print(f"   边界框: {bbox}")
        
        results = classifier.classify_with_details(args.test, bbox, top_k=args.top_k)
        
        print(f"\n📊 Top-{args.top_k} 结果:")
        for i, r in enumerate(results):
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"   {emoji} [{i+1}] {r['label']}")
            print(f"       相似度: {r['score']:.4f}")
        
        # 最终标签
        label, score = classifier.classify(args.test, bbox)
        print(f"\n🏷️ 最终标签: {label} (score: {score:.4f})")
    
    if not args.build and not args.rebuild and not args.test:
        parser.print_help()


if __name__ == "__main__":
    main()

