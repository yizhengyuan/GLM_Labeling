#!/usr/bin/env python3
"""
🎯 CLIP + Chroma 交通标志分类器（真正的 RAG）

使用 CLIP 模型将标志图片编码为向量，存入 Chroma 向量数据库，
通过图像相似度检索实现精确的交通标志分类。

用法:
    # 1. 首次运行：建立向量库
    python scripts/clip_rag_classifier.py --build
    
    # 2. 测试分类
    python scripts/clip_rag_classifier.py --test path/to/sign.jpg
    
    # 3. 作为模块导入
    from scripts.clip_rag_classifier import CLIPSignClassifier
    classifier = CLIPSignClassifier()
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
import numpy as np
from PIL import Image

# ============================================================================
# 配置
# ============================================================================

# 标志图片目录
SIGNS_DIR = Path("raw_data/signs")

# Chroma 数据库目录
CHROMA_DB_DIR = Path("raw_data/chroma_signs_db")

# CLIP 模型配置（使用最先进的 ViT-L/14）
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "openai"  # 或 "laion2b_s32b_b82k" 更大规模预训练

# 69 个摩托车安全相关标志（与 sign_classifier_v2 保持一致）
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
# CLIP 模型封装
# ============================================================================

class CLIPEncoder:
    """CLIP 图像编码器"""
    
    def __init__(self, model_name: str = CLIP_MODEL_NAME, pretrained: str = CLIP_PRETRAINED):
        """
        初始化 CLIP 模型
        
        Args:
            model_name: 模型名称，推荐 ViT-L-14（更准确）或 ViT-B-32（更快）
            pretrained: 预训练权重来源
        """
        try:
            import open_clip
        except ImportError:
            raise ImportError("请安装 open_clip: pip install open_clip_torch")
        
        print(f"🔧 加载 CLIP 模型: {model_name} ({pretrained})")
        
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"   设备: {self.device}")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"   ✅ CLIP 模型加载完成")
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        编码单张图片
        
        Args:
            image: PIL Image 对象
        
        Returns:
            归一化后的特征向量 (numpy array)
        """
        with torch.no_grad():
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            features = self.model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
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
    
    def __init__(self, db_dir: Path = CHROMA_DB_DIR, collection_name: str = "traffic_signs"):
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
            documents=labels  # 用于文本搜索备用
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
        # 删除并重建集合
        self.client.delete_collection("traffic_signs")
        self.collection = self.client.create_collection(
            name="traffic_signs",
            metadata={"hnsw:space": "cosine"}
        )
        print("   🗑️ 已清空向量库")


# ============================================================================
# 主分类器
# ============================================================================

class CLIPSignClassifier:
    """
    CLIP + Chroma 交通标志分类器
    
    用法:
        classifier = CLIPSignClassifier()
        
        # 建库（首次运行）
        classifier.build_database()
        
        # 分类
        label, score = classifier.classify("image.jpg", [100, 200, 150, 250])
    """
    
    def __init__(
        self, 
        signs_dir: Path = SIGNS_DIR,
        db_dir: Path = CHROMA_DB_DIR,
        use_69_signs: bool = True,
        similarity_threshold: float = 0.5
    ):
        """
        初始化分类器
        
        Args:
            signs_dir: 标志图片目录
            db_dir: 向量数据库目录
            use_69_signs: 是否只使用 69 个摩托车安全相关标志
            similarity_threshold: 相似度阈值，低于此值返回 "other"
        """
        self.signs_dir = Path(signs_dir)
        self.db_dir = Path(db_dir)
        self.use_69_signs = use_69_signs
        self.similarity_threshold = similarity_threshold
        
        # 懒加载
        self._encoder = None
        self._db = None
    
    @property
    def encoder(self) -> CLIPEncoder:
        """懒加载 CLIP 编码器"""
        if self._encoder is None:
            self._encoder = CLIPEncoder()
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
        print("🏗️ 建立交通标志向量数据库")
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
            # 只使用 69 个核心标志
            sign_names = MOTORCYCLE_SAFETY_SIGNS
            print(f"   📋 使用 69 个摩托车安全相关标志")
        else:
            # 使用所有标志
            sign_names = [f.stem for f in sorted(self.signs_dir.glob("*.png"))]
            print(f"   📋 使用全部 {len(sign_names)} 个标志")
        
        # 编码并添加
        embeddings = []
        labels = []
        paths = []
        
        print(f"\n🔄 编码标志图片...")
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


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CLIP + Chroma 交通标志分类器")
    parser.add_argument("--build", action="store_true", help="建立向量数据库")
    parser.add_argument("--rebuild", action="store_true", help="重建向量数据库（清空现有数据）")
    parser.add_argument("--test", type=str, help="测试分类，指定图片路径")
    parser.add_argument("--bbox", type=str, help="边界框，格式: x1,y1,x2,y2")
    parser.add_argument("--all-signs", action="store_true", help="使用全部 188 个标志（默认只用 69 个）")
    parser.add_argument("--top-k", type=int, default=3, help="返回前 K 个结果（默认 3）")
    args = parser.parse_args()
    
    classifier = CLIPSignClassifier(use_69_signs=not args.all_signs)
    
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
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"   {emoji} {r['label']}")
            print(f"      相似度: {r['score']:.4f}")
        
        # 最终标签
        label, score = classifier.classify(args.test, bbox)
        print(f"\n🏷️ 最终标签: {label} (score: {score:.4f})")
    
    if not args.build and not args.rebuild and not args.test:
        parser.print_help()


if __name__ == "__main__":
    main()

