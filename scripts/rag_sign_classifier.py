#!/usr/bin/env python3
"""
交通标志 RAG 分类器

使用向量检索 + GLM-4.6V 精排实现细粒度交通标志识别

用法:
    # 构建向量库
    python3 rag_sign_classifier.py --build
    
    # 分类测试
    python3 rag_sign_classifier.py --test path/to/sign_image.jpg
"""

import os
import base64
from pathlib import Path
from PIL import Image
import chromadb
from chromadb.utils import embedding_functions


class TrafficSignRAG:
    """交通标志 RAG 检索与分类器"""
    
    def __init__(self, 
                 signs_dir: str = "raw_data/images/signs/highres/png2560px",
                 db_path: str = ".chromadb"):
        """
        初始化 RAG 分类器
        
        Args:
            signs_dir: 标准交通标志图片目录
            db_path: 向量数据库存储路径
        """
        self.signs_dir = Path(signs_dir)
        self.db_path = db_path
        
        # 使用 CLIP 作为 embedding 模型
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="clip-ViT-B-32"
        )
        
        # 初始化 ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # 获取或创建 collection
        self.collection = self.client.get_or_create_collection(
            name="traffic_signs",
            embedding_function=self.embedding_fn,
            metadata={"description": "Hong Kong Traffic Signs Database"}
        )
    
    def build_database(self):
        """构建向量数据库"""
        print("=" * 60)
        print("🔨 构建交通标志向量数据库")
        print("=" * 60)
        
        if not self.signs_dir.exists():
            print(f"❌ 找不到标志目录: {self.signs_dir}")
            return
        
        # 获取所有标志图片
        sign_files = list(self.signs_dir.glob("*.png")) + list(self.signs_dir.glob("*.jpg"))
        
        if not sign_files:
            print(f"❌ 目录中没有图片")
            return
        
        print(f"📁 找到 {len(sign_files)} 个标志图片")
        
        # 清空已有数据
        existing = self.collection.count()
        if existing > 0:
            print(f"⚠️  清空已有 {existing} 条记录...")
            # ChromaDB 需要通过 ID 删除
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)
        
        # 批量添加
        ids = []
        documents = []
        images_data = []
        metadatas = []
        
        for i, sign_path in enumerate(sign_files):
            # 标签：从文件名提取
            label = sign_path.stem.replace("_", " ").replace("-", " ")
            normalized_label = sign_path.stem.lower().replace(" ", "_").replace("-", "_")
            
            # 读取图片并编码为 base64（用于存储）
            with open(sign_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            
            # 为 CLIP 准备图片描述（用于 embedding）
            # CLIP 可以编码图片或文本，这里用文本描述
            description = f"traffic sign: {label}"
            
            ids.append(normalized_label)
            documents.append(description)
            metadatas.append({
                "label": label,
                "normalized_label": normalized_label,
                "file_path": str(sign_path),
                "category": self._get_sign_category(label)
            })
            
            if (i + 1) % 50 == 0:
                print(f"  处理中: {i + 1}/{len(sign_files)}")
        
        # 添加到数据库
        print(f"\n📤 添加到向量数据库...")
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"✅ 完成! 共添加 {self.collection.count()} 个标志")
        print("=" * 60)
    
    def _get_sign_category(self, label: str) -> str:
        """根据标签判断标志类别"""
        label_lower = label.lower()
        
        if any(k in label_lower for k in ["speed", "limit", "km"]):
            return "speed_limit"
        elif any(k in label_lower for k in ["no ", "prohibit", "forbidden"]):
            return "prohibition"
        elif any(k in label_lower for k in ["warn", "ahead", "danger"]):
            return "warning"
        elif any(k in label_lower for k in ["direct", "arrow", "way"]):
            return "direction"
        else:
            return "other"
    
    def search(self, query: str, n_results: int = 5) -> list:
        """
        根据描述搜索相似标志
        
        Args:
            query: 查询描述（如 "speed limit 70"）
            n_results: 返回结果数量
            
        Returns:
            相似标志列表
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        candidates = []
        for i, id_ in enumerate(results["ids"][0]):
            candidates.append({
                "id": id_,
                "label": results["metadatas"][0][i]["label"],
                "category": results["metadatas"][0][i]["category"],
                "distance": results["distances"][0][i] if results["distances"] else None
            })
        
        return candidates
    
    def search_by_image(self, image_path: str, n_results: int = 5) -> list:
        """
        根据图片搜索相似标志（需要先将图片转为描述）
        
        注意：这是一个简化版本，实际应该用 CLIP 直接编码图片
        这里我们用 GLM-4.6V 先描述图片，再用描述检索
        
        Args:
            image_path: 待识别的标志图片路径
            n_results: 返回结果数量
        """
        # 简化版：直接用 "traffic sign" 作为查询
        # 完整版应该用 CLIP 编码图片
        return self.search("traffic sign", n_results)
    
    def classify_with_glm(self, image_path: str, candidates: list, client) -> dict:
        """
        使用 GLM-4.6V 进行精细分类
        
        Args:
            image_path: 待分类图片路径
            candidates: RAG 检索的候选标志列表
            client: ZaiClient 实例
            
        Returns:
            分类结果
        """
        # 读取图片
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
        
        # 构建候选列表文本
        candidate_list = "\n".join([
            f"  {i+1}. {c['label']} (类别: {c['category']})"
            for i, c in enumerate(candidates)
        ])
        
        prompt = f"""请仔细观察这个交通标志图片，从以下候选项中选择最匹配的：

{candidate_list}

规则：
1. 如果能确定是哪个，请返回对应的标签
2. 如果都不匹配，返回 "unknown"
3. 注意观察：颜色、形状、文字、数字、符号

请只返回标签名称，不要其他解释。例如：speed limit 70"""

        response = client.chat.completions.create(
            model="glm-4.6v",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        
        result = response.choices[0].message.content.strip()
        
        # 标准化结果
        result_normalized = result.lower().replace(" ", "_").replace("-", "_")
        
        # 检查是否在候选列表中
        matched = None
        for c in candidates:
            if c["id"] == result_normalized or c["label"].lower() == result.lower():
                matched = c
                break
        
        return {
            "predicted_label": result,
            "normalized_label": result_normalized,
            "matched_candidate": matched,
            "candidates": candidates
        }
    
    def get_stats(self) -> dict:
        """获取数据库统计信息"""
        count = self.collection.count()
        
        # 获取所有记录
        all_data = self.collection.get()
        
        # 统计类别
        categories = {}
        for meta in all_data["metadatas"]:
            cat = meta.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_signs": count,
            "categories": categories
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="交通标志 RAG 分类器")
    parser.add_argument("--build", action="store_true", help="构建向量数据库")
    parser.add_argument("--stats", action="store_true", help="查看数据库统计")
    parser.add_argument("--search", type=str, help="搜索标志（输入描述）")
    parser.add_argument("--test", type=str, help="测试分类（输入图片路径）")
    args = parser.parse_args()
    
    rag = TrafficSignRAG()
    
    if args.build:
        rag.build_database()
    
    elif args.stats:
        stats = rag.get_stats()
        print("=" * 60)
        print("📊 数据库统计")
        print("=" * 60)
        print(f"总标志数: {stats['total_signs']}")
        print(f"\n类别分布:")
        for cat, count in stats["categories"].items():
            print(f"  {cat}: {count}")
    
    elif args.search:
        print(f"🔍 搜索: {args.search}")
        results = rag.search(args.search)
        print(f"\n找到 {len(results)} 个候选:")
        for r in results:
            print(f"  - {r['label']} ({r['category']}) [距离: {r['distance']:.4f}]")
    
    elif args.test:
        print(f"🏷️ 测试分类: {args.test}")
        
        # 需要 API Key
        api_key = os.getenv("ZAI_API_KEY")
        if not api_key:
            print("❌ 请设置 ZAI_API_KEY 环境变量")
            return
        
        from zai import ZaiClient
        client = ZaiClient(api_key=api_key)
        
        # 先搜索候选
        candidates = rag.search("traffic sign", n_results=10)
        print(f"\n📚 RAG 候选 ({len(candidates)} 个):")
        for c in candidates[:5]:
            print(f"  - {c['label']}")
        
        # GLM-4.6V 精排
        print(f"\n🤖 GLM-4.6V 精排...")
        result = rag.classify_with_glm(args.test, candidates, client)
        
        print(f"\n✅ 分类结果: {result['predicted_label']}")
        if result['matched_candidate']:
            print(f"   类别: {result['matched_candidate']['category']}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
