"""
细粒度检索模块的在线检索器
执行相似度计算和检索策略
"""

from typing import List, Dict, Any, Optional

from .clip_encoder import CLIPEncoder
from .offline_indexer import OfflineIndexer


class OnlineRetriever:
    """在线检索器，执行细粒度检索"""
    
    def __init__(self, clip_encoder: CLIPEncoder, offline_indexer: OfflineIndexer):
        self.clip_encoder = clip_encoder
        self.offline_indexer = offline_indexer
    
    def retrieve(self, category: str, method: str = "TopK", 
                 target_count: int = None, top_k: int = 5, 
                 threshold: float = 0.8,
                 candidate_images: List[str] = None,
                 attributes: List[str] = None) -> Dict[str, Any]:
        """
        在线计算阶段：结合意图模块提取的类别与所需数量，计算相似度。
        支持零样本泛化检索（未见类别）和属性条件精排序。
        """
        try:
            # 构建搜索查询文本
            search_query = f"{target_count}个{category}" if target_count else f"包含{category}的图片"
            query_feature = self.clip_encoder.encode_text(search_query)
            
            results = []
            
            # 第一阶段：粗检索
            # 如果有离线向量库，先进行离线检索
            if not self.offline_indexer.is_empty():
                for path, data in self.offline_indexer.get_db().items():
                    db_vector = data["vector"].to(self.clip_encoder.device)
                    # 计算余弦相似度
                    similarity = (query_feature @ db_vector.t()).item()
                    results.append({
                        "url": path, 
                        "score": similarity, 
                        "desc": data["description"],
                        "labels": data["labels_counts"]
                    })
            
            # 如果提供了新图像或离线库无结果，使用CLIP零样本能力
            if candidate_images:
                zero_shot_results = []
                for img_path in candidate_images:
                    img_feat = self.clip_encoder.encode_image(img_path)
                    if img_feat is not None:
                        similarity = (query_feature @ img_feat.t()).item()
                        zero_shot_results.append({
                            "url": img_path,
                            "score": similarity,
                            "desc": "零样本检索",
                            "labels": {}
                        })
                
                # 合并结果（零样本结果追加）
                if results:
                    results.extend(zero_shot_results)
                    results = sorted(results, key=lambda x: x["score"], reverse=True)
                else:
                    results = zero_shot_results
            
            # 检查是否有结果
            if not results:
                return {
                    "search_query": search_query,
                    "method": method,
                    "results": [],
                    "message": "未找到匹配结果，请检查特征库或提供候选图像"
                }
            
            # 按分数排序
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            
            # 根据检索方式筛选
            if method == "TopK":
                results = results[:top_k]
            elif method == "卡阈值":
                results = [r for r in results if r["score"] >= threshold]
            else:
                results = results[:top_k]  # 默认TopK
            
            return {
                "search_query": search_query,
                "method": method,
                "results": results,
                "has_attributes": bool(attributes)
            }
        except Exception as e:
            return {"error": f"细粒度检索失效: {str(e)}"}