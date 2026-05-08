"""
细粒度检索模块的在线检索器
执行相似度计算和检索策略（中文文本由 Taiyi-CLIP Chinese RoBERTa 直接编码）
"""

from typing import List, Dict, Any, Optional

from shared.clip_encoder import CLIPEncoder


class OnlineRetriever:
    """在线检索器，执行细粒度粗排检索（使用共享的离线向量库）"""

    def __init__(self, clip_encoder: CLIPEncoder, offline_indexer):
        self.clip_encoder = clip_encoder
        self.offline_indexer = offline_indexer

    def retrieve(self, category: str, top_k: int = 5,
                 candidate_images: Optional[List[str]] = None,
                 attributes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        粗排阶段：CLIP文本编码查询 → 与向量库中图像特征做相似度计算 → Top-K。
        精排阶段由 VL_Refine 完成。
        """
        try:
            # 粗排只用类别编码，属性条件留给 VL 精排阶段处理
            # CLIP 对属性理解粗糙，提前引入属性可能误杀 VL 能正确判断的候选
            search_query = category
            text_features = self.clip_encoder.encode_text(search_query)

            results = []
            logit_scale = self.clip_encoder.get_logit_scale()

            if not self.offline_indexer.is_empty():
                for url, data in self.offline_indexer.get_db().items():
                    image_features = data["image_feature"].to(self.clip_encoder.device)
                    similarity = (logit_scale * image_features @ text_features.t()).item()
                    results.append({"url": url, "score": similarity})

            if candidate_images:
                for img_path in candidate_images:
                    img_feat = self.clip_encoder.encode_image(img_path)
                    if img_feat is not None:
                        similarity = (logit_scale * img_feat @ text_features.t()).item()
                        results.append({"url": img_path, "score": similarity})

            if not results:
                return {
                    "search_query": search_query,
                    "method": "TopK",
                    "results": [],
                    "message": "未找到匹配结果"
                }

            results = sorted(results, key=lambda x: x["score"], reverse=True)
            results = results[:top_k]

            return {
                "search_query": search_query,
                "method": "TopK",
                "results": results,
                "has_attributes": bool(attributes)
            }
        except Exception as e:
            return {"error": f"细粒度检索失效: {str(e)}"}
