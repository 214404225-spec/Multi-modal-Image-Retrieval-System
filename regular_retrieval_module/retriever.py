"""
常规检索模块的在线检索器
执行文本查询和相似度计算（中文文本由 Taiyi-CLIP Chinese RoBERTa 直接编码）
"""

from typing import List, Dict, Any

from shared.clip_encoder import CLIPEncoder
from .offline_indexer import OfflineIndexer


class Retriever:
    """在线检索器，执行文本到图像的检索"""

    def __init__(self, clip_encoder: CLIPEncoder, offline_indexer: OfflineIndexer):
        self.clip_encoder = clip_encoder
        self.offline_indexer = offline_indexer

    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        基于中文文本查询检索图像库。

        Args:
            query: 中文查询文本
            top_k: 返回的结果数量

        Returns:
            排好序的匹配结果
        """
        if self.offline_indexer.is_empty():
            return {"error": "特征库为空，请先运行离线索引(offline_indexing)。"}

        try:
            # 1. 编码查询文本
            text_features = self.clip_encoder.encode_text(query)

            # 2. 计算相似度
            results = []
            logit_scale = self.clip_encoder.get_logit_scale()

            for url, data in self.offline_indexer.get_db().items():
                image_features = data["image_feature"].to(self.clip_encoder.device)
                similarity = (logit_scale * image_features @ text_features.t()).item()
                results.append({"url": url, "score": similarity})

            # 3. Top-K 返回
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            final_results = results[:top_k]

            return {
                "query": query,
                "method": "TopK",
                "results": final_results
            }

        except Exception as e:
            return {"error": f"常规检索出现异常: {str(e)}"}