"""
常规检索模块的在线检索器
执行文本查询和相似度计算
"""

from typing import List, Dict, Any, Optional
from PIL import Image
import requests

from .text_encoder import TaiyiTextEncoder
from .image_encoder import CLIPImageEncoder
from .offline_indexer import OfflineIndexer


class Retriever:
    """在线检索器，执行文本到图像的检索"""
    
    def __init__(self, text_encoder: TaiyiTextEncoder, image_encoder: CLIPImageEncoder, 
                 offline_indexer: OfflineIndexer):
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.offline_indexer = offline_indexer
    
    def retrieve(self, query: str, method: str = "TopK", 
                 top_k: int = 5, threshold: float = 0.5,
                 image_urls: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        基于中文文本查询检索图像库。
        
        Args:
            query: 中文查询文本
            method: 检索方式，"TopK" 或 "卡阈值"
            top_k: TopK模式返回的结果数量
            threshold: 卡阈值模式的相似度阈值
            image_urls: 可选，临时指定待匹配的图像URL或路径列表
            
        Returns:
            排好序的匹配结果
        """
        if self.offline_indexer.is_empty() and not image_urls:
            return {"error": "特征库为空，请先运行离线索引(offline_indexing)。"}
        
        try:
            # 1. 编码查询文本
            text_features = self.text_encoder.encode(query)
            
            # 2. 计算相似度
            results = []
            logit_scale = self.image_encoder.get_logit_scale()
            
            # 如果提供了临时image_urls，则实时编码这些图像
            if image_urls:
                for url in image_urls:
                    try:
                        image_features = self.image_encoder.encode(url)
                        if image_features is not None:
                            similarity = (logit_scale * image_features @ text_features.t()).item()
                            results.append({"url": url, "score": similarity})
                        else:
                            results.append({"url": url, "score": 0.0, "error": "编码失败"})
                    except Exception as e:
                        results.append({"url": url, "score": 0.0, "error": str(e)})
            else:
                # 使用预建的离线向量库
                for url, data in self.offline_indexer.get_db().items():
                    image_features = data["image_feature"].to(self.text_encoder.device)
                    # 计算余弦相似度
                    similarity = (logit_scale * image_features @ text_features.t()).item()
                    results.append({"url": url, "score": similarity})
            
            # 3. 根据检索策略返回结果
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            
            if method == "TopK":
                final_results = results[:top_k]
            elif method == "卡阈值":
                final_results = [r for r in results if r["score"] >= threshold]
            else:
                final_results = results[:top_k]  # 默认TopK
            
            return {
                "query": query,
                "method": method,
                "results": final_results
            }
            
        except Exception as e:
            return {"error": f"常规检索出现异常: {str(e)}"}