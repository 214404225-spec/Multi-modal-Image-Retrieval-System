"""
常规检索模块主类
整合所有子组件，提供统一的检索接口
"""

from typing import List, Dict, Any, Optional

from .text_encoder import ChineseRobertaTextEncoder
from .image_encoder import ClipViTImageEncoder
from .offline_indexer import OfflineIndexer
from .attribute_refiner import AttributeRefiner
from .retriever import Retriever


class RegularRetrievalModule:
    """
    常规检索模块：基于中文RoBERTa（Taiyi-CLIP文本编码器）和clip_ViT（Taiyi-CLIP图像编码器），
    支持通用的文本到图像匹配。
    
    架构说明：
    - 中文RoBERTa文本编码器：基于RoBERTa架构，专为中文语义理解优化
    - clip_ViT图像编码器：基于CLIP ViT架构，用于图像特征提取
    - 离线向量库：图像库预先用clip_ViT图像编码器提取特征
    - 在线检索：中文RoBERTa文本编码器提取查询特征 → 相似度计算 → 检索策略
    """
    def __init__(self, device: str = None):
        # 初始化各子组件
        self.text_encoder = ChineseRobertaTextEncoder(device)
        self.image_encoder = ClipViTImageEncoder(device)
        self.offline_indexer = OfflineIndexer(self.image_encoder)
        self.attribute_refiner = AttributeRefiner(self.text_encoder, self.image_encoder)
        self.retriever = Retriever(self.text_encoder, self.image_encoder, self.offline_indexer)
    
    def offline_indexing(self, image_urls: List[str]) -> None:
        """
        离线阶段：预建图像特征向量库
        流程：图像库 → CLIP图像编码器 → 离线向量库1
        """
        self.offline_indexer.index(image_urls)
    
    def retrieve(self, query: str, method: str = "TopK", 
                 top_k: int = 5, threshold: float = 0.5,
                 image_urls: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        基于中文文本查询检索图像库。
        """
        return self.retriever.retrieve(query, method, top_k, threshold, image_urls)
    
    def refine_by_attributes(self, results: List[Dict], category: str, 
                              attributes: List[str], top_k: int = None,
                              alpha: float = 0.4, beta: float = 0.6) -> List[Dict]:
        """
        两阶段检索的第二阶段：根据属性条件重排序
        
        Args:
            results: 粗检索结果列表
            category: 类别名称
            attributes: 属性条件列表
            top_k: 返回的结果数量
            alpha: 粗检索分数权重（默认0.4）
            beta: 属性分数权重（默认0.6）
        """
        return self.attribute_refiner.refine(results, category, attributes, top_k, alpha, beta)
