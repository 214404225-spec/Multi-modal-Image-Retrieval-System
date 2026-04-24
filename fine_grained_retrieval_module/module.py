"""
细粒度检索模块主类
整合所有子组件，提供统一的细粒度检索接口
"""

from typing import List, Dict, Any, Optional

from .vl_models import VLModelManager
from .clip_encoder import CLIPEncoder
from .offline_indexer import OfflineIndexer
from .attribute_refiner import AttributeRefiner
from .online_retriever import OnlineRetriever


class FineGrainedRetrievalModule:
    """
    细粒度检索模块：
    此模块拆分为离线阶段和在线阶段。
    主要用于精确的数量匹配或组合特征查询。
    
    架构对应v1（v2版本）：
    - 离线：图像库 → Qwen2.5-VL → 核心类别标签 → CLIP文本编码器 → 离线向量库2
    - 在线：查询 → CLIP文本编码器 → 相似度计算 → 检索策略
    - 新增：支持属性条件两阶段检索（粗检索+精排序）
    - 新增：支持CLIP零样本泛化检索（未见类别）
    """
    def __init__(self, device: str = None):
        # 初始化各子组件
        self.vl_manager = VLModelManager(device)
        # 初始化Ollama VL模型（推荐方式）
        self.vl_manager.init_ollama_model()
        self.clip_encoder = CLIPEncoder(device)
        self.offline_indexer = OfflineIndexer(self.vl_manager, self.clip_encoder)
        self.attribute_refiner = AttributeRefiner(self.clip_encoder)
        self.online_retriever = OnlineRetriever(self.clip_encoder, self.offline_indexer)
    
    def offline_indexing(self, image_paths: List[str]) -> None:
        """
        离线阶段：处理离线图片库。
        流程：图片库 → Qwen2.5-VL (生成标签和数量) → CLIP文本编码器 → 离线向量库2
        """
        self.offline_indexer.index(image_paths)
    
    def online_retrieval(self, category: str, method: str = "TopK", 
                         target_count: int = None, top_k: int = 5, 
                         threshold: float = 0.8,
                         candidate_images: List[str] = None,
                         attributes: List[str] = None) -> Dict[str, Any]:
        """
        在线计算阶段：结合意图模块提取的类别与所需数量，计算相似度。
        """
        return self.online_retriever.retrieve(
            category, method, target_count, top_k, threshold, candidate_images, attributes
        )
    
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
