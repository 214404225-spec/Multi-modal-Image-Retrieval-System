"""
细粒度检索模块主类
整合所有子组件，提供统一的细粒度两阶段检索接口
"""

from typing import List, Dict, Any, Optional

from shared.clip_encoder import CLIPEncoder
from .vl_models import VLModelManager, VLRefiner
from .online_retriever import OnlineRetriever
from .constants import CLIP_MODEL_PATH


class FineGrainedRetrievalModule:
    """
    细粒度两阶段检索模块（对应 v1.md 架构）：

    - 离线：共享常规检索模块的 CLIP 图像编码向量库
    - 在线粗排：查询 → CLIP文本编码器 → 相似度计算 → 检索策略
    - 在线精排：VL_Refine（VL 模型验证属性条件）→ 加权融合 → Top-K
    """
    def __init__(self, device: str = None, clip_encoder: CLIPEncoder = None,
                 offline_indexer=None):
        if clip_encoder is not None:
            self.clip_encoder = clip_encoder
        else:
            self.clip_encoder = CLIPEncoder(device, CLIP_MODEL_PATH)

        # VL 模型管理器（供 VL_Refine 精排阶段使用，__init__ 自动初始化 Ollama）
        self.vl_manager = VLModelManager()

        # VL 精排器（对应 v1.md 的 VL_Refine）
        self.vl_refiner = VLRefiner(self.vl_manager)

        # 在线粗排检索器（使用共享的离线向量库）
        self.online_retriever = OnlineRetriever(self.clip_encoder, offline_indexer)

    def online_retrieval(self, category: str, method: str = "TopK",
                         target_count: int = None, top_k: int = 5,
                         threshold: float = 0.8,
                         candidate_images: List[str] = None,
                         attributes: List[str] = None) -> Dict[str, Any]:
        """
        在线粗排阶段：CLIP文本编码查询 → 相似度计算 → 检索策略。
        精排由 refine_by_attributes（VL_Refine）完成。
        """
        return self.online_retriever.retrieve(
            category, method, target_count, top_k, threshold, candidate_images, attributes
        )

    def refine_by_attributes(self, results: List[Dict], category: str,
                              attributes: List[str], top_k: int = None,
                              alpha: float = 0.4, beta: float = 0.6,
                              min_vl_score: float = 0.2,
                              progress_callback=None) -> List[Dict]:
        """
        两阶段检索的精排阶段（VL_Refine）：
        使用 VL 模型验证属性条件，加权融合分数。
        vl_score 低于 min_vl_score 的候选将被硬过滤。
        """
        return self.vl_refiner.refine(results, category, attributes, top_k, alpha, beta, min_vl_score, progress_callback=progress_callback)
