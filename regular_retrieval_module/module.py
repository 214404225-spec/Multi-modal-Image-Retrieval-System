"""
常规检索模块主类
整合所有子组件，提供统一的检索接口
"""

import os
from typing import List, Dict, Any, Optional

from shared.clip_encoder import CLIPEncoder
from .offline_indexer import OfflineIndexer
from .retriever import Retriever
from .constants import CLIP_MODEL_PATH


class RegularRetrievalModule:
    """
    常规检索模块：基于CLIP文本/图像编码器，支持通用的文本到图像匹配。

    架构（对应 v1.md）：
    - 离线阶段：图像库 → CLIP图像编码器 → 离线向量库
    - 在线阶段：查询 → CLIP文本编码器 → 相似度计算 → 按数量Top-K
    """
    def __init__(self, device: str = None, model_path: str = None, clip_encoder: CLIPEncoder = None):
        if clip_encoder is not None:
            self.clip_encoder = clip_encoder
        else:
            clip_path = model_path or CLIP_MODEL_PATH
            self.clip_encoder = CLIPEncoder(device, clip_path)
        self.offline_indexer = OfflineIndexer(self.clip_encoder)
        self.retriever = Retriever(self.clip_encoder, self.offline_indexer)

    def offline_indexing(self, image_urls: List[str]) -> float:
        """离线阶段：预建图像特征向量库。返回总用时（秒）。"""
        return self.offline_indexer.index(image_urls)

    def retrieve(self, query: str, method: str = "TopK",
                 top_k: int = 5, threshold: float = 0.5) -> Dict[str, Any]:
        """基于文本查询检索图像库，直接返回 Top-K 结果"""
        return self.retriever.retrieve(query, method, top_k, threshold)
