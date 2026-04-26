"""
细粒度检索模块
基于VL模型的离线标注和CLIP编码的两阶段检索
"""

from .module import FineGrainedRetrievalModule

__all__ = ["FineGrainedRetrievalModule"]