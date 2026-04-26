"""
共享组件包
- CLIPEncoder: Taiyi-CLIP 双模型编码器（Chinese RoBERTa 文本 + CLIP ViT 图像）
"""

from .clip_encoder import CLIPEncoder

__all__ = ["CLIPEncoder"]