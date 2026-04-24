"""
常规检索模块的文本编码器
使用CLIP-ViT的文本编码器进行文本特征提取，确保与图像编码器维度一致
"""

import os
import torch
from transformers import CLIPProcessor, CLIPModel

from .constants import CLIP_MODEL_NAME, LOCAL_MODEL_CACHE


class TaiyiTextEncoder:
    """CLIP文本编码器，用于文本特征提取（与图像编码器使用相同模型确保维度一致）"""
    
    def __init__(self, device: str = None, cache_dir: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 优先使用本地缓存路径
        model_path = LOCAL_MODEL_CACHE.get('clip_vit')
        if cache_dir:
            model_path = os.path.join(cache_dir, 'clip_vit')
        
        use_local = model_path is not None and os.path.exists(model_path)
        model_source = model_path if use_local else CLIP_MODEL_NAME
        
        # 加载模型（带重试机制）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 加载CLIP模型（包含文本和图像编码器）
                self.model = CLIPModel.from_pretrained(
                    model_source,
                    cache_dir=cache_dir,
                    local_files_only=use_local,
                    resume_download=True,
                    force_download=False
                ).to(self.device).eval()
                self.processor = CLIPProcessor.from_pretrained(
                    model_source,
                    cache_dir=cache_dir,
                    local_files_only=use_local,
                    resume_download=True,
                    force_download=False
                )
                print(f"[文本编码器] CLIP文本模型加载成功 {'(本地)' if use_local else '(网络)'}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[文本编码器] 加载失败 (尝试 {attempt + 1}/{max_retries}): {type(e).__name__}")
                    print(f"等待 {wait_time} 秒后重试...")
                    import time
                    time.sleep(wait_time)
                else:
                    print(f"[文本编码器] 加载失败，已达到最大重试次数")
                    raise
    
    def encode(self, query: str) -> torch.Tensor:
        """
        使用CLIP文本编码器提取文本特征
        
        Args:
            query: 查询文本
            
        Returns:
            归一化的文本特征向量（1024维，与图像编码器一致）
        """
        text_inputs = self.processor(text=[query], return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            # 归一化文本特征
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features