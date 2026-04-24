"""
常规检索模块的图像编码器
使用CLIP-ViT进行图像特征提取
"""

import os
import time
import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

from .constants import CLIP_MODEL_NAME, LOCAL_MODEL_CACHE


class CLIPImageEncoder:
    """CLIP图像编码器，用于图像特征提取"""
    
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
                # 加载原版 CLIP 图像模型和处理器
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
                print(f"[图像编码器] CLIP模型加载成功 {'(本地)' if use_local else '(网络)'}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[图像编码器] 加载失败 (尝试 {attempt + 1}/{max_retries}): {type(e).__name__}")
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"[图像编码器] 加载失败，已达到最大重试次数")
                    print("=" * 60)
                    print("解决方案：")
                    print("1. 手动下载模型到本地：")
                    print(f"   huggingface-cli download {CLIP_MODEL_NAME} --local-dir ./models/clip_vit")
                    print("2. 在 constants.py 中配置本地路径：")
                    print(f"   LOCAL_MODEL_CACHE['clip_vit'] = './models/clip_vit'")
                    print("=" * 60)
                    raise
    
    def encode(self, image_path: str) -> torch.Tensor:
        """
        使用CLIP图像编码器提取图像特征
        
        Args:
            image_path: 图像路径或URL
            
        Returns:
            归一化的图像特征向量
        """
        try:
            # 加载图像（支持本地文件和网络图片）
            if image_path.startswith("http"):
                image = Image.open(requests.get(image_path, stream=True, timeout=10).raw)
            else:
                image = Image.open(image_path)
            
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
            return image_features
        except Exception as e:
            print(f"[图像编码] 编码失败 [{image_path}]: {str(e)}")
            return None
    
    def get_logit_scale(self) -> float:
        """获取温度缩放参数"""
        return self.model.logit_scale.exp().item()