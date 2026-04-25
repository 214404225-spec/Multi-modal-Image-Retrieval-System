"""
细粒度检索模块的CLIP编码器
使用clip_ViT提供文本和图像特征提取
"""

import io
import os
import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from typing import cast, Optional

from .constants import CLIP_VIT_MODEL_ID, LOCAL_MODEL_CACHE


class CLIPEncoder:
    """CLIP编码器，提供文本和图像特征提取"""
    
    def __init__(self, device: str | None = None, cache_dir: str | None = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor: Optional[CLIPProcessor] = None
        self.model: Optional[CLIPModel] = None
        
        # 优先使用本地缓存路径
        model_path: str | None = LOCAL_MODEL_CACHE.get('clip_ViT')
        if cache_dir:
            model_path = os.path.join(cache_dir, 'clip_ViT')
        
        use_local = model_path is not None and os.path.exists(model_path)
        
        # 确定模型来源
        if use_local:
            assert model_path is not None
            model_source = model_path
        else:
            assert CLIP_VIT_MODEL_ID is not None, "clip_ViT模型路径未配置"
            model_source = CLIP_VIT_MODEL_ID
        
        model_source = cast(str, model_source)
        
        if not use_local:
            print("[警告] clip_ViT模型本地路径不存在，将尝试从网络下载")
            print("[提示] 建议运行: python scripts/download_models.py --model clip_ViT")
        
        print(f"[细粒度检索] 加载clip_ViT模型: {'(本地)' if use_local else '(网络)'} {model_source}")
        
        # 初始化 CLIP 模型（文本编码器 + 图像编码器）
        try:
            self.processor = CLIPProcessor.from_pretrained(  # type: ignore[assignment]
                model_source,
                cache_dir=cache_dir,
                local_files_only=use_local,
            )
            self.model = CLIPModel.from_pretrained(  # type: ignore[assignment]
                model_source,
                cache_dir=cache_dir,
                local_files_only=use_local,
            )
            self.model = self.model.to(self.device).eval()  # type: ignore[assignment]
            print(f"[细粒度检索] clip_ViT模型加载成功 {'(本地)' if use_local else '(网络)'}")
        except Exception as e:
            print(f"[细粒度检索] clip_ViT模型加载失败: {type(e).__name__}")
            print("=" * 60)
            print("解决方案：")
            print("1. 下载模型到本地：")
            print(f"   python scripts/download_models.py --model clip_ViT")
            print("2. 在 constants.py 中配置本地路径：")
            print(f"   LOCAL_MODEL_CACHE['clip_ViT'] = './models/clip_ViT'")
            print("=" * 60)
            self.processor = None
            self.model = None
    
    def encode_text(self, text: str) -> torch.Tensor:
        """使用clip_ViT文本编码器提取文本特征"""
        if self.processor is None or self.model is None:
            # Mock模式：返回随机向量
            return torch.randn(1, 768)
        
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)  # type: ignore[call-arg]
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            # 处理可能的不同返回类型
            if hasattr(outputs, 'last_hidden_state'):
                text_features = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] token
            elif hasattr(outputs, 'pooler_output'):
                text_features = outputs.pooler_output
            elif isinstance(outputs, torch.Tensor):
                text_features = outputs
            else:
                raise ValueError(f"Unexpected output type: {type(outputs)}")
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def encode_image(self, image_path: str) -> torch.Tensor | None:
        """
        使用clip_ViT图像编码器实时编码图像（用于未见类别的零样本检索）
        
        Args:
            image_path: 图像路径或URL
            
        Returns:
            归一化的图像特征向量，失败返回None
        """
        if self.processor is None or self.model is None:
            return None
        try:
            # 加载图像（支持本地文件和网络图片）
            if image_path.startswith("http"):
                response = requests.get(image_path, stream=True, timeout=10)
                image = Image.open(io.BytesIO(response.content))
            else:
                image = Image.open(image_path)
            
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)  # type: ignore[call-arg]
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                # 处理可能的不同返回类型
                if hasattr(outputs, 'last_hidden_state'):
                    image_features = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] token
                elif hasattr(outputs, 'pooler_output'):
                    image_features = outputs.pooler_output
                elif isinstance(outputs, torch.Tensor):
                    image_features = outputs
                else:
                    raise ValueError(f"Unexpected output type: {type(outputs)}")
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features
        except Exception as e:
            print(f"[细粒度检索] 图像编码失败 [{image_path}]: {str(e)}")
            return None
