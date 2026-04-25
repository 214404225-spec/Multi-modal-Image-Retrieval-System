"""
常规检索模块的图像编码器
使用clip_ViT（Taiyi-CLIP图像编码器）进行图像特征提取
该编码器基于 CLIP ViT 架构，用于将图像转换为特征向量
"""

import io
import os
import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from typing import cast, List

from .constants import CLIP_VIT_MODEL_ID, LOCAL_MODEL_CACHE


class ClipViTImageEncoder:
    """
    clip_ViT图像编码器（Taiyi-CLIP图像编码器）
    用于将图像转换为特征向量，与中文RoBERTa文本编码器配合使用
    """
    
    def __init__(self, device: str | None = None, cache_dir: str | None = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 打印设备信息
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[clip_ViT图像编码器] 检测到GPU: {gpu_name} ({gpu_memory:.1f}GB显存)")
            print(f"[clip_ViT图像编码器] 将使用CUDA加速")
        else:
            print(f"[clip_ViT图像编码器] 未检测到GPU，将使用CPU（速度较慢）")
        
        # 获取本地模型路径
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
        
        # 使用cast告诉类型检查器model_source是str类型
        model_source = cast(str, model_source)
        
        if not use_local:
            print("[警告] clip_ViT图像编码器本地路径不存在，将尝试从网络下载")
            print(f"[提示] 建议运行: python scripts/download_models.py --model clip_ViT")
        
        print(f"[clip_ViT图像编码器] 加载模型: {'(本地)' if use_local else '(网络)'} {model_source}")
        
        # 加载模型（仅一次尝试，本地模式）
        try:
            # 加载clip_ViT图像模型和处理器
            self.model: CLIPModel = CLIPModel.from_pretrained(  # type: ignore[assignment]
                model_source,
                cache_dir=cache_dir,
                local_files_only=use_local,
            )
            self.model = self.model.to(self.device).eval()  # type: ignore[assignment]
            self.processor = CLIPProcessor.from_pretrained(  # type: ignore[assignment]
                model_source,
                cache_dir=cache_dir,
                local_files_only=use_local,
            )
            print(f"[clip_ViT图像编码器] 加载成功 {'(本地)' if use_local else '(网络)'}")
        except Exception as e:
            print(f"[clip_ViT图像编码器] 加载失败: {type(e).__name__}")
            print("=" * 60)
            print("解决方案：")
            print("1. 下载模型到本地：")
            print(f"   python scripts/download_models.py --model clip_ViT")
            print("2. 确保 constants.py 中 LOCAL_MODEL_CACHE['clip_ViT'] 路径正确")
            print("=" * 60)
            raise
    
    def encode(self, image_path: str) -> torch.Tensor | None:
        """
        使用clip_ViT图像编码器提取图像特征
        
        Args:
            image_path: 图像路径或URL
            
        Returns:
            归一化的图像特征向量，失败时返回None
        """
        try:
            # 加载图像（支持本地文件和网络图片）
            if image_path.startswith("http"):
                response = requests.get(image_path, stream=True, timeout=10)
                image = Image.open(io.BytesIO(response.content))
            else:
                image = Image.open(image_path)
            
            return self.encode_image(image)
        except Exception as e:
            print(f"[图像编码] 编码失败 [{image_path}]: {str(e)}")
            return None
    
    def encode_image(self, image: Image.Image) -> torch.Tensor | None:
        """
        使用clip_ViT图像编码器提取已加载图像的特征
        
        Args:
            image: 已加载的PIL Image对象
            
        Returns:
            归一化的图像特征向量，失败时返回None
        """
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)  # type: ignore[call-arg]
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                # get_image_features 应该返回 torch.Tensor，但如果配置问题可能返回 BaseModelOutputWithPooling
                if hasattr(outputs, 'last_hidden_state'):
                    # 如果返回的是 BaseModelOutputWithPooling，提取 last_hidden_state
                    image_features = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] token
                elif hasattr(outputs, 'pooler_output'):
                    # 如果有 pooler_output，使用它
                    image_features = outputs.pooler_output
                elif isinstance(outputs, torch.Tensor):
                    image_features = outputs
                else:
                    raise ValueError(f"Unexpected output type: {type(outputs)}")
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features
        except Exception as e:
            print(f"[图像编码] 编码失败: {str(e)}")
            return None
    
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor | None:
        """
        批量编码多张图像（比逐张编码快3-5倍）
        
        Args:
            images: 已加载的PIL Image对象列表
            
        Returns:
            归一化的图像特征向量列表，形状为 (batch_size, feature_dim)
        """
        if not images:
            return None
        
        try:
            # 批量处理图像
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)  # type: ignore[call-arg]
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    image_features = outputs.last_hidden_state[:, 0, :]
                elif hasattr(outputs, 'pooler_output'):
                    image_features = outputs.pooler_output
                elif isinstance(outputs, torch.Tensor):
                    image_features = outputs
                else:
                    raise ValueError(f"Unexpected output type: {type(outputs)}")
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features
        except Exception as e:
            print(f"[批量图像编码] 编码失败: {str(e)}")
            return None
    
    def get_logit_scale(self) -> float:
        """获取温度缩放参数"""
        return self.model.logit_scale.exp().item()