"""
共享编码器模块
- 文本编码：Taiyi-CLIP 的 Chinese RoBERTa（中文原生支持）
- 图像编码：CLIP ViT-L/14（与 RoBERTa 共享 embedding 空间）
"""

import io
import os
import torch
from PIL import Image
import requests
from transformers import (
    CLIPProcessor, CLIPModel,
    BertForSequenceClassification, BertTokenizer,
)
from typing import cast, Optional, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模型路径
CLIP_VISION_MODEL_ID = "openai/clip-vit-large-patch14"
DEFAULT_CLIP_VISION_PATH = os.path.join(PROJECT_ROOT, "models", "clip_ViT")

TAIYI_TEXT_MODEL_ID = "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese"
DEFAULT_TAIYI_TEXT_PATH = os.path.join(PROJECT_ROOT, "models", "Chinese_RoBERTa")


class CLIPEncoder:
    """
    双模型编码器（Taiyi-CLIP 架构）：

    - 文本编码：Chinese RoBERTa + 投影头 → CLIP embedding 空间
    - 图像编码：CLIP ViT-L/14 → 同一 CLIP embedding 空间
    - 中文文本直接编码，无需翻译
    """

    def __init__(self, device: Optional[str] = None,
                 clip_vision_path: Optional[str] = None,
                 taiyi_text_path: Optional[str] = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # --- 文本编码器：Chinese RoBERTa ---
        text_path = taiyi_text_path or DEFAULT_TAIYI_TEXT_PATH
        use_local_text = os.path.exists(text_path)
        text_source = text_path if use_local_text else TAIYI_TEXT_MODEL_ID
        print(f"[编码器] 加载文本模型: {'(本地)' if use_local_text else '(网络)'} {text_source}")

        self.text_tokenizer = BertTokenizer.from_pretrained(
            text_source, local_files_only=use_local_text
        )
        self.text_encoder = BertForSequenceClassification.from_pretrained(
            text_source, local_files_only=use_local_text
        ).to(self.device).eval()

        # --- 图像编码器：CLIP ViT ---
        vision_path = clip_vision_path or DEFAULT_CLIP_VISION_PATH
        use_local_vis = os.path.exists(vision_path)
        vis_source = vision_path if use_local_vis else CLIP_VISION_MODEL_ID
        print(f"[编码器] 加载图像模型: {'(本地)' if use_local_vis else '(网络)'} {vis_source}")

        self.clip_model = CLIPModel.from_pretrained(
            vis_source, local_files_only=use_local_vis
        ).to(self.device).eval()

        self.clip_processor = CLIPProcessor.from_pretrained(
            vis_source, local_files_only=use_local_vis
        )
        self.vision_config = self.clip_model.config.vision_config

        print(f"[编码器] 加载完成 "
              f"(文本: {self.text_encoder.config.hidden_size}→{self.text_encoder.classifier.out_features} 维, "
              f"图像: {self.vision_config.hidden_size} 维)")

    # ============================================================
    # 文本编码
    # ============================================================

    def encode_text(self, text: str) -> torch.Tensor:
        """
        用 Chinese RoBERTa 编码中文文本。
        输入：原始中文文本（无需翻译）
        输出：L2 归一化后的文本特征向量
        """
        inputs = self.text_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            logits = self.text_encoder(**inputs).logits
        return logits / logits.norm(dim=-1, keepdim=True)

    # ============================================================
    # 图像编码
    # ============================================================

    @staticmethod
    def _normalize(features) -> torch.Tensor:
        """提取特征张量并 L2 归一化（兼容 BaseModelOutputWithPooling）"""
        if hasattr(features, 'pooler_output'):
            features = features.pooler_output
        return features / features.norm(dim=-1, keepdim=True)

    def encode_image(self, image_path: str) -> Optional[torch.Tensor]:
        """编码单张图像"""
        try:
            if image_path.startswith("http"):
                response = requests.get(image_path, stream=True, timeout=10)
                image = Image.open(io.BytesIO(response.content))
            else:
                image = Image.open(image_path)

            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
            return self._normalize(features)
        except Exception as e:
            print(f"[编码器] 图像编码失败 [{image_path}]: {str(e)}")
            return None

    def encode_images(self, images: List[Image.Image],
                      batch_size: int = 128) -> Optional[torch.Tensor]:
        """批量编码多张图像"""
        if not images:
            return None

        try:
            all_features = []
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                inputs = self.clip_processor(images=batch, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    features = self.clip_model.get_image_features(**inputs)
                all_features.append(self._normalize(features))
            return torch.cat(all_features, dim=0)
        except Exception as e:
            print(f"[编码器] 批量编码失败: {str(e)}")
            return None

    # ============================================================
    # 工具方法
    # ============================================================

    def get_logit_scale(self) -> float:
        """获取温度缩放参数（来自 CLIP 模型）"""
        return self.clip_model.logit_scale.exp().item()
