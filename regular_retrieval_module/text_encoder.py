"""
常规检索模块的文本编码器
使用中文RoBERTa（Taiyi-CLIP文本编码器）进行文本特征提取
该编码器基于 RoBERTa 架构，专为中文语义理解优化
"""

import os
import torch
from transformers import AutoModel, AutoTokenizer

from .constants import CHINESE_ROBERTA_MODEL_ID, LOCAL_MODEL_CACHE


class ChineseRobertaTextEncoder:
    """
    中文RoBERTa文本编码器（Taiyi-CLIP文本编码器）
    用于将中文文本转换为特征向量，与clip_ViT图像编码器配合使用
    """
    
    def __init__(self, device: str | None = None, cache_dir: str | None = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 获取本地模型路径
        model_path = LOCAL_MODEL_CACHE.get('Chinese_RoBERTa')
        if cache_dir:
            model_path = os.path.join(cache_dir, 'Chinese_RoBERTa')
        
        use_local = model_path is not None and os.path.exists(model_path)
        model_source = model_path if use_local else CHINESE_ROBERTA_MODEL_ID
        
        # 类型守卫：确保 model_source 不为 None
        if model_source is None:
            raise ValueError("Chinese_RoBERTa模型路径未配置")
        
        if not use_local:
            print("[警告] Chinese_RoBERTa模型本地路径不存在，将尝试从网络下载")
            print(f"[提示] 建议运行: python scripts/download_models.py --model Chinese_RoBERTa")
        
        print(f"[Chinese_RoBERTa文本编码器] 加载模型: {'(本地)' if use_local else '(网络)'} {model_source}")
        
        # 加载模型（仅一次尝试，本地模式）
        try:
            # 加载中文RoBERTa文本模型和tokenizer
            self.model: AutoModel = AutoModel.from_pretrained(  # type: ignore[assignment]
                model_source,
                cache_dir=cache_dir,
                local_files_only=use_local,
            )
            self.model = self.model.to(self.device).eval()  # type: ignore[assignment]
            self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[assignment]
                model_source,
                cache_dir=cache_dir,
                local_files_only=use_local,
            )
            print(f"[Chinese_RoBERTa文本编码器] 加载成功 {'(本地)' if use_local else '(网络)'}")
        except Exception as e:
            print(f"[Chinese_RoBERTa文本编码器] 加载失败: {type(e).__name__}")
            print("=" * 60)
            print("解决方案：")
            print("1. 下载模型到本地：")
            print(f"   python scripts/download_models.py --model Chinese_RoBERTa")
            print("2. 确保 constants.py 中 LOCAL_MODEL_CACHE['Chinese_RoBERTa'] 路径正确")
            print("=" * 60)
            raise
    
    def encode(self, query: str) -> torch.Tensor:
        """
        使用中文RoBERTa文本编码器提取文本特征
        
        Args:
            query: 查询文本
            
        Returns:
            归一化的文本特征向量（与图像编码器维度一致）
        """
        text_inputs = self.tokenizer(
            query,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**text_inputs)  # type: ignore[operator]
            # 使用 [CLS] token 的输出作为句子表示
            text_features = outputs.last_hidden_state[:, 0, :]
            # 归一化文本特征
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features