"""
常规检索模块的常量定义
"""

import os

# ============================================================
# 模型配置说明
# ============================================================
# Chinese_roberta: Taiyi-CLIP 的文本编码器，基于 RoBERTa 架构，专为中文优化
# clip_ViT:    Taiyi-CLIP 的图像编码器，基于 CLIP ViT 架构，用于图像特征提取
# ============================================================

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模型标识符
# Chinese_roberta文本编码器模型标识
CHINESE_ROBERTA_MODEL_ID = "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese"

# clip_ViT图像编码器模型标识
CLIP_VIT_MODEL_ID = "openai/clip-vit-large-patch14"

# HuggingFace国内镜像源配置（推荐按顺序尝试）
HF_MIRROR_URLS = [
    "https://huggingface.co",          # 官方源
    "https://hf-mirror.com",           # 镜像
]

# ModelScope镜像源配置（备用方案，国内更稳定）
MODELSCOPE_ENDPOINT = "https://api-inference.modelscope.cn"

# 下载超时设置（秒）
HF_DOWNLOAD_TIMEOUT = 120

# ============================================================
# 本地模型缓存路径配置
# ============================================================
# 说明：系统优先从以下路径加载本地模型，无需联网
# 如果路径不存在，请先运行以下命令下载模型：
#   python scripts/download_models.py --output-dir ./models
# ============================================================
LOCAL_MODEL_CACHE = {
    'Chinese_RoBERTa': os.path.join(PROJECT_ROOT, 'models', 'Chinese_RoBERTa'),
    'clip_ViT': os.path.join(PROJECT_ROOT, 'models', 'clip_ViT'),
}

# 默认检索参数
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.5