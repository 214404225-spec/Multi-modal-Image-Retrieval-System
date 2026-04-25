"""
细粒度检索模块的常量定义
"""

import os

# ============================================================
# 模型配置说明
# ============================================================
# clip_ViT:    Taiyi-CLIP 的图像编码器，基于 CLIP ViT 架构，用于图像特征提取
# qwen_vl:     Qwen2.5-VL 视觉语言模型，用于图像理解和属性提取
# ============================================================

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# clip_ViT图像编码器模型标识
CLIP_VIT_MODEL_ID = "openai/clip-vit-large-patch14"

# Qwen2.5-VL模型配置
QWEN_VL_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_VL_OLLAMA_MODEL = "qwen2.5vl:3b"

# ============================================================
# 本地模型缓存路径配置
# ============================================================
# 说明：系统优先从以下路径加载本地模型，无需联网
# 如果路径不存在，请先运行以下命令下载模型：
#   python scripts/download_models.py --output-dir ./models
# ============================================================
LOCAL_MODEL_CACHE = {
    'clip_ViT': os.path.join(PROJECT_ROOT, 'models', 'clip_ViT'),
    'qwen_vl': None  # Qwen2.5-VL 可通过 Ollama 运行，无需本地下载
}

# 默认检索参数
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.8