"""
常规检索模块的常量定义
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CLIP ViT 图像编码器路径（文本编码使用 shared/clip_encoder.py 中的 Chinese RoBERTa）
CLIP_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "clip_ViT")

# 离线索引磁盘缓存
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

