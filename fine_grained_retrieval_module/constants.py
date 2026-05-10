"""
细粒度检索模块的常量定义
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CLIP ViT 图像编码器路径（文本编码使用 shared/clip_encoder.py 中的 Chinese RoBERTa）
CLIP_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "clip_ViT")

# VL Ollama 模型配置（环境变量 VL_MODEL 可覆盖）
VL_OLLAMA_MODEL = os.environ.get("VL_MODEL", "qwen3-vl:8b")

# VL 精排并行 worker 数（默认串行=1；Ollama 单 GPU 下并行 HTTP 请求无性能收益，详见 README）
VL_PARALLEL_WORKERS = int(os.environ.get("VL_PARALLEL_WORKERS", "1"))

