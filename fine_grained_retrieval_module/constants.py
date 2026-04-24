"""
细粒度检索模块的常量定义
"""

# OpenAI CLIP模型配置
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

# Qwen2.5-VL模型配置
QWEN_VL_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_VL_OLLAMA_MODEL = "qwen2.5-vl:3b"

# 本地模型缓存路径配置（可选，设置为None则使用默认缓存路径）
LOCAL_MODEL_CACHE = {
    'clip_vit': 'C:/Users/21440/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41',
    'qwen_vl': None
}

# 默认检索参数
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.8
