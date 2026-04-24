"""
常规检索模块的常量定义
"""

# Taiyi-CLIP模型配置
TAIYI_CLIP_MODEL_NAME = "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese"

# OpenAI CLIP模型配置
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

# HuggingFace国内镜像源配置（推荐按顺序尝试）
HF_MIRROR_URLS = [
    "https://huggingface.co",          # 官方源
    "https://hf-mirror.com",           # 镜像1
]

# ModelScope镜像源配置（备用方案，国内更稳定）
MODELSCOPE_ENDPOINT = "https://api-inference.modelscope.cn"

# 下载超时设置（秒）
HF_DOWNLOAD_TIMEOUT = 120  # 增加到120秒

# 本地模型缓存路径配置（可选，设置为None则使用默认缓存路径）
# 示例: {"taiyi_clip": "D:/models/taiyi_clip", "clip_vit": "D:/models/clip_vit"}
LOCAL_MODEL_CACHE = {
    'taiyi_clip': 'C:/Users/21440/.cache/huggingface/hub/models--IDEA-CCNL--Taiyi-CLIP-Roberta-large-326M-Chinese/snapshots/2d54979689151036d4624e20e5f104cf73eadcf1',
    'clip_vit': 'C:/Users/21440/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
}

# 默认检索参数
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.5
