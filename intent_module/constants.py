"""
意图识别模块的常量定义
中文数字映射、模型配置
"""

import os

# 意图识别 LLM（环境变量 INTENT_MODEL 可覆盖）
INTENT_MODEL_NAME = os.environ.get("INTENT_MODEL", "qwen3:8b")

# 中文数字映射
CHINESE_NUMBERS = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "两": 2
}